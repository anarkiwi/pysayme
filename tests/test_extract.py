"""AST normalization, definitions, constants and module records."""

import ast
from textwrap import dedent

import numpy as np
import pytest

from pysayme import extract

BASE = """
def f(a, b):
    '''Doc.'''
    out = []
    for x in a:
        if x.size > b:
            out.append(x.size * 2)
    return out, "tag"
"""


def sh(src):
    return extract.shingles(ast.parse(dedent(src)).body[0])


def lit(src):
    return extract.literal(ast.parse(src, mode="eval").body)


def test_shingle_hashes_deterministic_sorted_unique():
    toks = ["a", "b", "c", "d", "a", "b", "c", "d", "e"]
    h = extract.shingle_hashes(toks)
    assert h.dtype == np.uint64 and len(h) == 5
    assert np.all(np.diff(h.astype(np.float64)) > 0)
    np.testing.assert_array_equal(h, extract.shingle_hashes(list(toks)))
    assert len(extract.shingle_hashes(toks[:3])) == 0


@pytest.mark.parametrize(
    "src, same",
    [
        (
            BASE.replace("out", "acc")
            .replace("x", "item")
            .replace("(a, b)", "(p, q)")
            .replace("in a", "in p")
            .replace("> b", "> q"),
            True,
        ),
        (BASE.replace("'''Doc.'''", "'''Something else entirely.'''"), True),
        (BASE.replace("    '''Doc.'''\n", ""), True),
        (BASE.replace(".size", ".length"), False),
        (BASE.replace("* 2", "* 3"), False),
        (BASE.replace('"tag"', '"other"'), False),
        (BASE.replace("[]", "list()"), False),
    ],
)
def test_shingles_rename_invariant(src, same):
    assert np.array_equal(sh(src), sh(BASE)) is same


def test_long_strings_collapse():
    a = BASE.replace('"tag"', '"' + "x" * 30 + '"')
    b = BASE.replace('"tag"', '"' + "y" * 30 + '"')
    np.testing.assert_array_equal(sh(a), sh(b))


def test_walk_defs_public_flags():
    tree = ast.parse(dedent("""
            def pub():
                def inner(): pass
            def _priv():
                pass
            class C:
                def m(self): pass
                def _h(self): pass
                class D:
                    def deep(self): pass
            class _K:
                def m(self): pass
            async def co(): pass
            """))
    got = {q: p for q, _, p in extract.walk_defs(tree)}
    assert got == {
        "pub": True,
        "pub.inner": False,
        "_priv": False,
        "C": True,
        "C.m": True,
        "C._h": False,
        "C.D": True,
        "C.D.deep": True,
        "_K": False,
        "_K.m": False,
        "co": True,
    }


def test_functions_min_shingles_and_module():
    tree = ast.parse(dedent(BASE) + "\ndef tiny():\n    return 1\n")
    fs = extract.functions(tree, "p.py", "pkg.mod")
    assert [(f.qualname, f.where()) for f in fs] == [("pkg.mod:f", "p.py:2")]
    assert [f.qualname for f in extract.functions(tree, "p.py", min_shingles=0)] == [
        "f",
        "tiny",
    ]


def test_parse_failure(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def (:\n")
    assert extract.parse(bad) is None
    assert extract.parse(tmp_path / "missing.py") is None


def test_constants():
    tree = ast.parse(dedent("""
            A, B = "alpha", ("x", "y")
            C: int = 4096
            D = frozenset(["q", "p", "r"])
            E = {"b": 2, "a": 1}
            F = [3, 1, 2]
            G = {"z", "y"}
            T = True
            N = None
            ONE = 1
            S = "ab"
            EMPTY = ()
            __version__ = "1.2.3"
            X = some_call()
            Y = 2 ** 10
            Z: int
            a.b = 5
            P, Q = R = 7, 8
            """))
    got = {c.qualname: c.value for c in extract.constants(tree, "m.py", "m")}
    assert got == {
        "m:A": "alpha",
        "m:B": ("x", "y"),
        "m:C": 4096,
        "m:D": ("p", "q", "r"),
        "m:E": {"b": 2, "a": 1},
        "m:F": (1, 2, 3),
        "m:G": ("y", "z"),
        "m:P": 7,
        "m:Q": 8,
        "m:R": (7, 8),
    }


def test_literal_errors():
    with pytest.raises(ValueError):
        lit("f(x)")
    assert lit("tuple([2, 1])") == (1, 2)


@pytest.mark.parametrize(
    "src, module, package, want",
    [
        (
            "from ._core import a, b as c\nfrom .sub.x import d\n",
            "pkg",
            True,
            [("pkg._core", "a", "a"), ("pkg._core", "b", "c"), ("pkg.sub.x", "d", "d")],
        ),
        (
            "from ._core import a\nfrom .. import b\n",
            "pkg.sub.mod",
            False,
            [("pkg.sub._core", "a", "a"), ("pkg", "b", "b")],
        ),
        ("from os.path import join\n", "m", False, [("os.path", "join", "join")]),
        ("from ... import a\n", "pkg.mod", False, []),
        (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from ._t import T\n"
            "try:\n    from ._c import fast\nexcept ImportError:\n"
            "    from ._p import fast\n"
            "else:\n    from ._d import d\nfinally:\n    from ._e import e\n",
            "pkg",
            True,
            [
                ("typing", "TYPE_CHECKING", "TYPE_CHECKING"),
                ("pkg._t", "T", "T"),
                ("pkg._c", "fast", "fast"),
                ("pkg._p", "fast", "fast"),
                ("pkg._d", "d", "d"),
                ("pkg._e", "e", "e"),
            ],
        ),
    ],
)
def test_module_info_imports(src, module, package, want):
    assert list(extract.module_info(ast.parse(src), module, package).imports) == want


def info(src, module="pkg", package=True):
    return extract.module_info(ast.parse(dedent(src)), module, package)


def test_module_info_stars_and_bound():
    m = info("""
        import os.path
        import numpy as np
        from . import _sub
        from ._a import *
        from .b import *
        def f(): pass
        class C: pass
        X, _y = 1, 2
        Z: int = 3
        def _hidden(): pass
        """)
    assert m.name == "pkg" and m.stars == ("pkg._a", "pkg.b")
    assert m.bound == {"os", "np", "f", "C", "X", "Z"}
    assert m.all_names is None and m.all_from == ()


@pytest.mark.parametrize(
    "src, names, mods",
    [
        ("__all__ = ['a', 'b']", {"a", "b"}, ()),
        ("__all__: list = ('a',)\n__all__.append('b')", {"a", "b"}, ()),
        (
            "from . import _x, _y\n__all__ = _x.__all__.copy()\n"
            "__all__ += _y.__all__\n__all__.extend(['z'])",
            {"z"},
            ("pkg._x", "pkg._y"),
        ),
        (
            "import pkg._q as q\nimport pkg.r\n"
            "__all__ = ['a', *q.__all__] + list(pkg.r.__all__)",
            {"a"},
            ("pkg._q", "pkg.r"),
        ),
        ("__all__ = sorted({'b', 'a'})\n__all__.remove('a')", {"a", "b"}, ()),
    ],
)
def test_module_info_all(src, names, mods):
    m = info(src)
    assert (m.all_names, m.all_from) == (names, mods)


@pytest.mark.parametrize(
    "src",
    [
        "__all__ = names()",
        "__all__ = [s for s in dir()]",
        "__all__ = ['a', f()]",
        "__all__ = [*g()]",
        "__all__ = ['a'] + h()",
        "__all__ = other.__all__",
        "__all__ = f().__all__",
        "__all__: list",
        "__all__ = ['a']\n__all__ += k()",
        "__getattr__, __all__ = attach(__name__)",
        "__all__ = ['a']\n__all__.extend(z)",
    ],
)
def test_module_info_all_unknown(src):
    m = info(src)
    assert (m.all_names, m.all_from) == (None, ())


def test_module_info_all_ignores_other_calls():
    m = info("__all__ = ['a']\nprint(1)\nx.append('b')\n__all__.sort()\nf(1)")
    assert m.all_names == {"a"}


def test_private_module():
    assert extract.private_module("a._b.c") and not extract.private_module("a.b")


def test_registered_defs_are_not_public():
    tree = ast.parse(dedent("""
        @overload(np.sum)
        def np_sum(a):
            pass
        @registry.lower_cast(types.Integer, types.Float)
        def cast(context, builder, fromty, toty, val):
            pass
        @intrinsic
        def fast(typingctx, x):
            pass
        @infer_global(len)
        class LenTemplate:
            def generic(self):
                pass
        @register_jitable
        def helper(x):
            pass
        @functools.lru_cache()
        def cached():
            pass
        """))
    got = {q: p for q, _, p in extract.walk_defs(tree)}
    assert got == {
        "np_sum": False,
        "cast": False,
        "fast": False,
        "LenTemplate": False,
        "LenTemplate.generic": False,
        "helper": True,
        "cached": True,
    }


def test_wrapped_set_is_order_insensitive():
    assert lit("frozenset(('q', 'p', 'r'))") == lit("{'r', 'q', 'p'}")


def free(src):
    return extract.free_names(ast.parse(dedent(src)).body[0])


def test_free_names_ignore_locals_and_keep_callees():
    base = free("def f(a, b):\n    t = a + b\n    return np.sum([x for x in t])\n")
    renamed = free("def g(p, q):\n    u = p + q\n    return np.sum([y for y in u])\n")
    other = free("def h(a, b):\n    t = a + b\n    return _helper([x for x in t])\n")
    nested = free("def k(a):\n    def inner():\n        return a\n    return inner()\n")
    assert base == renamed != other
    assert nested == free("def m(z):\n    return z\n")
