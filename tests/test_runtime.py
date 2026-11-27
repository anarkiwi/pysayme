"""Runtime exports: object records, module exports, child isolation, merging."""

# pylint: disable=too-few-public-methods

import functools
import math
import os
import time
import types

import numpy as np
import pytest

from conftest import write_tree
from pysayme import probe, runtime


def plain(a, b):
    return a + b


@functools.wraps(plain)
def wrapper(*a):
    return plain(*a)


class Lying:
    """A class whose ``__module__`` names another module."""

    def method(self):
        return self


Lying.__module__ = "elsewhere"


class Calls:
    """A Python callable instance."""

    def __call__(self):
        return 1


class Empty:
    """A class with no Python methods."""


def test_describe_python_objects():
    here = os.path.realpath(__file__)
    assert runtime.describe(wrapper) == ("def", here, "plain")
    assert runtime.describe(Lying().method) == ("def", here, "Lying.method")
    assert runtime.describe(Lying) == ("class", "elsewhere", "Lying", here, "Lying")
    assert runtime.describe(Empty)[3:] == (None, None)
    for obj in (Calls(), functools.partial(plain, 1), math, 3, "s"):
        assert runtime.describe(obj) is None


def test_describe_compiled():
    kind, owner, name, typ, sig, doc = runtime.describe(np.sin)
    assert (kind, owner, name, typ) == ("compiled", "numpy", "sin", "ufunc")
    assert sig.startswith("(x, /") and doc.startswith("sin(")
    assert runtime.describe(math.sqrt)[1:4] == (
        "math",
        "sqrt",
        "builtin_function_or_method",
    )
    assert runtime.describe(np.add.reduce)[1] is None
    assert runtime.describe(dict.fromkeys)[4] == "(iterable, value=None, /)"
    assert runtime.describe(getattr)[4] is None
    no_doc = runtime.describe(np.frompyfunc(plain, 2, 1))
    assert no_doc[0] == "compiled" and isinstance(no_doc[5], str)


def test_exports():
    mod = types.ModuleType("m")
    mod.plain, mod._hidden, mod.n = plain, plain, 1

    def broken(name):
        raise RuntimeError(name)

    want = ("plain", runtime.describe(plain), probe.spec(plain))
    assert runtime.exports(mod) == [want]
    mod.__all__ = ["n", "_hidden", "gone", 7, "Lying"]
    mod.__getattr__ = broken
    mod.Lying = Lying
    assert runtime.exports(mod) == [
        ("_hidden", *want[1:]),
        ("Lying", runtime.describe(Lying), None),
    ]


def test_spec_errors(monkeypatch):
    def boom(*_):
        raise RuntimeError

    monkeypatch.setattr(probe, "spec", boom)
    assert runtime.exports(types.SimpleNamespace(__name__="m", f=plain)) == [
        ("f", runtime.describe(plain), None)
    ]


@pytest.fixture(name="rt")
def fixture_rt(tmp_path, monkeypatch):
    path = write_tree(
        tmp_path / "rt",
        {
            "rtmods/__init__.py": "def ok(x):\n    return x\n",
            "rtmods/boom.py": "import os\nos._exit(3)\n",
            "rtmods/hang.py": "import time\ntime.sleep(60)\n",
            "rtmods/bad.py": "raise ImportError('no')\n",
            "rtmods/noisy.py": "print('noise')\nimport sys\nprint('x', file=sys.stderr)\n",
            "rtmods/hog.py": "import time\nheld = b'\\x01' * (1 << 30)\ntime.sleep(60)\n",
        },
    )
    monkeypatch.syspath_prepend(str(path))
    return path


def test_collect_isolates_crashes_and_hangs(rt):
    mods = ["rtmods.boom", "rtmods", "rtmods.hang", "rtmods.bad", "rtmods.noisy"]
    got = runtime.collect(mods, timeout=5, progress=False)
    assert set(got) == {"rtmods", "rtmods.bad", "rtmods.noisy"}
    here = os.path.realpath(rt / "rtmods/__init__.py")
    assert got["rtmods"] == [("ok", ("def", here, "ok"), probe.spec(rt_ok))]
    assert got["rtmods.bad"] == []


def rt_ok(x):
    return x


@pytest.mark.usefixtures("rt")
def test_workers_isolate_and_reuse():
    tasks = ["rtmods", "rtmods.boom", "rtmods.noisy", "rtmods.hang", "rtmods"]
    with runtime.Workers(runtime.module_exports, 2) as pool:
        got = pool.map(tasks, lambda t: 5, progress=False)
        assert [g is None for g in got] == [False, True, False, True, False]
        assert pool.map(["rtmods"], 5, progress=False)[0] == got[0]
        assert not pool.map([], 5, progress=False)


@pytest.mark.usefixtures("rt")
def test_workers_end_a_child_past_its_memory():
    with runtime.Workers(runtime.module_exports, 1, memory=512 << 20) as pool:
        start = time.monotonic()
        got = pool.map(["rtmods.hog", "rtmods"], 30, progress=False)
        assert got[0] is None and got[1] is not None
        assert time.monotonic() - start < 30


def test_memory_reading(tmp_path):
    assert runtime.resident(os.getpid()) > 0
    assert runtime.resident(2**22 + 1) == 0
    assert (
        0
        < runtime.memory_total()
        <= os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    )
    (tmp_path / "max").write_text("max\n")
    (tmp_path / "cap").write_text("1048576\n")
    limit = runtime._limit_bytes  # pylint: disable=protected-access
    assert limit(tmp_path / "max") is None
    assert limit(tmp_path / "cap") == 1 << 20
    assert limit(tmp_path / "missing") is None
    assert runtime.Workers(plain, 3).memory == runtime.memory_total() // 4


def test_introspect_merges_records(monkeypatch, tmp_path):
    impl, pub = tmp_path / "_impl.py", tmp_path / "pub.py"
    entries = [
        (impl, "d/_impl.py", "pkg._impl"),
        (pub, "d/pub.py", "pkg.pub"),
        (tmp_path / "pub.pyi", "d/pub.pyi", "pkg.pub"),
        (tmp_path / "x.py", "d/x.py", "pkg"),
    ]
    impl_path = os.path.realpath(impl)
    other = ("compiled", "other", "f", "ufunc", None, "")
    canned = {
        "pkg": [
            ("f", ("def", impl_path, "f")),
            ("g", ("def", impl_path, "renamed")),
            ("h", ("def", "/elsewhere.py", "h")),
            ("C", ("class", "pkg", "C", impl_path, "C")),
            ("D", ("class", "pkg._impl", "D", None, None)),
            ("E", ("class", "nowhere", "E", None, None)),
            ("sin", ("compiled", "pkg", "sin", "ufunc", "(x)", "sine")),
            ("sqrt", ("compiled", "math", "sqrt", "builtin", None, "")),
            ("g2", ("compiled", None, "g2", "builtin", None, "")),
            ("f2", other),
        ],
        "pkg.pub": [
            ("sin", ("compiled", "pkg", "sin", "ufunc", "(x)", "sine")),
            ("f", ("def", impl_path, "f")),
            ("x", ("def", os.path.realpath(pub), "x")),
        ],
    }

    spec = probe.Spec((probe.UNKNOWN,))
    canned = {
        m: [(n, r, spec if r[0] in ("def", "compiled") else None) for n, r in items]
        for m, items in canned.items()
    }

    def collect(modules, timeout, desc, progress, jobs):
        assert modules == ["pkg", "pkg.pub"] and timeout == runtime.IMPORT_TIMEOUT
        assert (desc, progress, jobs) == ("import", True, 2)
        return canned

    monkeypatch.setattr(runtime, "collect", collect)
    exports, compiled, probed = runtime.introspect(entries, jobs=2)
    assert exports == {
        ("pkg._impl", "f"): {"pkg", "pkg.pub"},
        ("pkg._impl", "C"): {"pkg"},
        ("pkg._impl", "D"): {"pkg"},
    }
    assert compiled == [
        runtime.Compiled("pkg:g2", "builtin", None, ""),
        runtime.Compiled("pkg:sin", "ufunc", "(x)", "sine"),
    ]
    assert probed == [
        runtime.Callable("pkg.pub:x", spec),
        runtime.Callable("pkg:f", spec),
        runtime.Callable("pkg:g", spec),
        runtime.Callable("pkg:g2", spec),
        runtime.Callable("pkg:sin", spec),
    ]
