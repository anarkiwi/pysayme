"""Input kinds, probe inputs, output digests and in-process probe runs."""

import ast
import sys
import time
import typing

import numpy as np
import pytest

from conftest import write_tree
from pysayme import probe, runtime

F, C, B, I, R, Y = "f8[]", "c16[]", "bool[]", "int", "float", "bytes"
NONE = frozenset({"?"})


@pytest.mark.parametrize(
    "ann, strict, want",
    [
        ("bytes", True, {Y}),
        ("np.ndarray", True, {F}),
        ("npt.NDArray[np.complex128]", True, {C}),
        ("NDArray[np.bool_]", True, {B}),
        ("NDArray[np.int64]", True, NONE),
        ("int | None", True, {I}),
        ("Optional[float]", True, {R}),
        ("Union[bytes, np.ndarray]", True, {Y, F}),
        ("'np.ndarray'", True, {F}),
        ("list[float]", True, {F}),
        ("Sequence[str]", True, NONE),
        ("Annotated[int, 'n']", True, {I}),
        ("str", True, NONE),
        ("Phy", True, NONE),
        ("Phy", False, probe.UNKNOWN),
        ("Phy | int", False, {I}),
        ("Any", True, probe.UNKNOWN),
        ("_ArrayLikeFloat_co", False, {F}),
        ("f(x)", True, NONE),
        ("f(x)", False, probe.UNKNOWN),
        ("'not valid('", True, probe.UNKNOWN),
    ],
)
def test_annotation_kinds(ann, strict, want):
    node = ast.parse(ann, mode="eval").body
    assert probe.annotation_kinds(node, strict) == frozenset(want)


def test_annotation_objects():
    assert probe.annotation_kinds(int) == {I}
    assert probe.annotation_kinds(typing.Optional[bytes]) == {Y}
    assert probe.annotation_kinds("np.ndarray") == {F}
    assert probe.annotation_kinds(None) == probe.UNKNOWN


DOC = """Summary.

    Parameters
    ----------
    x, y : array_like
        Values.
    n : int, optional
    mode : {'wrap', 'reflect'}, optional
        Edges.
    *args : tuple
    z : array_like of bool
    w : array_like of complex
    s : str
    c : int or sequence of floats
    e :

    Returns
    -------
    out : ndarray
    """


def test_doc_types_and_kinds():
    types = probe.doc_types(DOC)
    assert set(types) == {"x", "y", "n", "mode", "args", "z", "w", "s", "c", "e"}
    assert probe.doc_kinds(types["x"]) == ({F}, ())
    assert probe.doc_kinds(types["n"]) == ({I}, ())
    assert probe.doc_kinds(types["mode"]) == (NONE, ("reflect", "wrap"))
    assert probe.doc_kinds(types["z"])[0] == {B}
    assert probe.doc_kinds(types["w"])[0] == {C}
    assert probe.doc_kinds(types["s"])[0] == NONE
    assert probe.doc_kinds(types["c"])[0] == {I, F}
    assert probe.doc_kinds(types["e"])[0] == probe.UNKNOWN
    assert probe.doc_kinds("{oops")[0] == NONE
    assert not probe.doc_kinds("{1, [2]}")[1]
    assert not probe.doc_types(None) and not probe.doc_types("Parameters\n---")


def sample(x, n: int, *rest, flag=True, mode="wrap", scale=1.0, **kw):
    """Parameters
    ----------
    x : array_like
    mode : {'wrap', 'reflect'}
    """
    return x, n, flag, mode, rest, scale, kw


def gen(x):
    yield x


async def coro(x):
    return x


def test_spec():
    s = probe.spec(sample)
    assert s.kinds == ({F}, {I})
    assert s.bindings == ((), (("flag", False),), (("mode", "reflect"),))
    assert probe.spec(np.sin) == probe.Spec(({F},))
    assert probe.spec(np.add).kinds == ({F}, {F})
    for bad in (gen, coro, dict, lambda: 0, lambda a, b, c, d: 0, getattr):
        assert probe.spec(bad) is None
    assert probe.spec(lambda a, *, b: 0) is None
    assert probe.spec(lambda a, /, b=1: 0).bindings == ((),)
    assert probe.spec(np.frompyfunc(sample, 4, 1)) is None


def test_stub_kinds(tmp_path, monkeypatch):
    write_tree(
        tmp_path,
        {
            "stubbed.py": "def f(a, b):\n    return a\n",
            "stubbed.pyi": "def f(a: bytes, b: Mystery) -> int: ...\n"
            "def f(a: int, b) -> int: ...\n",
            "broken.py": "def g(a):\n    return a\n",
            "broken.pyi": "def g(a: ) ->",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import broken  # pylint: disable=import-outside-toplevel,import-error
    import stubbed  # pylint: disable=import-outside-toplevel,import-error

    assert probe.spec(stubbed.f, "stubbed").kinds == ({Y, I}, probe.UNKNOWN)
    assert probe.spec(broken.g, "nowhere").kinds == (probe.UNKNOWN,)
    del sys.modules["stubbed"], sys.modules["broken"]


def test_kind_tuples():
    assert probe.kind_tuples((probe.UNKNOWN, {Y})) == [(F, Y), (I, Y)]
    assert not probe.kind_tuples((NONE,))


def test_project_specs():
    tree = ast.parse("""
import pytest
def one(a: bytes, b=1): pass
def kw(a, *, b): pass
def none(): pass
def test_x(a): pass
@pytest.fixture
def fix(a): pass
class K:
    @staticmethod
    def s(a: int, b, c): pass
    def m(self, a): pass
""")
    got = list(probe.project_specs(tree))
    assert got == [
        ("one", 3, ({Y},)),
        ("K.s", 11, ({I}, probe.UNKNOWN, probe.UNKNOWN)),
    ]


def test_make_is_seeded():
    for kind in probe.KINDS:
        a = probe.make(kind, 0, 1, 2)
        b = probe.make(kind, 0, 1, 2)
        assert probe.digest(a) == probe.digest(b)
        assert probe.digest(a) != probe.digest(probe.make(kind, 1, 1, 2))
    assert len(probe.make(Y, 0, 0, 3)) == 17 and probe.make(I, 0, 1, 0) == 8
    assert (
        probe.make(F, 0, 0, 1).shape == (64,) and probe.make(B, 0, 0, 0).dtype == bool
    )
    assert probe.make(C, 0, 2, 0).dtype == np.complex128


def test_digest():
    x = np.random.default_rng(0).standard_normal(9)
    d = probe.digest
    assert d(x) == d(x * (1 + 1e-14)) and d(x) != d(x * 1.001)
    assert d(np.arange(3, dtype=np.uint8)) == d(np.arange(3))
    assert d(np.arange(3)) != d(np.arange(3.0))
    assert d(3) == d(np.int64(3)) and d(2.5) == d(np.float32(2.5)) and d(True)[1]
    assert d(x, True) == d(-x * -4.0, True) and d(x) != d(2 * x)
    assert d(np.zeros(3), True) == d(np.zeros(3))
    assert d(np.array([1.0, np.nan, np.inf])) != d(np.array([1.0, np.inf, np.nan]))
    assert d(np.zeros(0))[1] and d(x + 1j * x)[1]
    assert d({1: "a", 2: b"b"}) == d({2: b"b", 1: "a"}) and d({1, 2}) == d({2, 1})
    assert d([1, (2,)]) == d((1, [2])) and d("ab") != d(b"ab")
    assert (
        d(None)[1]
        and not d(object())[1]
        and d([1, object()])[1]
        and not d([object()])[1]
    )
    assert not d(np.array(["s"]))[1] and not d(probe.Raised("E"))[1]
    assert not d(b"x" * (probe.OUTPUT_CAP + 1))[1]
    assert not d(np.zeros(probe.OUTPUT_CAP + 1))[1]


def test_fingerprint_gates():
    ins = [[np.arange(n)] for n in (1, 2, 3, 4)]
    fp = probe.fingerprint([0, 1, 4, 9], ins)
    assert fp.exact_ok and fp.scale_ok
    assert not probe.fingerprint([5, 5, 5, 5], ins).exact_ok
    same = probe.fingerprint([a[0] for a in ins], ins)
    assert not same.exact_ok
    doubled = probe.fingerprint([2 * a[0] for a in ins], ins)
    assert doubled.exact_ok and not doubled.scale_ok
    raised = [probe.Raised("ValueError")] * 2 + [probe.Raised("TypeError")] * 2
    assert not probe.fingerprint(raised, ins).exact_ok
    arrays = [[probe.make(F, 0, 0, i)] for i in range(probe.N_PROBES)]
    assert probe.fingerprint([np.cumsum(a[0]) for a in arrays], arrays).exact_ok
    for fn in (len, lambda x: [x], lambda x: (x, np.ones(len(x)))):
        assert not probe.fingerprint([fn(a[0]) for a in arrays], arrays).exact_ok
    with np.errstate(divide="ignore"):
        inf = [np.where(a[0] > 0, 1 / a[0], -np.inf) for a in arrays]
    other = [np.where(a[0] > 0, a[0], -np.inf) for a in arrays]
    assert (
        probe.fingerprint(inf, arrays).scale != probe.fingerprint(other, arrays).scale
    )


def slow(n):
    time.sleep(n / 100)
    return n + 1


def test_run(tmp_path, monkeypatch):
    here = (None, __name__, "slow")
    got = probe.run((here, ((),), ((I,),), 0))
    assert got[(), (I,)].exact_ok
    write_tree(tmp_path, {"loose.py": "def rev(b):\n    return b[::-1] + b'!'\n"})
    loose = (str(tmp_path), str(tmp_path / "loose.py"), "rev")
    fps = probe.run((loose, ((),), ((Y,), (I,)), 1))
    assert fps[(), (Y,)].exact_ok and not fps[(), (I,)].exact_ok
    assert probe.run((loose, ((),), ((Y,),), 1)) == {((), (Y,)): fps[(), (Y,)]}
    assert probe.run(((None, __name__, "missing"), ((),), ((I,),), 0)) is None
    monkeypatch.setattr(probe, "CALL_TIMEOUT", 0.01)
    assert probe.run((here, ((),), ((I,),), 0)) is None


def test_watchdog_ends_held_calls(tmp_path, monkeypatch):
    write_tree(
        tmp_path,
        {
            "held.py": "import signal, time\n"
            "def hold(n):\n"
            "    signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGALRM])\n"
            "    time.sleep(60)\n"
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    task = ((None, "held", "hold"), ((),), ((I,),), 0)
    start = time.monotonic()
    with runtime.Workers(probe.run, 1) as pool:
        assert pool.map([task], 300, progress=False) == [None]
    assert time.monotonic() - start < 30
