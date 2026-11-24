"""Vectorize mode: loops numpy should replace, the rewrites it proposes, and the gate."""

# pylint: disable=redefined-outer-name,use-implicit-booleaness-not-comparison
import ast
import textwrap

import numpy as np
import pytest

from pysayme import Config, check, load_config, report, vectorize
from pysayme.cli import main

HEAD = "import math, cmath\nimport numpy as np\n"

# (name, source, idiom, numpy target, args) -- each rewrite is run against its loop.
CASES = {
    "magnitude": (
        """
        def f(s: np.ndarray):
            out = []
            for v in s:
                out.append(math.hypot(v.real, v.imag))
            return out
        """,
        "MAP",
        "hypot",
        lambda r: (r.normal(size=300) + 1j * r.normal(size=300),),
    ),
    "dot": (
        """
        def f(x: np.ndarray, y: np.ndarray):
            acc = 0.0
            for i in range(len(x)):
                acc += x[i] * y[i]
            return acc
        """,
        "REDUCE",
        "dot",
        lambda r: (r.normal(size=200), r.normal(size=200)),
    ),
    "diff": (
        """
        def f(x: np.ndarray):
            out = []
            for i in range(len(x) - 1):
                out.append(x[i + 1] - x[i])
            return out
        """,
        "STENCIL",
        "diff",
        lambda r: (r.normal(size=200),),
    ),
    "cumsum": (
        """
        def f(x: np.ndarray):
            acc = 0.0
            out = []
            for v in x:
                acc += v
                out.append(acc)
            return out
        """,
        "SCAN",
        "cumsum",
        lambda r: (r.normal(size=200),),
    ),
    "recurrence": (
        """
        def f(x: np.ndarray):
            out = np.zeros(len(x))
            out[0] = x[0]
            for i in range(1, len(x)):
                out[i] = out[i - 1] + x[i]
            return out
        """,
        "SCAN",
        "cumsum",
        lambda r: (r.normal(size=200),),
    ),
    "select_index": (
        """
        def f(x: np.ndarray, t):
            return [i for i in range(len(x)) if x[i] > t]
        """,
        "SELECT",
        "flatnonzero",
        lambda r: (r.normal(size=200), 0.3),
    ),
    "select_value": (
        """
        def f(x: np.ndarray, t):
            out = []
            for v in x:
                if v > t:
                    out.append(v)
            return out
        """,
        "SELECT",
        "compress",
        lambda r: (r.normal(size=200), 0.3),
    ),
    "count": (
        """
        def f(x: np.ndarray, t):
            n = 0
            for v in x:
                if v > t:
                    n += 1
            return n
        """,
        "REDUCE",
        "count_nonzero",
        lambda r: (r.normal(size=200), 0.3),
    ),
    "argmax": (
        """
        def f(x: np.ndarray):
            best = x[0]
            at = 0
            for i in range(1, len(x)):
                if x[i] > best:
                    best = x[i]
                    at = i
            return at
        """,
        "ARGEXT",
        "argmax",
        lambda r: (r.normal(size=200),),
    ),
    "energy": (
        """
        def f(x):
            x = np.asarray(x)
            return sum(abs(v) ** 2 for v in x)
        """,
        "REDUCE",
        "sum",
        lambda r: (r.normal(size=200) + 1j * r.normal(size=200),),
    ),
    "phase": (
        """
        def f(x: np.ndarray):
            return np.array([math.atan2(v.imag, v.real) for v in x])
        """,
        "MAP",
        "arctan2",
        lambda r: (r.normal(size=200) + 1j * r.normal(size=200),),
    ),
    "where": (
        """
        def f(x: np.ndarray):
            return np.array([v if v > 0 else -v for v in x])
        """,
        "MAP",
        "where",
        lambda r: (r.normal(size=200),),
    ),
    "zip_dot": (
        """
        def f(x: np.ndarray, y: np.ndarray):
            return sum(a * b for a, b in zip(x, y))
        """,
        "REDUCE",
        "dot",
        lambda r: (r.normal(size=200), r.normal(size=200)),
    ),
    "builtin_sum": (
        """
        def f(x: np.ndarray):
            return sum(x)
        """,
        "DEMOTE",
        "sum",
        lambda r: (r.normal(size=200),),
    ),
}

QUIET = {
    "crc": """
        def f(data: np.ndarray):
            reg = 0xFFFF
            for b in data:
                reg ^= int(b) << 8
                for _ in range(8):
                    reg = (reg << 1) ^ 0x1021 if reg & 0x8000 else reg << 1
            return reg
        """,
    "lfsr": """
        def f(bits: np.ndarray):
            out = []
            state = 1
            for b in bits:
                state = ((state << 1) | int(b)) & 0xFF
                out.append(state)
            return out
        """,
    "iir": """
        def f(x: np.ndarray):
            out = []
            y = 0.0
            for v in x:
                y = 0.9 * y + v
                out.append(y)
            return out
        """,
    "small_range": """
        def f():
            out = []
            for i in range(8):
                out.append(math.sin(i))
            return out
        """,
    "small_array": """
        def f():
            t = np.array([1.0, 2.0, 3.0])
            return [math.sin(v) for v in t]
        """,
    "list_of_arrays": """
        def f(spans: list[np.ndarray], held: np.ndarray):
            return [s for s, k in zip(spans, held) if k]
        """,
    "plain_lists": """
        def f(words, sizes):
            out = []
            for i in range(len(words)):
                out.append(len(words[i]) * sizes[i])
            return sum(len(w) for w in words), out
        """,
    "unknown_call": """
        def f(x: np.ndarray):
            return [open(v) for v in x]
        """,
    "break": """
        def f(x: np.ndarray):
            for v in x:
                if v > 3:
                    return v
        """,
    "numba": """
        import numba

        @numba.njit
        def f(x: np.ndarray):
            acc = 0.0
            for i in range(len(x)):
                acc += x[i] * x[i]
            return acc
        """,
    "invariant_array": """
        def f(x: np.ndarray, w: np.ndarray):
            return [v * w for v in x]
        """,
}


def scan(source, **kw):
    return vectorize.scan(HEAD + textwrap.dedent(source), "m.py", Config(**kw))


@pytest.mark.parametrize("name", CASES)
def test_idiom_found_and_rewrite_equals_loop(name):
    source, idiom, target, args = CASES[name]
    found = scan(source)
    assert [(f.idiom, f.target) for f in found] == [(idiom, target)]
    ns = {}
    exec(HEAD + textwrap.dedent(source), ns)  # pylint: disable=exec-used
    values = args(np.random.default_rng(0))
    names = list(ns["f"].__code__.co_varnames[: len(values)])
    scope = {"np": np, **dict(zip(names, values))}
    want = ns["f"](*values)
    after = found[0].after.split("  #")[0]
    stmt = ast.parse(after).body[0]
    if isinstance(stmt, ast.Expr):
        got = eval(after, scope)  # pylint: disable=eval-used
    else:
        scope["acc"] = scope["n"] = 0
        first = values[0]
        scope["out"] = np.zeros(len(first), dtype=first.dtype)
        scope["out"][0] = first[0]
        exec(after, scope)  # pylint: disable=exec-used
        tgt = stmt.target if isinstance(stmt, ast.AugAssign) else stmt.targets[0]
        got = scope[tgt.id] if isinstance(tgt, ast.Name) else scope[tgt.value.id]
    np.testing.assert_allclose(
        np.asarray(got, dtype=complex), np.asarray(want, dtype=complex)
    )


@pytest.mark.parametrize("name", QUIET)
def test_legal_loops_stay_quiet(name):
    assert scan(QUIET[name]) == []


NESTS = {
    "dft": (
        """
        def f(x: np.ndarray):
            N = len(x)
            out = []
            for k in range(N):
                acc = 0j
                for n in range(len(x)):
                    acc += x[n] * cmath.exp(-2j * math.pi * k * n / N)
                out.append(acc)
            return out
        """,
        "sum",
        lambda r: (r.normal(size=40) + 1j * r.normal(size=40),),
    ),
    "outer_sin": (
        """
        def f(a: np.ndarray, b: np.ndarray):
            out = np.zeros((len(a), len(b)))
            for i in range(len(a)):
                for j in range(len(b)):
                    out[i, j] = math.sin(a[i] * b[j])
            return out
        """,
        "sin",
        lambda r: (r.normal(size=12), r.normal(size=9)),
    ),
    "chained_index": (
        """
        def f(a: np.ndarray, b: np.ndarray):
            out = np.zeros((len(a), len(b)))
            for i in range(len(a)):
                for j in range(len(b)):
                    out[i][j] = a[i] + b[j] * 2
            return out
        """,
        "add",
        lambda r: (r.normal(size=12), r.normal(size=9)),
    ),
    "three_axes": (
        """
        def f(n):
            out = np.zeros((n, n, n))
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        out[i, j, k] = i * 100 + j * 10 + k
            return out
        """,
        "add",
        lambda r: (6,),
    ),
    "row_reduce": (
        """
        def f(m: np.ndarray):
            out = np.zeros(m.shape[0])
            for i in range(m.shape[0]):
                acc = 0.0
                for j in range(m.shape[1]):
                    acc += m[i, j] * m[i, j]
                out[i] = acc
            return out
        """,
        "sum",
        lambda r: (r.normal(size=(7, 11)),),
    ),
    "offset_seed": (
        """
        def f(m: np.ndarray):
            out = np.zeros(m.shape[0])
            for i in range(m.shape[0]):
                acc = 1.5
                for j in range(m.shape[1]):
                    acc += m[i, j]
                out[i] = acc
            return out
        """,
        "sum",
        lambda r: (r.normal(size=(7, 11)),),
    ),
}


@pytest.mark.parametrize("name", NESTS)
def test_nested_loops_become_one_broadcast_expression(name):
    source, target, args = NESTS[name]
    (found,) = scan(source)
    assert (found.idiom, found.target) == ("OUTER", target)
    assert "builds an array of shape" in found.after
    ns = {}
    exec(HEAD + textwrap.dedent(source), ns)  # pylint: disable=exec-used
    values = args(np.random.default_rng(1))
    names = list(ns["f"].__code__.co_varnames[: len(values)])
    scope = {"np": np, **dict(zip(names, values))}
    want = np.asarray(ns["f"](*values))
    scope["out"] = np.zeros(want.shape, dtype=want.dtype)
    scope["N"] = (
        len(values[0]) if hasattr(values[0], "__len__") else 0
    )  # DFT: `N = len(x)`
    if "m" in scope:  # the reduction forms read the array's shape, as the loop did
        scope["m"] = values[0]
    exec(found.after.split("  #")[0], scope)  # pylint: disable=exec-used
    np.testing.assert_allclose(scope["out"], want)


def test_triangular_and_dependent_nests_are_left_alone():
    assert not scan("""
        def f(n):
            out = np.zeros((n, n))
            for i in range(n):
                for j in range(i):
                    out[i, j] = i + j
            return out
        """)


def test_nest_that_cannot_be_proved_stays_silent():
    assert not scan("""
        def f(x: np.ndarray, n):
            out = []
            for k in range(n):
                acc = 0.0
                for i in range(len(x)):
                    acc += x[i] * helper(k)
                out.append(acc)
            return out
        """)


def test_grow_and_demote():
    found = scan("""
        def f(n):
            a = np.zeros(0)
            for i in range(n):
                a = np.append(a, i)
            return a

        def g():
            a = np.random.rand(1000)
            return max(a)
        """)
    assert {(f.qualname, f.idiom) for f in found} == {("f", "GROW"), ("g", "DEMOTE")}


def test_helper_is_inlined_one_level():
    found = scan("""
        def gain(v):
            return math.sqrt(v * v + 1.0)

        def f(x: np.ndarray):
            return np.array([gain(v) for v in x])
        """)
    assert [(f.idiom, f.after) for f in found] == [("MAP", "np.sqrt(x * x + 1.0)")]


def test_python_call_per_element_is_likely_only(tmp_path):
    src = """
        def gate(n):
            share = n / 4
            return share * 2 if n else float("inf")

        def f(tones: np.ndarray):
            return np.array([gate(int(one)) for one in tones])
        """
    (found,) = scan(src)
    assert (found.idiom, found.tier) == ("MAPCALL", vectorize.LIKELY)
    (tmp_path / "m.py").write_text(HEAD + textwrap.dedent(src))
    files = [(tmp_path / "m.py", "m.py", "")]
    assert vectorize.find(files, Config()) == {}
    assert len(vectorize.find(files, Config(vectorize_confidence="likely"))) == 1


def test_cast_only_comprehension_wants_tolist():
    (found,) = scan("def f(x: np.ndarray):\n    return [int(v) for v in x]\n")
    assert (found.idiom, found.tier) == ("TOLIST", vectorize.LIKELY)


def test_no_numpy_import_no_findings():
    assert (
        vectorize.scan(
            "def f(x):\n    return sum(v * v for v in x)\n", "m.py", Config()
        )
        == []
    )


def test_min_trip():
    src = "def f():\n    return [math.sin(i) for i in range(%d)]\n"
    assert scan(src % 100) == []  # range alone proves nothing about an array
    body = "def f(x: np.ndarray):\n    return [v + 1 for v in x[:%d]]\n"
    assert len(scan(body % 100)) == 1


@pytest.fixture
def naive_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pysayme]\nroots = ["src", "tests"]\ndists = ["numpy"]\n'
        "[tool.pysayme.allow]\n"
        '"src/oracle.py:slow_energy -> numpy:sum" = "reference the fast path is tested against"\n'
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "oracle.py").write_text(
        HEAD
        + "def slow_energy(x: np.ndarray):\n    return sum(abs(v) ** 2 for v in x)\n"
    )
    (src / "fast.py").write_text(
        HEAD + "def slow(x: np.ndarray, y: np.ndarray):\n"
        "    acc = 0.0\n    for i in range(len(x)):\n        acc += x[i] * y[i]\n    return acc\n"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ref.py").write_text(
        HEAD + "def test_ref(x: np.ndarray):\n    return [math.sin(v) for v in x]\n"
    )
    return tmp_path


def test_gate_uses_the_allowlist(naive_project):
    cfg = load_config(naive_project / "pyproject.toml", progress=False, vectorize=True)
    found = vectorize.find(
        [
            (naive_project / p, p, "")
            for p in ("src/oracle.py", "src/fast.py", "tests/test_ref.py")
        ],
        cfg,
    )
    assert set(found) == {
        "src/oracle.py:slow_energy -> numpy:sum",
        "src/fast.py:slow -> numpy:dot",
    }  # the test file is excluded by default
    result = check(cfg, {k: str(v) for k, v in found.items()})
    assert set(result.new) == {"src/fast.py:slow -> numpy:dot"}
    assert result.stale == [] and result.ok is False
    fixed = check(cfg, {})
    assert fixed.stale == [
        "src/oracle.py:slow_energy -> numpy:sum"
    ]  # the list only shrinks


def test_cli_vectorize(naive_project, capsys):
    cfg = str(naive_project / "pyproject.toml")
    assert main(["vectorize", "--config", cfg]) == 1
    out = capsys.readouterr().out
    assert "src/fast.py:5 slow REDUCE" in out and "np.dot(x, y)" in out
    assert "slow_energy" not in out and "[tool.pysayme.allow]" in out
    assert main(["vectorize", "--config", cfg, "--all"]) == 1
    assert "slow_energy" in capsys.readouterr().out
    (naive_project / "src" / "fast.py").write_text(
        HEAD + "def slow(x, y):\n    return np.dot(x, y)\n"
    )
    assert main(["vectorize", "--config", cfg]) == 0


def test_report_gate_flag(naive_project):
    cfg = load_config(naive_project / "pyproject.toml", progress=False)
    assert not any("numpy:dot" in k for k in report(cfg))
    cfg.vectorize = True
    assert any("numpy:dot" in k for k in report(cfg))
