"""Advise mode: ranked refactoring advice for loops (never a gate)."""

# pylint: disable=redefined-outer-name,duplicate-code
import textwrap

import pytest

from pysayme import Config, advise
from pysayme.cli import main

HEAD = "import math\nimport numpy as np\nimport requests\n"

HEAVY = """
    def analyze(path):
        total = 0
        for line in open(path):
            for tok in line.split():
                total += len(tok)
        return total
"""

CASES = {
    "PARALLEL": HEAVY
    + """
    def per_file(paths):
        out = []
        for p in paths:
            out.append(analyze(p))
        return out
    """,
    "CONCURRENT": """
    def fetch_all(urls):
        got = []
        for u in urls:
            got.append(requests.get(u).text)
        return got
    """,
    "JIT": """
    def iir(x: np.ndarray):
        out = []
        y = 0.0
        for v in x:
            y = 0.9 * y + v
            out.append(y)
        return out
    """,
    "MEMBERSHIP": """
    def dedupe(items):
        seen = []
        out = []
        for x in items:
            if x not in seen:
                seen.append(x)
                out.append(x)
        return out
    """,
    "PAIRWISE": """
    def closest(points):
        best = 1e9
        for i in range(len(points)):
            for j in range(len(points)):
                best = min(best, math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]))
        return best
    """,
    "LIST_HEAD": """
    def drain(q):
        out = []
        for _ in range(1000):
            out.append(q.pop(0))
        return out
    """,
    "STRCAT": """
    def build(words):
        s = ""
        for w in words:
            s += w + ","
        return s
    """,
    "REPEATED": """
    def rescan(keys, table):
        out = []
        for k in keys:
            out.append(max(table) + k)
        return out
    """,
    "VECTORIZE": """
    def dot(x: np.ndarray, y: np.ndarray):
        acc = 0.0
        for i in range(len(x)):
            acc += x[i] * y[i]
        return acc
    """,
}

QUIET = {
    "shared_state": HEAVY
    + """
    class Acc:
        def go(self, files):
            for f in files:
                self.total += analyze(f)
    """,
    "small_literal": HEAVY
    + """
    def tiny():
        return [analyze(i) for i in range(5)]
    """,
    "cheap_body": """
    def f(items):
        out = []
        for v in items:
            out.append(v + 1)
        return out
    """,
    "early_exit": HEAVY
    + """
    def first(paths):
        for p in paths:
            if analyze(p):
                return p
    """,
    "set_membership": """
    def dedupe(items):
        seen = set()
        for x in items:
            if x not in seen:
                seen.add(x)
    """,
}


def run(source, **kw):
    cfg = Config(vectorize_min_trip=64)
    return advise.scan(HEAD + textwrap.dedent(source), "m.py", cfg, **kw)


@pytest.mark.parametrize("kind", CASES)
def test_each_kind_is_advised(kind):
    kinds = {a.kind for a in run(CASES[kind])}
    assert kind in kinds


@pytest.mark.parametrize("name", QUIET)
def test_safe_or_small_loops_stay_quiet(name):
    # `analyze` is the heavy callee fixture; only the loop around it is under test
    assert [a for a in run(QUIET[name]) if a.qualname != "analyze"] == []


def test_pairwise_replaces_the_jit_note_on_the_same_loop():
    found = run(CASES["PAIRWISE"])
    assert [a.kind for a in found] == ["PAIRWISE"]


def test_comprehension_variables_are_not_carried_state():
    found = run("""
        def f(items, table):
            out = []
            for it in items:
                out.append(sorted({e for e in table[it] if e > 0}))
                out.append(sum(analyze(e) for e in table[it]))
            return out
        """ + HEAVY)
    assert not [a for a in found if a.kind in ("SERIAL", "JIT")]


def test_serial_names_the_carried_variable():
    found = run(
        HEAVY + """
        def chain(paths):
            state = 0
            for p in paths:
                state = analyze(p) + state * 2
            return state
        """,
        serial_cost=1000,
    )
    serial = [a for a in found if a.kind == "SERIAL"]
    assert serial and "carried: state" in serial[0].why


def test_libs_hint_and_ranking():
    found = run(CASES["PARALLEL"], libs=["joblib"])
    top = max(found, key=lambda a: a.score)
    assert "joblib" in top.why
    cost = {a.kind: a.score for a in run(CASES["PAIRWISE"]) + run(CASES["STRCAT"])}
    assert cost["PAIRWISE"] > cost["STRCAT"]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pysayme]\nroots = ["src"]\ndists = ["numpy"]\n'
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(
        HEAD + textwrap.dedent(CASES["PAIRWISE"] + CASES["STRCAT"])
    )
    return tmp_path


def test_advise_ranks_and_filters(project):
    cfg_path = project / "pyproject.toml"
    from pysayme import load_config, sources  # pylint: disable=import-outside-toplevel

    cfg = load_config(cfg_path)
    files = sources.project_files(cfg.roots, cfg.base, cfg.exclude)
    got = advise.advise(files, cfg, top=0, min_score=0)
    assert [a.kind for a in got][:1] == ["PAIRWISE"]
    only = advise.advise(files, cfg, top=0, min_score=0, kinds=("STRCAT",))
    assert [a.kind for a in only] == ["STRCAT"]
    assert len(advise.advise(files, cfg, top=1, min_score=0)) == 1


def test_cli_advise_never_fails(project, capsys):
    cfg = str(project / "pyproject.toml")
    assert main(["advise", "--config", cfg, "--min-score", "0"]) == 0
    out = capsys.readouterr().out
    assert "  1." in out and "PAIRWISE" in out
    assert main(["advise", "--config", cfg, "--kind", "PARALLEL"]) == 0
    assert "nothing worth refactoring" in capsys.readouterr().out
