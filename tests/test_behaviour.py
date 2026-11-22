"""Behavioural fingerprinting end to end on a fake library and project."""

from types import SimpleNamespace

import pytest

from conftest import install, write_tree
from pysayme import behaviour, load_config, probe, report, runtime
from pysayme.cli import main

LIB = {
    "behavelib/__init__.py": """
        from ._impl import checksum, mean_a, mean_b, pick, ramp, square, unit

        __all__ = ["checksum", "mean_a", "mean_b", "pick", "ramp", "square", "unit"]
    """,
    "behavelib/_impl.py": '''
        import numpy as np


        def checksum(data: bytes) -> int:
            out = 0
            for i, b in enumerate(data):
                out = (out * 31 + b * (i + 1)) % 65521
            return out


        def ramp(n: int, reverse=False):
            r = np.arange(n, dtype=float) ** 2
            return r[::-1] if reverse else r


        def pick(x, mode="first"):
            """One end of `x`.

            Parameters
            ----------
            x : array_like
                Values.
            mode : {'first', 'last'}
                Which end.
            """
            return float(x[0] if mode == "first" else x[-1])


        def square(n: int) -> int:
            return n * n


        def unit(x: np.ndarray):
            return x / np.linalg.norm(x)


        def mean_a(x: np.ndarray) -> float:
            return float(np.mean(x))


        def mean_b(x: np.ndarray) -> float:
            return float(np.sum(x) / len(x))
    ''',
}

PROJECT = {
    "pyproject.toml": """
        [tool.pysayme]
        dists = ["behavelib"]
        roots = ["app"]
        self = false
        jobs = 2
        behaviour = true
    """,
    "app/__init__.py": "",
    "app/calc.py": """
        import functools

        import numpy as np

        import behavelib


        def adler_ish(data: bytes) -> int:
            step = lambda acc, pair: (acc * 31 + pair[0] * pair[1]) % 65521
            return functools.reduce(step, enumerate(data, 1), 0)


        def via_lib(data: bytes) -> int:
            return behavelib.checksum(data)


        def descending(n: int):
            return np.array([float(k * k) for k in range(n - 1, -1, -1)])


        def scaled_ramp(n: int):
            return 3.0 * np.arange(n) ** 2


        def avg(x: np.ndarray) -> float:
            return float(sum(x) / len(x))


        def lucky(n: int) -> int:
            return n * n if n in (3, 8, 5, 12) else -1


        def flat(x: np.ndarray) -> int:
            return 7


        def same(x: np.ndarray):
            return x.copy()


        def last(x):
            return float(x[len(x) - 1])


        def normalise(x: np.ndarray):
            return x / np.sqrt((x * x).sum())


        def named(text: str) -> str:
            return text


        class Holder:
            @staticmethod
            def sq(n: int) -> int:
                return n**2
    """,
}

KEYS = {
    "app/calc.py:adler_ish -> behavelib:checksum",
    "app/calc.py:descending -> behavelib:ramp",
    "app/calc.py:scaled_ramp -> behavelib:ramp",
    "app/calc.py:last -> behavelib:pick",
    "app/calc.py:normalise -> behavelib:unit",
    "app/calc.py:Holder.sq -> behavelib:square",
}


@pytest.fixture(name="proj")
def fixture_proj(tmp_path, monkeypatch):
    site = install(tmp_path / "site", "behavelib", LIB)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    return write_tree(tmp_path / "project", PROJECT)


def test_report(proj):
    cfg = load_config(proj / "pyproject.toml", progress=False)
    found = report(cfg)
    assert set(found) == KEYS
    assert found["app/calc.py:adler_ish -> behavelib:checksum"] == (
        "app/calc.py:8 adler_ish behaves like behavelib:checksum on the probe set"
    )
    descending = found["app/calc.py:descending -> behavelib:ramp"]
    assert "behavelib:ramp(reverse=True) on" in descending
    assert found["app/calc.py:scaled_ramp -> behavelib:ramp"].endswith(" up to scale")
    assert "pick(mode='last')" in found["app/calc.py:last -> behavelib:pick"]
    cfg.ambiguous = True
    extra = report(cfg)
    assert set(extra) - KEYS == {"app/calc.py:avg -> behavelib:mean_a"}
    assert extra["app/calc.py:avg -> behavelib:mean_a"].endswith("(one of 2 alike)")


def test_cli_flag(proj, capsys):
    text = (proj / "pyproject.toml").read_text().replace("behaviour = true", "")
    (proj / "pyproject.toml").write_text(text)
    cfg = str(proj / "pyproject.toml")
    assert main(["--config", cfg]) == 0
    assert main(["--config", cfg, "--behaviour", "--no-introspect"]) == 1
    assert len(capsys.readouterr().out.splitlines()) == len(KEYS)


def test_importable(tmp_path):
    root = write_tree(
        tmp_path, {"pkg/__init__.py": "", "pkg/sub/__init__.py": "", "pkg/sub/m.py": ""}
    )
    root = root.resolve()
    assert behaviour.importable(root / "pkg/sub/m.py") == (str(root), "pkg.sub.m")
    assert behaviour.importable(root / "pkg/sub/__init__.py") == (str(root), "pkg.sub")
    (root / "s.py").write_text("")
    assert behaviour.importable(root / "s.py") == (str(root), str(root / "s.py"))


class FakePool:  # pylint: disable=too-few-public-methods
    """Records the tasks it is given and fingerprints each as `fp`."""

    def __init__(self, fp):
        self.fp, self.tasks = fp, []

    def map(self, tasks, *_):
        self.tasks += tasks
        return [{(b, t): self.fp for b in task[1] for t in task[2]} for task in tasks]


def test_library_fingerprints_cached(cache):
    spec = probe.Spec((probe.UNKNOWN, frozenset({"int"})), ((), (("k", True),)))
    lone = probe.Spec((frozenset({"bytes"}),))
    idx = [
        SimpleNamespace(
            key="k1",
            callables=[runtime.Callable("m:f", spec), runtime.Callable("m:g", lone)],
        )
    ]
    fp = probe.Fingerprint((b"a",), (b"b",), True, True)
    demand = [("f8[]", "int"), ("int", "int"), ("int", "f8[]"), ("bytes",)]
    pool = FakePool(fp)
    got = behaviour.library_fingerprints(idx, demand, pool)
    assert {k[2] for k in got if k[0] == "m:f"} == {("f8[]", "int"), ("int", "int")}
    assert len(got) == 5 and len(pool.tasks) == 2
    assert (cache / "behaviour-k1.pkl").is_file()
    again = FakePool(None)
    assert behaviour.library_fingerprints(idx, demand, again) == got
    assert not again.tasks
    fresh = behaviour.library_fingerprints(idx, demand, FakePool(None), cache=False)
    assert set(fresh) == set(got) and set(fresh.values()) == {None}
