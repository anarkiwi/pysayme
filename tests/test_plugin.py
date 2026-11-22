"""The pytest plugin's gate item, run in a nested pytest session."""

import pytest

from conftest import KEYS, PROJECT, write_tree

pytestmark = pytest.mark.usefixtures("site")


@pytest.fixture(name="proj")
def fixture_proj(pytester, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(pytester.path.parent))
    write_tree(pytester.path, PROJECT)
    pytester.makepyfile(test_ok="def test_ok():\n    assert True\n")
    return pytester


def allow(pytester, entries):
    path = pytester.path / "pyproject.toml"
    lines = "".join(f'"{k}" = "{v}"\n' for k, v in entries.items())
    path.write_text(path.read_text() + "[tool.pysayme.allow]\n" + lines)


def test_gate_fails_listing_duplicates(proj):
    res = proj.runpytest()
    res.assert_outcomes(passed=1, failed=1)
    res.stdout.fnmatch_lines(
        [
            "pyproject.toml F*",
            "*proj/mod.py:15 mysum duplicates fakelib:accumulate*",
            "*Each line above is project code*",
            "FAILED pyproject.toml::pysayme - *",
        ]
    )
    res.stdout.no_fnmatch_line("*grid_of*")
    res.stdout.no_fnmatch_line("*Traceback*")


def test_gate_passes_when_allowlisted(proj):
    allow(proj, {k: "deliberate" for k in KEYS})
    proj.runpytest().assert_outcomes(passed=2)


def test_gate_stale_and_unreasoned(proj):
    allow(proj, {**{k: "deliberate" for k in KEYS}, "gone.py:f -> x:y": " "})
    res = proj.runpytest()
    res.assert_outcomes(passed=1, failed=1)
    res.stdout.fnmatch_lines(
        ["*no longer match:*", "*gone.py:f -> x:y*", "*need a reason*"]
    )


@pytest.mark.parametrize(
    "args, gated",
    [
        (["test_ok.py"], False),
        (["test_ok.py", "--pysayme"], True),
        (["--no-pysayme"], False),
        (["--pysayme", "--no-pysayme"], False),
        (["-m", "not pysayme"], False),
        (["-m", "pysayme"], True),
    ],
)
def test_gate_selection(proj, args, gated):
    res = proj.runpytest(*args)
    outcomes = res.parseoutcomes()
    assert outcomes.get("failed", 0) == int(gated)
    assert outcomes.get("passed", 0) == int(args != ["-m", "pysayme"])


def test_no_section_no_gate(pytester):
    pytester.makepyfile(test_ok="def test_ok():\n    assert True\n")
    pytester.makepyprojecttoml("[project]\nname = 'x'\n")
    pytester.runpytest().assert_outcomes(passed=1)


def test_gate_runs_once_under_xdist(proj):
    res = proj.runpytest_subprocess("-n", "2", "-rf")
    res.assert_outcomes(passed=1, failed=1)
    res.stdout.fnmatch_lines(["*2 workers*"])
    assert sum("FAILED pyproject.toml::pysayme" in ln for ln in res.outlines) == 1
