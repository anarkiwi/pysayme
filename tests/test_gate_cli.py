"""The gate end to end on the fixture project, and the command line."""

import pytest

from conftest import KEYS, UNRELATED
from pysayme import Config, check, load_config, report
from pysayme.cli import main


def test_report_on_project(project):
    cfg = load_config(project / "pyproject.toml", progress=False)
    found = report(cfg)
    assert set(found) == KEYS
    assert found["proj/mod.py:mysum -> fakelib:accumulate"].startswith(
        "proj/mod.py:15 mysum duplicates fakelib:accumulate"
    )
    assert " inlines " in found["proj/mod.py:pipeline -> fakelib:transform"]
    assert "redefines fakelib:COLORS" in found["proj/mod.py:PALETTE -> fakelib:COLORS"]
    cfg.private = True
    assert set(report(cfg)) - KEYS == {"proj/mod.py:grid_of -> fakelib._hidden:secret"}


def test_report_self(project, capsys):
    copy = UNRELATED.replace("def unrelated(", "def elsewhere(")
    (project / "proj" / "other.py").write_text(copy)
    cfg = load_config(project / "pyproject.toml", progress=False)
    key = "proj/mod.py:unrelated -> proj/other.py:elsewhere"
    found = report(cfg)
    assert set(found) - KEYS == {key}
    assert found[key] == (
        "proj/mod.py:1 unrelated duplicates proj/other.py:2 elsewhere score=1.00"
    )
    assert "call one from the" in str(check(cfg, found))
    cfg.self_join = False
    assert set(report(cfg)) == KEYS
    args = ["--config", str(project / "pyproject.toml"), "--no-self"]
    assert main(args) == 1
    assert "elsewhere" not in capsys.readouterr().out


def test_check():
    found = {"a": "A dup", "b": "B dup"}
    res = check(Config(allow={"a": "why", "gone": "old", "c": " "}), found)
    assert (res.new, res.stale, res.unreasoned) == (
        {"b": "B dup"},
        ["c", "gone"],
        ["c"],
    )
    assert not res.ok
    text = str(res)
    assert text.startswith("B dup\n\nEach line above")
    assert "no longer match:\n  c\n  gone" in text and "need a reason:\n  c" in text
    assert "A dup" not in text
    ok = check(Config(allow={"a": "why", "b": "ok"}), found)
    assert ok.ok and str(ok) == ""
    stale = str(check(Config(allow={"x": "r"}), {}))
    assert stale.startswith("remove from") and "Each line" not in stale


def test_cli_check(project, capsys):
    cfg = str(project / "pyproject.toml")
    assert main(["--config", cfg]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == len(KEYS) and out == sorted(out)
    text = (project / "pyproject.toml").read_text() + "[tool.pysayme.allow]\n"
    allowed = text + "".join(f'"{k}" = "deliberate"\n' for k in sorted(KEYS))
    (project / "pyproject.toml").write_text(allowed)
    assert main(["check", "--config", cfg]) == 0
    assert capsys.readouterr().out == ""
    assert main(["--config", cfg, "--all", "--no-cache"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == len(KEYS)
    (project / "pyproject.toml").write_text(allowed + '"x.py:f -> y:g" = "old"\n')
    assert main(["--config", cfg, "--jobs", "1"]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "  x.py:f -> y:g"
    assert main(["--config", cfg, "--private", "--copy", "0.99"]) == 1
    assert "grid_of duplicates fakelib._hidden:secret" in capsys.readouterr().out


def test_cli_no_dists(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "none.toml")]) == 2
    assert "no dependencies" in capsys.readouterr().err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0 and capsys.readouterr().out.strip()


def test_cli_rank(project, capsys):
    assert (
        main(["rank", "--config", str(project / "pyproject.toml"), "--top", "2"]) == 0
    )
    out = capsys.readouterr().out.splitlines()
    ranks = {}
    for line in out:
        if line.startswith("  "):
            ranks[key].append(line.split()[1])
        else:
            key = line
            ranks[key] = []
    assert ranks["proj/mod.py:pipeline"][0] == "fakelib:transform"
    assert ranks["proj/mod.py:grid_of"][0] == "fakelib._hidden:secret"
    assert set(ranks) == {"proj/mod.py:pipeline", "proj/mod.py:grid_of"}
    assert all(len(v) <= 2 for v in ranks.values())
