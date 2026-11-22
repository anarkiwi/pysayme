"""Distribution and project file discovery, indexing, caching and public names."""

import subprocess

import pytest

from conftest import write_tree
from pysayme import sources


def test_dist_files(site):
    got = {(rel, mod) for _, rel, mod in sources.dist_files("fakelib")}
    assert got == {
        ("fakelib/fakelib/__init__.py", "fakelib"),
        ("fakelib/fakelib/_core.py", "fakelib._core"),
        ("fakelib/fakelib/_hidden.py", "fakelib._hidden"),
        ("fakelib/fakelib/util.py", "fakelib.util"),
    }
    assert all(p.is_relative_to(site) for p, _, _ in sources.dist_files("fakelib"))
    assert sources.dist_files("no-such-dist-pysayme") is None


def test_dist_files_without_record(tmp_path, monkeypatch):
    write_tree(
        tmp_path,
        {
            "toplib/__init__.py": "X = 1\n",
            "toplib/sub/m.py": "Y = 2\n",
            "toplib/tests/test_t.py": "Z = 3\n",
            "topmod.py": "W = 4\n",
            "toplib-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: toplib\n"
            "Version: 1.0\n",
            "toplib-1.0.dist-info/top_level.txt": "toplib\ntopmod\nmissingmod\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    got = {(rel, mod) for _, rel, mod in sources.dist_files("toplib")}
    assert got == {
        ("toplib/toplib/__init__.py", "toplib"),
        ("toplib/toplib/sub/m.py", "toplib.sub.m"),
        ("toplib/topmod.py", "topmod"),
    }


@pytest.mark.usefixtures("site")
def test_dist_index_cache(cache, monkeypatch):
    first = sources.dist_index("fakelib", jobs=1, progress=False)
    (pkl,) = cache.glob("fakelib-*.pkl")
    names = sorted(f.qualname for f in first[0])
    assert names == [
        "fakelib._core:accumulate",
        "fakelib._core:transform",
        "fakelib._hidden:secret",
        "fakelib.util:helper",
    ]

    def boom(*a, **k):
        raise AssertionError("re-indexed despite cache")

    with monkeypatch.context() as m:
        m.setattr(sources, "index_files", boom)
        assert sorted(f.qualname for f in sources.dist_index("fakelib")[0]) == names
    pkl.write_bytes(b"garbage")
    assert sorted(f.qualname for f in sources.dist_index("fakelib")[0]) == names
    assert pkl.read_bytes() != b"garbage"


@pytest.mark.usefixtures("site")
def test_dist_index_uncached(cache, tmp_path, monkeypatch):
    assert sources.dist_index("fakelib", cache=False)[0]
    assert not cache.exists()
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("PYSAYME_CACHE", str(blocker / "sub"))
    with pytest.warns(UserWarning, match="cannot cache"):
        assert sources.dist_index("fakelib")[0]
    assert sources.dist_index("no-such-dist-pysayme") is None


def test_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("PYSAYME_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert sources.cache_dir() == tmp_path / "pysayme"


@pytest.mark.usefixtures("site")
def test_index_files_parallel_equals_serial(monkeypatch):
    entries = sources.dist_files("fakelib") * 3

    def key(out):
        return [sorted((x.qualname, x.path) for x in part) for part in out[:2]] + [
            sorted(out[2])
        ]

    serial = sources.index_files(entries, "s", jobs=1, progress=False)
    monkeypatch.setattr(sources, "PARALLEL_MIN_FILES", 2)
    assert key(sources.index_files(entries, "p", jobs=2)) == key(serial)
    assert len(serial[0]) == 12


def test_index_file_unparsable(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def (:\n")
    assert sources.index_file((bad, "bad.py", "bad")) == ([], [], [])


def test_public_paths():
    reexp = [
        ("a.b", "a._x", "f", "f"),
        ("a", "a.b", "f", "f"),
        ("a.c", "a._x", "f", "f"),
        ("pkg", "pkg.sub.mod", "g", "g"),
        ("pkg.sub", "pkg.sub.mod", "g", "g"),
        ("_p", "_p._q", "h", "h"),
        ("a", "a._y", "k", "alias"),
        ("a.d", "a.d", "self", "self"),
        ("pkg.deep.er", "pkg.x", "m", "m"),
    ]
    assert sources.public_paths(reexp) == {
        ("a._x", "f"): "a",
        ("a.b", "f"): "a",
        ("pkg.sub.mod", "g"): "pkg",
    }


@pytest.mark.usefixtures("site")
def test_library_index():
    with pytest.warns(UserWarning, match="not installed"):
        funcs, consts = sources.library_index(["fakelib", "no-such-dist-pysayme"])
    pub = {f.qualname: f.public for f in funcs + consts}
    assert pub == {
        "fakelib:accumulate": True,
        "fakelib:transform": True,
        "fakelib:helper": True,
        "fakelib._hidden:secret": False,
        "fakelib:COLORS": True,
        "fakelib:TIMEOUT": True,
        "fakelib:STEPS": True,
    }


LAYOUT = {
    "src/a.py": "",
    "src/b.txt": "",
    "src/gen.py": "",
    "src/pkg/c.py": "",
    "src/.hidden/d.py": "",
    "src/__pycache__/e.py": "",
    "src/venv/pyvenv.cfg": "",
    "src/venv/lib/f.py": "",
    "src/skip_me.py": "",
    "other/g.py": "",
    "top.py": "",
    ".gitignore": "gen.py\nvenv/\n",
}


def rels(files):
    return [rel for _, rel, mod in files if mod == ""]


def test_project_files_walk(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    write_tree(tmp_path / "w", LAYOUT)
    got = sources.project_files(["src", "top.py", "src"], tmp_path / "w", ["*skip*"])
    assert rels(got) == ["src/a.py", "src/gen.py", "src/pkg/c.py", "top.py"]
    assert all(p == tmp_path / "w" / r for p, r, _ in got)


def test_project_files_git(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    base = write_tree(tmp_path / "g", LAYOUT)
    subprocess.run(["git", "init", "-q"], cwd=base, check=True)
    subprocess.run(["git", "add", "src/a.py"], cwd=base, check=True)
    (base / "src/a.py").unlink()
    got = sources.project_files(["src"], base, ["*skip*"])
    assert rels(got) == ["src/.hidden/d.py", "src/__pycache__/e.py", "src/pkg/c.py"]
