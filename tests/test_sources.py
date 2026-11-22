"""Distribution and project file discovery, indexing, caching and public names."""

import subprocess

import pytest

from conftest import write_tree
from pysayme import extract, sources


def test_dist_files(site):
    got = {(rel, mod) for _, rel, mod in sources.dist_files("fakelib")}
    assert got == {
        ("fakelib/fakelib/__init__.py", "fakelib"),
        ("fakelib/fakelib/_core.py", "fakelib._core"),
        ("fakelib/fakelib/_hidden.py", "fakelib._hidden"),
        ("fakelib/fakelib/util.py", "fakelib.util"),
        ("fakelib/fakelib/stars.py", "fakelib.stars"),
        ("fakelib/fakelib/_star.py", "fakelib._star"),
        ("fakelib/fakelib/lazy/__init__.py", "fakelib.lazy"),
        ("fakelib/fakelib/lazy/__init__.pyi", "fakelib.lazy"),
        ("fakelib/fakelib/lazy/_impl.py", "fakelib.lazy._impl"),
        ("fakelib/fakelib/dyn.py", "fakelib.dyn"),
        ("fakelib/fakelib/_dyn.py", "fakelib._dyn"),
        ("fakelib/fakelib/ext.py", "fakelib.ext"),
    }
    assert all(p.is_relative_to(site) for p, _, _ in sources.dist_files("fakelib"))
    assert sources.dist_files("no-such-dist-pysayme") is None


def test_dist_files_without_record(tmp_path, monkeypatch):
    write_tree(
        tmp_path,
        {
            "toplib/__init__.py": "X = 1\n",
            "toplib/sub/m.py": "Y = 2\n",
            "toplib/sub/m.pyi": "Y: int\n",
            "toplib/sub/conftest.py": "",
            "toplib/sub/m.pyc": "",
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
        ("toplib/toplib/sub/m.pyi", "toplib.sub.m"),
        ("toplib/topmod.py", "topmod"),
    }


@pytest.mark.usefixtures("site")
def test_dist_index_cache(cache, monkeypatch):
    first = sources.dist_index("fakelib", jobs=1, progress=False)
    (pkl,) = cache.glob("fakelib-*.pkl")
    names = sorted(f.qualname for f in first.funcs)
    assert names == [
        "fakelib._core:accumulate",
        "fakelib._core:transform",
        "fakelib._dyn:dynamic",
        "fakelib._hidden:secret",
        "fakelib._star:starred",
        "fakelib._star:unlisted",
        "fakelib.ext:len_impl",
        "fakelib.lazy._impl:stubbed",
        "fakelib.util:helper",
    ]
    assert first.exports[("fakelib._dyn", "dynamic")] == {"fakelib.dyn"}
    assert first.compiled == []

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
            sorted(m.name for m in out[2])
        ]

    serial = sources.index_files(entries, "s", jobs=1, progress=False)
    monkeypatch.setattr(sources, "PARALLEL_MIN_FILES", 2)
    assert key(sources.index_files(entries, "p", jobs=2)) == key(serial)
    assert len(serial[0]) == 27


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


def test_public_paths_ties_runtime_and_owner():
    reexp = [
        ("q.b", "q.c", "t", "t"),
        ("q.a", "q.c", "t", "t"),
        ("q.z", "q._p", "u", "u"),
        ("q.y", "q._p", "u", "u"),
        ("other", "q.c", "v", "v"),
        ("mine", "_impl.m", "w", "w"),
        ("vendor", "q._p", "x", "x"),
    ]
    exports = {("q._r", "r"): {"q.s.t", "q.s"}, ("q._p", "x"): {"q.k.l"}}
    owner = {"vendor": 1}
    assert sources.public_paths(reexp, exports, owner) == {
        ("q._p", "u"): "q.y",
        ("q._r", "r"): "q.s",
        ("q._p", "x"): "q.k.l",
        ("_impl.m", "w"): "mine",
    }


def record(name, imports=(), stars=(), bound=(), all_names=None, all_from=()):
    return extract.Module(
        name,
        tuple(imports),
        tuple(stars),
        frozenset(bound),
        None if all_names is None else frozenset(all_names),
        tuple(all_from),
    )


def test_reexports():
    mods = [
        record(
            "p", [("p._a", "f", "f"), ("p._a", "g", "g")], ["p._s"], ["f", "g"], ["f"]
        ),
        record("p._s", stars=["p._t", "missing"], bound=["s"]),
        record("p._t", bound=["t", "u"], all_names=["t"], all_from=["p._u"]),
        record("p._u", bound=["v"]),
        record("p.cyc", stars=["p.cyc2"], bound=["c"]),
        record("p.cyc2", stars=["p.cyc"], bound=["d"]),
        record("p.lazy", bound=["x"]),
        record("p.lazy", [("p.lazy._i", "y", "y")], all_names=["y"]),
    ]
    got = set(sources.reexports(mods))
    assert got == {
        ("p", "p._a", "f", "f"),
        ("p._s", "p._t", "t", "t"),
        ("p._s", "p._t", "v", "v"),
        ("p.cyc", "p.cyc2", "d", "d"),
        ("p.cyc", "p.cyc2", "c", "c"),
        ("p.cyc2", "p.cyc", "c", "c"),
        ("p.cyc2", "p.cyc", "d", "d"),
        ("p.lazy", "p.lazy._i", "y", "y"),
    }


def test_index_file_stub(tmp_path):
    stub = tmp_path / "__init__.pyi"
    stub.write_text("from ._x import f as f\nBIG = (1, 5, 9)\n")
    funcs, consts, (info,) = sources.index_file((stub, "d/__init__.pyi", "pkg"))
    assert (funcs, consts) == ([], [])
    assert info.imports == (("pkg._x", "f", "f"),)


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
        "fakelib.stars:starred": True,
        "fakelib._star:unlisted": False,
        "fakelib.lazy:stubbed": True,
        "fakelib.dyn:dynamic": True,
        "fakelib.ext:len_impl": False,
    }
    funcs, _ = sources.library_index(["fakelib"], introspect=False, progress=False)
    assert {f.qualname: f.public for f in funcs}["fakelib._dyn:dynamic"] is False
    assert sources.compiled_callables(["fakelib"], progress=False) == []


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
