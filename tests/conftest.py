"""A fake installed distribution ``fakelib`` and a project that duplicates it."""

from pathlib import Path
from textwrap import dedent

import pytest

pytest_plugins = ["pytester", "pysayme.plugin"]

UNRELATED = '''
def unrelated(path, mode):
    """Not in any dependency."""
    with open(path, mode) as fh:
        lines = [ln.rstrip() for ln in fh if not ln.startswith("#")]
    counts = {}
    for ln in lines:
        counts[ln.split("=")[0]] = counts.get(ln.split("=")[0], 0) + 1
    return max(counts, key=counts.get) if counts else None
'''

LIB = {
    "fakelib/__init__.py": """
        from ._core import COLORS, accumulate, transform
        from .util import helper

        __all__ = ["COLORS", "STEPS", "TIMEOUT", "accumulate", "helper", "transform"]
        TIMEOUT = 300
        STEPS = (1, 2, 3, 4)
    """,
    "fakelib/_core.py": """
        COLORS = ("crimson", "teal", "amber")


        def accumulate(values, scale):
            total = 0
            for item in values:
                if item.weight > scale:
                    total += item.weight * scale
                else:
                    total -= item.offset
            return total


        def transform(records, limit):
            out = []
            for rec in records:
                key = rec.name.lower().strip()
                if not key or key in out:
                    continue
                size = len(rec.payload) + rec.header.length
                if size > limit:
                    out.append((key, size // limit, size % limit))
                elif rec.flags & 0x10:
                    out.append((key, rec.flags >> 4, rec.checksum))
                else:
                    out.append((key, 0, rec.checksum ^ size))
            return sorted(out, key=lambda t: t[1])
    """,
    "fakelib/_hidden.py": """
        def secret(rows, width):
            grid = {}
            for row in rows:
                cell = row.cells[width] if width < len(row.cells) else None
                grid.setdefault(row.group, []).append(cell)
            return {k: v for k, v in grid.items() if v and v[0] is not None}
    """,
    "fakelib/util.py": """
        def helper(text, sep):
            parts = text.split(sep)
            return [p.strip().upper() for p in parts if p and p.isidentifier()]
    """,
    "fakelib/tests/test_x.py": UNRELATED,
    "fakelib/__main__.py": UNRELATED,
}

PROJECT = {
    "pyproject.toml": """
        [project]
        name = "proj"

        [tool.pysayme]
        dists = ["fakelib", "proj"]
        roots = ["proj"]
    """,
    "proj/mod.py": dedent(UNRELATED) + dedent("""
        TIMEOUT = 300
        PALETTE = ("crimson", "teal", "amber")
        SEQ = (1, 2, 3, 4)


        def mysum(xs, factor):
            acc = 0
            for it in xs:
                if it.weight > factor:
                    acc += it.weight * factor
                else:
                    acc -= it.offset
            return acc


        def pipeline(source, cap, verbose, sink):
            if verbose:
                print("starting pipeline", cap, file=sink)
            out = []
            for rec in source:
                key = rec.name.lower().strip()
                if not key or key in out:
                    continue
                size = len(rec.payload) + rec.header.length
                if size > cap:
                    out.append((key, size // cap, size % cap))
                elif rec.flags & 0x10:
                    out.append((key, rec.flags >> 4, rec.checksum))
                else:
                    out.append((key, 0, rec.checksum ^ size))
            ranked = sorted(out, key=lambda t: t[1])
            totals = {name: a * b for name, a, b in ranked if a and b}
            for name, total in sorted(totals.items(), reverse=True):
                sink.write(f"{name}: {total:>12}\\n")
                sink.flush()
            while ranked and ranked[-1][2] < 0:
                ranked.pop()
            return ranked, totals, len(totals) / max(len(ranked), 1)


        def grid_of(lines, col):
            grid = {}
            for row in lines:
                cell = row.cells[col] if col < len(row.cells) else None
                grid.setdefault(row.group, []).append(cell)
            return {k: v for k, v in grid.items() if v and v[0] is not None}
        """),
}

KEYS = {
    "proj/mod.py:mysum -> fakelib:accumulate",
    "proj/mod.py:pipeline -> fakelib:transform",
    "proj/mod.py:TIMEOUT -> fakelib:TIMEOUT",
    "proj/mod.py:PALETTE -> fakelib:COLORS",
}


def write_tree(root, files):
    root = Path(root)
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(text).lstrip(), encoding="utf-8")
    return root


def install_fakelib(site):
    """Write ``fakelib`` and its dist-info under `site`."""
    write_tree(site, LIB)
    info = Path(site, "fakelib-1.0.dist-info")
    info.mkdir()
    info.joinpath("METADATA").write_text(
        "Metadata-Version: 2.1\nName: fakelib\nVersion: 1.0\n"
    )
    record = [f"{rel},," for rel in LIB] + ["fakelib-1.0.dist-info/METADATA,,"]
    info.joinpath("RECORD").write_text("\n".join(record) + "\n", "utf-8")
    return Path(site)


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    path = tmp_path / "cache"
    monkeypatch.setenv("PYSAYME_CACHE", str(path))
    return path


@pytest.fixture(name="site")
def fixture_site(tmp_path, monkeypatch):
    """Directory on sys.path holding the installed ``fakelib``."""
    path = install_fakelib(tmp_path / "site")
    monkeypatch.syspath_prepend(str(path))
    monkeypatch.setenv("PYTHONPATH", str(path))
    return path


@pytest.fixture
def project(site, monkeypatch):
    """A project, outside any git work tree, duplicating ``fakelib``."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(site.parent))
    return write_tree(site.parent / "project", PROJECT)
