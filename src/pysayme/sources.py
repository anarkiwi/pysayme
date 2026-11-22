"""Finding and indexing the Python files of installed distributions and of a project,
with per-distribution indexes cached on disk."""

from __future__ import annotations

import hashlib
import multiprocessing
import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fnmatch import fnmatch
from importlib import metadata
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from . import extract

INDEX_VERSION = 5
PARALLEL_MIN_FILES = 64
_SKIP_DIRS = frozenset(("tests", "test", "testing", "__pycache__"))


def _skip_lib(rel):
    parts = PurePosixPath(rel).parts
    return rel.endswith("__main__.py") or any(
        p in _SKIP_DIRS or p.startswith("test_") for p in parts
    )


def module_name(rel):
    """Dotted module of a path relative to its import root."""
    mod = ".".join(PurePosixPath(rel).with_suffix("").parts)
    return mod.removesuffix(".__init__")


def _top_level(dist):
    names = {
        k
        for k, v in metadata.packages_distributions().items()
        if dist.metadata["Name"] in v
    }
    for name in sorted(names):
        spec = importlib.util.find_spec(name)
        if spec is None:
            continue
        if spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                root = Path(loc).parent
                for p in sorted(Path(loc).rglob("*.py")):
                    yield p, p.relative_to(root).as_posix()
        elif spec.origin and spec.origin.endswith(".py"):
            yield Path(spec.origin), Path(spec.origin).name


def dist_files(name):
    """(path, ``dist/rel``, module) of the importable non-test .py files of the
    installed distribution `name`, or None when it is not installed."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    found = [
        (Path(dist.locate_file(f)), f.as_posix())
        for f in dist.files or ()
        if f.suffix == ".py" and ".." not in f.parts
    ]
    if not found:
        found = list(_top_level(dist))
    return [
        (path, f"{name}/{rel}", module_name(rel))
        for path, rel in found
        if not _skip_lib(rel) and path.is_file()
    ]


def index_file(entry, min_shingles=12, k=extract.K):
    """(functions, constants, re-exports) of one (path, display, module) entry."""
    path, display, module = entry
    tree = extract.parse(path)
    if tree is None:
        return [], [], []
    package = Path(path).name == "__init__.py"
    return (
        extract.functions(tree, display, module, min_shingles, k),
        extract.constants(tree, display, module),
        [(module, *r) for r in extract.reexports(tree, module, package)],
    )


def _index_star(args):
    return index_file(*args)


def index_files(entries, desc, jobs=None, min_shingles=12, k=extract.K, progress=True):
    """Concatenated (functions, constants, re-exports) of `entries`, in parallel
    when many."""
    funcs, consts, reexp = [], [], []
    jobs = jobs or os.cpu_count() or 1
    work = [(e, min_shingles, k) for e in entries]
    opts = {"desc": desc, "unit": "file", "total": len(work), "leave": False}
    opts.update(file=sys.stderr, disable=not progress)
    if jobs > 1 and len(work) >= PARALLEL_MIN_FILES:
        ctx = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(jobs, mp_context=ctx) as pool:
            results = pool.map(_index_star, work, chunksize=16)
            for f, c, r in tqdm(results, **opts):
                funcs += f
                consts += c
                reexp += r
    else:
        for f, c, r in tqdm(map(_index_star, work), **opts):
            funcs += f
            consts += c
            reexp += r
    return funcs, consts, reexp


def cache_dir():
    env = os.environ.get("PYSAYME_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "pysayme"


def _cache_key(name, entries, params):
    h = hashlib.sha256(repr((INDEX_VERSION, name, params)).encode())
    for path, display, module in entries:
        st = path.stat()
        h.update(f"{display}\0{module}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def dist_index(
    name, jobs=None, min_shingles=12, k=extract.K, cache=True, progress=True
):
    """(functions, constants, re-exports) of installed distribution `name`, cached by the size
    and mtime of each of its files; None when it is not installed."""
    entries = dist_files(name)
    if entries is None:
        return None
    path = None
    if cache:
        key = _cache_key(name, entries, (min_shingles, k))
        path = cache_dir() / f"{metadata.distribution(name).metadata['Name']}-{key}.pkl"
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
            pass
    out = index_files(entries, name, jobs, min_shingles, k, progress)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
            os.replace(f.name, path)
        except OSError as e:
            warnings.warn(f"pysayme: cannot cache index of {name}: {e}")
    return out


def _private(module):
    return any(p.startswith("_") for p in module.split("."))


def public_paths(reexp):
    """{(module, name): module} mapping each re-exported definition to the public
    module with the shortest path that exports it, directly or through a chain."""
    exporters = defaultdict(set)
    for exporter, src, name, alias in reexp:
        if alias == name and exporter != src:
            exporters[src, name].add(exporter)
    best = {}
    for src, name in exporters:
        seen, todo = set(), [src]
        while todo:
            for e in exporters.get((todo.pop(), name), ()):
                if e not in seen:
                    seen.add(e)
                    todo.append(e)
        public = [m for m in seen if not _private(m)]
        if public:
            top = min(public, key=lambda m: (m.count("."), m))
            if _private(src) or top.count(".") < src.count("."):
                best[src, name] = top
    return best


def _publish(items, best):
    """`items` renamed to their public import path; those only reachable through a
    private module are marked not public."""
    for it in items:
        mod, _, qual = it.qualname.partition(":")
        mod = best.get((mod, qual.split(".", 1)[0]), mod)
        it.qualname = f"{mod}:{qual}"
        it.public = it.public and not _private(mod)
    return items


def library_index(dists, **kw):
    """Concatenated (functions, constants) of the installed distributions `dists`,
    named by their public import path; names not installed are warned and skipped."""
    funcs, consts, reexp = [], [], []
    off = not kw.get("progress", True)
    for name in tqdm(dists, desc="dependencies", unit="dist", leave=False, disable=off):
        got = dist_index(name, **kw)
        if got is None:
            warnings.warn(f"pysayme: distribution {name!r} is not installed")
            continue
        funcs += got[0]
        consts += got[1]
        reexp += got[2]
    best = public_paths(reexp)
    return _publish(funcs, best), _publish(consts, best)


def _git_files(roots, base):
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
            + ["--", *map(str, roots)],
            cwd=base,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode().split("\0") if p]


def _walk_files(roots, base):
    for root in roots:
        top = Path(base, root)
        if top.is_file():
            yield top.relative_to(base).as_posix()
        for dirpath, dirs, files in os.walk(top):
            dirs[:] = sorted(
                d
                for d in dirs
                if not d.startswith(".")
                and d != "__pycache__"
                and not Path(dirpath, d, "pyvenv.cfg").exists()
            )
            for fn in sorted(files):
                yield Path(dirpath, fn).relative_to(base).as_posix()


def project_files(roots, base=".", exclude=()):
    """(path, relative posix path, "") of the project's .py files under `roots`:
    git's tracked and unignored files when `base` is in a work tree, else a walk."""
    base = Path(base).resolve()
    rels = _git_files(roots, base)
    if rels is None:
        rels = _walk_files(roots, base)
    seen = set()
    out = []
    for rel in rels:
        if not rel.endswith(".py") or rel in seen:
            continue
        if any(fnmatch(rel, pat) for pat in exclude):
            continue
        seen.add(rel)
        path = base / rel
        if path.is_file():
            out.append((path, rel, ""))
    return sorted(out, key=lambda e: e[1])
