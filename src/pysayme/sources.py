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
from typing import NamedTuple

from tqdm import tqdm

from . import extract, runtime

INDEX_VERSION = 7
SOURCE_SUFFIXES = (".py", ".pyi")
PARALLEL_MIN_FILES = 64
_SKIP_DIRS = frozenset(("tests", "test", "testing", "__pycache__"))


def _skip_lib(rel):
    parts = PurePosixPath(rel).parts
    return rel.endswith(("__main__.py", "conftest.py")) or any(
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
                for p in sorted(Path(loc).rglob("*.py*")):
                    if p.suffix in SOURCE_SUFFIXES:
                        yield p, p.relative_to(root).as_posix()
        elif spec.origin and spec.origin.endswith(".py"):
            yield Path(spec.origin), Path(spec.origin).name


def dist_files(name):
    """(path, ``dist/rel``, module) of the importable non-test .py and .pyi files of
    the installed distribution `name`, or None when it is not installed."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    found = [
        (Path(dist.locate_file(f)), f.as_posix())
        for f in dist.files or ()
        if f.suffix in SOURCE_SUFFIXES and ".." not in f.parts
    ]
    if not found:
        found = list(_top_level(dist))
    return [
        (path, f"{name}/{rel}", module_name(rel))
        for path, rel in found
        if not _skip_lib(rel) and path.is_file()
    ]


class Index(NamedTuple):
    """One distribution's functions, constants, Module records, runtime exports
    {(module, name): exporting modules} and compiled public callables."""

    funcs: list
    consts: list
    modules: list
    exports: dict
    compiled: list


def index_file(entry, min_shingles=12, k=extract.K):
    """(functions, constants, [Module]) of one (path, display, module) entry; a
    ``.pyi`` stub contributes its Module record only."""
    path, display, module = entry
    tree = extract.parse(path)
    if tree is None:
        return [], [], []
    path = Path(path)
    info = [extract.module_info(tree, module, path.stem == "__init__")]
    if path.suffix == ".pyi":
        return [], [], info
    return (
        extract.functions(tree, display, module, min_shingles, k),
        extract.constants(tree, display, module),
        info,
    )


def _index_star(args):
    return index_file(*args)


def index_files(entries, desc, jobs=None, min_shingles=12, k=extract.K, progress=True):
    """Concatenated (functions, constants, Modules) of `entries`, in parallel when
    many."""
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
    name,
    jobs=None,
    min_shingles=12,
    k=extract.K,
    cache=True,
    progress=True,
    introspect=True,
):
    """Index of installed distribution `name`, cached by the size and mtime of
    each of its files; None when it is not installed. `introspect` adds what
    importing its public modules exports."""
    entries = dist_files(name)
    if entries is None:
        return None
    path = None
    if cache:
        params = (min_shingles, k, introspect, sys.version)
        key = _cache_key(name, entries, params)
        path = cache_dir() / f"{metadata.distribution(name).metadata['Name']}-{key}.pkl"
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
            pass
    out = Index(
        *index_files(entries, name, jobs, min_shingles, k, progress),
        *(
            runtime.introspect(entries, desc=f"{name} import", progress=progress)
            if introspect
            else ({}, [])
        ),
    )
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
            os.replace(f.name, path)
        except OSError as e:
            warnings.warn(f"pysayme: cannot cache index of {name}: {e}")
    return out


def reexports(modules):
    """(exporter, source, name, alias) of each re-export of `modules` (a ``.pyi``
    record merged with its module's), ``*`` imports expanded, filtered by the
    exporter's ``__all__`` or, without one, its public names (a fixed point, so
    cycles of ``*`` imports resolve)."""
    mods = {}
    for m in modules:
        old = mods.get(m.name)
        if old is not None:
            known = [r for r in (old, m) if r.all_names is not None]
            m = extract.Module(
                m.name,
                old.imports + m.imports,
                old.stars + m.stars,
                old.bound | m.bound,
                frozenset().union(*(r.all_names for r in known)) if known else None,
                tuple(x for r in known for x in r.all_from),
            )
        mods[m.name] = m
    exported = dict.fromkeys(mods, frozenset())
    changed = True
    while changed:
        changed = False
        for name, m in mods.items():
            base, via = (
                (m.bound, m.stars) if m.all_names is None else (m.all_names, m.all_from)
            )
            new = base.union(*(exported.get(s, ()) for s in via))
            if new != exported[name]:
                exported[name] = new
                changed = True

    out = []
    for name, m in mods.items():
        allowed = exported[name]
        out += [(name, *r) for r in m.imports if r[2] in allowed]
        out += [
            (name, s, n, n)
            for s in m.stars
            for n in exported.get(s, ())
            if n in allowed
        ]
    return out


def public_paths(reexp, exports=None, owner=None):
    """{(module, name): module}: the shortest public module (own, then lexicographic
    on ties) of each definition's own and exporters, via static chains or runtime
    `exports`. With `owner` {module: distribution}, exporters count only in the
    definition's distribution and top-level package, unless that is private."""
    owner = owner or {}

    def counts(exporter, src):
        top, src_top = exporter.split(".", 1)[0], src.split(".", 1)[0]
        same = top == src_top or src_top.startswith("_")
        return exporter != src and same and owner.get(exporter) == owner.get(src)

    edges = defaultdict(set)
    for exporter, src, name, alias in reexp:
        if alias == name:
            edges[src, name].add(exporter)
    for key, mods in (exports or {}).items():
        edges[key] |= mods
    edges = {k: {e for e in v if counts(e, k[0])} for k, v in edges.items()}
    best = {}
    for src, name in list(edges):
        seen, todo = set(), [src]
        while todo:
            for e in edges.get((todo.pop(), name), ()):
                if e not in seen:
                    seen.add(e)
                    todo.append(e)
        public = [m for m in seen | {src} if not extract.private_module(m)]
        if public:
            top = min((m.count("."), m != src, m) for m in public)[2]
            if top != src:
                best[src, name] = top
    return best


def _publish(items, best):
    """`items` renamed to their public import path; those only reachable through a
    private module are marked not public."""
    for it in items:
        mod, _, qual = it.qualname.partition(":")
        mod = best.get((mod, qual.split(".", 1)[0]), mod)
        it.qualname = f"{mod}:{qual}"
        it.public = it.public and not extract.private_module(mod)
    return items


def indexes(dists, **kw):
    """Index of each installed distribution of `dists`; names not installed are
    warned and skipped."""
    out = []
    off = not kw.get("progress", True)
    for name in tqdm(dists, desc="dependencies", unit="dist", leave=False, disable=off):
        got = dist_index(name, **kw)
        if got is None:
            warnings.warn(f"pysayme: distribution {name!r} is not installed")
        else:
            out.append(got)
    return out


def naming(idx):
    """public_paths of the Index list `idx`, static and runtime exports merged."""
    exports = defaultdict(set)
    for i in idx:
        for key, mods in i.exports.items():
            exports[key] |= mods
    owner = {m.name: n for n, i in enumerate(idx) for m in i.modules}
    return public_paths(reexports(m for i in idx for m in i.modules), exports, owner)


def library_index(dists, **kw):
    """Concatenated (functions, constants) of the installed distributions `dists`,
    named by their public import path."""
    idx = indexes(dists, **kw)
    best = naming(idx)
    funcs = [f for i in idx for f in i.funcs]
    consts = [c for i in idx for c in i.consts]
    return _publish(funcs, best), _publish(consts, best)


def compiled_callables(dists, **kw):
    """[runtime.Compiled] public callables of `dists` that have no Python source."""
    return [c for i in indexes(dists, **kw) for c in i.compiled]


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
