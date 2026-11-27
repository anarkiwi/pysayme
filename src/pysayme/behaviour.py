"""Behavioural fingerprinting: project functions that compute what one public
library callable computes, found by running both on the same seeded probes."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from . import extract, probe, runtime, sources
from .match import allow_key

PROBE, VERIFY = 0, 1


class Candidate(NamedTuple):
    """A project function to probe, with the names its body references."""

    path: str
    qualname: str
    line: int
    target: tuple
    tuples: tuple
    names: frozenset


class Finding(NamedTuple):
    """A project function that behaves like a library callable."""

    project: Candidate
    library: str
    binding: tuple
    scale: bool
    rivals: int

    def key(self):
        return allow_key(self.project, self.library)

    def __str__(self):
        p = self.project
        bound = ", ".join(f"{k}={v!r}" for k, v in self.binding)
        out = (
            f"{p.path}:{p.line} {p.qualname} behaves like {self.library}"
            f"{f'({bound})' if bound else ''} on the probe set"
        )
        if self.scale:
            out += " up to scale"
        if self.rivals > 1:
            out += f" (one of {self.rivals} alike)"
        return out


def importable(path):
    """(sys.path entry, dotted module or file to load) of a project file."""
    path = Path(path).resolve()
    if not (path.parent / "__init__.py").is_file():
        return str(path.parent), str(path)
    top, parts = path.parent, [] if path.stem == "__init__" else [path.stem]
    while (top / "__init__.py").is_file():
        parts.insert(0, top.name)
        top = top.parent
    return str(top), ".".join(parts)


def _body_names(tree, qualname):
    node = tree
    for part in qualname.split("."):
        node = next(n for n in node.body if getattr(n, "name", None) == part)
    return frozenset(
        n.id if isinstance(n, ast.Name) else n.attr
        for stmt in node.body
        for n in ast.walk(stmt)
        if isinstance(n, (ast.Name, ast.Attribute))
    )


def project_candidates(files):
    """Candidates of the project's (path, display, module) `files`."""
    out = []
    for path, rel, _ in files:
        tree = extract.parse(path)
        if tree is None:
            continue
        target = importable(path)
        for qualname, line, kinds in probe.project_specs(tree):
            tuples = tuple(probe.kind_tuples(kinds))
            if tuples:
                names = _body_names(tree, qualname)
                out.append(
                    Candidate(rel, qualname, line, (*target, qualname), tuples, names)
                )
    return out


def _target(path):
    module, _, name = path.partition(":")
    return None, module, name


def _timeout(task):
    calls = len(task[1]) * len(task[2]) * probe.N_PROBES
    return runtime.IMPORT_TIMEOUT + calls * probe.CALL_TIMEOUT


def _wanted(spec, demand):
    every = frozenset(probe.KINDS)
    return tuple(
        t
        for t in demand
        if len(t) == len(spec.kinds)
        and all(k in (s or every) for k, s in zip(t, spec.kinds))
    )


def library_fingerprints(idx, demand, pool, cache=True, progress=True):
    """{(path, binding, kinds): Fingerprint or None} of the probeable callables of
    the Index list `idx` on the kind tuples in `demand`, cached per distribution."""
    out, todo, owners = {}, [], []
    for n, i in enumerate(idx):
        saved = sources.load(_cache_path(i)) if cache and i.key else None
        have = saved or {}
        owners.append(have)
        for c in i.callables:
            tuples = _wanted(c.spec, demand)
            miss = [t for t in tuples if (c.path, (), t) not in have]
            if miss:
                todo.append((n, c, tuple(miss)))
            out.update(
                ((c.path, b, t), have[c.path, b, t])
                for t in tuples
                if (c.path, (), t) in have
                for b in c.spec.bindings
            )
    tasks = [(_target(c.path), c.spec.bindings, t, PROBE) for _, c, t in todo]
    got = pool.map(tasks, _timeout, "library behaviour", progress, "callable")
    dirty = set()
    for (n, c, tuples), res in zip(todo, got):
        for b in c.spec.bindings:
            for t in tuples:
                owners[n][c.path, b, t] = out[c.path, b, t] = (res or {}).get((b, t))
        dirty.add(n)
    for n in dirty:
        if cache and idx[n].key:
            sources.save(_cache_path(idx[n]), owners[n])
    return out


def _cache_path(index):
    return sources.cache_dir() / f"behaviour-{index.key}.pkl"


def _joins(lib):
    exact, scale = defaultdict(set), defaultdict(set)
    for (path, binding, kinds), fp in lib.items():
        if fp is not None:
            exact[kinds, fp.exact].add((path, binding))
            scale[kinds, fp.scale].add((path, binding))
    return exact, scale


def _hits(fps, exact, scale):
    """(scale?, [(path, binding, kinds)]) of a project function's fingerprints:
    exact matches, else scale-normalised ones."""
    for is_scale, table in ((False, exact), (True, scale)):
        hits = [
            (path, binding, kinds)
            for (_, kinds), fp in fps.items()
            if (fp.scale_ok if is_scale else fp.exact_ok)
            for path, binding in table.get(
                (kinds, fp.scale if is_scale else fp.exact), ()
            )
        ]
        if hits:
            return is_scale, sorted(hits, key=_order)
    return False, []


def _order(hit):
    path, binding, _ = hit
    return path.count("."), path, len(binding), repr(hit)


def _name(path):
    return path.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def _agree(a, b, is_scale):
    if a is None or b is None:
        return False
    if is_scale:
        return a.scale_ok and a.scale == b.scale
    return a.exact_ok and a.exact == b.exact


def find(
    project, idx, jobs=None, cache=True, ambiguous=False, progress=True, memory=None
):
    """Findings of the project Candidates whose fingerprints join library callables
    of a single name (several if `ambiguous`) that agree with them again on fresh
    probes; a function that calls a callable it behaves like is a wrapper. A probe
    whose child holds more than `memory` resident bytes gives no fingerprint."""
    demand = sorted({t for c in project for t in c.tuples})
    with runtime.Workers(probe.run, jobs, memory) as pool:
        lib = library_fingerprints(idx, demand, pool, cache, progress)
        tasks = [(c.target, ((),), c.tuples, PROBE) for c in project]
        got = pool.map(tasks, _timeout, "project behaviour", progress, "function")
        exact, scale = _joins(lib)
        pending = []
        for c, fps in zip(project, got):
            is_scale, hits = _hits(fps or {}, exact, scale)
            names = {_name(h[0]) for h in hits}
            if not names or names & c.names:
                continue
            if len(names) == 1 or ambiguous:
                pending.append((c, is_scale, hits, len(names)))
        checks = {}
        for c, _, hits, _ in pending:
            checks[c.target, ((),), _kinds(hits), VERIFY] = None
            checks.update(((_target(p), (b,), (t,), VERIFY), None) for p, b, t in hits)
        done = pool.map(checks, _timeout, "verify", progress, "function")
    results = dict(zip(checks, done))
    out = []
    for c, is_scale, hits, rivals in pending:
        mine = results[c.target, ((),), _kinds(hits), VERIFY] or {}
        for path, binding, kinds in hits:
            theirs = results[_target(path), (binding,), (kinds,), VERIFY] or {}
            if _agree(mine.get(((), kinds)), theirs.get((binding, kinds)), is_scale):
                out.append(Finding(c, path, binding, is_scale, rivals))
                break
    return out


def _kinds(hits):
    return tuple(sorted({h[2] for h in hits}))
