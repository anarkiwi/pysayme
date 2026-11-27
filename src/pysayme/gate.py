"""The CI gate: duplicates found, minus the allowlist, plus stale allowlist entries."""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import behaviour, sources, vectorize
from .match import allow_key, find, find_constants, find_self

HOW = """\
Each line above is project code that duplicates code in an installed dependency.
Delete the copy and import/call the dependency's version. If the dependency's version
is not quite right, change the call site, or the dependency, rather than keep a copy.
A line naming two project locations is a copy within the project: call one from the
other, or move it to a module both import. Only if the duplication is deliberate, add
its key to [tool.pysayme.allow] in pyproject.toml with the reason as the value."""


VECTORIZE_HOW = """\
Each line above is a Python loop over a numpy array that numpy does in one call. Replace
it with the call shown. If the loop is deliberate (a naive reference implementation a
fast path is tested against, say), add its key to [tool.pysayme.allow] in
pyproject.toml with the reason as the value."""


def report(cfg):
    """{allowlist key: message} of every project function and constant that
    duplicates one in `cfg.dists`, if `cfg.self_join` another project one, and if
    `cfg.behaviour` every project function that behaves like a library callable."""
    jobs = cfg.jobs or os.cpu_count() or 1
    kw = {
        "jobs": jobs,
        "min_shingles": cfg.min_shingles,
        "k": cfg.shingle,
        "progress": cfg.progress,
    }
    introspect = cfg.introspect or cfg.behaviour
    idx = sources.indexes(cfg.dists, cache=cfg.cache, introspect=introspect, **kw)
    lib_funcs, lib_consts = sources.publish(idx)
    files = sources.project_files(cfg.roots, cfg.base, cfg.exclude)
    proj_funcs, proj_consts, _ = sources.index_files(files, "project", **kw)
    matches = find(
        proj_funcs,
        lib_funcs,
        cfg.copy,
        cfg.inlined,
        cfg.min_inlined,
        cfg.private,
        cfg.ambiguous,
        jobs,
    )
    if cfg.self_join:
        matches += find_self(
            proj_funcs, cfg.copy, cfg.inlined, cfg.min_inlined, cfg.ambiguous, jobs
        )
    out = {m.key(): str(m) for m in matches}
    for p, lib in find_constants(proj_consts, lib_consts, cfg.private):
        out[allow_key(p, lib.qualname)] = (
            f"{p.where()} {p.qualname} = {p.value!r} redefines "
            f"{lib.qualname} ({lib.where()})"
        )
    if cfg.behaviour:
        project = behaviour.project_candidates(files)
        found = behaviour.find(
            project,
            idx,
            kw["jobs"],
            cfg.cache,
            cfg.ambiguous,
            cfg.progress,
            cfg.child_memory or None,
        )
        for f in found:
            out.setdefault(f.key(), str(f))
    if cfg.vectorize:
        for f in vectorize.find(files, cfg).values():
            out.setdefault(f.key(), str(f))
    return out


@dataclass
class Result:
    """Outcome of checking a report against an allowlist."""

    found: dict
    new: dict
    stale: list
    unreasoned: list

    @property
    def ok(self):
        return not (self.new or self.stale or self.unreasoned)

    def __str__(self):
        parts = []
        if self.new:
            parts += [*sorted(self.new.values()), "", HOW]
        if self.stale:
            parts += ["", "remove from [tool.pysayme.allow], they no longer match:"]
            parts += [f"  {k}" for k in self.stale]
        if self.unreasoned:
            parts += ["", "[tool.pysayme.allow] entries need a reason:"]
            parts += [f"  {k}" for k in self.unreasoned]
        return "\n".join(parts).strip("\n")


def check(cfg, found=None):
    """Result of `cfg`: new duplicates, and allowlist entries that are stale (so the
    list only shrinks) or that lack a reason."""
    found = report(cfg) if found is None else found
    allow = cfg.allow
    return Result(
        found=found,
        new={k: v for k, v in found.items() if k not in allow},
        stale=sorted(set(allow) - set(found)),
        unreasoned=sorted(
            k for k, r in allow.items() if not (isinstance(r, str) and r.strip())
        ),
    )
