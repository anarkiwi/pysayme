"""The CI gate: duplicates found, minus the allowlist, plus stale allowlist entries."""

from __future__ import annotations

from dataclasses import dataclass

from . import sources
from .match import find, find_constants

HOW = """\
Each line above is project code that duplicates code in an installed dependency.
Delete the copy and import/call the dependency's version. If the dependency's version
is not quite right, change the call site, or the dependency, rather than keep a copy.
Only if the duplication is deliberate, add its key to [tool.pysayme.allow] in
pyproject.toml with the reason as the value."""


def report(cfg):
    """{allowlist key: message} of every project function and constant that
    duplicates one in `cfg.dists`."""
    kw = {
        "jobs": cfg.jobs or None,
        "min_shingles": cfg.min_shingles,
        "k": cfg.shingle,
        "progress": cfg.progress,
    }
    lib_funcs, lib_consts = sources.library_index(cfg.dists, cache=cfg.cache, **kw)
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
    )
    out = {m.key(): str(m) for m in matches}
    for p, lib in find_constants(proj_consts, lib_consts, cfg.private):
        out[f"{p.path}:{p.qualname} -> {lib.qualname}"] = (
            f"{p.where()} {p.qualname} = {p.value!r} redefines "
            f"{lib.qualname} ({lib.where()})"
        )
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
