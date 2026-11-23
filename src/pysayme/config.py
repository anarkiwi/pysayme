"""``[tool.pysayme]`` configuration and dependency discovery."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from . import extract, match


@dataclass
class Config:
    """What to check against what, and the matcher's thresholds."""

    base: Path = Path(".")
    roots: list = field(default_factory=lambda: ["."])
    dists: list = field(default_factory=list)
    allow: dict = field(default_factory=dict)
    exclude: list = field(default_factory=list)
    copy: float = match.COPY
    inlined: float = match.INLINED
    min_shingles: int = 12
    min_inlined: int = match.MIN_INLINED
    shingle: int = extract.K
    private: bool = False
    ambiguous: bool = False
    self_join: bool = True
    behaviour: bool = False
    jobs: int = 0
    cache: bool = True
    introspect: bool = True
    progress: bool = True


def _requirement(line):
    try:
        return Requirement(line).name
    except InvalidRequirement:
        return None


def requirement_names(path):
    """Distribution names a requirements file names, following ``-r`` includes."""
    path = Path(path)
    out = []
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    for line in text.splitlines():
        line = line.split(" #", 1)[0].strip()
        opt, _, arg = line.partition(" ")
        if opt in ("-r", "--requirement"):
            out += requirement_names(path.parent / arg.strip())
        elif line and not line.startswith(("#", "-")):
            name = _requirement(line.split(" --", 1)[0])
            if name:
                out.append(name)
    return out


def project_dependencies(project):
    """Names of a PEP 621 ``[project]`` table's dependencies and extras."""
    reqs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        reqs += extra
    return [n for n in map(_requirement, reqs) if n]


def installed(name):
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def unique(names, drop=()):
    seen = {canonicalize_name(d) for d in drop}
    out = []
    for n in names:
        c = canonicalize_name(n)
        if c not in seen:
            seen.add(c)
            out.append(n)
    return out


def discover_dists(cfg, doc, base):
    """``dists``, else names in ``requirements`` file(s), else the ``[project]``
    dependencies and extras that are installed, else ``requirements.txt``."""
    if "dists" in cfg:
        names = list(cfg["dists"])
    elif "requirements" in cfg:
        reqs = cfg["requirements"]
        reqs = [reqs] if isinstance(reqs, str) else reqs
        names = [n for r in reqs for n in requirement_names(base / r)]
    else:
        project = doc.get("project", {})
        names = [n for n in project_dependencies(project) if installed(n)]
        if not names and (base / "requirements.txt").is_file():
            names = requirement_names(base / "requirements.txt")
    names += cfg.get("extra-dists", [])
    drop = list(cfg.get("ignore-dists", []))
    if doc.get("project", {}).get("name"):
        drop.append(doc["project"]["name"])
    return unique(names, drop)


_KEYS = {
    f.name.replace("_", "-"): f.name
    for f in fields(Config)
    if f.name not in ("base", "progress")
}
_KEYS["self"] = _KEYS.pop("self-join")


def load_config(pyproject="pyproject.toml", **overrides):
    """Config from `pyproject`'s ``[tool.pysayme]``; `overrides` that are not None
    win. Paths are relative to the file's directory."""
    path = Path(pyproject)
    doc = {}
    if path.is_file():
        with open(path, "rb") as f:
            doc = tomllib.load(f)
    cfg = dict(doc.get("tool", {}).get("pysayme", {}))
    base = path.resolve().parent
    if overrides.get("dists") is not None:
        cfg["dists"] = overrides.pop("dists")
    kw = {_KEYS[k]: v for k, v in cfg.items() if k in _KEYS and k != "dists"}
    kw.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(cfg) - set(_KEYS) - {"requirements", "extra-dists", "ignore-dists"}
    if unknown:
        raise ValueError(f"unknown [tool.pysayme] keys: {', '.join(sorted(unknown))}")
    return Config(base=base, dists=discover_dists(cfg, doc, base), **kw)


def has_section(pyproject):
    path = Path(pyproject)
    if not path.is_file():
        return False
    with open(path, "rb") as f:
        return "pysayme" in tomllib.load(f).get("tool", {})
