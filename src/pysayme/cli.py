"""Command line: ``pysayme check`` (the gate) and ``pysayme rank`` (semantic leads)."""

from __future__ import annotations

import argparse
import sys

from . import __version__, advise, gate, semantic, sources, vectorize
from .config import load_config

COMMANDS = ("check", "rank", "vectorize", "advise")


def _common(p):
    p.add_argument("--config", default="pyproject.toml", help="pyproject.toml to read")
    p.add_argument("--dist", nargs="+", dest="dists", help="installed distributions")
    p.add_argument("--jobs", type=int, help="indexing processes (default: all CPUs)")
    p.add_argument(
        "--no-introspect",
        dest="introspect",
        action="store_false",
        default=None,
        help="do not import dependencies to find their public names",
    )
    p.add_argument("roots", nargs="*", help="project directories or files")


def parser():
    ap = argparse.ArgumentParser(
        prog="pysayme",
        description="Find project code that duplicates its dependencies' code.",
    )
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="report duplicates; exit 1 if any are new")
    _common(c)
    c.add_argument("--copy", type=float, help="Jaccard threshold for a copy")
    c.add_argument("--inlined", type=float, help="containment threshold for inlining")
    c.add_argument("--private", action="store_true", default=None)
    c.add_argument("--no-cache", dest="cache", action="store_false", default=None)
    c.add_argument(
        "--no-self",
        dest="self_join",
        action="store_false",
        default=None,
        help="do not report duplicates within the project",
    )
    c.add_argument(
        "--behaviour",
        action="store_true",
        default=None,
        help="also run functions to find ones behaving like library callables",
    )
    c.add_argument(
        "--vectorize",
        action="store_true",
        default=None,
        help="also report pure-Python loops that numpy would do faster",
    )
    c.add_argument("--all", action="store_true", help="include allowlisted matches")
    d = sub.add_parser("advise", help="rank loops worth refactoring (never fails)")
    _common(d)
    d.add_argument("--top", type=int, default=20, help="items to show (0: all)")
    d.add_argument(
        "--min-score", type=float, default=10.0, help="decibel-style score floor"
    )
    d.add_argument("--kind", nargs="+", help="only these kinds, e.g. PARALLEL JIT")
    d.add_argument(
        "--min-trip", type=int, help="smallest literal loop length considered"
    )
    v = sub.add_parser("vectorize", help="report loops that numpy should replace")
    _common(v)
    v.add_argument("--likely", action="store_true", help="include unproved matches")
    v.add_argument("--min-trip", type=int, help="smallest literal loop length reported")
    v.add_argument("--all", action="store_true", help="include allowlisted matches")
    r = sub.add_parser("rank", help="rank library look-alikes of project functions")
    _common(r)
    r.add_argument("--depth", type=int, default=2, help="library call depth")
    r.add_argument("--top", type=int, default=3)
    r.add_argument("--min-score", type=float, default=0.3)
    return ap


def _advise(a):
    cfg = load_config(a.config, roots=a.roots or None, vectorize_min_trip=a.min_trip)
    files = sources.project_files(cfg.roots, cfg.base, cfg.exclude)
    found = advise.advise(files, cfg, a.top, a.min_score, tuple(a.kind or ()))
    for i, item in enumerate(found, 1):
        print(f"{i:3d}. {item}")
    if not found:
        print("pysayme: nothing worth refactoring at this score")
    return 0


def _vectorize(a):
    over = {
        "vectorize": True,
        "vectorize_min_trip": a.min_trip,
        "vectorize_confidence": "likely" if a.likely else None,
    }
    cfg = load_config(a.config, roots=a.roots or None, **over)
    files = sources.project_files(cfg.roots, cfg.base, cfg.exclude)
    found = {k: str(f) for k, f in vectorize.find(files, cfg).items()}
    result = gate.check(cfg, found)
    for line in sorted((result.found if a.all else result.new).values()):
        print(line)
    if result.new:
        print("\n" + gate.VECTORIZE_HOW)
    if result.stale or result.unreasoned:
        print(str(gate.Result({}, {}, result.stale, result.unreasoned)))
    return 0 if result.ok else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS + ("-h", "--help", "--version"):
        argv.insert(0, "check")
    a = parser().parse_args(argv)
    over = {k: getattr(a, k, None) for k in ("dists", "jobs", "copy", "inlined")}
    flags = ("private", "cache", "self_join", "behaviour", "vectorize")
    over.update({k: getattr(a, k, None) for k in flags})
    over["introspect"] = a.introspect
    if a.command == "vectorize":
        return _vectorize(a)
    if a.command == "advise":
        return _advise(a)
    cfg = load_config(a.config, roots=a.roots or None, **over)
    if not cfg.dists:
        print("pysayme: no dependencies to check against", file=sys.stderr)
        return 2
    if a.command == "rank":
        files = sources.project_files(cfg.roots, cfg.base, cfg.exclude)
        lib, proj, background = semantic.corpus(
            cfg.dists, files, a.depth, cfg.introspect
        )
        ranks = semantic.ranked(lib, proj, background, a.top)
        for key, hits in sorted(ranks.items()):
            hits = [h for h in hits if h[0] >= a.min_score]
            if hits:
                print(key)
                for score, lk in hits:
                    print(f"  {score:.2f} {lk}")
        return 0
    result = gate.check(cfg)
    shown = result.found if a.all else result.new
    for line in sorted(shown.values()):
        print(line)
    if result.stale or result.unreasoned:
        print(str(gate.Result({}, {}, result.stale, result.unreasoned)))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
