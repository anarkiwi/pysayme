"""Public names found by importing a distribution's public modules in child
processes, so library imports cannot pollute, hang or crash the caller."""

from __future__ import annotations

import functools
import importlib
import inspect
import multiprocessing
import os
import sys
import warnings
from collections import defaultdict
from typing import NamedTuple

from tqdm import tqdm

from . import extract

IMPORT_TIMEOUT = 60.0


class Compiled(NamedTuple):
    """A public callable with no Python source: a ufunc, builtin, or Cython or
    Rust function."""

    path: str
    kind: str
    signature: str | None
    doc: str


def _first_line(doc):
    return doc.strip().split("\n", 1)[0].strip() if isinstance(doc, str) else ""


def _signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _class_code(cls):
    """(file, class qualname) from the first Python method of `cls`, for classes
    whose ``__module__`` names a re-exporting module."""
    for v in vars(cls).values():
        v = getattr(v, "__func__", getattr(v, "fget", v))
        if inspect.isfunction(v):
            owner, _, _ = v.__code__.co_qualname.rpartition(".")
            if owner:
                return os.path.realpath(v.__code__.co_filename), owner
    return None, None


def describe(obj):
    """A picklable record of an exported object: ``("def", file, qualname)``,
    ``("class", module, qualname, file, qualname)``, ``("compiled", module, name,
    kind, signature, doc)``, or None for anything else."""
    obj = inspect.unwrap(obj)
    if inspect.ismethod(obj):
        obj = obj.__func__
    if inspect.isfunction(obj):
        code = obj.__code__
        return "def", os.path.realpath(code.co_filename), code.co_qualname
    if inspect.isclass(obj):
        return "class", obj.__module__, obj.__qualname__, *_class_code(obj)
    if (
        not callable(obj)
        or inspect.ismodule(obj)
        or isinstance(obj, functools.partial)
        or inspect.isfunction(getattr(type(obj), "__call__", None))
    ):
        return None
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "")
    owner = getattr(obj, "__module__", None)
    return (
        "compiled",
        owner if isinstance(owner, str) else None,
        name,
        type(obj).__name__,
        _signature(obj),
        _first_line(getattr(obj, "__doc__", None)),
    )


def exports(mod):
    """[(name, record)] of `mod`'s ``__all__``, else its public names."""
    names = getattr(mod, "__all__", None)
    if not isinstance(names, (list, tuple)):
        names = [n for n in vars(mod) if not n.startswith("_")]
    out = []
    for n in names:
        try:
            rec = describe(getattr(mod, n))
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if rec is not None and isinstance(n, str):
            out.append((n, rec))
    return out


def _serve(modules, conn):
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    warnings.simplefilter("ignore")
    for name in modules:
        try:
            got = exports(importlib.import_module(name))
        except BaseException:  # pylint: disable=broad-exception-caught
            got = []
        conn.send(got)
    conn.close()


def collect(modules, timeout=IMPORT_TIMEOUT, desc="import", progress=True):
    """{module: [(name, record)]} of each of `modules` imported in a spawned child;
    one that times out or kills its child is skipped and a new child continues."""
    ctx = multiprocessing.get_context("spawn")
    out, pos = {}, 0
    opts = {"desc": desc, "unit": "module", "leave": False, "file": sys.stderr}
    with tqdm(total=len(modules), disable=not progress, **opts) as shown:
        while pos < len(modules):
            recv, send = ctx.Pipe(duplex=False)
            child = ctx.Process(target=_serve, args=(modules[pos:], send), daemon=True)
            child.start()
            send.close()
            try:
                while pos < len(modules) and recv.poll(timeout):
                    out[modules[pos]] = recv.recv()
                    pos += 1
                    shown.update()
            except EOFError:
                pass
            if pos < len(modules):
                pos += 1
                shown.update()
            child.kill()
            child.join()
            recv.close()
    return out


def introspect(entries, timeout=IMPORT_TIMEOUT, desc="import", progress=True):
    """({(module, name): exporting modules}, [Compiled]) of a distribution's
    (path, display, module) `entries`: the public modules that bind each of its
    top-level definitions under its own name, and its compiled public callables."""
    files = {os.path.realpath(p): m for p, _, m in entries if str(p).endswith(".py")}
    known = {m for _, _, m in entries}
    tops = {m.split(".", 1)[0] for m in known}
    modules = sorted(m for m in known if m and not extract.private_module(m))
    found = defaultdict(set)
    compiled = {}
    for mod, items in collect(modules, timeout, desc, progress).items():
        for name, rec in items:
            if rec[0] == "compiled":
                owner = rec[1]
                if owner is None or owner.split(".", 1)[0] in tops:
                    path = f"{mod}:{name}"
                    ident = rec[1:4]
                    old = compiled.get(ident)
                    rank = (path.count("."), path)
                    if old is None or rank < (old.path.count("."), old.path):
                        compiled[ident] = Compiled(path, rec[3], rec[4], rec[5])
                continue
            if rec[0] == "def":
                src, qual = files.get(rec[1]), rec[2]
            elif rec[3] in files:
                src, qual = files[rec[3]], rec[4]
            else:
                src, qual = (rec[1], rec[2]) if rec[1] in known else (None, None)
            if src is not None and qual == name and src != mod:
                found[src, name].add(mod)
    return dict(found), sorted(compiled.values())
