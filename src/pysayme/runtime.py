"""Public names found by importing a distribution's public modules in child
processes, so library imports cannot pollute, hang or crash the caller, and the
worker pool those children come from."""

from __future__ import annotations

import functools
import importlib
import inspect
import multiprocessing
import os
import sys
import tempfile
import threading
import warnings
from collections import defaultdict
from typing import NamedTuple

from tqdm import tqdm

from . import extract, probe

IMPORT_TIMEOUT = 60.0
CHILD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
}


class Compiled(NamedTuple):
    """A public callable with no Python source: a ufunc, builtin, or Cython or
    Rust function."""

    path: str
    kind: str
    signature: str | None
    doc: str


class Callable(NamedTuple):
    """A public callable that can be probed, under its shortest public path."""

    path: str
    spec: probe.Spec


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


def _spec(rec, obj, module):
    if rec[0] not in ("def", "compiled"):
        return None
    try:
        return probe.spec(obj, module)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def exports(mod):
    """[(name, record, probe Spec or None)] of `mod`'s ``__all__``, else its public
    names."""
    names = getattr(mod, "__all__", None)
    if not isinstance(names, (list, tuple)):
        names = [n for n in vars(mod) if not n.startswith("_")]
    out = []
    for n in names:
        try:
            obj = getattr(mod, n)
            rec = describe(obj)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if rec is not None and isinstance(n, str):
            out.append((n, rec, _spec(rec, obj, mod.__name__)))
    return out


def module_exports(name):
    try:
        return exports(importlib.import_module(name))
    except BaseException:  # pylint: disable=broad-exception-caught
        return []


def _serve(fn, home, conn):
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    warnings.simplefilter("ignore")
    os.chdir(home)
    while True:
        try:
            task = conn.recv()
        except EOFError:
            break
        try:
            out = fn(task)
        except BaseException:  # pylint: disable=broad-exception-caught
            out = None
        conn.send(out)


class Workers:
    """Spawned children, isolated from the caller's output, working directory and
    thread pools, that run `fn` on tasks. A task whose child dies or overruns its
    timeout yields None, and a fresh child takes the next task."""

    def __init__(self, fn, jobs=1):
        self.fn, self.jobs = fn, max(1, jobs or os.cpu_count() or 1)
        self._ctx = multiprocessing.get_context("spawn")
        self._home = tempfile.TemporaryDirectory(  # pylint: disable=consider-using-with
            prefix="pysayme-"
        )
        self._lock = threading.Lock()
        self._idle = []

    def _start(self):
        recv, send = self._ctx.Pipe()
        args = (self.fn, self._home.name, send)
        child = self._ctx.Process(target=_serve, args=args, daemon=True)
        with self._lock:
            saved = {k: os.environ.get(k) for k in CHILD_ENV}
            os.environ.update(CHILD_ENV)
            try:
                child.start()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k)
                    else:
                        os.environ[k] = v
        send.close()
        return child, recv

    @staticmethod
    def _stop(worker):
        worker[0].kill()
        worker[0].join()
        worker[1].close()

    def map(self, tasks, timeout, desc="work", progress=True, unit="task"):
        """[result or None] of each of `tasks`; `timeout` is seconds, or a function
        of the task giving them."""
        tasks = list(tasks)
        out = [None] * len(tasks)
        todo = iter(enumerate(tasks))
        limit = timeout if callable(timeout) else lambda _: timeout
        opts = {"desc": desc, "unit": unit, "leave": False, "file": sys.stderr}

        def lane(shown):
            with self._lock:
                worker = self._idle.pop() if self._idle else None
                item = next(todo, None)
            while item is not None:
                worker = worker or self._start()
                try:
                    worker[1].send(item[1])
                    done = worker[1].poll(limit(item[1]))
                    if done:
                        out[item[0]] = worker[1].recv()
                except (EOFError, OSError):
                    done = False
                if not done:
                    self._stop(worker)
                    worker = None
                shown.update()
                with self._lock:
                    item = next(todo, None)
            if worker is not None:
                with self._lock:
                    self._idle.append(worker)

        with tqdm(total=len(tasks), disable=not progress, **opts) as shown:
            lanes = [
                threading.Thread(target=lane, args=(shown,))
                for _ in range(min(self.jobs, len(tasks)))
            ]
            for t in lanes:
                t.start()
            for t in lanes:
                t.join()
        return out

    def close(self):
        for worker in self._idle:
            self._stop(worker)
        self._idle = []
        self._home.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def collect(modules, timeout=IMPORT_TIMEOUT, desc="import", progress=True, jobs=1):
    """{module: [(name, record, spec)]} of each of `modules` imported in a spawned
    child; one that times out or kills its child is skipped."""
    with Workers(module_exports, jobs) as pool:
        got = pool.map(modules, timeout, desc, progress, "module")
    return {m: g for m, g in zip(modules, got) if g is not None}


def _rank(path):
    return path.count("."), path


def public_modules(entries):
    """The public modules of (path, display, module) `entries`, sorted so that
    modules of one subpackage are imported by the same child in turn."""
    return sorted({m for _, _, m in entries if m and not extract.private_module(m)})


def introspect(entries, timeout=IMPORT_TIMEOUT, desc="import", progress=True, jobs=1):
    """({(module, name): exporting modules}, [Compiled], [Callable]) of a
    distribution's (path, display, module) `entries`: the public modules that bind
    each of its top-level definitions under its own name, its compiled public
    callables, and its public callables that can be probed."""
    collected = collect(public_modules(entries), timeout, desc, progress, jobs)
    return resolve(entries, collected)


def resolve(entries, collected):
    """introspect()'s result from `collected` {module: [(name, record, spec)]}."""
    files = {os.path.realpath(p): m for p, _, m in entries if str(p).endswith(".py")}
    known = {m for _, _, m in entries}
    tops = {m.split(".", 1)[0] for m in known}
    found = defaultdict(set)
    compiled, probed = {}, {}
    for mod, items in collected.items():
        for name, rec, spec in items:
            path = f"{mod}:{name}"
            if rec[0] == "compiled":
                owner = rec[1]
                if owner is None or owner.split(".", 1)[0] in tops:
                    _keep(compiled, rec[1:4], Compiled(path, *rec[3:]))
                    _keep(probed, rec[1:4], spec and Callable(path, spec))
                continue
            if rec[0] == "def":
                src, qual = files.get(rec[1]), rec[2]
                if src is not None:
                    _keep(probed, rec[1:], spec and Callable(path, spec))
            elif rec[3] in files:
                src, qual = files[rec[3]], rec[4]
            else:
                src, qual = (rec[1], rec[2]) if rec[1] in known else (None, None)
            if src is not None and qual == name and src != mod:
                found[src, name].add(mod)
    return dict(found), sorted(compiled.values()), sorted(probed.values())


def _keep(table, ident, item):
    """Record `item` under `ident` unless one with a shorter path is there."""
    old = table.get(ident)
    if item and (old is None or _rank(item.path) < _rank(old.path)):
        table[ident] = item
