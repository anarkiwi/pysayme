"""Advise mode: rank the loops of a project by how much a refactor could win.

``pysayme advise`` never fails. It lists, most valuable first, the loops where a manual
change is worth the effort, and says which change:

VECTORIZE   a numpy call replaces the loop (the `vectorize` detector's proved findings)
PARALLEL    independent iterations with a heavy body: spread them over cores
CONCURRENT  independent iterations that wait on I/O: threads or asyncio
JIT         a scalar recurrence numpy cannot express: numba, or scipy.signal.lfilter
SERIAL      a heavy loop held serial by a carried dependence: name the dependence
MEMBERSHIP  ``x in a_list`` / ``.index`` / ``.count`` inside a loop: O(n*m), use a set
PAIRWISE    all-pairs loops over one collection: O(n^2)
LIST_HEAD   ``pop(0)`` / ``insert(0, ...)`` in a loop: use a deque
STRCAT      ``s += ...`` on a string in a loop: use ``"".join``
REPEATED    a scan or sort of a loop-invariant collection on every iteration

The cost model is a static proxy, not a measurement: an iterable of unknown size counts
as `UNKNOWN_TRIP` items, an array as `ARRAY_TRIP`, and each node, call and nested loop
adds weight. Independence of iterations assumes the callees are pure; the advice says so.
"""

# pylint: disable=too-many-branches,too-many-return-statements,too-many-locals
# pylint: disable=protected-access,too-many-boolean-expressions

from __future__ import annotations

import ast
import math
import pstats
from pathlib import Path
from collections import defaultdict
from typing import NamedTuple

from . import vectorize
from .vecexpr import ARRAY, LISTED, _text

ARRAY_TRIP, UNKNOWN_TRIP = 1000, 100
PER_ITER, MIN_COST = 60, 5000  # heavy enough to be worth the cores
CHEAP = frozenset(
    "len range int float abs min max bool str isinstance enumerate zip tuple list "
    "round divmod pow sum any all ord chr hash id type iter next reversed".split()
)
IO_CALLS = frozenset(
    "open print input time.sleep os.system os.walk os.listdir shutil".split()
)
IO_ROOTS = frozenset(
    ("requests", "urllib", "httpx", "socket", "subprocess", "http", "aiohttp")
)
IO_ATTRS = frozenset(
    """read write readline readlines read_text write_text read_bytes write_bytes send
    recv execute fetchall fetchone commit get post put urlopen connect""".split()
)
ASSOC = (ast.Add, ast.Mult, ast.BitOr, ast.BitAnd, ast.BitXor)
COLLECT = frozenset(("append", "extend", "add", "update"))
MUTATE = frozenset(("pop", "remove", "clear", "insert", "sort", "reverse", "discard"))
CONCURRENCY = (
    "concurrent.futures",
    "multiprocessing",
    "joblib",
    "numba",
    "ThreadPool",
    "asyncio",
)
MULT = {  # how much of a loop's cost a refactor of this kind could remove
    "VECTORIZE": 3.0, "PARALLEL": 1.0, "CONCURRENT": 1.0, "JIT": 2.0, "SERIAL": 0.3,
    "MEMBERSHIP": 4.0, "PAIRWISE": 4.0, "LIST_HEAD": 2.0, "STRCAT": 1.0, "REPEATED": 2.0,
}  # fmt: skip
CUMULATIVE_KINDS = frozenset({"PARALLEL", "CONCURRENT", "SERIAL"})  # the callees' time is the prize
ADVICE = {
    "PARALLEL": "independent iterations, heavy body: run them across cores "
    "(process pool); assumes the callees are pure",
    "CONCURRENT": "independent iterations that wait on I/O: threads or asyncio",
    "JIT": "scalar recurrence numpy cannot express: numba @njit, or "
    "scipy.signal.lfilter if it is a linear filter",
    "SERIAL": "heavy loop held serial by a carried dependence: restructure it (blocking, "
    "a tree reduction, or hoist the part that does not depend on it)",
    "MEMBERSHIP": "scans a list on every iteration, O(n*m): build a set or dict once, "
    "outside the loop",
    "PAIRWISE": "all-pairs over one collection, O(n^2): numpy broadcasting, sort and "
    "sweep, a hash join, or scipy.spatial for distances",
    "LIST_HEAD": "O(n) per operation: collections.deque",
    "STRCAT": "quadratic string building: collect the pieces and ''.join them",
    "REPEATED": "rescans or re-sorts a loop-invariant collection every iteration: "
    "compute it once, outside the loop",
}


class Profile:  # pylint: disable=too-few-public-methods
    """Measured time per function, from a pstats file (``python -m cProfile -o out.prof``).

    A pstats file has no per-loop times, so a function's time is shared among its advised
    loops by their static cost. Which time depends on what the advice removes: kinds that
    run the body across cores (`CUMULATIVE_KINDS`) gain from the callees' time too, while
    a rewrite that only removes interpreter overhead (vectorize, JIT, a better algorithm)
    leaves the callees as slow as they were, so it is scored on the function's own time."""

    def __init__(self, path):
        stats = pstats.Stats(str(path)).stats  # pylint: disable=no-member
        self.rows = [
            (Path(f).as_posix(), line, name, ct, tt, calls)
            for (f, line, name), (_, calls, tt, ct, _) in stats.items()
        ]

    def find(self, rel, name, lo, hi):
        """(cumulative seconds, own seconds, calls) of the function `name` defined on lines
        lo..hi of `rel`, or None when the profile never saw it run."""
        hits = [
            (ct, tt, calls)
            for f, line, n, ct, tt, calls in self.rows
            if n == name and lo <= line <= hi and (f == rel or f.endswith("/" + rel))
        ]
        if not hits:
            return None
        return tuple(sum(h[i] for h in hits) for i in range(3))


class Advice(NamedTuple):
    """One place worth refactoring."""

    path: str
    line: int
    qualname: str
    kind: str
    score: float
    why: str
    action: str
    source: str

    def __str__(self):
        return (
            f"{self.score:5.1f}  {self.kind:<10} {self.path}:{self.line} {self.qualname}\n"
            f"          {self.why}\n          -> {self.action}\n          `{self.source}`"
        )


class _Flow(ast.NodeVisitor):
    """What a loop body reads and writes, in execution order."""

    def __init__(self, targets, written):
        self.first, self.loads, self.aug = {}, set(), defaultdict(list)
        self.targets, self.written = set(targets), set(written)
        self.shared, self.early, self.io, self.collect = [], False, False, 0

    def visit_Name(self, node):  # pylint: disable=invalid-name
        if isinstance(node.ctx, ast.Load):
            self.first.setdefault(node.id, "load")
            self.loads.add(node.id)
        else:
            self.first.setdefault(node.id, "store")

    def visit_Assign(self, node):  # pylint: disable=invalid-name
        self.visit(node.value)
        for t in node.targets:
            self.store(t)

    def visit_AugAssign(self, node):  # pylint: disable=invalid-name
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.first.setdefault(node.target.id, "aug")
            self.aug[node.target.id].append(node.op)
        else:
            self.store(node.target)

    def visit_For(self, node):  # pylint: disable=invalid-name
        self.visit(node.iter)
        self.visit(node.target)
        for s in node.body + node.orelse:
            self.visit(s)

    def visit_Call(self, node):  # pylint: disable=invalid-name
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            if f.attr in COLLECT:
                self.collect += 1
            elif f.attr in MUTATE and f.value.id not in self.targets:
                self.shared.append(f"{f.value.id}.{f.attr}()")
        self.generic_visit(node)

    def visit_ListComp(self, node):  # pylint: disable=invalid-name
        """Generators bind before the element reads them."""
        for g in node.generators:
            self.visit(g.iter)
            self.visit(g.target)
            for cond in g.ifs:
                self.visit(cond)
        for part in (node.elt,) if hasattr(node, "elt") else (node.key, node.value):
            self.visit(part)

    visit_SetComp = visit_GeneratorExp = visit_DictComp = visit_ListComp  # type: ignore

    def visit_Break(self, node):  # pylint: disable=invalid-name,unused-argument
        self.early = True

    visit_Return = visit_Yield = visit_Await = visit_Break  # type: ignore

    def visit_FunctionDef(self, node):  # pylint: disable=invalid-name,unused-argument
        return None

    visit_AsyncFunctionDef = visit_ClassDef = visit_Lambda = visit_FunctionDef

    def store(self, target):
        if isinstance(target, ast.Attribute):
            self.shared.append(_text(target))
        elif isinstance(target, ast.Subscript):
            names = {n.id for n in ast.walk(target.slice) if isinstance(n, ast.Name)}
            if not names & self.targets:
                self.shared.append(_text(target))
            self.visit(target.value)
            self.visit(target.slice)
        else:
            self.visit(target)

    def carried(self):
        out = []
        for name, first in self.first.items():
            if first == "load" and name in self.written:
                out.append(name)
            elif first == "aug" and (
                name in self.loads
                or not all(isinstance(o, ASSOC) for o in self.aug[name])
            ):
                out.append(name)
        return sorted(out)


def _flow(loop, targets):
    written = {
        n.id
        for s in loop.body
        for n in ast.walk(s)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    } | {
        s.target.id
        for s in loop.body
        for s in [s]
        if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name)
    }
    fl = _Flow(targets, written)
    for s in loop.body:
        fl.visit(s)
    return fl


class _Advisor:
    """Advice for one function."""

    def __init__(self, scope, rel, qual, ctx, measured=None):
        self.sc, self.rel, self.qual, self.ctx = scope, rel, qual, ctx
        self.measured, self.fn_cost = measured, None
        self.out, self.marked = [], {}
        self.listish, self.strs = self.containers()
        self.vec = {
            f.line: f
            for f in vectorize._Finder(scope, rel, qual).run()
            if f.tier == vectorize.PROVED
        }  # pylint: disable=protected-access

    # -- names --------------------------------------------------------------------
    def containers(self):
        """Names holding lists/tuples (membership is linear) and strings."""
        lists, other, strs = set(), set(), set()
        for p in self.sc.params:
            ann = ast.unparse(p.annotation) if p.annotation is not None else ""
            if ann.split("[", maxsplit=1)[0].rsplit(".", 1)[-1] in (
                "list",
                "List",
                "Sequence",
                "tuple",
                "Tuple",
            ):
                lists.add(p.arg)
        for n in vectorize._own(self.sc.body):  # pylint: disable=protected-access
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                name, v = n.targets[0].id, n.value
                call = self.sc.res(v.func) if isinstance(v, ast.Call) else None
                if isinstance(v, (ast.List, ast.ListComp, ast.Tuple)) or call in (
                    "list",
                    "sorted",
                    "tuple",
                ):
                    lists.add(name)
                elif isinstance(
                    v, (ast.Set, ast.SetComp, ast.Dict, ast.DictComp)
                ) or call in ("set", "frozenset", "dict"):
                    other.add(name)
                elif isinstance(v, ast.Constant) and isinstance(v.value, str):
                    strs.add(name)
                elif self.sc.kind(v) == ARRAY:
                    other.add(name)
        return lists - other, strs

    # -- cost ---------------------------------------------------------------------
    def trip(self, node):
        if isinstance(node, ast.While):
            return UNKNOWN_TRIP
        return self.iter_trip(node.iter)

    def iter_trip(self, it):
        if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
            return len(it.elts)
        if isinstance(it, ast.Call) and self.sc.res(it.func) == "range":
            last = it.args[-1] if it.args else None
            if (
                isinstance(last, ast.Constant)
                and isinstance(last.value, int)
                and len(it.args) <= 2
            ):
                lo = (
                    it.args[0].value
                    if len(it.args) == 2 and isinstance(it.args[0], ast.Constant)
                    else 0
                )
                return max(last.value - lo, 0)
        if self.sc.kind(vectorize._array(it)) in (
            ARRAY,
            LISTED,
        ):  # pylint: disable=protected-access
            return ARRAY_TRIP
        return UNKNOWN_TRIP

    def call_weight(self, n):
        sc = self.sc
        full = sc.res(n.func) or ""
        head = full.split(".")[0]
        if full in self.ctx["defs"]:
            return 150 if self.ctx["defs"][full] else 40
        if full in IO_CALLS or head in IO_ROOTS:
            return 200
        if full in CHEAP or head in ("math", "cmath"):
            return 3
        if head in sc.roots:
            return 30
        if isinstance(n.func, ast.Attribute):
            return 25 if n.func.attr in IO_ATTRS else 15
        return 40

    def node_weight(self, n):
        if isinstance(n, ast.Call):
            return self.call_weight(n)
        return (
            1
            if isinstance(
                n, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Subscript, ast.Attribute)
            )
            else 0
        )

    def expr_cost(self, node):
        total, todo = 0.0, [node]
        while todo:
            n = todo.pop()
            if isinstance(
                n, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)
            ):
                inner = self.expr_cost(n.elt if hasattr(n, "elt") else n.value)
                for g in n.generators:
                    inner = (
                        inner + sum(self.expr_cost(c) for c in g.ifs)
                    ) * self.iter_trip(g.iter)
                total += inner
                continue
            if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue
            total += self.node_weight(n)
            todo.extend(ast.iter_child_nodes(n))
        return total

    def block_cost(self, stmts):
        return sum(self.stmt_cost(s) for s in stmts)

    def stmt_cost(self, s):
        if isinstance(s, (ast.For, ast.While)):
            head = self.expr_cost(s.iter if isinstance(s, ast.For) else s.test)
            return self.trip(s) * (head + self.block_cost(s.body)) + self.block_cost(
                s.orelse
            )
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return 0.0
        subs = [c for c in ast.iter_child_nodes(s) if isinstance(c, ast.stmt)]
        subs += [
            h
            for c in ast.iter_child_nodes(s)
            if isinstance(c, ast.ExceptHandler)
            for h in c.body
        ]
        rest = [
            c
            for c in ast.iter_child_nodes(s)
            if not isinstance(c, (ast.stmt, ast.ExceptHandler))
        ]
        return sum(self.expr_cost(c) for c in rest) + self.block_cost(subs)

    # -- walking ------------------------------------------------------------------
    def run(self):
        self.walk(self.sc.body, [])
        pairs = {a.line for a in self.out if a.kind == "PAIRWISE"}
        return [a for a in self.out if a.kind == "PAIRWISE" or a.line not in pairs]

    def walk(self, stmts, chain):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            self.algorithmic(s, chain)
            if isinstance(s, (ast.For, ast.While)):
                self.loop(s, chain)
                self.walk(s.body + s.orelse, chain + [s])
            else:
                subs = [c for c in ast.iter_child_nodes(s) if isinstance(c, ast.stmt)]
                subs += [
                    h
                    for c in ast.iter_child_nodes(s)
                    if isinstance(c, ast.ExceptHandler)
                    for h in c.body
                ]
                self.walk(subs, chain)

    def add(self, node, kind, cost, why, action=None, source=None):
        score = 10 * math.log10(1 + cost * MULT[kind] / 1000)
        if self.ctx["profile"] is not None:
            if self.measured is None:
                if not self.ctx["keep_unprofiled"]:
                    return
                score = 0.0  # no measurement: below everything measured
                why += "; not seen running in the profile"
            else:
                if self.fn_cost is None:
                    self.fn_cost = max(self.block_cost(self.sc.body), 1.0)
                total, own, calls = self.measured
                seconds = total if kind in CUMULATIVE_KINDS else own
                share = min(1.0, cost / self.fn_cost)
                score = 10 * math.log10(1 + seconds * share * 1e6 * MULT[kind] / 1000)
                basis = "cumulative" if kind in CUMULATIVE_KINDS else "own (callees excluded)"
                why += f"; measured {seconds:.3g}s {basis} over {calls} calls in this function"
        self.out.append(
            Advice(
                self.rel, node.lineno, self.qual, kind, round(score, 1),
                why, action or ADVICE[kind], source or _text(node),
            )
        )  # fmt: skip

    # -- loops --------------------------------------------------------------------
    def loop(self, node, chain):
        trip = self.trip(node)
        if trip < self.ctx["min_trip"]:
            return
        cost = self.stmt_cost(node)
        if any(self.marked.get(id(c)) for c in chain):
            return  # the outermost loop already carries the advice
        found = self.vec.get(node.lineno)
        if found:
            self.marked[id(node)] = "VECTORIZE"
            self.add(
                node,
                "VECTORIZE",
                cost,
                f"{found.idiom}, ~{trip} array elements",
                f"{found.after}  (numpy.{found.target})",
            )
            return
        targets = (
            {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            if isinstance(node, ast.For)
            else set()
        )
        fl = _flow(node, targets)
        carried = fl.carried()
        per_iter = self.block_cost(node.body)
        base = f"~{trip} iterations, per-iteration weight {per_iter:.0f}"
        if carried:
            if self.numeric(node):
                self.marked[id(node)] = "JIT"
                self.add(node, "JIT", cost, f"{base}; carried: {', '.join(carried)}")
            elif cost >= self.ctx["serial_cost"]:
                self.add(node, "SERIAL", cost, f"{base}; carried: {', '.join(carried)}")
            return
        if fl.shared or fl.early or per_iter < PER_ITER or cost < MIN_COST:
            return
        io = self.has_io(node)
        kind = "CONCURRENT" if io else "PARALLEL"
        self.marked[id(node)] = kind
        extra = (
            f"; project already uses {', '.join(self.ctx['libs'])}"
            if self.ctx["libs"]
            else ""
        )
        self.add(node, kind, cost, f"{base}; no carried state, no shared writes{extra}")

    def has_io(self, node):
        for n in (m for s in node.body for m in ast.walk(s)):
            if isinstance(n, ast.Call):
                full = self.sc.res(n.func) or ""
                if full in IO_CALLS or full.split(".")[0] in IO_ROOTS:
                    return True
                if (
                    isinstance(n.func, ast.Attribute)
                    and n.func.attr in IO_ATTRS
                    and self.sc.kind(n.func.value) != ARRAY
                ):
                    if n.func.attr in (
                        "read_text",
                        "write_text",
                        "read_bytes",
                        "write_bytes",
                        "urlopen",
                        "fetchall",
                        "execute",
                        "recv",
                        "send",
                    ):
                        return True
        return False

    def numeric(self, node):
        """Only arithmetic, subscripts, math calls: something a JIT compiles well."""
        for n in ast.walk(node):
            if isinstance(
                n, (ast.JoinedStr, ast.Dict, ast.Set, ast.Lambda, ast.Try, ast.With)
            ):
                return False
            if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)):
                return False
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute) and n.func.attr in (
                    "append",
                    "real",
                    "imag",
                ):
                    continue
                full = self.sc.res(n.func) or ""
                head = full.split(".")[0]
                if not (full in CHEAP or head in ("math", "cmath", "numpy")):
                    return False
        return True

    # -- algorithmic patterns -----------------------------------------------------
    def algorithmic(self, s, chain):
        chain_trip = 1
        for c in chain:
            chain_trip *= self.trip(c)
        if not chain or chain_trip < self.ctx["min_trip"]:
            if isinstance(s, ast.For):
                self.pairwise(s, chain)
            return
        exprs = {
            ast.For: lambda x: [x.iter],
            ast.While: lambda x: [x.test],
            ast.If: lambda x: [x.test],
            ast.With: lambda x: [i.context_expr for i in x.items],
            ast.Try: lambda x: [],
        }.get(type(s), lambda x: [x])(s)
        seen = set()
        for e in exprs:
            for n in ast.walk(e):
                self.pattern(n, s, chain, chain_trip, seen)
        if isinstance(s, ast.For):
            self.pairwise(s, chain)

    def pattern(
        self, n, s, chain, chain_trip, seen
    ):  # pylint: disable=too-many-arguments
        stored = {
            t.id
            for c in chain
            for x in ast.walk(c)
            for t in [x]
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)
        }
        if isinstance(n, ast.Compare) and any(
            isinstance(o, (ast.In, ast.NotIn)) for o in n.ops
        ):
            right = n.comparators[-1]
            if (
                isinstance(right, ast.Name)
                and right.id in self.listish
                and ("m", right.id) not in seen
            ):
                seen.add(("m", right.id))
                self.add(
                    s,
                    "MEMBERSHIP",
                    chain_trip * UNKNOWN_TRIP,
                    f"`{_text(n, 60)}` scans list `{right.id}` on each of ~{chain_trip} iterations",
                    source=_text(n),
                )
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
        ):
            name, attr = n.func.value.id, n.func.attr
            if (
                attr in ("index", "count", "remove")
                and name in self.listish
                and ("m", name) not in seen
            ):
                seen.add(("m", name))
                self.add(
                    s,
                    "MEMBERSHIP",
                    chain_trip * UNKNOWN_TRIP,
                    f"`{name}.{attr}(...)` is linear, on each of ~{chain_trip} iterations",
                )
            elif (
                attr == "pop"
                and n.args
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == 0
            ) or (
                attr == "insert"
                and n.args
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == 0
            ):
                self.add(
                    s,
                    "LIST_HEAD",
                    chain_trip * UNKNOWN_TRIP,
                    f"`{name}.{attr}(0 ...)` shifts the whole list, ~{chain_trip} times",
                )
            elif attr == "sort" and name not in stored and name in self.listish:
                self.add(
                    s,
                    "REPEATED",
                    chain_trip * UNKNOWN_TRIP,
                    f"`{name}.sort()` on every iteration",
                    source=_text(n),
                )
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("sorted", "max", "min", "sum")
            and len(n.args) == 1
        ):
            a = n.args[0]
            if isinstance(a, ast.Name) and a.id not in stored and a.id not in self.strs:
                if ("r", a.id, n.func.id) not in seen:
                    seen.add(("r", a.id, n.func.id))
                    self.add(
                        s,
                        "REPEATED",
                        chain_trip * UNKNOWN_TRIP,
                        f"`{n.func.id}({a.id})`, loop-invariant, ~{chain_trip} times",
                    )
        if n is s and isinstance(s, ast.AugAssign) and isinstance(s.op, ast.Add):
            if isinstance(s.target, ast.Name) and s.target.id in self.strs:
                self.add(
                    s,
                    "STRCAT",
                    chain_trip * UNKNOWN_TRIP,
                    f"`{s.target.id} += ...` builds a string over ~{chain_trip} iterations",
                )

    def pairwise(self, s, chain):
        """A loop over the same collection as an enclosing one (or its tail)."""
        key = self.iter_key(s.iter)
        if key is None:
            return
        for c in chain:
            if (
                isinstance(c, ast.For)
                and self.iter_key(c.iter) == key
                and self.trip(s) >= self.ctx["min_trip"]
            ):
                self.add(
                    c,
                    "PAIRWISE",
                    self.stmt_cost(c),
                    f"nested loops over the same collection `{key}`, ~{self.trip(c)}^2 pairs",
                    source=_text(c),
                )
                self.marked.setdefault(id(c), "PAIRWISE")
                return

    @staticmethod
    def iter_key(it):
        """A name for what a loop iterates, normalising range(len(X)) and range(i + 1, len(X))."""
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and it.args
        ):
            tail = it.args[-1]
            if (
                isinstance(tail, ast.Call)
                and isinstance(tail.func, ast.Name)
                and tail.func.id == "len"
                and tail.args
            ):
                return ast.unparse(tail.args[0])
            return None
        if isinstance(it, (ast.Name, ast.Attribute)):
            return ast.unparse(it)
        if (
            isinstance(it, ast.Subscript)
            and isinstance(it.slice, ast.Slice)
            and isinstance(it.value, (ast.Name, ast.Attribute))
        ):
            return ast.unparse(it.value)
        return None


def _libs(texts):
    return [k for k in CONCURRENCY if any(k in t for t in texts)]


def _defs_cost(tree):
    """{name: whether the function contains a loop} for the module's functions."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = any(
                isinstance(x, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
                for x in ast.walk(n)
            )
    return out


def scan(
    source, rel, cfg, libs=(), serial_cost=50_000, profile=None, keep_unprofiled=False
):
    """Advice for one file's `source`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    aliases = vectorize._aliases(tree)  # pylint: disable=protected-access
    helpers = vectorize._helpers(tree)  # pylint: disable=protected-access
    ctx = {
        "defs": _defs_cost(tree),
        "min_trip": cfg.vectorize_min_trip,
        "libs": list(libs),
        "profile": profile,
        "keep_unprofiled": keep_unprofiled,
        "serial_cost": serial_cost,
    }
    scope = vectorize._Scope(
        aliases, helpers, cfg, (), tree.body
    )  # pylint: disable=protected-access
    module = profile.find(rel, "<module>", 1, 1) if profile else None
    out = _Advisor(scope, rel, "<module>", ctx, module).run()
    for qual, fn in vectorize._defs(tree.body):  # pylint: disable=protected-access
        if vectorize._decorated(fn, aliases):  # pylint: disable=protected-access
            continue
        a = fn.args
        sc = vectorize._Scope(
            aliases, helpers, cfg, a.posonlyargs + a.args + a.kwonlyargs, fn.body
        )  # pylint: disable=protected-access
        first = min([d.lineno for d in fn.decorator_list] + [fn.lineno])
        measured = profile.find(rel, fn.name, first, fn.lineno) if profile else None
        out += _Advisor(sc, rel, qual, ctx, measured).run()
    return out


def advise(
    files, cfg, top=20, min_score=10.0, kinds=(), profile=None, keep_unprofiled=False
):
    """The `top` advice items of `files` scoring at least `min_score`, best first. With a
    `profile` (a `Profile`), scores come from measured time and functions that never ran
    are dropped, unless `keep_unprofiled`."""
    parsed = [(text, rel) for rel, text in vectorize.read_sources(files, cfg)]
    libs = _libs([text for text, _ in parsed])
    found = []
    for text, rel in parsed:
        found += scan(
            text, rel, cfg, libs, profile=profile, keep_unprofiled=keep_unprofiled
        )
    found = [
        a for a in found if a.score >= min_score and (not kinds or a.kind in kinds)
    ]
    found.sort(key=lambda a: (-a.score, a.path, a.line))
    return found[:top] if top else found
