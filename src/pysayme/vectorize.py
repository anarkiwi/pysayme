"""Vectorize mode: pure-Python loops over numpy arrays that numpy would do faster.

Three static steps per function (see docs/vectorize.md):

1. *Provenance*: which local names hold ndarrays, which hold one of their elements.
2. *Legality*: a loop over an array is matched to an idiom only if its body is a pure
   expression of same-index reads (or one recognised accumulator). Translating the
   body to numpy and checking legality are the same walk (`_Vec`): anything it cannot
   translate raises `_Opaque` and the loop stays silent.
3. *Cost*: loops over small literal ranges and arrays are skipped.

Each `Finding` is keyed ``path:qualname -> numpy:callable`` like the other modes, so
the allowlist, its reasons, and its shrink-only rule apply unchanged.
"""

# pylint: disable=too-many-branches,too-many-return-statements

from __future__ import annotations

import ast
from collections import defaultdict
from fnmatch import fnmatch

from .vecexpr import (  # pylint: disable=unused-import
    ARRAY,
    ELEM,
    LISTED,
    ARRAYISH,
    DEMOTED,
    EXCLUDE,
    Finding,
    GROWERS,
    LIKELY,
    NUMBA,
    PROVED,
    REDUCERS,
    REDUCE_OPS,
    SCALARS,
    Src,
    _Opaque,
    _Vec,
    _aliases,
    _array,
    _call,
    _dotted,
    _is_array_annotation,
    _np,
    _offset,
    _own,
    _stores,
    _target,
    _text,
)


class _Scope:
    """Provenance and loop recognition for one function (or module body)."""

    def __init__(self, aliases, helpers, cfg, params=(), body=()):
        self.al, self.helpers, self.cfg = aliases, helpers, cfg
        self.env, self.lens, self.small = {}, {}, set()
        self.roots = tuple(cfg.array_libs)
        self.params = list(params)
        self.body = list(body)
        for _ in range(3):
            self._pass()

    # -- names --------------------------------------------------------------------
    def res(self, node):
        d = _dotted(node)
        if not d:
            return None
        head, _, rest = d.partition(".")
        full = self.al.get(head, head)
        return f"{full}.{rest}" if rest else full

    def kind(self, e):  # pylint: disable=too-many-return-statements,too-many-branches
        if isinstance(e, ast.Name):
            return self.env.get(e.id)
        if isinstance(e, ast.Call):
            full = self.res(e.func) or ""
            if full.split(".")[0] in self.roots:
                return None if full.rsplit(".", 1)[-1] in SCALARS else ARRAY
            if isinstance(e.func, ast.Attribute) and self.kind(e.func.value) == ARRAY:
                m = e.func.attr
                return LISTED if m == "tolist" else None if m in REDUCERS else ARRAY
            if (
                full in ("list", "sorted", "tuple")
                and len(e.args) == 1
                and self.kind(e.args[0]) in ARRAYISH
            ):
                return LISTED
            return None
        if isinstance(e, (ast.BinOp, ast.Compare)):
            parts = [e.left] + (
                [e.right] if isinstance(e, ast.BinOp) else e.comparators
            )
            ks = {self.kind(p) for p in parts}
            return ARRAY if ARRAY in ks else ELEM if ELEM in ks else None
        if isinstance(e, ast.UnaryOp):
            return None if isinstance(e.op, ast.Not) else self.kind(e.operand)
        if isinstance(e, ast.Subscript):
            if self.kind(e.value) == ARRAY:
                sl = e.slice
                masked = self.kind(sl) == ARRAY
                sliced = isinstance(sl, ast.Slice) or (
                    isinstance(sl, ast.Tuple)
                    and any(isinstance(x, ast.Slice) for x in sl.elts)
                )
                return ARRAY if masked or sliced else ELEM
            return ELEM if self.kind(e.value) == LISTED else None
        if isinstance(e, ast.Attribute):
            return (
                ARRAY
                if self.kind(e.value) == ARRAY and e.attr in ("T", "real", "imag")
                else None
            )
        if isinstance(e, ast.IfExp):
            a, b = self.kind(e.body), self.kind(e.orelse)
            return a if a == b else None
        return None

    def _pass(self):
        sets = defaultdict(set)
        for p in self.params:
            if p.annotation is not None and _is_array_annotation(p.annotation):
                sets[p.arg].add(ARRAY)
        lens, small = {}, set()
        for n in _own(self.body):
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                name = n.targets[0].id
                sets[name].add(self.kind(n.value))
                arr = self.length_of(n.value)
                if arr is not None:
                    lens[name] = arr
                if self.is_small(n.value):
                    small.add(name)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                sets[n.target.id].add(self.kind(ast.BinOp(n.target, n.op, n.value)))
            elif isinstance(n, ast.For):
                src = self.source(n.iter, n.target, check_small=False)
                for v in (src.elems if src else {}):
                    sets[v].add(ELEM)
                if src and src.idx:
                    sets[src.idx].add(None)
        self.env = {
            k: next(iter(v)) for k, v in sets.items() if len(v) == 1 and None not in v
        }
        self.lens, self.small = lens, small

    # -- sizes --------------------------------------------------------------------
    def is_small(self, e):
        if not isinstance(e, ast.Call):
            return False
        full = self.res(e.func) or ""
        if (
            full in ("numpy.array", "numpy.asarray")
            and e.args
            and isinstance(e.args[0], (ast.List, ast.Tuple))
        ):
            return len(e.args[0].elts) < self.cfg.vectorize_min_trip
        if (
            full in ("numpy.zeros", "numpy.ones", "numpy.arange", "numpy.empty")
            and e.args
        ):
            a = e.args[-1] if full == "numpy.arange" and len(e.args) == 1 else e.args[0]
            return (
                isinstance(a, ast.Constant)
                and isinstance(a.value, int)
                and a.value < self.cfg.vectorize_min_trip
            )
        return False

    def length_of(self, e):
        """The array whose length `e` is: ``len(A)``, ``A.size``, ``A.shape[0]``."""
        if isinstance(e, ast.Call) and self.res(e.func) == "len" and len(e.args) == 1:
            return e.args[0]
        if isinstance(e, ast.Attribute) and e.attr == "size":
            return e.value
        if (
            isinstance(e, ast.Subscript)
            and isinstance(e.value, ast.Attribute)
            and e.value.attr == "shape"
        ):
            if isinstance(e.slice, ast.Constant) and e.slice.value == 0:
                return e.value.value
        return None

    def bound(self, e):
        arr = self.length_of(e)
        if arr is not None:
            return arr, 0
        if isinstance(e, ast.Name) and e.id in self.lens:
            return self.lens[e.id], 0
        if (
            isinstance(e, ast.BinOp)
            and isinstance(e.op, ast.Sub)
            and isinstance(e.right, ast.Constant)
        ):
            arr, off = self.bound(e.left)
            if arr is not None and isinstance(e.right.value, int):
                return arr, off + e.right.value
        return None, 0

    # -- loops --------------------------------------------------------------------
    def source(
        self, it, target, check_small=True
    ):  # pylint: disable=too-many-return-statements
        src = self._source(it, target)
        if src is None or not check_small:
            return src
        arrays = list(src.elems.values()) + ([src.base] if src.base is not None else [])
        if any(
            isinstance(_array(a), ast.Name) and _array(a).id in self.small
            for a in arrays
        ):
            return None
        return src

    def _source(self, it, target):  # pylint: disable=too-many-return-statements
        if isinstance(it, ast.Call):
            full = self.res(it.func)
            args = it.args
            if (
                full == "enumerate"
                and len(args) == 1
                and self.kind(_array(args[0])) in ARRAYISH
            ):
                if (
                    isinstance(target, ast.Tuple)
                    and len(target.elts) == 2
                    and all(isinstance(t, ast.Name) for t in target.elts)
                ):
                    i, v = target.elts
                    return Src(i.id, {v.id: args[0]}, _array(args[0]))
                return None
            if (
                full == "zip"
                and len(args) >= 2
                and any(self.kind(a) == ARRAY for a in args)
            ):
                ok = all(self.kind(_array(a)) in ARRAYISH for a in args)
                if (
                    ok
                    and isinstance(target, ast.Tuple)
                    and len(target.elts) == len(args)
                ):
                    if all(isinstance(t, ast.Name) for t in target.elts):
                        return Src(
                            None, {t.id: a for t, a in zip(target.elts, args)}, None
                        )
                return None
            if full == "range" and 1 <= len(args) <= 2 and isinstance(target, ast.Name):
                lo = 0
                if len(args) == 2:
                    if not (
                        isinstance(args[0], ast.Constant)
                        and isinstance(args[0].value, int)
                    ):
                        return None
                    lo = args[0].value
                stop = args[-1]
                arr, hi = self.bound(stop)
                if arr is not None and self.kind(arr) in ARRAYISH:
                    return Src(target.id, {}, _array(arr), lo, hi)
                if isinstance(stop, ast.Constant) and isinstance(stop.value, int):
                    if stop.value - lo < self.cfg.vectorize_min_trip:
                        return None
                return Src(target.id, {}, None, lo, 0)
        if self.kind(_array(it)) in ARRAYISH and isinstance(target, ast.Name):
            return Src(None, {target.id: it}, None)
        return None


class _Finder:
    """Findings of one function body."""

    def __init__(self, scope, rel, qualname):
        self.sc, self.rel, self.qual = scope, rel, qualname
        self.out = []
        self.seen = set()

    def add(self, node, idiom, target, after, tier=PROVED, source=None):
        src = source if source is not None else _text(node)
        self.out.append(
            Finding(self.rel, node.lineno, self.qual, idiom, target, src, after, tier)
        )

    def run(self):
        sc = self.sc
        nodes = list(_own(sc.body))
        done = set()
        for n in nodes:
            if isinstance(n, ast.Call):
                self.call(n, done)
        for n in nodes:
            if isinstance(n, (ast.ListComp, ast.GeneratorExp)) and id(n) not in done:
                if isinstance(n, ast.ListComp):
                    self.comp(n, None, n, done)
        for n in nodes:
            if isinstance(n, (ast.For, ast.While)):
                self.grow(n)
            if isinstance(n, ast.For) and id(n) not in done:
                f = self.loop(n, done)
                if f:
                    self.add(n, *f)
        return self.out

    # -- calls: builtin reducers on arrays, comprehension consumers ------------------
    def call(self, n, done):
        sc = self.sc
        full = sc.res(n.func) or ""
        if len(n.args) != 1 or n.keywords:
            return
        a = n.args[0]
        if full in DEMOTED and sc.kind(a) == ARRAY:
            t = DEMOTED[full]
            meth = (
                f"{ast.unparse(a)}.{t}()"
                if t != "sort"
                else f"np.sort({ast.unparse(a)})"
            )
            self.add(n, "DEMOTE", t, meth)
        elif isinstance(a, (ast.ListComp, ast.GeneratorExp)):
            if full in ("sum", "min", "max", "any", "all", "len") or full in (
                "numpy.array",
                "numpy.asarray",
                "list",
                "tuple",
                "sorted",
            ):
                self.comp(a, full, n, done)

    def comp(self, c, consumer, whole, done):  # pylint: disable=too-many-locals
        sc = self.sc
        if len(c.generators) != 1 or c.generators[0].is_async:
            return
        g = c.generators[0]
        src = sc.source(g.iter, g.target)
        if src is None or (not src.elems and src.base is None):
            return
        done.add(id(c))
        v = _Vec(sc, src, set())
        try:
            masks = [v.go(t) for t in g.ifs]
            mask = None
            for m in masks:
                mask = m if mask is None else ast.BinOp(mask, ast.BitAnd(), m)
            if consumer == "len":
                if mask is None:
                    return
                self.add(
                    whole,
                    "REDUCE",
                    "count_nonzero",
                    ast.unparse(_call("count_nonzero", mask)),
                )
                return
            if mask is not None and isinstance(c.elt, ast.Name) and c.elt.id == src.idx:
                pick = _call("flatnonzero", mask)
                if src.lo:
                    pick = ast.BinOp(pick, ast.Add(), ast.Constant(src.lo))
                if consumer in (
                    None,
                    "numpy.array",
                    "numpy.asarray",
                    "list",
                    "tuple",
                    "sorted",
                ):
                    self.add(whole, "SELECT", "flatnonzero", ast.unparse(pick))
                return
            e = v.go(c.elt)
            if not v.dep:
                return
        except _Opaque as o:
            if o.why == "call-scalar" and any(
                sc.kind(_array(a)) == ARRAY for a in src.elems.values()
            ):
                self.add(
                    whole,
                    "MAPCALL",
                    "ufunc",
                    "a Python call per element: give the callee an array-valued form",
                    LIKELY,
                )
            return
        pick = e if mask is None else ast.Subscript(e, mask, ast.Load())
        plain = consumer in (None, "list", "tuple", "sorted")
        if consumer in (
            None,
            "numpy.array",
            "numpy.asarray",
            "list",
            "tuple",
            "sorted",
        ):
            idiom = "MAP" if mask is None else "SELECT"
            target = _target(e) if mask is None else "compress"
            after = ast.unparse(pick) + (
                "  # .tolist() if a list is needed" if plain else ""
            )
            elt = c.elt
            cast = (
                isinstance(elt, ast.Call)
                and sc.res(elt.func) in ("int", "float", "complex", "bool")
                and len(elt.args) == 1
                and isinstance(elt.args[0], ast.Name)
                and elt.args[0].id in src.elems
            )
            if cast and plain and mask is None:
                self.add(
                    whole,
                    "TOLIST",
                    "ndarray.tolist",
                    f"{ast.unparse(pick)}.tolist()",
                    LIKELY,
                )
                return
            self.add(whole, idiom, target, after)
        elif consumer == "sum":
            self.reduce(whole, "sum", pick, e, mask)
        elif consumer in ("min", "max", "any", "all"):
            self.add(whole, "REDUCE", consumer, ast.unparse(_call(consumer, pick)))

    def reduce(self, node, op, pick, e, mask):
        if (
            op == "sum"
            and mask is None
            and isinstance(e, ast.BinOp)
            and isinstance(e.op, ast.Mult)
        ):
            if not isinstance(e.left, ast.Constant) and not isinstance(
                e.right, ast.Constant
            ):
                self.add(
                    node, "REDUCE", "dot", ast.unparse(_call("dot", e.left, e.right))
                )
                return
        self.add(node, "REDUCE", op, ast.unparse(_call(op, pick)))

    # -- statement loops --------------------------------------------------------------
    def loop(self, node, done):  # pylint: disable=too-many-return-statements
        sc = self.sc
        if node.orelse:
            return None
        src = sc.source(node.iter, node.target)
        if src is None:
            return None
        body = [s for s in node.body if not isinstance(s, ast.Pass)]
        if (not src.elems and src.base is None) or any(
            isinstance(s, ast.For) for s in body
        ):
            return self.outer(body, done)
        v = _Vec(sc, src, _stores(body))
        try:
            return self.match(body, src, v)
        except _Opaque:
            return None

    def outer(self, body, done):
        inner = [s for s in body if isinstance(s, ast.For)]
        if len(inner) != 1:
            return None
        f = self.loop(inner[0], done)
        if not f or f[0] not in ("REDUCE", "MAP", "STENCIL"):
            return None
        rest = [s for s in body if s is not inner[0]]
        if not all(isinstance(s, (ast.Assign, ast.Expr, ast.AugAssign)) for s in rest):
            return None
        done.add(id(inner[0]))
        return (
            "OUTER",
            "einsum",
            "broadcast the outer index against the inner array; reduce along an axis",
        )

    @staticmethod
    def appended(s):
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            c = s.value
            if (
                isinstance(c.func, ast.Attribute)
                and c.func.attr == "append"
                and isinstance(c.func.value, ast.Name)
                and len(c.args) == 1
            ):
                return c.func.value.id, c.args[0]
        return None

    @staticmethod
    def stored(s, idx):
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Subscript)
        ):
            t = s.targets[0]
            if (
                isinstance(t.value, ast.Name)
                and isinstance(t.slice, ast.Name)
                and t.slice.id == idx
            ):
                return t.value.id, s.value
        return None

    def match(
        self, body, src, v
    ):  # pylint: disable=too-many-return-statements,too-many-branches
        if len(body) == 2:
            return self.scan_pair(body, src, v)
        if len(body) != 1:
            return None
        s = body[0]
        ap = self.appended(s) or (self.stored(s, src.idx) if src.idx else None)
        if ap:
            out, expr = ap
            if src.idx and self.stored(s, src.idx):
                rec = self.scan_recurrence(out, expr, src, v)
                if rec:
                    return rec
            e = v.go(expr)
            if not v.dep:
                return None
            if len(v.offsets) > 1:
                t = (
                    "diff"
                    if isinstance(e, ast.BinOp)
                    and isinstance(e.op, ast.Sub)
                    and v.offsets == {0, 1}
                    else "convolve"
                )
                return "STENCIL", t, f"{out} = {ast.unparse(e)}"
            return "MAP", _target(e), f"{out} = {ast.unparse(e)}"
        if (
            isinstance(s, ast.AugAssign)
            and isinstance(s.target, ast.Name)
            and type(s.op) in REDUCE_OPS
        ):
            e = v.go(s.value)
            if not v.dep:
                return None
            op = REDUCE_OPS[type(s.op)]
            if op == "sum" and isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
                return (
                    "REDUCE",
                    "dot",
                    f"{s.target.id} += {ast.unparse(_call('dot', e.left, e.right))}",
                )
            fn = (
                _call(op, e)
                if op in ("sum", "prod")
                else ast.Call(ast.Attribute(_np(op), "reduce", ast.Load()), [e], [])
            )
            return "REDUCE", op, f"{s.target.id} {ast.unparse(s.op)}= {ast.unparse(fn)}"
        if isinstance(s, ast.If) and not s.orelse and len(s.body) in (1, 2):
            return self.guarded(s, src, v)
        return None

    def scan_pair(self, body, src, v):
        a, b = body
        if not (
            isinstance(a, ast.AugAssign)
            and isinstance(a.target, ast.Name)
            and isinstance(a.op, (ast.Add, ast.Mult))
        ):
            return None
        out = self.appended(b) or (self.stored(b, src.idx) if src.idx else None)
        if not out or not (isinstance(out[1], ast.Name) and out[1].id == a.target.id):
            return None
        v.written.discard(a.target.id)
        v.written.discard(out[0])
        e = v.go(a.value)
        if not v.dep:
            return None
        fn = "cumsum" if isinstance(a.op, ast.Add) else "cumprod"
        join = "+" if fn == "cumsum" else "*"
        return (
            "SCAN",
            fn,
            f"{out[0]} = {a.target.id} {join} {ast.unparse(_call(fn, e))}",
        )

    def scan_recurrence(self, out, expr, src, v):
        if not (
            isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Mult))
        ):
            return None
        for prev, rest in ((expr.left, expr.right), (expr.right, expr.left)):
            if (
                isinstance(prev, ast.Subscript)
                and isinstance(prev.value, ast.Name)
                and prev.value.id == out
            ):
                if _offset(src.idx, prev.slice) == -1:
                    v.written.discard(out)
                    e = v.go(rest)
                    fn = "cumsum" if isinstance(expr.op, ast.Add) else "cumprod"
                    join = "+" if fn == "cumsum" else "*"
                    seed = f"{out}[{src.lo - 1}]"
                    return (
                        "SCAN",
                        fn,
                        f"{out}[{src.lo}:] = {seed} {join} {ast.unparse(_call(fn, e))}",
                    )
        return None

    def guarded(self, s, src, v):  # pylint: disable=too-many-return-statements
        inner = s.body
        if len(inner) == 1:
            t = inner[0]
            ap = self.appended(t)
            if ap:
                m = v.go(s.test)
                out, expr = ap
                if isinstance(expr, ast.Name) and expr.id == src.idx:
                    pick = _call("flatnonzero", m)
                    return (
                        "SELECT",
                        "flatnonzero",
                        f"{out} = {ast.unparse(pick)}"
                        + (f" + {src.lo}" if src.lo else ""),
                    )
                e = v.go(expr)
                return (
                    "SELECT",
                    "compress",
                    f"{out} = {ast.unparse(ast.Subscript(e, m, ast.Load()))}",
                )
            if (
                isinstance(t, ast.AugAssign)
                and isinstance(t.target, ast.Name)
                and isinstance(t.op, ast.Add)
            ):
                m = v.go(s.test)
                if isinstance(t.value, ast.Constant):
                    return (
                        "REDUCE",
                        "count_nonzero",
                        f"{t.target.id} += {ast.unparse(_call('count_nonzero', m))}",
                    )
                e = v.go(t.value)
                return (
                    "REDUCE",
                    "sum",
                    f"{t.target.id} += "
                    + ast.unparse(_call("sum", ast.Subscript(e, m, ast.Load()))),
                )
        return self.argext(s, src, v)

    def argext(self, s, src, v):
        t = s.test
        if not (
            isinstance(t, ast.Compare)
            and len(t.ops) == 1
            and isinstance(t.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE))
        ):
            return None
        left, right, op = t.left, t.comparators[0], t.ops[0]
        best = x = None
        if isinstance(right, ast.Name) and right.id in v.written:
            best, x = right.id, left
        elif isinstance(left, ast.Name) and left.id in v.written:
            best, x, op = (
                left.id,
                right,
                {ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Lt: ast.Gt, ast.LtE: ast.GtE}[
                    type(op)
                ](),
            )
        if best is None:
            return None
        assigns = {}
        for a in s.body:
            if not (
                isinstance(a, ast.Assign)
                and len(a.targets) == 1
                and isinstance(a.targets[0], ast.Name)
            ):
                return None
            assigns[a.targets[0].id] = a.value
        if best not in assigns or ast.unparse(assigns[best]) != ast.unparse(x):
            return None
        pos = [
            n
            for n, val in assigns.items()
            if n != best and isinstance(val, ast.Name) and val.id == src.idx
        ]
        if len(assigns) != 1 + len(pos) or len(pos) > 1:
            return None
        fn = "argmax" if isinstance(op, (ast.Gt, ast.GtE)) else "argmin"
        e = v.go(x)
        who = pos[0] if pos else "i"
        shift = f"{src.lo} + " if src.lo else ""
        return "ARGEXT", fn, f"{who} = {shift}int({ast.unparse(_call(fn, e))})"

    def grow(self, loop):
        if isinstance(loop, ast.For):
            src = self.sc.source(loop.iter, loop.target)
            if src is None and not self.sc.kind(_array(loop.iter)):
                if not (
                    isinstance(loop.iter, ast.Call)
                    and self.sc.res(loop.iter.func) == "range"
                ):
                    return
        for s in _own(loop.body):
            if (
                isinstance(s, ast.Assign)
                and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name)
            ):
                v = s.value
                if isinstance(v, ast.Call) and self.sc.res(v.func) in GROWERS:
                    name = s.targets[0].id
                    if any(
                        isinstance(n, ast.Name) and n.id == name
                        for a in v.args
                        for n in ast.walk(a)
                    ):
                        self.add(
                            s,
                            "GROW",
                            "concatenate",
                            "collect the pieces in a list, `np.concatenate` once",
                        )
                        return


def _decorated(fn, aliases):
    for d in fn.decorator_list:
        node = d.func if isinstance(d, ast.Call) else d
        dotted = _dotted(node) or ""
        full = aliases.get(dotted.split(".")[0], dotted.split(".")[0])
        if (
            full.split(".")[0] in ("numba", "cython")
            or dotted.rsplit(".", 1)[-1] in NUMBA
        ):
            return True
    return False


def _helpers(tree):
    """Module-level one-expression functions that may be inlined, by dotted name."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and not n.decorator_list:
            body = [
                s
                for s in n.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            ]
            a = n.args
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and body[0].value is not None
            ):
                if not (a.kwonlyargs or a.vararg or a.kwarg or a.defaults):
                    out[n.name] = n
    return out


def _defs(body, prefix=""):
    """(qualname, def) of every function under `body`, methods and nested ones included."""
    for n in _own(body):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield prefix + n.name, n
            yield from _defs(n.body, f"{prefix}{n.name}.")
        elif isinstance(n, ast.ClassDef):
            yield from _defs(n.body, f"{prefix}{n.name}.")


def scan(source, rel, cfg):
    """Findings in one file's `source`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    aliases = _aliases(tree)
    if not any(v.split(".")[0] in cfg.array_libs for v in aliases.values()):
        return []
    helpers = _helpers(tree)
    out = _Finder(_Scope(aliases, helpers, cfg, (), tree.body), rel, "<module>").run()
    for qual, fn in _defs(tree.body):
        if _decorated(fn, aliases):
            continue
        a = fn.args
        sc = _Scope(
            aliases, helpers, cfg, a.posonlyargs + a.args + a.kwonlyargs, fn.body
        )
        out += _Finder(sc, rel, qual).run()
    return out


def find(files, cfg):
    """Findings of every (path, rel, _) in `files`, deduplicated by allowlist key and
    filtered to `cfg.vectorize_confidence`."""
    out = {}
    for path, rel, _ in files:
        if any(fnmatch(rel, p) for p in cfg.vectorize_exclude):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for f in scan(text, rel, cfg):
            if f.tier == LIKELY and cfg.vectorize_confidence != LIKELY:
                continue
            out.setdefault(f.key(), f)
    return out
