"""Broadcast rewrite of loop nests: every loop variable becomes an axis.

Vectorizing the innermost loop adds an axis to the body; repeating that outward turns a
loop nest into one array expression over all its axes (the inside-out idea of
Liu, Liu and Wang, "Vectorizer", arXiv 2609.08088, specialised to this module's
provenance and without masks). Two shapes are rewritten:

MAP     ``for i..: for j..: out[i, j] = e``          ->  ``out[:, :] = e`` over (i, j)
REDUCE  ``for k..: acc = c; for n..: acc += e; out.append(acc)``  ->  a sum over axis n

The rewrite is a hint with a stated size: it builds one intermediate array with an
element per loop-nest iteration, so a nest whose product is large wants chunking.
"""

# pylint: disable=too-many-return-statements,too-many-branches

from __future__ import annotations

import ast
from typing import NamedTuple

from .vecexpr import (
    ARRAY,
    Src,
    _array,
    _call,
    _np,
    _offset,
    _Opaque,
    _stores,
    _target,
    _text,
    _Vec,
)


class Axis(NamedTuple):
    """One loop of the nest: its variable, bounds, and the array it walks (if any)."""

    var: str
    lo: ast.AST
    stop: ast.AST
    array: ast.AST | None  # set for ``for v in A``; None for ``range``


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _axis(scope, loop):
    """The `Axis` of a ``for`` over a range or an ndarray, else None."""
    if loop.orelse or not isinstance(loop.target, ast.Name):
        return None
    it, var = loop.iter, loop.target.id
    if (
        isinstance(it, ast.Call)
        and scope.res(it.func) == "range"
        and 1 <= len(it.args) <= 2
    ):
        lo = it.args[0] if len(it.args) == 2 else ast.Constant(0)
        return Axis(var, lo, it.args[-1], None)
    if scope.kind(it) == ARRAY:
        return Axis(
            var, ast.Constant(0), ast.Call(ast.Name("len", ast.Load()), [it], []), it
        )
    return None


def _slice(ax, off=0):
    """The slice of an array an axis (shifted by `off`) walks."""

    def shift(e):
        if off == 0:
            return e
        return ast.BinOp(e, ast.Add() if off > 0 else ast.Sub(), ast.Constant(abs(off)))

    lo = (
        None
        if isinstance(ax.lo, ast.Constant) and ax.lo.value == 0 and off == 0
        else shift(ax.lo)
    )
    stop = shift(ax.stop)
    if lo is None and ax.array is not None and ast.dump(stop) == ast.dump(ax.stop):
        return ast.Slice()
    return ast.Slice(lo, stop)


class _VecN(_Vec):
    """`_Vec` over several axes: variables and subscripts become broadcastable arrays."""

    def __init__(self, scope, axes, written):
        super().__init__(scope, Src(None, {}, None), written)
        self.axes = axes
        self.index = {a.var: k for k, a in enumerate(axes)}

    def place(self, node, used):
        """`node`, whose dimensions are the axes `used`, expanded to every axis."""
        n = len(self.axes)
        if list(used) == list(range(n)):
            return node
        elts = [ast.Slice() if k in used else ast.Constant(None) for k in range(n)]
        return ast.Subscript(node, ast.Tuple(elts, ast.Load()), ast.Load())

    def go(self, e, depth=0):
        if isinstance(e, ast.Name) and e.id in self.index:
            k = self.index[e.id]
            ax = self.axes[k]
            self.dep = True
            if ax.array is not None:
                return self.place(_array(ax.array), [k])
            return self.place(_call("arange", ax.lo, ax.stop), [k])
        return super().go(e, depth)

    def subscript(self, e):
        idx, node = [], e
        while isinstance(node, ast.Subscript):
            sl = node.slice
            idx = (list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]) + idx
            node = node.value
        if not isinstance(node, ast.Name) or node.id in self.written:
            raise _Opaque("subscript base")
        if self.scope.kind(node) != ARRAY:
            raise _Opaque("subscript of a non-array")
        out, used = [], []
        for part in idx:
            hit = self.axis_of(part)
            if hit is None:
                if _names(part) & (set(self.index) | self.written):
                    raise _Opaque("index")
                out.append(part)
                continue
            k, off = hit
            if used and k <= used[-1]:
                raise _Opaque("axes out of order")
            used.append(k)
            sl = _slice(self.axes[k], off)
            if off == 0 and sl.lower is None and _text(sl.upper) == f"len({node.id})":
                sl = ast.Slice()
            out.append(sl)
        if not used:
            return e
        self.dep = True
        whole = ast.Subscript(
            node, out[0] if len(out) == 1 else ast.Tuple(out, ast.Load()), ast.Load()
        )
        if all(
            isinstance(p, ast.Slice) and p.lower is None and p.upper is None
            for p in out
        ):
            whole = node
        return self.place(whole, used)

    def axis_of(self, part):
        """(axis number, offset) when `part` is an axis variable plus a literal."""
        for var, k in self.index.items():
            off = _offset(var, part)
            if off is not None:
                if self.axes[k].array is not None:
                    raise _Opaque("element variable used as an index")
                return k, off
        return None


def _store(stmt, names):
    """``(out, slices)`` when `stmt` stores into ``out[v0, v1, ...]`` / ``out[v0][v1]``
    with exactly the loop variables `names`, in order; ``out.append`` gives slices None.
    """
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        node, idx = stmt.targets[0], []
        while isinstance(node, ast.Subscript):
            sl = node.slice
            idx = (list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]) + idx
            node = node.value
        if (
            isinstance(node, ast.Name)
            and [getattr(i, "id", None) for i in idx] == names
        ):
            return node.id, stmt.value
    return None


def _shape(axes):
    parts = []
    for ax in axes:
        parts.append(
            _text(ax.stop)
            if isinstance(ax.lo, ast.Constant) and ax.lo.value == 0
            else f"{_text(ax.stop)} - {_text(ax.lo)}"
        )
    return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"


def rewrite(scope, node, done):
    """``(idiom, target, after)`` for a loop nest rooted at `node`, else None.

    Every ``for`` of the nest is added to `done`."""
    axes, loops, cur = [], [], node
    while True:
        ax = _axis(scope, cur)
        if ax is None:
            return None
        axes.append(ax)
        loops.append(cur)
        body = [s for s in cur.body if not isinstance(s, ast.Pass)]
        if len(body) == 1 and isinstance(body[0], ast.For):
            cur = body[0]
            continue
        break
    reduce = _reduction(scope, body, axes, loops)
    if len(axes) + (reduce is not None) < 2:
        return None
    lits = [
        a.stop.value
        for a in axes
        if isinstance(a.stop, ast.Constant) and isinstance(a.stop.value, int)
    ]
    if len(lits) == len(axes):
        total = 1
        for n in lits:
            total *= n
        if total < scope.cfg.vectorize_min_trip:
            return None
    if reduce is not None:
        axes, loops, outer_vars, op, init, out, expr = reduce
    else:
        got = _store(body[0], [a.var for a in axes]) if len(body) == 1 else None
        if got is None:
            return None
        outer_vars, op, init = [a.var for a in axes], None, None
        out, expr = got
    written = _stores([s for loop in loops for s in loop.body])
    vec = _VecN(scope, axes, written)
    for a in axes:
        if (_names(a.lo) | _names(a.stop)) & (set(vec.index) | written):
            return None  # triangular or dependent bounds
    try:
        e = vec.go(expr)
    except _Opaque:
        return None
    if not vec.dep:
        return None
    done.update(id(loop) for loop in loops)
    note = f"  # builds an array of shape {_shape(axes)}"
    if op is None:
        k = len(axes)
        target = ast.Subscript(
            ast.Name(out, ast.Load()),
            ast.Tuple([_slice(a) for a in axes], ast.Load()),
            ast.Load(),
        )
        if k == 1:
            target.slice = _slice(axes[0])
        return "OUTER", _target(e), f"{ast.unparse(target)} = {ast.unparse(e)}{note}"
    name = "sum" if isinstance(op, ast.Add) else "prod"
    full = ast.unparse(
        ast.Call(_np(name), [e], [ast.keyword("axis", ast.Constant(-1))])
    )
    neutral = 0 if name == "sum" else 1
    if not (isinstance(init, ast.Constant) and init.value == neutral):
        full = f"{ast.unparse(init)} {'+' if name == 'sum' else '*'} {full}"
    outer_axes = axes[:-1]
    if outer_vars is None:
        return "OUTER", name, f"{out} = {full}{note}"
    tgt = ast.Subscript(
        ast.Name(out, ast.Load()),
        ast.Tuple([_slice(a) for a in outer_axes], ast.Load()),
        ast.Load(),
    )
    if len(outer_axes) == 1:
        tgt.slice = _slice(outer_axes[0])
    return "OUTER", name, f"{ast.unparse(tgt)} = {full}{note}"


def _appends(stmt, acc):
    """Whether `stmt` is ``<list>.append(acc)``."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    call = stmt.value
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == acc
    )


def _reduction(scope, body, axes, loops):
    """Match ``acc = c; for r..: acc += e; <store acc>`` at the bottom of the nest."""
    if len(body) != 3:
        return None
    init, inner, tail = body
    if not (
        isinstance(init, ast.Assign)
        and len(init.targets) == 1
        and isinstance(init.targets[0], ast.Name)
    ):
        return None
    if not isinstance(init.value, ast.Constant) or not isinstance(inner, ast.For):
        return None
    acc = init.targets[0].id
    red = _axis(scope, inner)
    ibody = [s for s in inner.body if not isinstance(s, ast.Pass)]
    if red is None or len(ibody) != 1 or not isinstance(ibody[0], ast.AugAssign):
        return None
    aug = ibody[0]
    if not (
        isinstance(aug.target, ast.Name)
        and aug.target.id == acc
        and isinstance(aug.op, (ast.Add, ast.Mult))
    ):
        return None
    outer_names = [a.var for a in axes]
    out = None
    if len(axes) == 1 and _appends(tail, acc):
        return (
            axes + [red],
            loops + [inner],
            None,
            aug.op,
            init.value,
            tail.value.func.value.id,
            aug.value,
        )
    got = (
        _store(ast.Assign(targets=tail.targets, value=tail.value), outer_names)
        if isinstance(tail, ast.Assign)
        else None
    )
    if got is None or not (isinstance(got[1], ast.Name) and got[1].id == acc):
        return None
    out = got[0]  # pylint: disable=unused-variable
    return (
        axes + [red],
        loops + [inner],
        outer_names,
        aug.op,
        init.value,
        out,
        aug.value,
    )
