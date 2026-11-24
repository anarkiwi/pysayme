"""Expression translation for vectorize mode: provenance kinds, constants, and `_Vec`,
which turns a loop-body expression into a numpy one or raises `_Opaque`."""

# pylint: disable=too-many-branches,too-many-return-statements

from __future__ import annotations

import ast
from collections import defaultdict
from fnmatch import fnmatch
from typing import NamedTuple

ARRAY, ELEM, LISTED = "array", "elem", "listed"
ARRAYISH = (ARRAY, LISTED)
PROVED, LIKELY = "proved", "likely"
NUMBA = frozenset(("jit", "njit", "vectorize", "guvectorize", "stencil", "cfunc"))
SCALARS = frozenset(
    """sum prod mean std var max min amax amin argmax argmin dot vdot inner any all
    count_nonzero median norm size ndim shape float16 float32 float64 int8 int16 int32
    int64 uint8 uint16 uint32 uint64 complex64 complex128 bool_ isscalar item
    result_type issubdtype dtype finfo iinfo ptp trace cumulative_sum""".split()
)
REDUCERS = frozenset(
    """sum prod mean std var max min argmax argmin any all item ptp trace tolist dot
    tobytes nonzero""".split()
)
MATH = {
    "sin": "sin", "cos": "cos", "tan": "tan", "asin": "arcsin", "acos": "arccos",
    "atan": "arctan", "atan2": "arctan2", "sinh": "sinh", "cosh": "cosh",
    "tanh": "tanh", "exp": "exp", "log": "log", "log10": "log10", "log2": "log2",
    "sqrt": "sqrt", "hypot": "hypot", "floor": "floor", "ceil": "ceil",
    "fabs": "abs", "pow": "power", "degrees": "degrees", "radians": "radians",
    "copysign": "copysign", "fmod": "fmod", "trunc": "trunc", "phase": "angle",
}  # fmt: skip
UFUNCS = frozenset(MATH.values()) | frozenset(
    """abs absolute angle conj conjugate real imag sign maximum minimum where clip
    square cbrt remainder floor_divide arcsinh arccosh arctanh expm1 log1p""".split()
)
CONSTANTS = frozenset(("pi", "e", "tau", "inf", "nan"))
OPS = {
    ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide",
    ast.FloorDiv: "floor_divide", ast.Mod: "remainder", ast.Pow: "power",
    ast.LShift: "left_shift", ast.RShift: "right_shift", ast.BitAnd: "bitwise_and",
    ast.BitOr: "bitwise_or", ast.BitXor: "bitwise_xor",
    ast.USub: "negative", ast.Invert: "invert", ast.Not: "logical_not",
    ast.Gt: "greater", ast.GtE: "greater_equal", ast.Lt: "less",
    ast.LtE: "less_equal", ast.Eq: "equal", ast.NotEq: "not_equal",
}  # fmt: skip
GROWERS = frozenset(
    ("numpy.append", "numpy.concatenate", "numpy.vstack", "numpy.hstack")
)
DEMOTED = {
    "sum": "sum", "max": "max", "min": "min", "sorted": "sort", "any": "any",
    "all": "all",
}  # fmt: skip
REDUCE_OPS = {
    ast.Add: "sum", ast.Mult: "prod", ast.BitOr: "bitwise_or",
    ast.BitXor: "bitwise_xor", ast.BitAnd: "bitwise_and",
}  # fmt: skip
EXCLUDE = ("tests/*", "*/tests/*", "test_*.py", "*/test_*.py", "conftest.py")


class _Opaque(Exception):
    """The loop body cannot be proved vectorizable."""

    def __init__(self, why=""):
        super().__init__(why)
        self.why = why


class Finding(NamedTuple):
    """One loop that numpy should replace."""

    path: str
    line: int
    qualname: str
    idiom: str
    target: str  # numpy callable, e.g. "cumsum"
    source: str  # the offending code, one line
    after: str  # the numpy form
    tier: str = PROVED

    def key(self):
        return f"{self.path}:{self.qualname} -> numpy:{self.target}"

    def __str__(self):
        return (
            f"{self.path}:{self.line} {self.qualname} {self.idiom}: "
            f"`{self.source}` -> `{self.after}` (numpy.{self.target})"
        )


class Src(NamedTuple):
    """What a loop iterates: element variables, an index variable, bounding array."""

    idx: str | None
    elems: dict  # variable -> array expression it takes elements of
    base: ast.AST | None  # array whose length bounds `idx`
    lo: int = 0
    hi: int = 0  # the loop runs over [lo, len(base) - hi)


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return f"{head}.{node.attr}" if head else None
    return None


def _aliases(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0]
                )
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            for a in n.names:
                out[a.asname or a.name] = f"{n.module}.{a.name}"
    return out


CONTAINERS = frozenset(
    "list List tuple Tuple dict Dict set Sequence Iterable Mapping Iterator".split()
)


def _is_array_annotation(ann):
    """`np.ndarray`, `NDArray[...]`, `ArrayLike` (optionally ``| None``), but not a
    list or tuple *of* arrays."""
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            ann = ast.parse(ann.value, mode="eval").body
        except SyntaxError:
            return False
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return any(_is_array_annotation(x) for x in (ann.left, ann.right))
    if isinstance(ann, ast.Subscript):
        ann = ann.value
    last = (_dotted(ann) or "").rsplit(".", 1)[-1]
    return last in ("ndarray", "NDArray", "ArrayLike")


def _np(name):
    return ast.Attribute(ast.Name("np", ast.Load()), name, ast.Load())


def _call(name, *args):
    return ast.Call(_np(name), list(args), [])


def _text(node, width=90):
    head, *rest = (ln.strip() for ln in ast.unparse(node).splitlines())
    text = " ".join([head, "; ".join(rest)]).strip()
    return text if len(text) <= width else text[: width - 3] + "..."


def _array(node):
    """The array an iterable expression stands for (`A.tolist()` is `A`)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "tolist" and not node.args:
            return node.func.value
    return node


def _own(body):
    """Nodes of `body` not inside a nested function, class or lambda."""
    todo = list(body)
    while todo:
        n = todo.pop(0)
        yield n
        if not isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            todo.extend(ast.iter_child_nodes(n))


def _stores(body):
    return {
        n.id
        for s in body
        for n in ast.walk(s)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _target(e):
    """The numpy callable that names a translated expression."""
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute):
        return e.func.attr
    if isinstance(e, ast.BinOp):
        return OPS.get(type(e.op), "add")
    if isinstance(e, (ast.UnaryOp,)):
        return OPS.get(type(e.op), "negative")
    if isinstance(e, ast.Compare):
        return OPS.get(type(e.ops[0]), "greater")
    return "copy"


class _Subst(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):  # pylint: disable=invalid-name
        return self.mapping.get(node.id, node)


def _offset(idx, sl):
    """`k` when `sl` is ``idx``, ``idx + k`` or ``idx - k`` with an int literal `k`."""
    if idx is None:
        return None
    if isinstance(sl, ast.Name) and sl.id == idx:
        return 0
    if isinstance(sl, ast.BinOp) and isinstance(sl.op, (ast.Add, ast.Sub)):
        l, r = sl.left, sl.right
        if (
            isinstance(l, ast.Name)
            and l.id == idx
            and isinstance(r, ast.Constant)
            and isinstance(r.value, int)
        ):
            return r.value if isinstance(sl.op, ast.Add) else -r.value
    return None


class _Vec:
    """Translate a loop-body expression into a numpy expression, or raise _Opaque."""

    def __init__(self, scope, src, written):
        self.scope, self.src, self.written = scope, src, written
        self.offsets = set()
        self.dep = False  # depends on an element or the index

    def go(self, e, depth=0):  # pylint: disable=too-many-return-statements
        sc, src = self.scope, self.src
        if isinstance(e, ast.Constant):
            return e
        if isinstance(e, ast.Name):
            if e.id in src.elems:
                self.dep = True
                return _array(src.elems[e.id])
            if e.id == src.idx:
                if src.base is None:
                    raise _Opaque("index as value without an array")
                self.dep = True
                n = ast.Call(ast.Name("len", ast.Load()), [_array(src.base)], [])
                if src.lo == 0 and src.hi == 0:
                    return _call("arange", n)
                stop = (
                    n if src.hi == 0 else ast.BinOp(n, ast.Sub(), ast.Constant(src.hi))
                )
                return _call("arange", ast.Constant(src.lo), stop)
            if e.id in self.written or sc.kind(e) in (ARRAY, LISTED):
                raise _Opaque(f"{e.id} is not loop-invariant")
            return e
        if isinstance(e, ast.Subscript):
            return self.subscript(e)
        if isinstance(e, ast.BinOp) and type(e.op) in OPS:
            return ast.BinOp(self.go(e.left, depth), e.op, self.go(e.right, depth))
        if isinstance(e, ast.UnaryOp) and type(e.op) in OPS:
            return ast.UnaryOp(e.op, self.go(e.operand, depth))
        if isinstance(e, ast.Compare) and len(e.ops) == 1 and type(e.ops[0]) in OPS:
            return ast.Compare(
                self.go(e.left, depth),
                e.ops,
                [self.go(c, depth) for c in e.comparators],
            )
        if isinstance(e, ast.BoolOp):
            op = ast.BitAnd() if isinstance(e.op, ast.And) else ast.BitOr()
            out = self.go(e.values[0], depth)
            for v in e.values[1:]:
                out = ast.BinOp(out, op, self.go(v, depth))
            return out
        if isinstance(e, ast.IfExp):
            return _call(
                "where",
                self.go(e.test, depth),
                self.go(e.body, depth),
                self.go(e.orelse, depth),
            )
        if isinstance(e, ast.Attribute):
            return self.attribute(e)
        if isinstance(e, ast.Call):
            return self.call(e, depth)
        raise _Opaque(type(e).__name__)

    def attribute(self, e):
        full = self.scope.res(e)
        if (
            full
            and full.split(".")[0] in ("math", "numpy", "cmath")
            and e.attr in CONSTANTS
        ):
            return _np(e.attr)
        if e.attr in ("real", "imag"):
            inner = self.go(e.value)
            if self.dep:
                return ast.Attribute(inner, e.attr, ast.Load())
        raise _Opaque("attribute")

    def subscript(self, e):
        sc, src = self.scope, self.src
        if not isinstance(e.value, ast.Name) or e.value.id in self.written:
            raise _Opaque("subscript base")
        base = e.value
        if sc.kind(base) != ARRAY:
            raise _Opaque("subscript of a non-array")
        off = self.index_offset(e.slice)
        if off is None:
            # an index that does not involve the loop variable is a loop invariant
            if self.mentions(e.slice) or isinstance(e.slice, ast.Slice):
                raise _Opaque("index")
            return e
        self.dep = True
        self.offsets.add(off)
        lo, stop = src.lo + off, off - src.hi
        if lo == 0 and stop == 0:
            return base
        return ast.Subscript(
            base,
            ast.Slice(
                ast.Constant(lo) if lo else None, ast.Constant(stop) if stop else None
            ),
            ast.Load(),
        )

    def mentions(self, node):
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        return bool(names & (self.written | set(self.src.elems) | {self.src.idx}))

    def index_offset(self, sl):
        return _offset(self.src.idx, sl)

    def call(self, e, depth):  # pylint: disable=too-many-return-statements
        if e.keywords or any(isinstance(a, ast.Starred) for a in e.args):
            raise _Opaque("keywords")
        sc = self.scope
        full = sc.res(e.func) or ""
        head, _, last = full.rpartition(".")
        args = e.args
        if full in sc.helpers and depth < 3:
            fn = sc.helpers[full]
            names = [a.arg for a in fn.args.args]
            if len(names) != len(args):
                raise _Opaque("arity")
            sub = _Subst(dict(zip(names, args))).visit(
                ast.parse(ast.unparse(fn.body[-1].value), mode="eval").body
            )
            return self.go(sub, depth + 1)
        if head in ("math", "cmath") and last in MATH:
            return _call(MATH[last], *[self.go(a, depth) for a in args])
        if head.split(".")[0] == "numpy" and last in UFUNCS:
            return _call(last, *[self.go(a, depth) for a in args])
        if full == "abs" and len(args) == 1:
            return _call("abs", self.go(args[0], depth))
        if full in ("min", "max") and len(args) == 2:
            return _call(
                "minimum" if full == "min" else "maximum",
                *[self.go(a, depth) for a in args],
            )
        if full == "pow" and len(args) == 2:
            return _call("power", *[self.go(a, depth) for a in args])
        if full in ("float", "int", "complex", "bool") and len(args) == 1:
            inner = self.go(args[0], depth)
            before = self.dep
            if before and not isinstance(inner, ast.Constant):
                return ast.Call(
                    ast.Attribute(inner, "astype", ast.Load()),
                    [ast.Name(full, ast.Load())],
                    [],
                )
            return ast.Call(e.func, [inner], [])
        scalar = any(
            isinstance(a, ast.Call)
            and sc.res(a.func) in ("int", "float", "complex", "bool")
            and any(
                isinstance(n, ast.Name) and n.id in self.src.elems for n in ast.walk(a)
            )
            for a in args
        )
        raise _Opaque("call-scalar" if scalar else "call")
