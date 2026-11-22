"""AST extraction: normalized shingle sets of functions, and module-level literal
constants."""

from __future__ import annotations

import ast
import builtins
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

K = 4
BUILTINS = frozenset(dir(builtins))
_MIX = np.uint64(0x100000001B3)
_TOKEN_IDS: dict[str, int] = {}
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
REGISTRATIONS = frozenset(
    (
        "box",
        "infer",
        "infer_getattr",
        "infer_global",
        "intrinsic",
        "lower",
        "lower_builtin",
        "lower_cast",
        "lower_constant",
        "lower_getattr",
        "lower_getattr_generic",
        "lower_setattr",
        "lower_setattr_generic",
        "overload",
        "overload_attribute",
        "overload_classmethod",
        "overload_method",
        "register_model",
        "type_callable",
        "unbox",
    )
)


@dataclass
class Func:
    """One function, method or nested function, with its sorted unique shingles."""

    qualname: str
    path: str
    line: int
    shingles: np.ndarray = field(repr=False, compare=False)
    public: bool = True
    free: int = 0

    def where(self):
        return f"{self.path}:{self.line}"


@dataclass
class Const:
    """A module-level ``NAME = literal`` (int, str, or a tuple/set/dict of them)."""

    qualname: str
    path: str
    line: int
    value: object
    public: bool = True

    def where(self):
        return f"{self.path}:{self.line}"


class _Tokens(ast.NodeVisitor):
    """Normalized preorder token stream: every non-builtin name is one token, so
    renames vanish; attribute names, builtins, operators and literals are kept."""

    def __init__(self):
        self.out = []

    @staticmethod
    def name(ident):
        return ident if ident in BUILTINS else "v"

    def generic_visit(self, node):
        self.out.append(type(node).__name__)
        super().generic_visit(node)

    def visit_Name(self, node):  # pylint: disable=invalid-name
        self.out.append(self.name(node.id))

    def visit_arg(self, node):
        self.out.append("arg:" + self.name(node.arg))

    def visit_Attribute(self, node):  # pylint: disable=invalid-name
        self.visit(node.value)
        self.out.append("." + node.attr)

    def visit_Constant(self, node):  # pylint: disable=invalid-name
        v = node.value
        if isinstance(v, bool) or v is None:
            self.out.append(repr(v))
        elif isinstance(v, int):
            self.out.append(f"i{v}")
        elif isinstance(v, (str, bytes)):
            self.out.append("s" if len(v) > 24 else f"s{v!r}")
        else:
            self.out.append(type(v).__name__)

    def visit_FunctionDef(self, node):  # pylint: disable=invalid-name,unused-argument
        self.out.append("def")

    visit_AsyncFunctionDef = visit_Lambda = visit_ClassDef = visit_FunctionDef


def _token_id(tok):
    tid = _TOKEN_IDS.get(tok)
    if tid is None:
        tid = _TOKEN_IDS[tok] = zlib.crc32(tok.encode())
    return tid


def shingle_hashes(tokens, k=K):
    """Sorted unique 64-bit hashes of every run of `k` consecutive tokens."""
    if len(tokens) < k:
        return np.empty(0, np.uint64)
    ids = np.fromiter(map(_token_id, tokens), np.uint64, len(tokens))
    n = len(ids) - k + 1
    h = ids[:n].copy()
    for i in range(1, k):
        h *= _MIX
        h ^= ids[i : i + n]
    return np.unique(h)


def body(fn):
    """`fn`'s statements without its docstring."""
    stmts = fn.body
    if stmts and isinstance(stmts[0], ast.Expr):
        if isinstance(stmts[0].value, ast.Constant):
            return stmts[1:]
    return stmts


def shingles(fn, k=K):
    t = _Tokens()
    for stmt in body(fn):
        t.visit(stmt)
    return shingle_hashes(t.out, k)


def registered(node):
    """Whether a def is decorated by a REGISTRATIONS name, bare or called: an
    implementation handed to a registry rather than API."""
    for d in node.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name in REGISTRATIONS:
            return True
    return False


def private_module(module):
    return any(p.startswith("_") for p in module.split("."))


def walk_defs(tree):
    """(qualname, node, public) of every def and class; `public` is importable API:
    no ``_private`` component, not nested in a function, not registered."""

    def walk(node, prefix, public):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _DEFS):
                name = f"{prefix}.{child.name}" if prefix else child.name
                pub = public and not child.name.startswith("_")
                pub = pub and not isinstance(node, _FUNCS) and not registered(child)
                yield name, child, pub
                yield from walk(child, name, pub)

    yield from walk(tree, "", True)


def parse(path):
    try:
        return ast.parse(Path(path).read_bytes(), filename=str(path))
    except (SyntaxError, ValueError, OSError):
        return None


def free_names(fn):
    """Hash of the names `fn`'s body loads but never binds: its globals and callees."""
    loads, bound = set(), {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
    for stmt in body(fn):
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name):
                (loads if isinstance(n.ctx, ast.Load) else bound).add(n.id)
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
            elif isinstance(n, _DEFS):
                bound.add(n.name)
    return zlib.crc32("\0".join(sorted(loads - bound)).encode())


def functions(tree, display, module="", min_shingles=12, k=K):
    out = []
    for name, node, pub in walk_defs(tree):
        if isinstance(node, _FUNCS):
            sh = shingles(node, k)
            if len(sh) >= min_shingles:
                q = f"{module}:{name}" if module else name
                out.append(Func(q, display, node.lineno, sh, pub, free_names(node)))
    return out


def trivial(v):
    if isinstance(v, bool) or v is None:
        return True
    if isinstance(v, str):
        return len(v) < 4
    return isinstance(v, (int, float, tuple)) and v in (0, 1, -1, ())


def _assignments(tree):
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                yield node.lineno, t.id, value
            elif isinstance(t, ast.Tuple) and isinstance(value, ast.Tuple):
                for e, v in zip(t.elts, value.elts):
                    if isinstance(e, ast.Name):
                        yield node.lineno, e.id, v


def literal(v):
    """The value of a literal expression (``frozenset((...))`` and friends unwrapped,
    sets and lists as sorted tuples), or raise ValueError."""
    if (
        isinstance(v, ast.Call)
        and isinstance(v.func, ast.Name)
        and v.func.id in ("frozenset", "set", "tuple")
        and len(v.args) == 1
        and not v.keywords
    ):
        unordered = v.func.id != "tuple"
        v = v.args[0]
    else:
        unordered = False
    try:
        lit = ast.literal_eval(v)
    except (TypeError, SyntaxError, MemoryError, RecursionError) as e:
        raise ValueError(str(e)) from e
    if isinstance(lit, (list, set, frozenset)) or (
        unordered and isinstance(lit, tuple)
    ):
        lit = tuple(sorted(lit, key=repr))
    return lit


def constants(tree, display, module=""):
    out = []
    for line, name, value in _assignments(tree):
        if name.startswith("__"):
            continue
        try:
            lit = literal(value)
        except ValueError:
            continue
        if not trivial(lit):
            q = f"{module}:{name}" if module else name
            out.append(Const(q, display, line, lit))
    return out


def _module_level(stmts):
    for node in stmts:
        yield node
        if isinstance(node, ast.If):
            yield from _module_level(node.body + node.orelse)
        elif isinstance(node, ast.Try):
            handlers = [x for h in node.handlers for x in h.body]
            yield from _module_level(
                node.body + handlers + node.orelse + node.finalbody
            )


class Module(NamedTuple):
    """What one module (or its ``.pyi`` stub) binds and exports, statically."""

    name: str
    imports: tuple
    stars: tuple
    bound: frozenset
    all_names: frozenset | None
    all_from: tuple


_SEQUENCE_CALLS = frozenset(("list", "tuple", "sorted", "set", "frozenset"))


def _dotted(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, aliases)
        return base and f"{base}.{node.attr}"
    return None


def _element(node, aliases):
    if isinstance(node, ast.Starred):
        return _all_parts(node.value, aliases)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}, []
    return None


def _unwrapped(node):
    """The sequence a ``list(x)``-style or ``x.copy()`` call copies, else None."""
    if not isinstance(node, ast.Call) or node.keywords or len(node.args) > 1:
        return None
    f = node.func
    if isinstance(f, ast.Name) and f.id in _SEQUENCE_CALLS and node.args:
        return node.args[0]
    if isinstance(f, ast.Attribute) and f.attr == "copy" and not node.args:
        return f.value
    return None


def _all_parts(node, aliases):
    """(names, modules whose ``__all__`` is included) of an ``__all__`` value, or
    None when it is not statically known."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        parts = [_element(e, aliases) for e in node.elts]
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        parts = [_all_parts(node.left, aliases), _all_parts(node.right, aliases)]
    elif isinstance(node, ast.Attribute) and node.attr == "__all__":
        mod = _dotted(node.value, aliases)
        parts = [mod and (set(), [mod])]
    else:
        inner = _unwrapped(node)
        parts = [None if inner is None else _all_parts(inner, aliases)]
    if None in parts:
        return None
    return set().union(*(p[0] for p in parts)), [m for p in parts for m in p[1]]


def _is_all(node):
    return isinstance(node, ast.Name) and node.id == "__all__"


def _all_updates(node):
    """The values `node` assigns or adds to ``__all__``: ``[]`` for none, None for
    one that is not statically known."""
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Tuple) and any(map(_is_all, t.elts)) for t in targets):
            return None
        hit = node.value is not None and any(map(_is_all, targets))
        return [node.value] if hit else []
    call = node.value if isinstance(node, ast.Expr) else None
    f = getattr(call, "func", None)
    if not (
        isinstance(call, ast.Call)
        and isinstance(f, ast.Attribute)
        and _is_all(f.value)
        and len(call.args) == 1
    ):
        return []
    return {"extend": call.args, "append": [ast.List([call.args[0]])]}.get(f.attr, [])


def _imported(node, parts):
    """(local name, dotted target, re-export or None) of each name an import
    binds, and the sources of its ``*`` imports."""
    if isinstance(node, ast.Import):
        tops = [(a, a.asname or a.name.split(".", 1)[0]) for a in node.names]
        return [(t, a.name if a.asname else t, None) for a, t in tops], []
    keep = len(parts) - node.level + 1 if node.level else 0
    if keep < 0:
        return [], []
    src = ".".join(parts[:keep] + ([node.module] if node.module else []))
    named = [a for a in node.names if a.name != "*"]
    got = [(a.asname or a.name, f"{src}.{a.name}", a) for a in named]
    stars = [src] if len(named) < len(node.names) else []
    return [(loc, t, (src, a.name, loc)) for loc, t, a in got], stars


def _assigned(node):
    if isinstance(node, _DEFS):
        return [node.name]
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return []
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    elts = [e for t in targets for e in (t.elts if isinstance(t, ast.Tuple) else [t])]
    return [e.id for e in elts if isinstance(e, ast.Name)]


def module_info(tree, module, package):
    """`tree`'s Module record: its module-level ``from M import`` re-exports and
    ``*`` imports (``if``/``try`` blocks included), public bound names and
    ``__all__``, whose parts may name other modules' ``__all__``."""
    parts = module.split(".") if module else []
    if not package:
        parts = parts[:-1]
    imports, stars, bound, aliases = [], [], set(), {}
    names, mods, known = set(), [], None
    for node in _module_level(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            got, star = _imported(node, parts)
            stars += star
            for local, target, imp in got:
                aliases[local] = target
                bound.add(local)
                imports += [imp] if imp else []
        bound.update(_assigned(node))
        updates = _all_updates(node)
        for got in (
            [None] if updates is None else (_all_parts(v, aliases) for v in updates)
        ):
            if got is None:
                known = False
            elif known is not False:
                known = True
                names |= got[0]
                mods += got[1]
    return Module(
        module,
        tuple(imports),
        tuple(stars),
        frozenset(b for b in bound if not b.startswith("_")),
        frozenset(names) if known else None,
        tuple(mods) if known else (),
    )
