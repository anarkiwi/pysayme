"""AST extraction: normalized shingle sets of functions, and module-level literal
constants."""

from __future__ import annotations

import ast
import builtins
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

K = 4
BUILTINS = frozenset(dir(builtins))
_MIX = np.uint64(0x100000001B3)
_TOKEN_IDS: dict[str, int] = {}
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class Func:
    """One function, method or nested function, with its sorted unique shingles."""

    qualname: str
    path: str
    line: int
    shingles: np.ndarray = field(repr=False, compare=False)
    public: bool = True

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


def walk_defs(tree):
    """(qualname, node, public) of every def and class; `public` is importable: no
    ``_private`` component and not nested in a function."""

    def walk(node, prefix, public):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _DEFS):
                name = f"{prefix}.{child.name}" if prefix else child.name
                pub = public and not child.name.startswith("_")
                pub = pub and not isinstance(node, _FUNCS)
                yield name, child, pub
                yield from walk(child, name, pub)

    yield from walk(tree, "", True)


def parse(path):
    try:
        return ast.parse(Path(path).read_bytes(), filename=str(path))
    except (SyntaxError, ValueError, OSError):
        return None


def functions(tree, display, module="", min_shingles=12, k=K):
    out = []
    for name, node, pub in walk_defs(tree):
        if isinstance(node, _FUNCS):
            sh = shingles(node, k)
            if len(sh) >= min_shingles:
                q = f"{module}:{name}" if module else name
                out.append(Func(q, display, node.lineno, sh, pub))
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


def _all(tree):
    for _, name, value in _assignments(tree):
        if name == "__all__":
            try:
                return set(literal(value))
            except (ValueError, TypeError):
                return None
    return None


def reexports(tree, module, package):
    """(source module, name, exported name) of each module-level ``from M import``
    of `tree` (``if``/``try`` blocks included) that ``__all__``, if any, lists."""
    parts = module.split(".") if module else []
    if not package:
        parts = parts[:-1]
    public = _all(tree)
    out = []
    for node in _module_level(tree.body):
        if not isinstance(node, ast.ImportFrom):
            continue
        keep = len(parts) - node.level + 1 if node.level else 0
        if keep < 0:
            continue
        src = ".".join(parts[:keep] + ([node.module] if node.module else []))
        out += [
            (src, a.name, a.asname or a.name)
            for a in node.names
            if a.name != "*" and (public is None or (a.asname or a.name) in public)
        ]
    return out
