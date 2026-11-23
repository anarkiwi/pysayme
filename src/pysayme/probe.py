"""Input kinds of callables' parameters, seeded probe inputs, and output digests:
what behavioural fingerprinting runs and compares, in parent and worker alike."""

from __future__ import annotations

import ast
import faulthandler
import functools
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import os
import signal
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

KINDS = ("bytes", "f8[]", "c16[]", "bool[]", "int", "float")
FALLBACK = frozenset(("f8[]", "int"))
UNKNOWN = frozenset()
MAX_REQUIRED = 3
CALL_TIMEOUT = 2.0
OUTPUT_CAP = 1 << 20
PROBES = (
    {"bytes": (0, 1, 5, 17), "array": (7, 64, 7, 64), "int": (3, 8, 5, 12)},
    {"bytes": (2, 9, 31, 64), "array": (11, 33, 11, 33), "int": (4, 6, 9, 7)},
)
N_PROBES = 4
_ARRAYS = (
    "ndarray",
    "NDArray",
    "ArrayLike",
    "array_like",
    "array-like",
    "array",
    "sequence",
)
_SEQUENCES = frozenset(("list", "tuple", "Sequence", "Iterable", "Collection"))
_NAMES = {
    "bytes": "bytes",
    "bytearray": "bytes",
    "memoryview": "bytes",
    "Buffer": "bytes",
    "ReadableBuffer": "bytes",
    "ByteString": "bytes",
    "buffer": "bytes",
    "bytes-like": "bytes",
    "int": "int",
    "ints": "int",
    "integer": "int",
    "integers": "int",
    "SupportsIndex": "int",
    "Integral": "int",
    "_IntLike_co": "int",
    "float": "float",
    "floats": "float",
    "scalar": "float",
    "number": "float",
    "Real": "float",
    "_FloatLike_co": "float",
}
_DOC = {k.lower(): v for k, v in _NAMES.items()}
_IGNORED = frozenset(("none", "nonetype", "optional", "default", "or", "of", "type"))
_UNSUPPORTED = frozenset(
    ("str", "string", "dict", "mapping", "callable", "function", "path", "pathlike")
    + ("bool", "boolean", "literal", "set", "slice", "file", "complex")
)


class Spec(NamedTuple):
    """Kinds of each required positional parameter (UNKNOWN: no evidence), and the
    keyword bindings to probe: the defaults first, then one alternative each."""

    kinds: tuple
    bindings: tuple = ((),)


def _element(words, default="f8[]"):
    """The array kind whose element type `words` name; None for integer elements
    or, unless `default`, unnamed ones."""
    low = " ".join(words).lower()
    if "complex" in low:
        return "c16[]"
    if "bool" in low:
        return "bool[]"
    if any(w in low for w in ("float", "real", "double", "inexact", "number")):
        return "f8[]"
    return None if "int" in low else default


def _arraylike(name):
    return name not in _NAMES and any(a in name for a in _ARRAYS)


def _word(name, inner=(), strict=True):
    """Kinds one type name means, with `inner` the names it is parametrised by:
    {"?"} for a name that is no probed kind, None for an unknown one unless
    `strict`."""
    if name in _NAMES:
        return {_NAMES[name]}
    if _arraylike(name):
        return {_element([name, *inner]) or "?"}
    if name in _SEQUENCES:
        return {_element(inner, None) or "?"}
    if name.lower() in _IGNORED:
        return set()
    return {"?"} if strict or name.lower() in _UNSUPPORTED else None


def _last(node):
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")


def _names(node):
    return [
        _last(n) for n in ast.walk(node) if isinstance(n, (ast.Name, ast.Attribute))
    ]


def annotation_kinds(node, strict=True):
    """Kinds an annotation AST, object or source text allows: UNKNOWN when it gives
    no evidence, ``{"?"}`` when it allows no probed kind. Unless `strict`, a name
    that is no known type is no evidence."""
    if node is None or node is inspect.Parameter.empty:
        return UNKNOWN
    if not isinstance(node, (str, ast.AST)):
        node = inspect.formatannotation(node)
    got = _ann(ast.Constant(node) if isinstance(node, str) else node, strict)
    return UNKNOWN if got is None else frozenset(got - {"?"}) or frozenset(("?",))


def _ann(node, strict):  # pylint: disable=too-many-return-statements
    if isinstance(node, ast.Constant):
        if node.value is None:
            return set()
        try:
            return _ann(ast.parse(str(node.value), mode="eval").body, strict)
        except SyntaxError:
            return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union(_ann(node.left, strict), _ann(node.right, strict))
    if isinstance(node, ast.Subscript):
        base = _last(node.value)
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base in ("Optional", "Union"):
            return _union(*(_ann(e, strict) for e in elts))
        if base == "Annotated":
            return _ann(elts[0], strict)
        return _word(base, _names(node.slice), strict)
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = _last(node)
        return None if name in ("Any", "object") else _word(name, (), strict)
    return {"?"} if strict else None


def _union(*parts):
    known = [p for p in parts if p is not None]
    return set().union(*known) if known else None


def merge(*evidence):
    """One parameter's kinds from several sources: the probed kinds any allows,
    else ``{"?"}`` when one rules them out, else UNKNOWN."""
    every = frozenset().union(*evidence)
    return (every & frozenset(KINDS)) or (every and frozenset(("?",)))


def doc_types(doc):
    """{parameter: type text} of a numpydoc ``Parameters`` section."""
    lines = inspect.cleandoc(doc).splitlines() if isinstance(doc, str) else []
    out = {}
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "Parameters" and _rule(lines[i + 1]):
            for j in range(i + 2, len(lines)):
                text = lines[j]
                if j + 1 < len(lines) and _rule(lines[j + 1]) and text.strip():
                    break
                if text and not text[0].isspace() and text.strip():
                    names, _, typ = text.partition(":")
                    for n in names.split(","):
                        out.setdefault(n.strip().lstrip("*"), typ.strip())
            break
    return out


def _rule(line):
    s = line.strip()
    return len(s) >= 3 and set(s) == {"-"}


def doc_kinds(typ):
    """(kinds, literal choices) of a numpydoc type line."""
    choices = ()
    lo, hi = typ.find("{"), typ.find("}")
    if 0 <= lo < hi:
        try:
            choices = tuple(sorted(ast.literal_eval(typ[lo : hi + 1]), key=repr))
        except (ValueError, SyntaxError, TypeError):
            choices = ()
        typ = typ[:lo] + typ[hi + 1 :]
    for ch in ",()[]":
        typ = typ.replace(ch, " ")
    words = typ.split()
    if not words:
        return UNKNOWN, choices
    kinds, skip = set(), 0
    for i, w in enumerate(words):
        if i < skip:
            continue
        if w.lower() in _DOC:
            kinds.add(_DOC[w.lower()])
        elif _arraylike(w.lower()):
            of = words[i + 1 : i + 2] == ["of"]
            kinds.add(_element(words[i + 2 : i + 3]) if of else "f8[]")
            skip = i + 3 if of else 0
    kinds.discard(None)
    return frozenset(kinds) or frozenset(("?",)), choices


@functools.lru_cache(maxsize=None)
def _stub(path):
    try:
        tree = ast.parse(Path(path).read_bytes())
    except (OSError, SyntaxError, ValueError):
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out.setdefault(node.name, []).append(node.args)
    return out


def _stub_kinds(obj, modules):
    """{parameter: kinds} of `obj` in the ``.pyi`` stub of one of `modules`,
    united over overloads."""
    name = getattr(obj, "__name__", None)
    for mod in modules:
        file = getattr(sys.modules.get(mod), "__file__", None)
        if not file or not name:
            continue
        found = _stub(str(Path(file).with_suffix(".pyi"))).get(name)
        if found:
            out = defaultdict(frozenset)
            for args in found:
                for a in args.posonlyargs + args.args:
                    out[a.arg] |= annotation_kinds(a.annotation, strict=False)
            return out
    return {}


def spec(obj, module=None):
    """Spec of a library callable, or None when it is not probed: a class, a
    generator or coroutine, no signature, or no 1 to MAX_REQUIRED required
    positional parameters (or any required keyword-only one)."""
    obj = inspect.unwrap(obj)
    if (
        inspect.isclass(obj)
        or inspect.isgeneratorfunction(obj)
        or inspect.iscoroutinefunction(obj)
        or inspect.isasyncgenfunction(obj)
    ):
        return None
    docs = doc_types(getattr(obj, "__doc__", None))
    if isinstance(obj, np.ufunc):
        names = [f"x{i + 1}" if obj.nin > 1 else "x" for i in range(obj.nin)]
        kinds = tuple(doc_kinds(docs.get(n, ""))[0] for n in names)
        return Spec(kinds) if 1 <= obj.nin <= MAX_REQUIRED else None
    try:
        params = list(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return None
    empty = inspect.Parameter.empty
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    required = [p for p in params if p.kind in positional and p.default is empty]
    if any(p.kind == p.KEYWORD_ONLY and p.default is empty for p in params):
        return None
    if not 1 <= len(required) <= MAX_REQUIRED:
        return None
    stub = _stub_kinds(obj, [module, getattr(obj, "__module__", None)])
    kinds = tuple(
        merge(
            annotation_kinds(p.annotation, strict=False),
            stub.get(p.name, UNKNOWN),
            doc_kinds(docs.get(p.name, ""))[0],
        )
        for p in required
    )
    bindings = [()]
    for p in params:
        if p.default is empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.kind == p.POSITIONAL_ONLY:
            continue
        if isinstance(p.default, bool):
            bindings.append(((p.name, not p.default),))
        else:
            for c in doc_kinds(docs.get(p.name, ""))[1]:
                if c != p.default:
                    bindings.append(((p.name, c),))
    return Spec(kinds, tuple(bindings))


def kind_tuples(kinds, unknown=FALLBACK):
    """Every kind tuple a spec's parameters allow, with `unknown` for each
    parameter without evidence; none when a parameter allows no probed kind."""
    sets = [sorted((k or unknown) & set(KINDS)) for k in kinds]
    return list(itertools.product(*sets))


def project_specs(tree):
    """(qualname, line, kinds) of `tree`'s module-level functions and the static
    methods of its module-level classes that take 1 to MAX_REQUIRED required
    positional parameters, pytest tests and fixtures excluded."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield from _project_spec(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and "staticmethod" in [
                    getattr(d, "id", None) for d in sub.decorator_list
                ]:
                    yield from _project_spec(sub, f"{node.name}.{sub.name}")


def _project_spec(node, qualname):
    a = node.args
    if node.name.startswith("test") or any(
        "fixture" in _names(d) for d in node.decorator_list
    ):
        return
    if any(d is None for d in a.kw_defaults):
        return
    pos = a.posonlyargs + a.args
    required = pos[: len(pos) - len(a.defaults)]
    if 1 <= len(required) <= MAX_REQUIRED:
        kinds = tuple(annotation_kinds(r.annotation) for r in required)
        yield qualname, node.lineno, kinds


def _rng(probe_set, kind, param, index):
    return np.random.default_rng([probe_set, KINDS.index(kind), param, index])


def make(kind, probe_set, param, index):
    """Probe input `index` of `kind` for parameter position `param`; numeric
    arrays hold one zero."""
    table = PROBES[probe_set]
    shifted = (index + param) % N_PROBES
    rng = _rng(probe_set, kind, param, index)
    if kind == "bytes":
        return rng.bytes(table["bytes"][shifted])
    if kind == "int":
        return table["int"][shifted]
    if kind == "float":
        return float(np.round(rng.uniform(-1.0, 3.0), 3))
    n = table["array"][index]
    if kind == "bool[]":
        return rng.random(n) < 0.5
    x = rng.standard_normal(n)
    if kind == "c16[]":
        x = x + 1j * rng.standard_normal(n)
    x[rng.integers(n)] = 0
    return x


class Raised(NamedTuple):
    """A call that raised, by exception type."""

    name: str


def _numeric(h, x, scale):
    kind = x.dtype.kind
    tag = {"u": "i"}.get(kind, kind)
    if scale and x.size >= 2 and kind in "iufc":
        x = x.astype(np.complex128 if kind == "c" else np.float64)
        finite = np.isfinite(x)
        norm = np.linalg.norm(x[finite])
        x = np.where(finite, x / (norm or 1.0), x)
        kind = tag = x.dtype.kind
    h.update(f"{tag}{x.shape}".encode())
    if kind in "biu":
        h.update(x.astype(np.int64).tobytes())
        return
    v = np.ascontiguousarray(x, np.complex128 if kind == "c" else np.float64)
    v = v.view(np.float64).ravel()
    finite = np.isfinite(v)
    if not finite.all():
        h.update(np.where(np.isnan(v), 1, np.where(v > 0, 2, 3) * ~finite).tobytes())
        v = np.where(finite, v, 0.0)
    top = float(np.abs(v).max()) if v.size else 0.0
    h.update(f"{top:.9g}".encode())
    if top:
        h.update(np.rint(v / top * 1e9).astype(np.int64).tobytes())


def _feed(h, x, scale, budget, mask):  # pylint: disable=too-many-return-statements
    """Hash `x`'s canonical form into `h`, a part whose digest is in `mask` as a
    marker; whether any part of it is informative."""
    if mask and digest(x, scale)[0] in mask:
        h.update(b"=")
        return False
    if isinstance(x, Raised):
        h.update(b"!" + x.name.encode())
        return False
    if x is None or isinstance(x, (bool, np.bool_)):
        h.update(repr(None if x is None else bool(x)).encode())
        return True
    if isinstance(x, (int, np.integer)):
        h.update(b"i%d" % int(x))
        return True
    if isinstance(x, (float, complex, np.number)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        budget[0] -= x.size
        if budget[0] < 0 or x.dtype.kind not in "biufc":
            h.update(b"?")
            return False
        _numeric(h, x, scale)
        return True
    if isinstance(x, (bytes, bytearray, memoryview, str)):
        data = x.encode() if isinstance(x, str) else bytes(x)
        budget[0] -= len(data)
        h.update(b"s" if isinstance(x, str) else b"y")
        h.update(b"%d:" % len(data) + data)
        return budget[0] >= 0
    if isinstance(x, (list, tuple)):
        h.update(b"(%d" % len(x))
        oks = [_feed(h, e, scale, budget, mask) for e in x]
        return any(oks)
    if isinstance(x, (dict, set, frozenset)):
        items = x.values() if isinstance(x, dict) else x
        keys = x.keys() if isinstance(x, dict) else [None] * len(x)
        parts = sorted(digest((k, v), scale, mask) for k, v in zip(keys, items))
        h.update(b"{%d" % len(parts) + b"".join(d for d, _ in parts))
        return any(ok for _, ok in parts)
    h.update(b"?")
    return False


def digest(x, scale=False, mask=frozenset()):
    """(8-byte hash, informative?) of an output's canonical form: numbers rounded
    to 1e-9 of their array's largest magnitude, numeric arrays divided by their
    norm if `scale`. Exceptions, opaque objects and `mask`ed parts inform nothing."""
    h = hashlib.blake2b(digest_size=8)
    ok = _feed(h, x, scale, [OUTPUT_CAP], mask)
    return h.digest(), ok


class Fingerprint(NamedTuple):
    """Output digests over a probe set, exact and scale-normalised, and whether
    each distinguishes the function (see `fingerprint`)."""

    exact: tuple
    scale: tuple
    exact_ok: bool
    scale_ok: bool


def fingerprint(outputs, inputs):
    """Fingerprint of `outputs` given each probe's `inputs`. It distinguishes when,
    with inputs passed through masked, 2+ distinct outputs inform, and two probes
    whose inputs differ only in array values give different outputs."""
    shapes = [
        tuple(np.shape(a) if isinstance(a, np.ndarray) else digest(a) for a in args)
        for args in inputs
    ]
    pairs = [
        (i, j) for j in range(len(shapes)) for i in range(j) if shapes[i] == shapes[j]
    ]
    parts = []
    for scale in (False, True):
        masks = [frozenset(digest(a, scale)[0] for a in args) for args in inputs]
        seen = [digest(o, scale, m) for o, m in zip(outputs, masks)]
        useful = {d for d, ok in seen if ok}
        varies = not pairs or any(seen[i][0] != seen[j][0] for i, j in pairs)
        exact = tuple(digest(o, scale)[0] for o in outputs)
        parts.append((exact, len(useful) >= 2 and varies))
    return Fingerprint(parts[0][0], parts[1][0], parts[0][1], parts[1][1])


class _Late(BaseException):
    """A probe call ran past CALL_TIMEOUT."""


def _alarm(*_):
    raise _Late


def _call(fn, args, kw):
    """`fn(*args, **kw)`, or its Raised; SIGALRM interrupts it after CALL_TIMEOUT,
    and faulthandler's C watchdog ends the process if C code holds on past that."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, CALL_TIMEOUT, CALL_TIMEOUT)
    faulthandler.dump_traceback_later(2 * CALL_TIMEOUT + 1, exit=True)
    try:
        with np.errstate(all="ignore"):
            return fn(*args, **kw)
    except _Late:
        raise
    except BaseException as e:  # pylint: disable=broad-exception-caught
        return Raised(type(e).__name__)
    finally:
        faulthandler.cancel_dump_traceback_later()
        signal.setitimer(signal.ITIMER_REAL, 0)


def resolve(target):
    """The callable a (sys.path entry, module or file, qualname) target names."""
    root, module, qualname = target
    if root and root not in sys.path:
        sys.path.insert(0, root)
    if module.endswith(".py"):
        name = "_pysayme_" + hashlib.blake2b(module.encode(), digest_size=8).hexdigest()
        mod = sys.modules.get(name)
        if mod is None:
            loc = importlib.util.spec_from_file_location(name, module)
            mod = importlib.util.module_from_spec(loc)
            sys.modules[name] = mod
            loc.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def run(task):
    """{(binding, kinds): Fingerprint} of a (target, bindings, kind tuples, probe
    set) task, or None when the target cannot be resolved or a call times out."""
    target, bindings, tuples, probe_set = task
    cwd = os.getcwd()
    try:
        fn = resolve(target)
        out = {}
        for binding, kinds in itertools.product(bindings, tuples):
            outs, ins = [], []
            for i in range(N_PROBES):
                args = [make(k, probe_set, j, i) for j, k in enumerate(kinds)]
                ins.append([make(k, probe_set, j, i) for j, k in enumerate(kinds)])
                outs.append(_call(fn, args, dict(binding)))
                os.chdir(cwd)
            out[binding, kinds] = fingerprint(outs, ins)
        return out
    except BaseException:  # pylint: disable=broad-exception-caught
        return None
    finally:
        os.chdir(cwd)
