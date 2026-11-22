"""Shape-independent ranking: what a function touches, not how it is laid out.

Features (comments and docstrings unused): ``sym:`` dependency-defined names, attributes
and raw literals equal to a public dependency constant; ``lit:`` distinctive literals;
``ex:`` normalized expression-subtree hashes. TF-IDF weighted, compared by cosine.
"""

from __future__ import annotations

import ast
import zlib
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from tqdm import tqdm

from . import extract, sources

MIN_EX = 4
COMMUTATIVE = (ast.Add, ast.Mult, ast.BitOr, ast.BitAnd, ast.BitXor)
CHUNK = 512


def vocabulary(trees):
    """Names the dependencies define (defs, classes, assigned attributes, module
    constants), and {value: names} of public int/bytes constants with rare values."""
    names, consts = set(), defaultdict(set)
    for tree in trees:
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
                names.add(n.attr)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                names.add(n.target.id)
        for c in extract.constants(tree, ""):
            names.add(c.qualname)
            v = c.value
            if not c.qualname.startswith("_") and isinstance(v, (int, bytes)):
                if not isinstance(v, int) or abs(v) > 16:
                    consts[v].add(c.qualname)
    return names - extract.BUILTINS, consts


class _Features(ast.NodeVisitor):
    def __init__(self, vocab, consts):
        self.vocab, self.consts, self.f = vocab, consts, Counter()

    def norm(self, node):
        """Canonical string of an expression subtree, and its node count."""
        if isinstance(node, ast.Name):
            keep = node.id in self.vocab or node.id in extract.BUILTINS
            return (node.id if keep else "v"), 1
        if isinstance(node, ast.Attribute):
            s, n = self.norm(node.value)
            return f"{s}.{node.attr}", n + 1
        if isinstance(node, ast.Constant):
            v = node.value
            return ("s" if isinstance(v, str) and len(v) >= 25 else repr(v)), 1
        parts, size = [], 1
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.expr_context):
                s, n = self.norm(child)
                parts.append(s)
                size += n
        if isinstance(node, ast.BinOp) and isinstance(node.op, COMMUTATIVE):
            parts = sorted(parts[:1] + parts[2:]) + parts[1:2]
        return f"{type(node).__name__}({','.join(parts)})", size

    def generic_visit(self, node):
        if isinstance(node, ast.expr):
            s, n = self.norm(node)
            if n >= MIN_EX:
                self.f[f"ex:{zlib.crc32(s.encode())}"] += 1
        if isinstance(node, ast.Name) and node.id in self.vocab:
            self.f["sym:" + node.id] += 1
        elif isinstance(node, ast.Attribute) and node.attr in self.vocab:
            self.f["sym:" + node.attr] += 1
        elif isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            self._constant(node.value)
        super().generic_visit(node)

    def _constant(self, v):
        if isinstance(v, (int, bytes)):
            for name in self.consts.get(v, ()):
                self.f["sym:" + name] += 1
        if (
            (isinstance(v, int) and abs(v) > 8)
            or isinstance(v, bytes)
            or (isinstance(v, str) and 3 < len(v) < 25)
        ):
            self.f[f"lit:{v!r}"] += 1

    def visit_FunctionDef(self, node):  # pylint: disable=invalid-name,unused-argument
        return None

    visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef


def _callees(node):
    return {
        n.func.id if isinstance(n.func, ast.Name) else n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
    }


def features(tree, vocab, consts):
    """{qualname: (Counter of features, called names, public)} of `tree`'s functions."""
    out = {}
    for name, node, public in extract.walk_defs(tree):
        if not isinstance(node, ast.ClassDef):
            v = _Features(vocab, consts)
            for stmt in extract.body(node):
                v.visit(stmt)
            out[name] = (v.f, _callees(node), public)
    return out


def _expand(feats, depth):
    """{key: [Counter at depth 0..depth]} of public functions: depth d adds the
    features of same-distribution functions reached through d unambiguous calls."""
    by_name = defaultdict(list)
    for k in feats:
        by_name[k.rsplit(":", 1)[1].rsplit(".", 1)[-1]].append(k)
    out = {}
    for k, (f, _, public) in feats.items():
        if not public:
            continue
        levels, total, frontier, seen = [f], Counter(f), [k], {k}
        for _ in range(depth):
            nxt = []
            for fk in frontier:
                for callee in feats[fk][1]:
                    hits = by_name.get(callee, ())
                    if len(hits) == 1 and hits[0] not in seen:
                        seen.add(hits[0])
                        nxt.append(hits[0])
                        total.update(feats[hits[0]][0])
            frontier = nxt
            if nxt:
                levels.append(Counter(total))
        out[k] = levels
    return out


def _parse_all(entries):
    for path, display, module in entries:
        tree = extract.parse(path)
        if tree is not None:
            yield tree, display, module


def corpus(dists, project_entries, depth=2, introspect=True):
    """(library {key: [Counter per depth]}, project {key: Counter}, [Counter of
    every library function], the documents idf is taken over with the project's)."""
    lib_trees = {}
    for name in dists:
        entries = [e for e in sources.dist_files(name) or [] if e[0].suffix == ".py"]
        lib_trees[name] = list(_parse_all(tqdm(entries, desc=name, leave=False)))
    vocab, consts = vocabulary(t for ts in lib_trees.values() for t, _, _ in ts)
    best = sources.naming(sources.indexes(dists, introspect=introspect))
    lib, background = {}, []
    for trees in lib_trees.values():
        feats = {}
        for tree, _, module in trees:
            for q, v in features(tree, vocab, consts).items():
                feats[f"{best.get((module, q.split('.', 1)[0]), module)}:{q}"] = v
        lib.update(_expand(feats, depth))
        background += [f for f, _, _ in feats.values()]
    proj = {}
    for tree, display, _ in _parse_all(project_entries):
        for q, (f, _, _) in features(tree, vocab, consts).items():
            proj[f"{display}:{q}"] = f
    return lib, proj, background


def _matrix(counters, columns, idf):
    rows, cols, vals = [], [], []
    for i, c in enumerate(counters):
        for k, x in c.items():
            rows.append(i)
            cols.append(columns[k])
            vals.append(1 + np.log(x))
    m = sparse.csr_matrix(
        (np.asarray(vals) * idf[cols], (rows, cols)), shape=(len(counters), len(idf))
    )
    norm = np.sqrt(np.asarray(m.multiply(m).sum(1))).ravel()
    norm[norm == 0] = 1
    return (sparse.diags(1 / norm) @ m).tocsr()


def ranked(lib, proj, background, top=5, min_info=8.0):
    """{project key: [(score, library key)] best first}, idf over `background` and
    `proj`; functions whose summed idf is below `min_info` are skipped."""
    docs = list(background) + list(proj.values())
    df = Counter(k for c in docs for k in c)
    columns = {k: i for i, k in enumerate(df)}
    idf = np.log(len(docs) / np.fromiter(df.values(), float, len(df)))

    def content(c):
        return sum(idf[columns[k]] for k in c)

    names, starts, lcounters = [], [], []
    for k, levels in lib.items():
        if content(levels[0]) >= min_info:
            names.append(k)
            starts.append(len(lcounters))
            lcounters += levels
    pkeys = [k for k, c in proj.items() if content(c) >= min_info]
    if not names or not pkeys:
        return {}
    lm = _matrix(lcounters, columns, idf)
    pm = _matrix([proj[k] for k in pkeys], columns, idf)
    out = {}
    for lo in tqdm(range(0, len(pkeys), CHUNK), desc="rank", leave=False):
        s = (pm[lo : lo + CHUNK] @ lm.T).toarray()
        best = np.maximum.reduceat(s, starts, axis=1)
        for r, row in enumerate(best):
            idx = np.argsort(-row, kind="stable")[:top]
            out[pkeys[lo + r]] = [(float(row[j]), names[j]) for j in idx if row[j]]
    return out
