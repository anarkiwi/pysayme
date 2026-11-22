"""Matching project functions and constants against dependencies, and each other.

Both function relations imply a share >= ``min(copy, inlined)`` of the library
function's shingles occurs in the project function, so candidates come from library
prefix filtering (rarest shingles first) and are scored exactly with sparse products.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .extract import Func

COPY = 0.45
INLINED = 0.70
MIN_INLINED = 30
CHUNK = 1 << 18


@dataclass
class Match:
    """A project function that duplicates a dependency function."""

    project: Func
    library: Func
    kind: str
    score: float
    rivals: int = 1
    local: bool = False

    def _target(self):
        lib = self.library
        return f"{lib.path}:{lib.qualname}" if self.local else lib.qualname

    def key(self):
        return f"{self.project.path}:{self.project.qualname} -> {self._target()}"

    def __str__(self):
        if self.local:
            verb = "duplicates" if self.kind == "copy" else "is inlined in"
            return (
                f"{self.project.where()} {self.project.qualname} {verb} "
                f"{self.library.where()} {self.library.qualname} "
                f"score={self.score:.2f}"
            )
        verb = "duplicates" if self.kind == "copy" else "inlines"
        return (
            f"{self.project.where()} {self.project.qualname} {verb} "
            f"{self.library.qualname} ({self.library.where()}) score={self.score:.2f}"
        )


def _incidence(sets, vocab):
    """CSR row-per-set 0/1 matrix over the sorted column vocabulary `vocab`."""
    sizes = np.fromiter((len(s) for s in sets), np.int64, len(sets))
    indptr = np.zeros(len(sets) + 1, np.int64)
    np.cumsum(sizes, out=indptr[1:])
    cols = np.searchsorted(vocab, np.concatenate(sets))
    data = np.ones(len(cols), np.float32)
    m = sparse.csr_matrix((data, cols, indptr), shape=(len(sets), len(vocab)))
    return m, sizes


def _prefix(m, sizes, rank, threshold):
    """Rows of `m` cut to their first tokens in `rank` order: enough that any set
    holding ``>= threshold * size`` of a row's tokens holds one of them."""
    rows = np.repeat(np.arange(m.shape[0]), sizes)
    order = np.lexsort((rank[m.indices], rows))
    pos = np.arange(len(order)) - m.indptr[rows]
    keep = pos < (sizes - np.ceil(threshold * sizes - 1e-9) + 1)[rows]
    sel = order[keep]
    return sparse.csr_matrix((m.data[sel], (rows[sel], m.indices[sel])), shape=m.shape)


def _overlap(p, lib, pi, li):
    """|p row ∩ lib row| of each pair, in bounded-memory chunks."""
    out = np.empty(len(pi), np.float64)
    for lo in range(0, len(pi), CHUNK):
        sl = slice(lo, lo + CHUNK)
        prod = p[pi[sl]].multiply(lib[li[sl]])
        out[sl] = np.asarray(prod.sum(axis=1)).ravel()
    return out


def _balanced(sizes, n):
    """Row bounds of at most `n` contiguous chunks with about equal total size."""
    cum = np.cumsum(sizes)
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, n) / n) if len(cum) else []
    return np.unique(np.concatenate(([0], cuts, [len(sizes)])))


def _join(project, library, copy, inlined, min_inlined, jobs=1):
    """(project index, library index, inlined?, score) of every qualifying pair.

    Candidates share a token in both prefixes (Jaccard) or in the library prefix
    (containment), the AllPairs bounds; a length filter precedes exact scoring.
    Project rows are joined in size-balanced chunks on `jobs` threads."""
    psets = [f.shingles for f in project]
    lsets = [f.shingles for f in library]
    vocab = np.unique(np.concatenate(psets + lsets))
    p, psize = _incidence(psets, vocab)
    lib, lsize = _incidence(lsets, vocab)
    df = np.bincount(p.indices, minlength=len(vocab))
    df += np.bincount(lib.indices, minlength=len(vocab))
    rank = np.empty(len(vocab), np.int64)
    rank[np.lexsort((np.arange(len(vocab)), df))] = np.arange(len(vocab))
    big = sparse.diags((lsize >= min_inlined).astype(np.float32))
    p_copy = _prefix(p, psize, rank, copy)
    l_copy = _prefix(lib, lsize, rank, copy).T.tocsc()
    l_inl = (big @ _prefix(lib, lsize, rank, inlined)).T.tocsc()

    def rows(lo, hi):
        cand = (p_copy[lo:hi] @ l_copy + p[lo:hi] @ l_inl).tocoo()
        pi, li = cand.row + lo, cand.col
        ps, ls = psize[pi], lsize[li]
        fits = ((ls >= copy * ps) & (copy * ls <= ps)) | (
            (ls >= min_inlined) & (inlined * ls <= ps)
        )
        pi, li = pi[fits], li[fits]
        shared = _overlap(p, lib, pi, li)
        jac = shared / (psize[pi] + lsize[li] - shared)
        inl = shared / lsize[li]
        is_copy = jac >= copy
        is_inl = ~is_copy & (inl >= inlined) & (lsize[li] >= min_inlined)
        ok = is_copy | is_inl
        return pi[ok], li[ok], is_inl[ok], np.where(is_inl, inl, jac)[ok]

    bounds = _balanced(psize, 4 * max(1, jobs))
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        parts = list(pool.map(rows, bounds[:-1], bounds[1:]))
    return tuple(np.concatenate(x) for x in zip(*parts))


def _rivals(pi, li, library, n, callees=True):
    """Per project function, the number of distinct shapes matched: a shingle set,
    with the free names its body references when `callees`, so copies count once
    and one template forwarding to different callees counts per callee."""
    shapes = {}
    shape = np.fromiter(
        (
            shapes.setdefault((f.shingles.tobytes(), callees and f.free), len(shapes))
            for f in library
        ),
        np.int64,
        len(library),
    )
    pairs = np.unique(np.stack([pi, shape[li]]), axis=1)[0]
    return np.bincount(pairs, minlength=n)


def find(
    project,
    library,
    copy=COPY,
    inlined=INLINED,
    min_inlined=MIN_INLINED,
    private=False,
    ambiguous=False,
    jobs=1,
):
    """Best library match of each project Func that has one, best score first.

    *copy*: shingle-set Jaccard >= `copy`. *inlined*: a share >= `inlined` of a
    library function's shingles (it having >= `min_inlined`) is in the project one.
    Only public matches unless `private`; only unambiguous ones unless `ambiguous`."""
    if not project or not library:
        return []
    pi, li, is_inl, score = _join(project, library, copy, inlined, min_inlined, jobs)
    rivals = _rivals(pi, li, library, len(project))
    if not private:
        public = np.fromiter((f.public for f in library), bool, len(library))[li]
        pi, li, is_inl, score = pi[public], li[public], is_inl[public], score[public]
    order = np.lexsort((li, -score, ~is_inl, pi))
    rows, first = np.unique(pi[order], return_index=True)
    first = order[first]
    if not ambiguous:
        keep = rivals[rows] == 1
        rows, first = rows[keep], first[keep]
    out = [
        Match(project[i], library[j], "inlined" if k else "copy", float(s), int(r))
        for i, j, k, s, r in zip(
            rows, li[first], is_inl[first], score[first], rivals[rows]
        )
    ]
    return sorted(out, key=lambda m: (-m.score, m.key()))


def _nested(funcs, pi, li):
    """Whether each pair is one function and itself, or one nested in the other."""
    out = pi == li
    for n in np.flatnonzero(~out):
        a, b = funcs[pi[n]], funcs[li[n]]
        if a.path == b.path:
            qa, qb = a.qualname + ".", b.qualname + "."
            out[n] = qa.startswith(qb) or qb.startswith(qa)
    return out


def find_self(
    funcs,
    copy=COPY,
    inlined=INLINED,
    min_inlined=MIN_INLINED,
    ambiguous=False,
    jobs=1,
):
    """Project Funcs duplicating other project Funcs, by path then qualname.

    One self-join: each copy pair once, each inlined pair as (helper, inliner). Only
    pairs whose copying sides each match one shape, unless `ambiguous`."""
    if not funcs:
        return []
    pi, li, is_inl, score = _join(funcs, funcs, copy, inlined, min_inlined, jobs)
    keep = ~_nested(funcs, pi, li)
    pi, li, is_inl, score = pi[keep], li[keep], is_inl[keep], score[keep]
    rivals = _rivals(pi, li, funcs, len(funcs), callees=False)
    rank = np.empty(len(funcs), np.int64)
    rank[np.lexsort(([f.qualname for f in funcs], [f.path for f in funcs]))] = (
        np.arange(len(funcs))
    )
    keep = is_inl | (rank[pi] < rank[li])
    if not ambiguous:
        keep &= (rivals[pi] == 1) & (is_inl | (rivals[li] == 1))
    a, b = np.where(is_inl, li, pi), np.where(is_inl, pi, li)
    sel = np.flatnonzero(keep)
    sel = sel[np.lexsort((rank[b[sel]], rank[a[sel]]))]
    return [
        Match(
            funcs[a[n]],
            funcs[b[n]],
            "inlined" if is_inl[n] else "copy",
            float(score[n]),
            int(rivals[pi[n]]),
            True,
        )
        for n in sel
    ]


def canonical(v):
    """Hashable, order-insensitive form of a literal value."""
    if isinstance(v, dict):
        items = ((canonical(k), canonical(x)) for k, x in v.items())
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(v, (tuple, list)):
        return ("tuple", tuple(canonical(x) for x in v))
    if isinstance(v, (set, frozenset)):
        return ("set", tuple(sorted(map(canonical, v), key=repr)))
    return (type(v).__name__, v)


def _progression(v):
    """Whether `v` is a tuple of ints in arithmetic progression: a ``range``."""
    if not (
        isinstance(v, tuple)
        and v
        and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    ):
        return False
    return len(set(np.diff(v).tolist())) <= 1


def _name(c):
    return c.qualname.rsplit(":", 1)[-1]


def find_constants(project, library, private=False):
    """(project Const, library Const) pairs: the same public name bound to the same
    value, or the same collection of 3+ items (not a range) under any name."""
    by_name, by_value = defaultdict(list), defaultdict(list)
    for c in library:
        name = _name(c)
        if not private and (name.startswith("_") or not c.public):
            continue
        v = canonical(c.value)
        by_name[name, v].append(c)
        if isinstance(c.value, (tuple, dict)) and len(c.value) >= 3:
            if not _progression(c.value):
                by_value[v].append(c)
    out = []
    for c in project:
        v = canonical(c.value)
        hits = by_name.get((_name(c), v)) or by_value.get(v)
        if hits:
            out.append((c, hits[0]))
    return out
