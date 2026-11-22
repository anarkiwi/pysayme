"""The prefix-filtered join against brute force; match selection; constants."""

import numpy as np
import pytest

from pysayme import match
from pysayme.extract import Const, Func


def funcs(sets, prefix, public=None):
    return [
        Func(f"{prefix}:f{i}", f"{prefix}.py", i, np.asarray(s, np.uint64), pub)
        for i, (s, pub) in enumerate(zip(sets, public or [True] * len(sets)))
    ]


def brute(psets, lsets, copy, inlined, min_inlined):
    """{(project, library): (inlined?, score)} by dense set algebra."""
    universe = np.unique(np.concatenate(psets + lsets))

    def dense(sets):
        m = np.zeros((len(sets), len(universe)), np.int64)
        for i, s in enumerate(sets):
            m[i, np.searchsorted(universe, s)] = 1
        return m

    p, lib = dense(psets), dense(lsets)
    shared = p @ lib.T
    ps, ls = p.sum(1)[:, None], lib.sum(1)[None, :]
    jac = shared / (ps + ls - shared)
    inl = shared / ls
    is_copy = jac >= copy
    is_inl = ~is_copy & (inl >= inlined) & (ls >= min_inlined)
    return {
        (i, j): (bool(is_inl[i, j]), float(inl[i, j] if is_inl[i, j] else jac[i, j]))
        for i, j in zip(*np.nonzero(is_copy | is_inl))
    }


def random_sets(rng, n_proj, n_lib, universe):
    """Library sets, and project sets that are noisy copies, supersets or random."""
    skew = rng.zipf(1.3, universe * 4) % universe
    lsets = [
        np.unique(rng.choice(skew, rng.integers(5, 80)).astype(np.uint64))
        for _ in range(n_lib)
    ]
    psets = []
    for _ in range(n_proj):
        base = lsets[rng.integers(n_lib)]
        keep = base[rng.random(len(base)) < rng.uniform(0.5, 1.0)]
        extra = rng.choice(universe, rng.integers(0, 2 * len(base) + 1))
        psets.append(
            np.unique(
                np.concatenate([keep, extra.astype(np.uint64), [np.uint64(universe)]])
            )
        )
    return psets, lsets


@pytest.mark.parametrize("seed", range(6))
def test_join_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    psets, lsets = random_sets(rng, 300, 60, 300)
    copy, inlined, min_inlined = [
        (match.COPY, match.INLINED, match.MIN_INLINED),
        (0.3, 0.5, 10),
    ][seed % 2]
    want = brute(psets, lsets, copy, inlined, min_inlined)
    assert want
    pi, li, inl, score = match._join(  # pylint: disable=protected-access
        funcs(psets, "p"), funcs(lsets, "l"), copy, inlined, min_inlined
    )
    got = {(int(i), int(j)): (bool(k), s) for i, j, k, s in zip(pi, li, inl, score)}
    assert got.keys() == want.keys()
    for key, (kind, s) in want.items():
        assert got[key][0] == kind and got[key][1] == pytest.approx(s)


def test_join_boundaries():
    """Jaccard exactly COPY and containment exactly INLINED both qualify."""
    r = np.arange
    lib = [r(0, 15), r(100, 130), r(200, 215)]
    proj = [
        np.concatenate([r(0, 9), r(50, 55)]),
        np.concatenate([r(100, 121), r(300, 400)]),
        np.concatenate([r(200, 208), r(500, 507)]),
    ]
    got = {
        (int(i), int(j)): (bool(k), round(float(s), 9))
        for i, j, k, s in zip(
            *match._join(  # pylint: disable=protected-access
                funcs(proj, "p"), funcs(lib, "l"), match.COPY, match.INLINED, 30
            )
        )
    }
    assert got == {(0, 0): (False, 0.45), (1, 1): (True, 0.7)}


def test_find_selection():
    r = np.arange
    a, b, c = r(0, 40), r(1000, 1040), r(2000, 2040)
    lib = funcs(
        [a, a.copy(), b, np.concatenate([b[:20], r(3000, 3020)]), c],
        "lib",
        [True, True, True, True, False],
    )
    proj = funcs(
        [a, np.concatenate([b[:20], r(3000, 3010), b[20:30]]), c, r(9000, 9040)],
        "proj",
    )
    found = match.find(proj, lib)
    assert [(m.project.qualname, m.library.qualname, m.kind) for m in found] == [
        ("proj:f0", "lib:f0", "copy")
    ]
    assert found[0].score == 1.0 and found[0].rivals == 1
    assert str(found[0]).startswith("proj.py:0 proj:f0 duplicates lib:f0 (lib.py:0)")
    every = {
        m.project.qualname: m
        for m in match.find(proj, lib, private=True, ambiguous=True)
    }
    assert set(every) == {"proj:f0", "proj:f1", "proj:f2"}
    assert every["proj:f1"].rivals == 2
    assert every["proj:f2"].library.qualname == "lib:f4"
    assert match.find([], lib) == [] and match.find(proj, []) == []


def test_find_inlined():
    r = np.arange
    lib = funcs([r(0, 40)], "lib")
    proj = funcs([np.concatenate([r(0, 30), r(100, 200)])], "proj")
    (m,) = match.find(proj, lib)
    assert (m.kind, m.score) == ("inlined", 0.75)
    assert " inlines " in str(m)
    assert match.find(proj, lib, min_inlined=41) == []


def test_canonical():
    c = match.canonical
    assert c({"a": 1, "b": (1, 2)}) == c({"b": (1, 2), "a": 1})
    assert c({1, 2, 3}) == c(frozenset((3, 2, 1)))
    assert c([1, 2]) == c((1, 2)) != c((2, 1))
    assert c(1) != c("1") and c((1,)) != c({1})


def const(q, v, public=True):
    return Const(q, "x.py", 1, v, public)


def test_find_constants():
    lib = [
        const("lib:TIMEOUT", 300),
        const("lib:COLORS", ("r", "g", "b")),
        const("lib:STEPS", (1, 2, 3, 4)),
        const("lib:EVENS", (2, 4, 8)),
        const("lib:_SECRET", ("p", "q", "r")),
        const("lib._m:HIDDEN", ("x", "y", "z"), public=False),
        const("lib:MAP", {"a": 1, "b": 2, "c": 3}),
        const("lib:PAIR", ("k", "v")),
    ]
    proj = [
        const("TIMEOUT", 300),
        const("DELAY", 300),
        const("PALETTE", ("r", "g", "b")),
        const("SEQ", (1, 2, 3, 4)),
        const("POW", (2, 4, 8)),
        const("SECRET", ("p", "q", "r")),
        const("HIDDEN", ("x", "y", "z")),
        const("TABLE", {"c": 3, "b": 2, "a": 1}),
        const("DUO", ("k", "v")),
        const("STEPS", (1, 2, 3, 4)),
    ]

    def names(found):
        return {(p.qualname, lb.qualname) for p, lb in found}

    public = {
        ("TIMEOUT", "lib:TIMEOUT"),
        ("PALETTE", "lib:COLORS"),
        ("POW", "lib:EVENS"),
        ("TABLE", "lib:MAP"),
        ("STEPS", "lib:STEPS"),
    }
    assert names(match.find_constants(proj, lib)) == public
    assert names(match.find_constants(proj, lib, private=True)) == public | {
        ("SECRET", "lib:_SECRET"),
        ("HIDDEN", "lib._m:HIDDEN"),
    }
