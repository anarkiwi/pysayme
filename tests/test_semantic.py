"""Shape-independent features and TF-IDF ranking."""

import ast
from textwrap import dedent

import pytest

from pysayme import semantic, sources

LIB = """
MAGIC = 0xCAFE
SMALL = 7
_PRIV = 0xBEEF
NAME: str = "label"

def outer(x):
    x.cached = 1
    return inner(x) + MAGIC

def inner(x):
    return x.size * 3 + x.depth

class Box:
    pass
"""


def tree(src):
    return ast.parse(dedent(src))


def test_vocabulary():
    names, consts = semantic.vocabulary([tree(LIB)])
    assert {"MAGIC", "SMALL", "outer", "inner", "Box", "cached", "NAME"} <= names
    assert "len" not in names and "x" not in names
    assert consts == {0xCAFE: {"MAGIC"}}


def test_features():
    vocab, consts = semantic.vocabulary([tree(LIB)])
    src = """
    def f(a, b):
        '''Not a feature.'''
        def nested():
            return 0xCAFE
        return inner(a) + 51966 + a.cached * "hello" + b"raw" + True

    def g(q, r):
        return 51966 + q.cached * "hello" + inner(q) + b"raw" + False
    """
    feats = semantic.features(tree(src), vocab, consts)
    assert set(feats) == {"f", "f.nested", "g"}
    f, callees, public = feats["f"]
    assert callees == {"inner"} and public and not feats["f.nested"][2]
    assert f["sym:inner"] == 1 and f["sym:cached"] == 1 and f["sym:MAGIC"] == 1
    assert f["lit:51966"] == 1 and f["lit:'hello'"] == 1 and f["lit:b'raw'"] == 1
    assert not any("Not a feature" in k or "True" in k for k in f)
    assert feats["f.nested"][0]["sym:MAGIC"] == 1
    ex_f = {k for k in f if k.startswith("ex:")}
    ex_g = {k for k in feats["g"][0] if k.startswith("ex:")}
    assert ex_f & ex_g


def test_commutative_normalization():
    vocab, consts = set(), {}
    a = semantic.features(
        tree("def f(p):\n    return p.x * p.y + p.z\n"), vocab, consts
    )
    b = semantic.features(
        tree("def f(q):\n    return q.z + q.y * q.x\n"), vocab, consts
    )
    c = semantic.features(
        tree("def f(q):\n    return q.z - q.y * q.x\n"), vocab, consts
    )
    assert a["f"][0] == b["f"][0] != c["f"][0]


def test_expand_follows_unambiguous_calls():
    vocab, consts = semantic.vocabulary([tree(LIB)])
    feats = {
        f"m:{q}": v for q, v in semantic.features(tree(LIB), vocab, consts).items()
    }
    out = semantic._expand(feats, 2)  # pylint: disable=protected-access
    assert len(out["m:outer"]) == 2
    assert out["m:outer"][1]["sym:MAGIC"] == 1
    assert out["m:outer"][1] - out["m:outer"][0] == feats["m:inner"][0]
    assert len(out["m:inner"]) == 1


@pytest.mark.usefixtures("site")
def test_ranked_finds_counterparts(project):
    files = sources.project_files(["proj"], project)
    lib, proj, background = semantic.corpus(["fakelib", "missing-pysayme"], files)
    assert set(lib) == {
        "fakelib:accumulate",
        "fakelib:transform",
        "fakelib:helper",
        "fakelib._hidden:secret",
        "fakelib._star:unlisted",
        "fakelib.dyn:dynamic",
        "fakelib.ext:overload",
        "fakelib.lazy:stubbed",
        "fakelib.stars:starred",
    }
    ranks = semantic.ranked(lib, proj, background, top=2, min_info=0)
    assert ranks["proj/mod.py:mysum"][0][1] == "fakelib:accumulate"
    assert ranks["proj/mod.py:pipeline"][0][1] == "fakelib:transform"
    assert ranks["proj/mod.py:grid_of"][0] == (
        pytest.approx(1.0),
        "fakelib._hidden:secret",
    )
    assert all(s < 0.5 for s, _ in ranks["proj/mod.py:unrelated"])
    assert not semantic.ranked(lib, proj, background, min_info=1e9)
