"""Requirements parsing, dependency discovery and ``[tool.pysayme]`` loading."""

from textwrap import dedent

import pytest

from conftest import write_tree
from pysayme import config, load_config


def test_requirement_names(tmp_path):
    write_tree(
        tmp_path,
        {
            "requirements.txt": """
                # comment
                Alpha>=1.0  # trailing
                beta[extra]; python_version > "3.8"
                --index-url https://example.invalid/simple
                -e ./local
                gamma==2.0 \\
                    --hash=sha256:abc
                -r sub/more.txt
                not a requirement !!
            """,
            "sub/more.txt": "--requirement ../base.txt\ndelta\n",
            "base.txt": "epsilon\n",
        },
    )
    assert config.requirement_names(tmp_path / "requirements.txt") == [
        "Alpha",
        "beta",
        "gamma",
        "epsilon",
        "delta",
    ]


def pyproject(tmp_path, text):
    path = tmp_path / "pyproject.toml"
    path.write_text(dedent(text))
    return path


@pytest.mark.parametrize(
    "text, files, want",
    [
        (
            """
            [project]
            name = "Me"
            dependencies = ["fakelib"]
            [tool.pysayme]
            dists = ["x", "me", "Y_z", "y-z"]
            requirements = "r.txt"
            extra-dists = ["w"]
            ignore-dists = ["W"]
            """,
            {"r.txt": "rr\n"},
            ["x", "Y_z"],
        ),
        (
            """
            [project]
            dependencies = ["fakelib"]
            [tool.pysayme]
            requirements = ["r.txt", "s.txt"]
            extra-dists = ["w"]
            """,
            {"r.txt": "rr\n", "s.txt": "ss\nrr\n"},
            ["rr", "ss", "w"],
        ),
        (
            """
            [project]
            dependencies = ["fakelib>=1", "not-installed-pysayme"]
            optional-dependencies = {x = ["pytest"]}
            [tool.pysayme]
            """,
            {"requirements.txt": "rr\n"},
            ["fakelib", "pytest"],
        ),
        (
            """
            [project]
            dependencies = ["not-installed-pysayme"]
            [tool.pysayme]
            """,
            {"requirements.txt": "rr\n"},
            ["rr"],
        ),
        ("[tool.pysayme]\n", {}, []),
    ],
)
@pytest.mark.usefixtures("site")
def test_discover_dists(tmp_path, text, files, want):
    write_tree(tmp_path, files)
    assert load_config(pyproject(tmp_path, text)).dists == want


def test_load_config_keys_and_overrides(tmp_path):
    path = pyproject(
        tmp_path,
        """
        [tool.pysayme]
        dists = ["a"]
        roots = ["src"]
        copy = 0.5
        min-shingles = 20
        [tool.pysayme.allow]
        "k -> v" = "reason"
        """,
    )
    cfg = load_config(path, copy=None, inlined=0.9, dists=["b"])
    assert (cfg.base, cfg.roots, cfg.dists) == (tmp_path, ["src"], ["b"])
    assert (cfg.copy, cfg.inlined, cfg.min_shingles) == (0.5, 0.9, 20)
    assert cfg.allow == {"k -> v": "reason"}
    assert config.has_section(path)
    assert not config.has_section(tmp_path / "missing.toml")
    pyproject(tmp_path, "[tool.other]\n")
    assert not config.has_section(path)
    assert load_config(tmp_path / "missing.toml").roots == ["."]


@pytest.mark.parametrize("key", ["bogus", "progress", "base"])
def test_unknown_keys(tmp_path, key):
    path = pyproject(tmp_path, f"[tool.pysayme]\n{key} = 1\n")
    with pytest.raises(ValueError, match=key):
        load_config(path)
