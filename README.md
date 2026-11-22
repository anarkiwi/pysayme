# pysayme

Finds project code that duplicates functions or constants that the project's own
installed dependencies already export, and fails a pytest test naming the library
function to call instead.

Clone detectors such as pylint R0801, jscpd and PMD CPD compare a project with itself,
and they compare text. pysayme compares a project with its dependencies, and it
compares normalized AST structure, so renamed copies still match.

## Install

```
pip install pysayme
```

## Use

### As a pytest gate

Add a `[tool.pysayme]` table to `pyproject.toml`:

```toml
[tool.pysayme]
roots = ["src", "tests"]

[tool.pysayme.allow]
"src/app/util.py:runs -> somelib.masks:runs" = "why this copy is deliberate"
```

A full `pytest` run (no path arguments) then includes one `pyproject.toml::pysayme`
item. The item fails if any of these hold:

- project code duplicates a public dependency function or constant;
- an allowlist entry no longer matches (so the list only shrinks);
- an allowlist entry has no reason.

| Flag | Effect |
|---|---|
| `--pysayme` | run the gate even when paths or tests are selected |
| `--no-pysayme` | never run it |
| `-m "not pysayme"` | deselect it |

### From the command line

```
pysayme                                   # check, using pyproject.toml
pysayme --dist numpy scipy -- src tests   # explicit dependencies and roots
pysayme check --all                       # include allowlisted matches
pysayme rank --top 3                      # rank shape-independent look-alikes
```

`check` exits 1 when there are new duplicates or allowlist problems.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `roots` | `["."]` | project paths to check; git-tracked and unignored `.py` files |
| `dists` | discovered | installed distributions to check against |
| `requirements` | | requirements file(s) to take `dists` from |
| `extra-dists`, `ignore-dists` | `[]` | add to / remove from the discovered `dists` |
| `exclude` | `[]` | `fnmatch` globs of project paths to skip |
| `allow` | `{}` | `"path:qualname -> module:qualname"` keys mapped to reasons |
| `copy` | `0.45` | shingle Jaccard for a copy |
| `inlined` | `0.70` | share of a library function found inside a project one |
| `min-shingles`, `min-inlined` | `12`, `30` | smallest function compared, and smallest one inlined |
| `private` | `false` | also match private library code |
| `ambiguous` | `false` | also report matches that fit several library functions |
| `jobs` | all CPUs | indexing processes |
| `cache` | `true` | cache dependency indexes (`$PYSAYME_CACHE`, else `~/.cache/pysayme`) |

When `dists` and `requirements` are both unset, pysayme uses the installed
`[project]` dependencies and extras, falling back to `requirements.txt`.

## Docs

- [docs/design.md](docs/design.md): how matching works
- [docs/validation.md](docs/validation.md): results on real projects
- [docs/releasing.md](docs/releasing.md): CI and PyPI publishing

## License

Apache-2.0
