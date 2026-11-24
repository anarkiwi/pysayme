# pysayme

Manage agentic code bloat by finding project code that duplicates functions or constants
that the project's own installed dependencies already export, or other functions of the
project, and fails a pytest test naming the function to call instead.

Clone detectors such as pylint R0801, jscpd and PMD CPD compare text. pysayme compares
normalized AST structure, so renamed copies still match, and it compares a project
with its dependencies as well as with itself.

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
"tools/a.py:f -> tools/b.py:g" = "why these two stay apart"
```

A full `pytest` run (no path arguments) then includes one `pyproject.toml::pysayme`
item. The item fails if any of these hold:

- project code duplicates a public dependency function or constant;
- a project function duplicates, or is inlined in, another project function;
- an allowlist entry no longer matches (so the list only shrinks);
- an allowlist entry has no reason.

| Flag | Effect |
|---|---|
| `--pysayme` | run the gate even when paths or tests are selected |
| `--pysayme-behaviour` | also gate on behavioural matches (see `behaviour`) |
| `--no-pysayme` | never run it |
| `-m "not pysayme"` | deselect it |

### From the command line

```
pysayme                                   # check, using pyproject.toml
pysayme --dist numpy scipy -- src tests   # explicit dependencies and roots
pysayme check --all                       # include allowlisted matches
pysayme check --no-self                   # skip duplicates within the project
pysayme check --behaviour                 # also run functions to match by behaviour
pysayme rank --top 3                      # rank shape-independent look-alikes
```

`check` exits 1 when there are new duplicates or allowlist problems.

```
pysayme vectorize                         # fail on loops over ndarrays numpy would replace
pysayme advise --top 20                   # rank loops worth refactoring; never fails
pysayme advise --kind PARALLEL JIT        # only some kinds of advice
```

`vectorize` is a gate: findings use the allowlist like duplicates do (`check --vectorize`,
`pytest --pysayme-vectorize`). `advise` is a research aid like `rank`: it prints, most
valuable first, the loops where a manual change pays (vectorize, spread independent
iterations over cores, JIT a scalar recurrence, or fix an O(n^2) pattern). See
[docs/vectorize.md](docs/vectorize.md).

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `roots` | `["."]` | project paths to check; git-tracked and unignored `.py` files |
| `dists` | discovered | installed distributions to check against |
| `requirements` | | requirements file(s) to take `dists` from |
| `extra-dists`, `ignore-dists` | `[]` | add to / remove from the discovered `dists` |
| `exclude` | `[]` | `fnmatch` globs of project paths to skip |
| `allow` | `{}` | `"path:qualname -> module:qualname"` (or `-> path:qualname`) keys mapped to reasons |
| `copy` | `0.45` | shingle Jaccard for a copy |
| `inlined` | `0.70` | share of a library function found inside a project one |
| `min-shingles`, `min-inlined` | `12`, `30` | smallest function compared, and smallest one inlined |
| `private` | `false` | also match private library code |
| `ambiguous` | `false` | also report matches that fit several functions |
| `self` | `true` | also match project functions against each other |
| `jobs` | all CPUs | worker processes and join threads |
| `cache` | `true` | cache dependency indexes (`$PYSAYME_CACHE`, else `~/.cache/pysayme`) |
| `behaviour` | `false` | also run project functions and dependency callables (compiled ones included) on seeded probes, and report project functions that behave like one (`--behaviour`); executes dependency code, so meant as a release gate |
| `introspect` | `true` | import dependencies' public modules in a child process to find their public names (`--no-introspect`) |

When `dists` and `requirements` are both unset, pysayme uses the installed
`[project]` dependencies and extras, falling back to `requirements.txt`.

## Docs

- [docs/design.md](docs/design.md): how matching works
- [docs/validation.md](docs/validation.md): results on real projects
- [docs/releasing.md](docs/releasing.md): CI and PyPI publishing

## License

Apache-2.0
