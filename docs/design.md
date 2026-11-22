# Design

pysayme's own detection is AST-based throughout: `ast`, normalized token hashing, and
set and cosine similarity. It uses no regular expressions.

## Index

Indexing covers every `.py` file of each named installed distribution, except test
directories, `conftest.py` and `__main__.py`. Its `.pyi` stubs are read for
re-exports only. Distributions installed without a file list (editable installs) are
found through their top-level import packages.

Each function body is normalized as follows:
- its docstring is dropped;
- every non-builtin name becomes a single token, `v`;
- attribute names, builtins, AST node types, operators and short literals are kept.

The token stream is cut into 4-token shingles, and each shingle is hashed to 64 bits
(CRC32 token ids mixed with an FNV multiplier, vectorized in numpy). Each function
is stored as a sorted unique `uint64` array.

Module-level `NAME = literal` assignments are indexed as constants. Lists and sets
are canonicalized as sorted tuples.

A distribution's index is cached in a pickle, together with its runtime exports and
compiled callables (see below). The cache key covers the index parameters, the Python
version, and the size and mtime of every file in the distribution, so upgrading or
editing a dependency re-indexes it.

## Public names

A definition's public name is the import path a user would write. pysayme finds the
modules that export each definition in two ways, and merges them.

**Statically**, each module (and its `.pyi` stub, such as a lazy loader's
`__init__.pyi`) records its module-level imports, including those in `if` and `try`
blocks:
- `from M import name` re-exports `name`;
- `from M import *` binds what `M` exports;
- `__all__` is evaluated from literals, `X.__all__` references, `+`, `+=`,
  `.extend`, `.append`, `list(...)` and `.copy()`. Any other value, such as
  `[s for s in dir() ...]`, makes it unknown.

A module exports its `__all__`, or, when that is absent or unknown, its public bound
names, including those bound by `*` imports. `*` imports and `X.__all__` references
are resolved across modules as a fixed point, so import cycles resolve too. A
re-export counts only when the exporter exports that name.

**At runtime** (`introspect = true`), each public module of a distribution is
imported in a spawned child process. A module is public when no component of its path
is `_private`. For each name the module exports (its `__all__`, else its public
names), pysayme takes `inspect.unwrap(obj)` and finds the definition:
- functions by `__code__.co_filename` and `co_qualname`, so a `__module__` that
  names a re-exporting module does not mislead it;
- classes by their first Python method's code, else `__module__` and `__qualname__`.

The export is recorded when the definition is one of the distribution's own and is
exported under its own name. Constants are named by the static pass alone. Each import has a timeout (`runtime.IMPORT_TIMEOUT`).
A module that times out or kills its child is skipped, and a fresh child continues
with the next. Library output and warnings are discarded.

The runtime pass also records an inventory of the public callables that have no
Python source: ufuncs, builtins, and Cython or Rust functions. Each is recorded with
its public path, type name, `inspect.signature` text (when available) and the first
docstring line, under the shortest path that exports it.
`sources.compiled_callables(dists)` returns this inventory.

**Merging.** An exporter counts only when it is in the definition's own distribution
and its own top-level package. The exception is a private top-level package, such as
`_pytest`, whose definitions any package of the same distribution can export. This
keeps `pylab` or a dependent library from renaming another package's API. Each
definition is named by the shortest public module among its own module and the
modules that export it, directly or through a chain. Ties go to its own module, then
to the lexicographically first. For example, `jennings.coverage._core:call` is
reported as `jennings.coverage:call`, and `scipy.signal._peak_finding:find_peaks` as
`scipy.signal:find_peaks`.

A definition is public when:
- its qualified name has no `_private` component;
- it is not nested in a function;
- it is not registered (see below);
- its module path, after renaming, has no private component.

**Registration is not API.** A def or class decorated with a registry decorator,
bare (`@intrinsic`) or called (`@overload(np.sum)`, `@registry.lower_cast(...)`), is
an implementation handed to a registry, not a name users call. Such a definition is
not public, and neither is anything nested in it. A decorator is a registry
decorator when its callee's last name is in `extract.REGISTRATIONS`. These are the
numba extension API names: `overload*`, `intrinsic`, `type_callable`, `infer*`,
`lower*`, `box`, `unbox` and `register_model`. `register_jitable` is not one of them,
because its functions remain callable from Python. A bare `@overload` is
`typing.overload` on a bodiless stub, which is too small to index either way.

Only public definitions are reported unless `private = true`.

## Function matching

Two relations are reported:

- **copy**: the Jaccard similarity of the two shingle sets is at least `copy` (0.45).
- **inlined**: at least `inlined` (0.70) of a library function's shingles occur in a
  project function, and the library function has at least `min-inlined` (30) shingles.

The join over all project and library pairs is exact. It uses prefix filtering, as in
the AllPairs/PPJoin set-similarity joins:
1. Shingles are ordered by ascending document frequency over the project and the
   library together.
2. A pair can only reach Jaccard `t` if both sets' prefixes of length
   `|x| - ceil(t|x|) + 1` share a shingle.
3. A library set can only be `t`-contained in a project set if its own prefix shares
   one with the project set.

Both prefix products are sparse matrix products (scipy). A length filter removes
pairs whose sizes cannot qualify, and the surviving candidates are scored exactly
with chunked sparse row products.

A project function is reported only when its qualifying pairs identify a single
library shape. The shape count is taken over all indexed code, private included. A
shape is a shingle set together with the free names its body references: the globals
and callees it loads but never binds. Normalization collapses those names, so this
restores the one distinction that matters for identity. Vendored or renamed copies of a
function therefore count once, while one template forwarding to different callees
counts once per callee. For example, scipy's `solve_banded`, `cho_solve`,
`cho_solve_banded` and `lu_solve` are each a two-statement unpack-and-forward wrapper
around a different private helper. A generic shape,
such as a two-line wrapper that unpacks a tuple and forwards it to a call, matches
many library functions, so it carries no evidence of copying any one of them.
`ambiguous = true` reports these matches too.

## Project self-join

With `self = true`, the project is also joined with itself by the same exact join, with
the same `copy`, `inlined` and `min-inlined` thresholds. Each reported pair names both
project locations, keyed `path:qualname -> path:qualname`:
- a copy pair is reported once, the side first by path then qualname on the left;
- an inlined pair is keyed from the inlined helper to the function that inlines it.

These pairs are dropped:
- a function paired with itself;
- a function paired with a function nested in it.

Tests are held to the same rule as other code: copied tests are reported, and the fix
is a shared helper or `pytest.mark.parametrize`.

The ambiguity rule applies as it does for library matches. A copy pair is reported
only when each side matches a single other project shape. Here a shape is the shingle
set alone: bodies that differ only in the callees they forward to are one repeated
template, fixed by a shared helper or `parametrize`. Library matching counts callees
because it must name the one function to call. An inlined pair is reported
only when the inlining function contains a single shape.

## Constant matching

A project constant matches when either of these holds:
- a public library constant has the same name and value;
- a public library collection of 3 or more items has the same value under any name.

Collections that are an arithmetic progression of integers (`range` values) never
match by value alone.

## Parallelism

`jobs` (default: all CPUs) bounds every parallel stage:
- **Parsing:** one process pool parses the uncached files of all distributions, and
  the project's files, so small distributions do not run serially.
- **Introspection:** a distribution's public modules are grouped by top-level
  subpackage (`scipy.signal`, `scipy.ndimage`) and packed, largest first, onto at
  most `jobs` import children. Each child imports a shared base package once.
- **Joins:** project rows are split into chunks of about equal total shingle count,
  and each chunk's candidate products and exact overlaps run on a thread pool. The
  scipy sparse kernels release the GIL.

## Semantic ranking (`pysayme rank`)

`rank` is a research aid for duplicates written in a different shape. It does not
gate. Each function's features are:

- `sym:`: dependency-defined names and attributes it references, plus raw literals
  equal to a public dependency constant (`0x4C` → `JMP`);
- `lit:`: distinctive literals;
- `ex:`: hashes of normalized expression subtrees (4 or more nodes, commutative
  operands sorted).

Features are TF-IDF weighted. IDF is taken over every library and project function.

Each public library function is scored at call depths 0 to `--depth`, and its best
depth counts. Depth `d` adds the features of the same-distribution functions reached
through `d` unambiguous calls, so a thin entry point that delegates to private
helpers is compared by what the helpers do.

Scores are cosine similarities from chunked sparse products.

## Behavioural fingerprinting (`behaviour = true`)

The shape matcher needs a Python body on both sides, so it cannot see a project
function that recomputes a ufunc, a builtin, or a Cython or Rust callable, nor one
written in a different shape. Behaviour mode runs both sides on the same inputs and
compares outputs. It executes dependency and project code, so it is opt-in: a
release gate rather than an every-commit check. It needs the runtime pass, and turns
it on.

**Candidates.** On the library side, the runtime pass records each public callable,
Python or compiled, under its shortest public path (one record per object). Classes,
generators and coroutines are skipped, as is anything whose signature cannot be
determined, that has a required keyword-only parameter, or that does not have 1 to
3 required positional parameters. A ufunc has `nin` of them. On the project side,
module-level functions and static methods of module-level classes qualify on the
same terms, read from the AST; pytest tests (`test*`) and fixtures are skipped.
A project file inside a package is imported by its dotted name, with the package's
parent on `sys.path`; a loose script is loaded from its file.

**Input kinds.** Each required parameter takes a set of kinds: bytes, 1-D float64,
complex128 or bool arrays, int, or float.
- Project parameters are read from annotations. An annotation naming no probed kind
  (`str`, a project class) excludes the function. An unannotated parameter is tried
  as a float array and as an int.
- Library parameters unite the evidence of runtime annotations, the `.pyi` stub next
  to the exporting or defining module (overloads united), and the numpydoc
  `Parameters` type line (`x : array_like`, `n : int`, `data : bytes`,
  `array_like of complex`), parsed with `ast`, `inspect` and string methods. A name
  the parser does not know (a stub's type variable) is no evidence, while a known
  unprobed type (`str`, `callable`) is. A callable none of whose evidence allows a
  probed kind is skipped.
- A library parameter without evidence is tried with every kind tuple some project
  function needs. Library probing is demand-driven: a callable runs only on the
  kind tuples that could join a project function.

**Probes.** Each kind has 4 seeded probes per probe set, with byte lengths
(0, 1, 5, 17), array lengths (7, 64, 7, 64) and ints (3, 8, 5, 12). Array
parameters of one call share a length, and scalar and bytes parameters are offset
by position, so two parameters of one kind differ. The verification set uses other
lengths, ints and seeds.

**Digests.** An output is canonicalized and hashed with BLAKE2b:
- numeric arrays and scalars by element class (bool, integer, float, complex),
  shape, and values rounded to 1e-9 of the array's largest magnitude, with NaN and
  infinities kept apart;
- bytes, str, lists and tuples recursively; dicts and sets order-free;
- an exception by its type name.

Anything else, or anything over the output cap, is opaque. The scale-normalised
digest first divides each numeric array of 2 or more elements by its norm, so
outputs equal up to a constant factor, or a per-input normalisation such as unit
area, also match.

A fingerprint is the tuple of digests over the probes. It distinguishes a function
only when at least two of its outputs are distinct, informative (a value, not an
exception or an opaque object) and not equal to one of the call's inputs. This rule
drops constant functions, exception-only ones and identities. It does not drop a
transform that maps empty input to empty output.

**Binding.** A library callable runs with its defaults. It also runs with one
keyword parameter changed at a time: a `bool` default flipped, or each other choice
of a numpydoc literal set (`{'wrap', 'reflect'}`). Ufunc keywords are machinery, not
behaviour, and are not varied. A match names the binding:
`behavelib:ramp(reverse=True)`.

**Matching.** Library fingerprints are put in dicts keyed by (kind tuple,
fingerprint), one for exact digests and one for scale digests. Each project
fingerprint is looked up there, with no pairwise comparison. Exact hits are used if
there are any, else scale hits ("up to scale"). Then:
1. A function whose body names a callable it matches is a wrapper that delegates,
   not a reimplementation, and is dropped.
2. As in shape matching, a project fingerprint shared by callables of several
   distinct names is generic: a predicate that happens to agree on four probes, or a
   reduction that `nan*` variants share. It is dropped unless `ambiguous = true`.
   One name exported at several paths (`numpy:diff`, `numpy.ma:diff`) counts once,
   reported under its shortest path.
3. Each surviving pair runs again on the verification set, and is reported only if
   the digests agree again and the project outputs still distinguish it.

**Execution.** Calls run in a pool of spawned children (`runtime.Workers`, sized by
`jobs`, also used by the runtime pass), with stdio on `/dev/null`, a scratch working
directory, single-threaded BLAS, OpenMP and numba, and the Agg matplotlib backend.
Each call has `probe.CALL_TIMEOUT` (2 s, against probes of at most 64 elements),
signalled by SIGALRM and raised as a `BaseException` so library `except Exception`
blocks do not swallow it. C code that holds the interpreter past twice that is ended
by faulthandler's C watchdog thread, which exits the child. Each task also has a
parent-side deadline as a backstop. A callable that times out, or kills its
child, is skipped, and a fresh child takes the next task. Each call gets fresh
inputs, so in-place mutation cannot leak into later probes.

**Caching.** Library fingerprints are cached per distribution, next to its index
and under the same key, as a dict from (path, binding, kind tuple) to fingerprint.
Entries a new project's demand adds are computed and merged in. Crashes and timeouts
are cached as such. Project fingerprints and verification are recomputed on every
run.
