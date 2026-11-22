# Design

pysayme's own detection is AST-based throughout: `ast`, normalized token hashing, and
set and cosine similarity. It uses no regular expressions.

## Index

Indexing covers every `.py` file of each named installed distribution, except test
directories and `__main__.py`. Distributions installed without a file list (editable
installs) are found through their top-level import packages.

Each function body is normalized as follows:
- its docstring is dropped;
- every non-builtin name becomes a single token, `v`;
- attribute names, builtins, AST node types, operators and short literals are kept.

The token stream is cut into 4-token shingles, and each shingle is hashed to 64 bits
(CRC32 token ids mixed with an FNV multiplier, vectorized in numpy). Each function
is stored as a sorted unique `uint64` array.

Module-level `NAME = literal` assignments are indexed as constants. Lists and sets
are canonicalized as sorted tuples.

A distribution's index is cached in a pickle. The cache key covers the index
parameters and the size and mtime of every file in the distribution, so upgrading or
editing a dependency re-indexes it.

## Public names

`from M import name` statements at module level are recorded, including those in
`if TYPE_CHECKING:` and `try` blocks, and filtered by `__all__` when the module
defines one. Each definition is named by the shortest public module that re-exports
it, directly or through a chain. For example, `jennings.coverage._core:call` is
reported as `jennings.coverage:call`.

A definition is public when:
- its qualified name has no `_private` component;
- it is not nested in a function;
- its module path, after re-export resolution, has no private component.

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
library shape. The shape count is taken over all indexed code, private included.
Library functions with identical shingle sets count as one shape. A generic shape,
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
only when each side matches a single other project shape. An inlined pair is reported
only when the inlining function contains a single shape.

## Constant matching

A project constant matches when either of these holds:
- a public library constant has the same name and value;
- a public library collection of 3 or more items has the same value under any name.

Collections that are an arithmetic progression of integers (`range` values) never
match by value alone.

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
