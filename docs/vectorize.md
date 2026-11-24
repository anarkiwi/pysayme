# Proposal: `vectorize` mode

Fail on pure-Python code that numpy would do much faster. Same contract as the rest of
pysayme: a static finding names the call to use instead, an allowlist entry carries a
reason, and the list can only shrink.

Status: prototype. `pysayme vectorize` (the gate) and `pysayme advise` (the ranked
advisory view) are implemented; the sections below are the design they follow. Not yet
built: nested-loop broadcasting rewrites (the OUTER finding is a hint), dynamic
confirmation, and re-weighting `advise` by a measured profile.

## Why it fits pysayme

pysayme already fails on code that re-implements what a dependency exports. A
hand-written `for i in range(len(x)): acc += x[i] * y[i]` is the same bloat, with
`numpy.dot` as the dependency function, and it is the form agent-written DSP code
takes most often (moving averages, DFTs, magnitude/phase, bit packing, argmax scans).
It differs from the existing modes in one way: there is no library body to shingle
against. The loop must be recognised by what it *does*, so the detector is a loop
legality analysis, not a similarity join.

## What the sigmfalyzer survey says

I ran throwaway AST scans over `tactira/sigmfalyzer` (about 100k lines, already
heavily vectorized: it is the realistic "good" codebase to hold false positives against).

| Structural pattern, no data-flow | Hits in `sigmfalyzer/` | Actually iterating an ndarray |
|---|---|---|
| `np.array([f(x) for x in ...])` | 140 | **2** (`preamble.py:327`, `:399`) |
| `sum/any/all/max/min(<comprehension>)` | 83 / 33 / 30 / 31 / 23 | mostly dicts, records, strings |
| `for i in range(len(...))` with subscripts | 6 | mostly strings and lists |
| `for` with `math.*` call and subscripts | 12 | mostly scalar tables, not sample loops |
| `.tolist()` | 119 | many legitimate (JSON, keys) |

Two conclusions drive the design:

1. **Pattern matching alone is useless.** It flags hundreds of lines, nearly all
   iterating small python collections. The only things worth failing on are loops over
   data *known to be a numpy array*, so the analysis needs provenance.
2. **A mature numpy codebase should come out nearly clean.** The two true hits above
   (`np.array([_lltf_gate(int(one)) for one in np.asarray(tones).reshape(-1)])`) are
   exactly the shape worth reporting: a Python call per element of an ndarray.
   Zero-to-few findings on sigmfalyzer at the default tier is the acceptance bar.

## Pipeline

Three static stages, per function, project code only. It needs no dependency index, so
it is as cheap as parsing.

### 1. Array provenance

A small forward dataflow over each function body assigns every local name one of
`ARRAY`, `ELEM` (a scalar taken from an array), `PYSEQ` (list/tuple/dict/str/set),
`SCALAR`, `UNKNOWN`. It is flow-insensitive within a block and joins to `UNKNOWN` on
disagreement.

`ARRAY` comes from:

- calls resolved through the module's import aliases to `numpy`, `scipy` (and
  configured `array-libs` such as `jax.numpy`, `cupy`), minus reducers that return
  scalars;
- parameters whose annotation or docstring says array-like. `probe.annotation_kinds`
  and `probe.doc_types` already classify `np.ndarray`, `NDArray`, `ArrayLike`, `f8[]`
  and so on, so this reuses them;
- arithmetic, comparison, slicing, `.T`, `.real`, `.imag`, `.reshape`, `.astype`,
  `.ravel`, `.copy` of an `ARRAY`;
- project callees whose return annotation or single-assignment return is `ARRAY`
  (one level of interprocedural inference, via the same function table the self-join
  builds).

`ELEM` comes from `for v in A`, `A[i]` with integer `i`, `enumerate(A)`, `zip(A, B)`,
`A.tolist()`, `A.flat`, `np.ndenumerate`, `map(f, A)`. `A.tolist()` and `list(A)` mark
the result `PYSEQ-from-ARRAY`, so a later loop over it still counts: demoting an array
to a list is itself a finding when the next use is iteration.

Names that stay `UNKNOWN` never produce a default-tier finding. That is the precision
lever.

### 2. Loop legality and idiom classification

This is the same question a compiler's loop vectorizer asks: *does any iteration read
what another iteration writes?* Per loop (`for`, comprehension, generator, `map`) over
an `ARRAY`/`ELEM` source, collect the body's reads and writes, then classify:

| Idiom | Shape the body must have | Suggested call |
|---|---|---|
| **MAP** | `out[i] = e`, `out.append(e)`, or a comprehension element, where `e` is a pure expression of same-index reads | one array expression; `math.*` to `np.*`, `x if c else y` to `np.where`, `abs`/`min`/`max` of two to `np.abs`/`np.minimum`/`np.maximum` |
| **REDUCE** | `acc += e`, `acc *= e`, `sum/min/max/any/all/len(...)` over a generator, `e` as above, optionally filtered by `if` | `.sum()`, `np.dot`, `np.vdot`, `np.max`, `np.any`, `np.count_nonzero`, a boolean mask |
| **STENCIL** | reads at `i ± const`, or a window slice `A[i:i+n]` reduced | `np.diff`, `np.convolve`, `np.correlate`, `sliding_window_view`, `scipy.ndimage` filters, the cumsum trick |
| **SCAN** | carried value that is a prefix reduction: `acc += x[i]; out.append(acc)`, `out[i] = out[i-1] + x[i]` | `np.cumsum`, `np.cumprod`, `ufunc.accumulate`; linear recurrence with a constant coefficient to `scipy.signal.lfilter` |
| **SELECT** | `if cond(v): out.append(v)` or `append(i)` | `A[mask]`, `np.flatnonzero` |
| **ARGEXT** | best-so-far loop, `max(range(n), key=lambda i: A[i])`, first-hit-then-`break` | `np.argmax`, `np.argmin`, `np.argmax(mask)`, `np.searchsorted` |
| **OUTER** | two nested loops indexing `a[i]`, `b[j]`, optionally a reduction over `k` | broadcasting, `np.outer`, `@`, `np.einsum` |
| **PACK** | `v = (v << 1) \| b` or `v += b << k` over a bit array | `np.packbits`, `bits @ (1 << np.arange(n))` |
| **GROW** | `np.append`/`np.concatenate`/`vstack` inside a loop, or an `np.zeros` output filled in a loop whose body is MAP | collect and concatenate once, or preallocate and use the MAP form |
| **DEMOTE** | builtin `sum/max/min/sorted/list` applied directly to an `ARRAY` | `A.sum()`, `A.max()`, `A.min()`, `np.sort` |

Legality rules, all of which must hold for a loop to be classified (otherwise it is
`OPAQUE` and silent):

- every call in the body resolves to a numpy ufunc, `math`/`cmath` function, builtin
  numeric (`abs`, `min`/`max` of two, `int`, `float`, `round`, `pow`), or a project
  function that is itself a pure scalar expression (inlined one level). An unresolved
  or impure callee makes the loop `OPAQUE`. This is what keeps CRC, LFSR, Viterbi,
  and state-machine loops quiet;
- no `break`, `continue`, `return`, `raise`, `yield`, `await`, `try`, I/O or global
  writes, except the single first-hit-`break` form that ARGEXT recognises;
- no write to anything but the loop's own accumulator or output;
- carried dependences are limited to an associative accumulator (REDUCE) or a prefix
  accumulator (SCAN). Any other loop-carried read of a written name is `OPAQUE`;
- no inner `while`.

Each classification carries a rewritten expression produced by substituting `A[i]` to
`A` in the loop's AST and unparsing it, so the message can show the replacement rather
than describe it.

### 3. Cost gate

A match is reported only if it is probably slow, so small collections stay quiet:

- skip loops whose bound is a literal or `len` of a literal below `min-trip` (default
  64); an `ARRAY` of unknown length counts as large;
- weight nesting: score is nesting depth times body operation count, reported at
  `min-score`;
- skip code under `@numba.njit`/`@jit`/`@vectorize`, `numba.prange` loops, Cython
  modules, and `tests/` by default. Test oracles are deliberately naive, so `roots`
  that include tests need an allowlist entry, or `vectorize-exclude`.

## Output and allowlist

Findings join the existing report dict, so `gate.check` needs no change: stale entries,
missing reasons and the shrink-only rule come for free. The key's right side is the
numpy callable, which fits the existing `path:qualname -> module:qualname` form:

```toml
[tool.pysayme.allow]
"src/dsp/oracle.py:slow_dft -> numpy.fft:fft" = "reference implementation the fast path is tested against"
```

```
src/dsp/magnitude.py:12 magnitude iterates ndarray `samples` element-wise (MAP, ~50-200x):
    out = [math.hypot(s.real, s.imag) for s in samples]   ->   np.abs(samples)
Delete the loop and call numpy, or add "src/dsp/magnitude.py:magnitude -> numpy:abs"
to [tool.pysayme.allow] with a reason.
```

One key per function and idiom, not per line, so editing a function does not churn the
list. The target is checked against the installed numpy's public names by the existing
runtime pass, so a stale suggestion cannot ship.

## Integration

| Piece | Change |
|---|---|
| `src/pysayme/vectorize.py` (new) | provenance, legality, classification; reuses `probe.annotation_kinds`, `extract` import-alias handling |
| `src/pysayme/idioms.py` (new) | the idiom table and `math` to `np` map, as data |
| `config.py` | `vectorize = false`, `vectorize-min-trip = 64`, `vectorize-min-score`, `vectorize-confidence = "proved"` (or `"likely"`), `vectorize-exclude`, `array-libs`. Auto-off when numpy is not in `dists` |
| `gate.report` | `if cfg.vectorize: out.update(vectorize.report(files, cfg))` |
| CLI / pytest | `pysayme check --vectorize`, `--pysayme-vectorize` |

Confidence tiers: `proved` requires `ARRAY` provenance and a legal body, and is the gate
default. `likely` also reports structure-only matches on `UNKNOWN` names
(`for i in range(len(x))` with arithmetic on `x[i]`), and is a research view like `rank`,
not a gate.

## Optional dynamic confirmation (phase 3)

Mirror `behaviour`: for MAP/REDUCE findings with an unparsed rewrite, run the original
function and a compiled copy of the rewrite on seeded probes from `probe.py`, require
equal outputs, and time both at `n = 10^4`. Fail only when the speed-up is at least
`min-speedup` (default 5x). It executes project code, so it stays opt-in as a release
gate, and it turns "looks vectorizable" into a measured claim, which also cuts the
allowlist churn.

## Rollout and validation

1. **Corpus.** Hand-label ~40 snippets in `tests/vectorize/`: positives per idiom
   (moving average, FIR, naive DFT, hypot/atan2 loops, cumulative sum, bit packing,
   first-crossing search), and negatives that must stay silent (CRC, LFSR, Viterbi,
   IIR with data-dependent branch, loops over `dict`/`str`, small literal ranges,
   numba-jitted loops, `tests/` oracles).
2. **Precision.** Run on sigmfalyzer. At the `proved` tier, every finding is read by
   hand; the target is single digits, starting with `preamble.py:327` and `:399`.
3. **Recall.** Re-introduce each idiom into a copy of sigmfalyzer by rewriting a
   vectorized function as its naive loop, and require the mode to find it.
4. Ship behind `vectorize = false`; turn on in pysayme's own CI last.

## Open questions

- **Alias resolution depth.** `from numpy import *` and `xp = backend.namespace(x)`
  (sigmfalyzer's device-agnostic style) hide the numpy origin. Proposal: treat any
  module-qualified call on a name annotated or assigned from an `array-libs` namespace
  as array-producing, and list `xp`-style factories in config.
- **GPU/other backends.** `jax`/`cupy`/`torch` loops are slow for the same reason, but
  the suggested call differs. Default to numpy-only suggestions, and make the mapping
  table per `array-lib`.
- **Is a naive reference a bug?** Oracles are legitimate. The allowlist reason is the
  record; an optional `# pysayme: oracle` pragma would be shorter but would bypass the
  shrink-only guarantee, so I recommend against it.

## `advise`: where to concentrate manual refactoring

`vectorize` only reports what numpy provably replaces. Most of the cost in a real project
is elsewhere: per-item work over lists and tuples of Python objects that could use every
core, scalar recurrences numpy cannot express, and algorithmic traps. `pysayme advise`
ranks all of these and never fails the build.

It reuses the same dependence analysis. A loop whose iterations touch no carried state
and write no shared state is *independent*; whether that means vectorize or parallelize
depends on the body:

| Kind | Evidence | Suggested change |
|---|---|---|
| VECTORIZE | the proved `vectorize` finding | the numpy call shown |
| PARALLEL | independent, heavy body (project calls that loop, numpy calls, inner loops) | process pool |
| CONCURRENT | independent, the body does I/O | threads or asyncio |
| JIT | carried scalar recurrence over a numeric-only body | numba, or `lfilter` if linear |
| SERIAL | heavy loop with a carried dependence, named in the message | restructure |
| MEMBERSHIP | `x in a_list`, `.index`, `.count` inside a loop | set or dict |
| PAIRWISE | nested loops over the same collection | broadcasting, sort and sweep, KD-tree |
| LIST_HEAD, STRCAT, REPEATED | `pop(0)`, string `+=`, re-scanning an invariant | deque, `join`, hoist |

Cost is a static proxy: unknown iterables count as 100 items, arrays as 1000, a project
call that itself loops as 150 operations, I/O as 200, and nested loops multiply. The
score is 10 x log10(cost x gain / 1000 + 1), so 10 is roughly a thousand operations of
wasted work and every 10 more is a factor of ten. The outermost eligible loop carries the
advice; loops inside it are not reported separately.

Caveats the output states or the numbers reflect: independence assumes callees are pure
(a callee that writes a global or a file is invisible); the "unknown means 100" default
overstates short lists; and a project that already parallelizes at a higher level will
see advice there that it has taken. Treat the ranking as where to look first, and confirm
with a profiler.
