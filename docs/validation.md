# Validation

Every run below uses the default thresholds.

## gravity-inversion

[anarkiwi/gravity-inversion](https://github.com/anarkiwi/gravity-inversion) is checked
with these settings:
- dependencies: `jennings`, `pysidtracker`, `deity-informant`, `vagrant-moses` and
  `wayward-eagle`;
- roots: `tools`, `tunes` and `tests`.

### Ground truth: before #83 (`b7a4038^`, with #83's pins)

Commit b7a4038 (#83) was a manual pass that replaced duplicates with library calls.
Before it, pysayme reports these:

| Project | Library |
|---|---|
| `tools/sidtrace.py:parse_labels` | `jennings.acme:read_symbol_list` |
| `tools/sidtrace.py:diff` | `pysidtracker.reglog:first_divergence` |
| `tools/disasm.py:_init_cover` | `jennings.coverage:call` |
| `tools/disasm.py:one_way` (inlined) | `jennings.coverage:one_way` |
| `tools/procedures.py:JMP` | `jennings.coverage.cfg:JMP` |

`RASTER` in `tools/disasm.py` equals a constant in `jennings.coverage._core`, a
private module that does not export it. It is reported only with `private = true`.
#83 removed it together with `_init_cover`, which is reported.

### Main (7cfcc2e)

pysayme reports these, all of which #83 missed:

| Project | Library |
|---|---|
| `tools/cflow.py:runs` | `deity_informant.masks:runs` |
| `JSR = 32` in `constructs`, `passes` and `residue` | `jennings.coverage.cfg:JSR` |
| `JMP = 76` in `gen_emomyst` | `jennings.coverage.cfg:JMP` |
| `COMPARE` | `wayward_eagle.census:COMPARE` |
| `MAX_CYCLES` | `jennings.coverage.components:MAX_CYCLES` |

Four further matches are judgement calls, kept with reasons in the allowlist:

| Project | Library |
|---|---|
| `flowgraph.reach` | `cgen.structure.subtree` |
| `eventlog.origins` | `acme.read_symbol_list` |
| `test_ghidra_ref._image` | `cgen.mlir.have_docker_image` |
| `irflow.ENTRY` | `cgen.traps.ENTRY` |

On both commits, the function and constant findings are identical to those of the
original prototype. The only differences are the public import names and the private
`RASTER` constant.

`pysayme rank` places these candidates first, each a different-shape duplicate:

| Project | Library |
|---|---|
| `cflow.full` | `wayward_eagle.c.model:sized` |
| `sidtrace.psid_parts` | `pysidtracker:load_psid` |
| `eventlog.song_mask` | `deity_informant.cgen.regions:code_bytes` |
| `Music._at` | `pysidtracker:SidImage.contains` |

## sigmfalyzer (4e23871)

pysayme checks the project against the installed `[project]` dependencies and
extras: 19 distributions, including numpy, scipy, scikit-image, numba, pandas,
matplotlib, scikit-learn and pymavlink. The index holds 40,746 library functions and
49,554 constants, compared against 330 project files.

The prototype's rules report 8 function matches and 3 constant matches. All of the
function matches are false positives or private-API look-alikes:

- **Generic forwarding wrappers.** For example, the 2-statement `fm._refined` matches
  `scipy.linalg:solve_banded`. Each of these project functions matches between 2 and
  11 distinct library shapes, so the ambiguity rule drops them.
- **Matches in private modules.** `backend._slice` matches
  `scipy.signal._arraytools.axis_slice`, a real equivalent that is not public scipy
  API. `occupied` matches `sklearn.covariance._robust_covariance.c_step`, and
  `bracket` matches `scipy._lib._array_api.xp_ravel`. The public-path rule drops
  these.
- **`LORA_CODING_RATES = (1, 2, 3, 4)`.** This equals a scipy constant by value, but
  it is a `range`.

What remains is `DOWN = 3`, at two sites, matching `matplotlib.sankey:DOWN`. It is a
C4FM dibit level, so it belongs in the allowlist.

With public-API resolution (`*` imports, computed `__all__`, `.pyi` stubs and runtime
introspection), every runtime-exported function of `scipy.signal` (132),
`scipy.ndimage`, `scipy.special`, `scipy.interpolate` and `skimage.measure` is public
under its documented path. Examples are `scipy.signal:find_peaks`,
`scipy.signal:welch`, `scipy.signal:firwin`, `scipy.ndimage:uniform_filter1d`,
`scipy.ndimage:label` and `numpy.fft:fftfreq`. The 852 numba functions registered
with `@overload`, `@lower_builtin` and similar decorators are no longer public.

Making `scipy.linalg:solve_banded` public exposed `fm._refined` as a match. Its
2-statement body is shared by `solve_banded`, `cho_solve`, `cho_solve_banded` and
`lu_solve`, so the name-aware shape count makes it ambiguous. The findings are the
same as without public-API resolution: the two `DOWN` sites.

Runtime of `pysayme check` on 2 CPUs:

| | Cold cache | Warm cache |
|---|---|---|
| with introspection | 34 s | 6 s |
| `--no-introspect` | 17 s | 6 s |

On gravity-inversion, introspection takes a cold run from 2.5 s to 5.4 s. Warm runs
take 1.1 s either way, and the findings above are unchanged.

## Behavioural fingerprinting

Both projects were checked with `pysayme check --behaviour --jobs 2`, pinned to 2
CPUs.

### sigmfalyzer (4e23871)

The run used the discovered dependencies plus `fastcrc`, which sigmfalyzer only
reaches transitively, so it needs `extra-dists = ["fastcrc"]`. This gave about 5,500
library callables, compiled ones included, and 1,378 project functions. Findings:

| Project | Behaves like | Judgement |
|---|---|---|
| `elrs_stream.py:crc8` | `fastcrc.crc8:dvb_s2` | true; Rust, invisible to shape matching |
| `frsky.py:chip_crc` | `fastcrc.crc16:cms` | true |
| `frames.py:cc2500_crc`, `frames.py:afhds2a_crc` | `fastcrc.crc16:cms`, `:spi_fujitsu` | true; deliberate test oracles, allowlist |
| `tests/test_elrs.py:bitwise_crc8` | `fastcrc.crc8:dvb_s2` | true; deliberate oracle, allowlist |
| `backend.py:uniform_filter1d` | `scipy.ndimage:uniform_filter1d(mode='wrap')` | true; array-API backend wrapper, allowlist |
| `backend.py:_hann` | `scipy.signal.windows:hann(sym=False)` | true; backend, allowlist |
| `bits.py:popcount` | `numpy:bitwise_count` | true; numpy 2 only while the floor is 1.26, allowlist |
| `extract.py:_sinc` | `numpy:sinc` | true; written out to match the C++, allowlist |
| `signalling.py:_energy` | `sklearn.utils.extmath:squared_norm` | true; deliberately avoids BLAS |
| `analysis.py:_offsets` | `pandas.core.arrays.categorical:coerce_indexer_dtype` | equal on the probe domain (both cast the first argument to integers); unrelated |
| `tests/test_classify.py:_inside` | `pymavlink.mavexpression:demix1` | equal on finite inputs (`(a + b) / 2`); unrelated |

These are not caught, for the stated reasons:
- `wifi_dsss.py:crc16` takes a bit array, and no dependency computes a CRC of bits.
- `boxer/taps.py:final_taps` equals `firwin` only with a derived tap count and
  Kaiser beta.
- `synth.py:gaussian_pulse` equals `windows.gaussian(M, std) / sum` only under a
  reparametrisation (M, std from sps, bt).
- `reductions.py:taper` equals `tukey(width, 2 * share)` only for non-default
  `alpha`, and it is built from `np.hanning` in float32.
- `wifi.py:_circular_prefix_sum`, `estimators.py:_quantiles` and
  `fm.py:components` take 2-D arrays.
- `fm.py:at_least` is the tail of `scipy.stats.poisson_binom`, a distribution
  object rather than a callable with an array parameter.

Probes are 1-D, and parameters are bound one keyword at a time, so none of these can
join.

None of the research notes' deliberate near-misses are reported, apart from the
backend wrappers, `bits.popcount` and the `frames.py` oracles above, which behave
identically and are allowlist material. `reductions.running` and `reductions.moving`
are not reported.

The value-dependence and pass-through rules were added after an earlier run. That run
also reported these, each a coincidence on the probe domain:
- `hops.readings` as `numpy:divide`, before probes held a zero;
- `preamble._runs_of` as `scipy.signal:unique_roots`, which passes its input through;
- `preamble.side_gate` as `scipy.special:entr` up to scale, a normalisation bug with
  infinities;
- `tests/test_survey._die` as `pandas.core.common:maybe_make_list`, which wraps its
  input;
- `channels.runs` as `matplotlib.cbook:contiguous_regions`.

| sigmfalyzer, 2 CPUs | Cold cache | Warm cache |
|---|---|---|
| `check --behaviour` | 200 s | 84 s |
| `check` | | 11 s |

Warm runs spend their time running the project's functions, which are never cached.

### gravity-inversion

The run probes 641 library callables and 369 project functions, and reports
nothing. Cold: 19 s; warm: 5.3 s.
