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

Runtime of `pysayme check` on 2 CPUs: 22 s with a cold cache, 8 s warm.
