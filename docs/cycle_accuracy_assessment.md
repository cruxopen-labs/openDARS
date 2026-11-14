# Is a cycle-accurate model worth it for openDARS?

Status: decision note, 2026-11-14. Scope: whether openDARS should replace or augment its
analytical evaluator with a cycle-accurate model. Every number below is reproducible with
`python tools/decision_margins.py`, which reads the checked-in reports and measurements.

## Short answer

**Not for the inner search loop; yes as a narrow, second-stage validator.** A cycle-accurate
simulator already exists in openTARS (`tars_mini/rtlsim.py`: Verilator + the RTL core), so the
question is integration and calibration cost, not writing a model from scratch. For the current
search the binding uncertainty is **PPA**, not cycle ordering. Improving latency fidelity alone
would refine one of three objectives while leaving area, power, timing closure and the
non-RTL mapping hypotheses untouched.

## What the current model is good at, and where it stops

The analytical model is fitted per descriptor class on real openTARS counters and predicts total
core cycles. Reproduced validation:

| Dataset | MAPE | Worst |
|---|---:|---:|
| Qwen training total cycles | 0.30% | 0.74% |
| Gemma held-out | 4.20% | 4.72% |
| Baseline/repeat/excluded | — | 6.34% |
| Qwen training per-class | — | 8.02% |

Those errors are for baseline-like hardware. The model's per-unit slopes, new clocks, MXU
costs and mapping mechanisms are explicit engineering priors, not measurements. It executes
descriptors serially and deliberately does not model inter-descriptor overlap: queue depth is an
area knob with no assumed concurrency. Fusion and GQA reuse are traffic-scaling hypotheses;
Block DeltaNet's triangular solve is an uncalibrated FDOT-equivalent proxy.

A cycle-accurate core simulation would remove the serialization assumption and expose
backpressure/overlap effects for configurations the RTL actually implements. That is a real
fidelity gain. It is not the same as validating a fabricated chip, and it does not remove the
PPA priors.

## The current decision is not limited by cycle accuracy

FPGA grid: 128 evaluated, 80 resource-feasible, 7 nondominated.

| Point | Normalized latency | LUT fraction | Power (W) |
|---|---:|---:|---:|
| 1 (fastest) | 0.859857 | 0.996918 | 4.6185 |
| 2 | 0.879335 | 0.945005 | 4.5043 |
| 3 | 0.943613 | 0.895483 | 4.4430 |
| 4 | 1.041238 | 0.871576 | 4.3979 |
| 5 | 1.173198 | 0.884896 | 4.3862 |
| 6 | 1.237476 | 0.835374 | 4.3249 |
| 7 | 1.335101 | 0.811467 | 4.2798 |

The fastest two are only **2.27%** apart in latency, **5.21%** in area and **2.47%** in power.
The fastest and third are 9.74% apart. With held-out error of 4-6%, a more accurate latency model
could at best resolve the 1-vs-2 ordering; it would not establish that either point is a better
hardware investment, because their area differs by more than their latency.

More importantly, the fastest point has **361 LUT of headroom** (116,759 of 117,120, 0.31%).
Its extra area over point 2 comes almost entirely from the uncalibrated hold-buffer slope
(80 LUT/KiB). A slope error of only **+5.9%** (80 → 84.8 LUT/KiB) or a VPU slope error of
**+6.2%** (1,450 → 1,540 LUT/lane) makes the fastest point resource-infeasible. Since those
slopes are explicitly priors, "the fastest feasible point" is not a robust result. No amount of
cycle-accurate latency modeling fixes this: it is an area-model calibration problem.

## What the RTL can and cannot cover

openTARS `rtl/tm_core.sv` exposes many of the same knobs, but not all. A cycle-accurate run is
only meaningful for the subset it can actually build:

| openDARS knob | RTL coverage |
|---|---|
| `mxu_rows` / `mxu_cols` | `R` / `C` array parameters — yes |
| `hold_kib` | `HB` hold-buffer beats — yes |
| `outstanding` | `OT` / `VQ` reads in flight — yes |
| `gemv_batch` | `GB` GEMV rows — yes |
| `vpu_lanes` | `VG` row-op groups (1/2/4) — yes |
| `gemv_bytes` / `write_bytes` | `RW` port-width parameters — partially |
| `queue_depth` | `OQ` output queue — related, not equivalent |
| `fdot_lanes` | fixed in the pinned W8A8 RTL revision — no top-level knob |
| `mhz` | synthesis/DRAM configuration — not an RTL parameter |
| `chunk`, `skip_head` | compiler-side queue changes — runnable |
| `fusion`, `gqa_reuse` | analytical traffic hypotheses — no RTL implementation |
| `block_delta` | prototype/schedule — not an executable openTARS queue |

So a full RTL sweep would not validate the mapping mechanisms that produce some of the more
interesting speedups (e.g. Qwen3.5 block DeltaNet at 1.187x). Validating those requires new RTL
kernels, not a better simulator.

## Cost of running the existing RTL over the search

The six default workloads total **337,380,152 core cycles** and **59,436 descriptors**. One
128-point grid is therefore **43.18 Gcycles** and **7.6 M descriptors** before any rebuild or
image-export overhead.

| Simulated throughput | One full grid | On 32 workers |
|---:|---:|---:|
| 10 kHz | 1,200 h | 37.5 h |
| 100 kHz | 120 h | 3.75 h |
| 1 MHz | 12 h | 22 min |
| 10 MHz | 1.2 h | 2.3 min |

These are throughput scenarios, not measurements: Verilator is not installed in this workspace.
The real number must be measured. Also note the hidden costs: the 32 distinct hardware points
need parameterized RTL builds (minutes each; the four mappings reuse a build), the model image
is about 1.5 GB and is exported by `tools/export_kv260.py`, and queues must be generated from
openTARS lowering rather than the shape-only IR. At 1 MHz and 32 workers the arithmetic looks
cheap, but the integration and rebuild pipeline is the realistic first-pass cost.

One of the four mappings (the fusion/GQA/block-Delta variant) assumes mechanisms the pinned RTL
does not implement, so the runnable subset is smaller; the 43 Gcycle figure is an upper bound for
the full search space.

openDARS is pinned to openTARS revision `36af670` (W8A8). The openTARS worktree currently also
contains uncommitted v0.4 XPU/W4A16 work; any RTL integration must pin a revision and decide
whether it targets the calibrated W8A8 design or the newer, uncalibrated one.

A descriptor-level trace model would be much cheaper: 7.6 M descriptor evaluations for the whole
grid, i.e. seconds to minutes in Python. That is the right granularity if the goal is to add
overlap/queue effects, but it is a new model that still needs calibration against RTL; it is not
"free accuracy".

## Decision framework

Let Δ be the relative difference between two candidates and σ the validated model uncertainty.
Use σ ≈ 4-6% for baseline-like hardware; treat new-hardware extrapolation as larger and unknown.

- **Δ > 2σ (≳ 10%)**: analytical ranking is safe; do not simulate.
- **σ < Δ < 2σ (≈ 4-10%)**: RTL or synthesis on the shortlist can change the decision.
- **Δ < σ (≲ 4%)**: neither analytical latency nor uncalibrated PPA can rank the pair; measure
  the actual design.

The top two are at Δ = 2.3%, inside σ. The top-vs-third gap is 9.7%, near the boundary. Beyond
the top three the frontier is separated by 21-55% from the fastest point, so a higher-fidelity
model would not change the coarse shape of the frontier.

## Recommendation

Ranked by return on effort:

1. **Synthesize the two top FPGA points first (highest ROI).** Report actual LUT/FF/DSP/BRAM,
   Fmax, and power. This resolves the 0.31%-headroom knife-edge and calibrates the hold-buffer
   and VPU slopes. A cycle-accurate model cannot answer the question the fastest point actually
   poses.
2. **Run targeted RTL microbenchmarks (days).** Measure the sensitivity of the analytical
   slopes for VPU lanes, FDOT, hold buffer, GEMV epilogue and outstanding reads. Update
   `dars/hardware.py` priors. This improves all 128 points at once and is reusable.
3. **Two-stage search (days-weeks).** Keep the analytical grid/NSGA-II as the screen. Run RTL
   simulation only on the 2-5 candidates whose Δ is inside the uncertainty band and that the RTL
   can actually build. Validate latency; keep PPA from synthesis.
4. **Descriptor-level trace model (2-3 weeks, conditional).** Worth building only if the design
   space is expanded to concurrency/queue-depth knobs that the current serialized model cannot
   rank. Acceptance target: within 3% of RTL on 5-10 representative queues, under a second per
   configuration.
5. **Full cycle-accurate model from scratch (2-3 months, not recommended).** The existing RTL
   simulator already provides ground truth for latency. Building another cycle model duplicates
   it and still requires calibration for any mechanism that is not in RTL.

## What would change this recommendation

- Measured Verilator throughput above ~5 MHz with enough workers would make a full-grid RTL
  sweep an overnight job. It would still be a validation sweep, not an inner-loop evaluator.
- Expanding the search to queue depth/concurrency, double buffering, or multi-queue scheduling
  would make the current serialized model structurally wrong; then a trace model or RTL sweep
  becomes necessary.
- Calibrated PPA and timing closure would make latency the dominant objective, raising the
  value of cycle accuracy.
- An ASIC target with tight area/power margins would make synthesis-based PPA even more
  important relative to cycle simulation.

## Bottom line

A cycle-accurate model is **high cost, medium value for search, high value for final
validation**. Do not build one for the inner loop. Use the existing openTARS RTL simulator as a
second-stage validator for a shortlist, and spend the next engineering cycle on PPA calibration
and targeted RTL microbenchmarks. That is where the current frontier's ranking is actually
undetermined.
