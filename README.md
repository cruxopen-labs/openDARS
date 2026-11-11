# openDARS

Design-space exploration for [openTARS](https://github.com/cruxopen-labs/openTARS).
HF configs → ordered W8A8 workload → parameterized hardware/mapping model →
calibrated core-cycle predictions → PPA estimates → Pareto frontier.

## Run

Python 3.10+ on Linux, no checkpoints/torch/FPGA needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
dars report --out reports
```

Run from a source checkout (configs and evidence are repository assets).
The equivalent module entry point is `python -m dars.cli`.

```bash
dars calibrate --out reports
dars analyze --out reports
dars search --method grid --target fpga --space configs/search.json --out reports
dars search --method nsga2 --target asic --seed 7 --population 32 --generations 12 --out reports
```

See [generated analysis](reports/report.md) and [evidence provenance](data/SOURCES.md).
JSON results retain every candidate, per-model decode/prefill latency, calibration
errors and mapping assumptions. Plots show the three-objective nondominated set.
`search --position 250 --tokens 20` controls the workload; `report` uses these defaults.

### Validated snapshot

- 19 tests passed, including all 12 openTARS queue comparisons and Block DeltaNet algebra.
- Qwen training total-cycle MAPE **0.30%**, worst **0.74%**.
- Gemma held-out MAPE **4.20%**, worst **4.72%**; excluded/repeat/baseline worst **6.34%**.
- Training per-class worst error **8.02%**, reported separately from total error.
- FPGA grid: **128** combinations, **80** estimated resource-feasible, **7** Pareto points.
  Seeded NSGA-II recovered all 7 distinct frontier objective tuples.
- Fastest modeled normalized latency **0.8599**, but estimated LUT use is **99.69%**.
  A **94.50%** LUT point gives **0.8793** normalized latency and is a better first
  synthesis experiment. Neither point is a demonstrated hardware speedup.

All 13 measurement snapshots were checked byte-for-byte against the openTARS source.
On the development Linux/Python 3.12 environment, full report generation takes about
70 seconds and 300 MiB peak RSS. These are observations, not performance guarantees.

## What is implemented

- **Workloads:** Qwen3-0.6B, Qwen3.5-0.8B (text), Gemma3-1B. Shape-only lowering,
  including GQA, rotary dimensions, recurrent DeltaNet, split Gemma down projections,
  equal-block LM heads, aligned prefill chunks and tails.
- **Cross-check:** 12 captured real openTARS queues, exact semantic descriptor order,
  shapes, control/epilogue/broadcast/stride fields. No fixture-driven lowering.
- **Performance:** serialized descriptor execution; GEMV batch/weight reuse, optional
  MXU tiles, FDOT fast/fallback paths, VPU/SFU service, row-op passes/buffers,
  descriptor overhead and bandwidth. Nonnegative per-class fit on Qwen measurements;
  Gemma and original/repeat runs are held out. No summing overlapping wait counters.
- **Mapping:** chunks 1/2/4 and intermediate-head skipping change workloads.
  Fusion and GQA KV read reuse are hypothetical traffic models. Block DeltaNet
  has a float64 mathematical prototype (unit-lower solve and batched state products),
  tested against token recurrence, and a capacity-gated analytical schedule.
  Its triangular-solve timing is uncalibrated; it is not an executable openTARS queue.
- **PPA:** FPGA resource/power estimates anchored at baseline totals; ASIC estimates
  explicitly uncalibrated. Resource pruning does not establish timing closure.
- **Search:** configurable exhaustive grid and seeded NSGA-II (non-dominated sorting,
  tournament selection, uniform crossover, mutation, crowding, elitism).
  Default space: 128 hardware/mapping combinations; all three models participate.

## Interpreting results

Core busy time is not end-to-end chat latency. Training fit is not validation:
the report gives both and retains the excluded outlier. Good baseline cycle accuracy
does not validate new hardware or mapping speedups. PPA slopes, clocks and MXUs need
RTL synthesis; power needs board measurement. ASIC absolute numbers are illustrative
28nm-equivalent assumptions, not technology-library estimates.

The default objectives minimize equal-weight mean normalized latency across six
workloads (three models × decode/prefill), area and power. Individual latencies
remain available so the aggregate cannot hide a workload regression. FPGA area is
LUT fraction; ASIC area is mm², never compare them directly.

Change domains/mappings in `configs/search.json`; unspecified hardware fields use
`dars.hardware.BASE`. Supported hardware fields include clock, GEMV width/batch,
VPU/FDOT/SFU lanes, DDR bandwidth, write width, hold buffer, outstanding reads,
queue depth, and `mxu: [rows, cols]`. Queue depth has area cost but no assumed
concurrent descriptor execution.

## Layout

| Path | Purpose |
|---|---|
| `dars/workload.py` | HF config loader and openTARS-aligned IR/lowering |
| `dars/hardware.py` | Hardware template and PPA assumptions |
| `dars/performance.py` | Service model, calibration and mapping hypotheses |
| `dars/delta.py` | Block DeltaNet algebra prototype and analytical schedule |
| `dars/search.py` | Grid, NSGA-II, Pareto sorting |
| `dars/report.py`, `cli.py` | Reproducible outputs and commands |
| `configs/` | Model snapshots and discrete search space |
| `data/measurements/` | Preserved FPGA evidence |
| `tests/fixtures/` | Independent openTARS lowering oracle |

To regenerate the oracle, in openTARS's environment:

```bash
source /path/to/openTARS/env.sh
python tools/dump_opentars_queue.py --opentars /path/to/openTARS \
  --out tests/fixtures/opentars_queues.json
```

This is intentionally heavier than ordinary tests: it constructs random weights
with real shapes. Normal tests consume the checked-in fixture without openTARS.
