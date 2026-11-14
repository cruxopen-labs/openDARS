"""Reproducible report/plots; evidence and extrapolations kept separate."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .hardware import BASE, ppa
from .performance import Mapping, Performance, samples
from .search import Space, Evaluator, grid, nsga2, pareto
from .workload import MODELS, ROOT, config, workload

MEASUREMENTS = ROOT / "data" / "measurements"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _measurement(stem, label):
    data = json.loads((MEASUREMENTS / f"{stem}.json").read_text())
    try:
        return next(sample for sample in data["samples"] if sample["label"] == label)
    except StopIteration:
        raise ValueError(f"missing measurement sample: {stem}: {label}") from None


def _operator(sample, name):
    try:
        return next(op for op in sample["operators"] if op["name"] == name)
    except StopIteration:
        raise ValueError(f"missing operator {name} in {sample['label']}") from None


def _pct(sample, *names):
    return sum(_operator(sample, name)["pct"] for name in names)


def observed_bottlenecks():
    """Narrative derived from the preserved measurements, not hand-copied numbers."""
    q3_16 = _measurement("qwen3_profile", "decode t=16")
    q3_250 = _measurement("qwen3_profile", "decode t=250")
    q35_16 = _measurement("qwen35_profile", "decode t=16")
    q3_p20 = _measurement("qwen3_prefill20", "prefill 20 tokens")
    q35_p20 = _measurement("qwen35_prefill20", "prefill 20 tokens")
    prefill_writes = [_operator(sample, "gemv")["waits_pct"]["wr_wait"]
                      for sample in (q3_p20, q35_p20)]
    fdot_nn_wait = _operator(q3_250, "fdot_nn")["waits_pct"]["rd_wait"]
    return [
        f"- Qwen3 t=16: GEMV {_pct(q3_16, 'gemv'):.1f}%; "
        f"t=250: GEMV {_pct(q3_250, 'gemv'):.1f}%, "
        f"FDOT {_pct(q3_250, 'fdot_nt', 'fdot_nn'):.1f}%. "
        "Long-context attention/request-path work matters more than increasing the INT8 MAC count.",
        f"- Qwen3.5 t=16: GEMV {_pct(q35_16, 'gemv'):.1f}%, ELT {_pct(q35_16, 'elt'):.1f}%; "
        f"prefill20: ELT {_pct(q35_p20, 'elt'):.1f}%. "
        "DeltaNet state updates are scalar/broadcast-heavy; MXU alone cannot fix them.",
        f"- GEMV prefill write-request waits: {prefill_writes[0]:.1f}% / {prefill_writes[1]:.1f}% "
        "of GEMV cycles. Prioritize epilogue/output acceptance and buffers, not only DDR read width.",
        f"- Qwen3 t=250 FDOT NN read-request wait: {fdot_nn_wait:.1f}% of that class. "
        "Wait counters overlap and are never added together.",
    ]


def analysis(perf):
    result = []
    alternatives = {
        "baseline": Mapping(), "chunk1": Mapping(chunk=1), "chunk2": Mapping(chunk=2),
        "skip_intermediate_head": Mapping(skip_head=True),
        "fusion_hypothesis": Mapping(fusion=True),
        "gqa_reuse_hypothesis": Mapping(gqa_reuse=True),
        "block_delta_prototype": Mapping(delta_block=True),
    }
    for model in MODELS:
        c = config(model)
        for mode in ("decode", "prefill"):
            ref = None
            for name, mapping in alternatives.items():
                # State + Q/K/V/intermediates: about 76.2 KiB for B=4.
                h = replace(BASE, hold_kib=80) if mapping.delta_block else BASE
                mp = mapping if mode == "prefill" else replace(mapping, chunk=1)
                ops = workload(c, mode, 250, 20, mapping.chunk, mapping.skip_head)
                pred = perf.predict(ops, h, mp, c)
                if ref is None:
                    ref = pred["latency_ms"]
                result.append(dict(model=model, mode=mode, alternative=name, hardware=asdict(h),
                                   mapping=asdict(mp), descriptors=len(ops), **pred,
                                   speedup=ref / pred["latency_ms"]))
    return result


def search_report(perf, out, method="grid", target="fpga", space_path=None,
                  population=32, generations=12, seed=7, position=250, tokens=20):
    space = Space(space_path)
    evaluator = Evaluator(perf, space, target, position, tokens)
    rows = grid(evaluator) if method == "grid" else nsga2(evaluator, population, generations, seed)
    front = pareto(rows)
    stem = f"{target}_{method}"
    payload = dict(target=target, method=method, seed=seed, search_size=space.size,
                   position=position, prefill_tokens=tokens,
                   objectives=["mean of six baseline-normalized latencies", "area", "power_w"],
                   evaluated=len(rows), feasible=sum(r["feasible"] for r in rows), pareto=front, points=rows)
    write_json(out / f"{stem}.json", payload)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    good = [r for r in rows if r["feasible"]]
    if good:
        scatter = ax.scatter([r["area"] for r in good], [r["latency_ratio"] for r in good],
                             c=[r["power_w"] for r in good], s=24, alpha=.6)
        fig.colorbar(scatter, ax=ax, label="Estimated power (W)")
    ax.scatter([r["area"] for r in front], [r["latency_ratio"] for r in front],
               facecolors="none", edgecolors="red", s=80, label="3-objective nondominated")
    ax.set(xlabel="LUT fraction" if target == "fpga" else "Uncalibrated area (mm²)",
           ylabel="Mean normalized core latency (lower is better)",
           title=f"{target.upper()} {method}: analytical predictions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=140)
    plt.close(fig)
    return payload


def generate(out, include_search=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    perf = Performance().fit()
    perf.save(out / "calibration.json")
    validation = perf.validate()
    write_json(out / "validation.json", validation)
    alternatives = analysis(perf)
    write_json(out / "mapping.json", alternatives)
    train = [abs(r["error_pct"]) for r in validation if r["train"]]
    gemma = [abs(r["error_pct"]) for r in validation if r["model"] == "gemma3"]
    other = [abs(r["error_pct"]) for r in validation if not r["train"] and r["model"] != "gemma3"]
    class_max = max(abs(v) for r in validation if r["train"] for v in r["class_error_pct"].values())
    lines = [
        "# openDARS analytical exploration report", "",
        "## Evidence and validation", "",
        "Runtime: W8A8, FP32 state, core busy time (not host latency). "
        "openTARS source revision: `36af670d9df0f211d5b6377505f3311c410d33f8`.",
        "The lowering test compares 12 real descriptor queues, excluding GEMM row-scale addresses.",
        "Fit: Qwen3/Qwen3.5 per-class counters, decode and 5/20-token prefill. "
        "Qwen3 t=8 is excluded as a documented one-off outlier. "
        "Gemma3 and baseline/repeat measurements never enter fitting.", "",
        f"- Training total-cycle MAPE: {np.mean(train):.2f}%; maximum: {max(train):.2f}%.",
        f"- Gemma holdout MAPE: {np.mean(gemma):.2f}%; maximum: {max(gemma):.2f}%.",
        f"- Other excluded/repeat/baseline maximum error: {max(other):.2f}%.",
        f"- Training per-class maximum absolute error: {class_max:.2f}%.",
        "- Per-class errors are preserved in `validation.json`; total accuracy can hide class errors.",
        "- These are empirical errors, not confidence intervals or hardware extrapolation guarantees.", "",
        "| Dataset | Workload | Measured ms | Predicted ms | Error |",
        "|---|---|---:|---:|---:|",
    ]
    for r in validation:
        lines.append(f"| {r['dataset']} | {r['label']} | {r['measured_cycles']/1e5:.3f} | "
                     f"{r['predicted_cycles']/1e5:.3f} | {r['error_pct']:+.2f}% |")
    lines += ["", "## Observed bottlenecks (not inferred utilization)", "",
              *observed_bottlenecks(), "",
              "## Baseline predictions and mapping alternatives", "",
              "Decode uses t=250; prefill uses 20 tokens. Speedups below are model predictions. "
              "Chunking/head skipping change descriptor workloads; fusion/GQA/resident DeltaNet "
              "need RTL validation. Block DeltaNet uses a unit-lower triangular solve "
              "and batched state products; its float64 algebra is tested against token recurrence. "
              "Its FDOT-equivalent triangular-solve cost is uncalibrated and is not an executable "
              "openTARS queue. An 80 KiB state/workspace buffer is charged to the hardware; "
              "decode retains token recurrence. "
              "GQA reduces KV reads, not dot-product arithmetic, so compute-bound cases need not improve.", "",
              "| Model | Mode | Alternative | Predicted ms | Speedup |",
              "|---|---|---|---:|---:|"]
    for r in alternatives:
        lines.append(f"| {r['model']} | {r['mode']} | {r['alternative']} | {r['latency_ms']:.3f} | {r['speedup']:.3f}x |")
    lines += ["", "## PPA assumptions", "",
              "FPGA baseline: 104,879 LUT, 903 DSP, 118 BRAM (original v3.3). "
              "The instrumented measurement build is 104,967 LUT, 903 DSP, 118 BRAM. "
              "FF scaling uses v3.4's 59,170 / 105,992 ratio. "
              "Power baseline is the v3.4 Vivado estimate: 4.443 W, including 2.435 W PS8 "
              "and 0.312 W static, NOT a board power measurement.", "",
              "Only total baseline resources/power are anchored to observations. Per-unit slopes, "
              "new clocks, MXU, and mappings have not been synthesized or calibrated. "
              "FPGA feasibility means estimated resource fit, NOT timing closure. "
              "ASIC is an explicitly UNCALIBRATED illustrative 28nm-equivalent area/power model; "
              "it uses no cell library, place/route, SRAM compiler, or measured ASIC data. "
              "Do not use its absolute values for tapeout or compare FPGA LUT fractions with ASIC mm².", "",
              "The baseline has ARR=0; an optional MXU includes tile fill/drain and resource cost. "
              "The small MXU is often slower than 128-byte batch GEMV for <=4 tokens. "
              "Historical 16x8 and 16x16 arrays exceeded KV260 LUT capacity. "
              "The template exposes GEMV width/batch, VPU/FDOT/SFU lanes, DDR bandwidth, "
              "write width, hold buffer, outstanding reads, queue depth, clock and MXU shape. "
              "Queue depth is a capacity/area knob, not an assumed execution-parallelism speedup.", ""]
    if include_search:
        lines += ["## Design-space search", "",
                  "Objectives: mean of six equally weighted baseline-normalized latencies "
                  "(3 models x decode/prefill), area, power. No claim of a global optimum "
                  "outside the configured discrete space. Grid prunes resource-infeasible "
                  "designs before performance evaluation. NSGA-II is seeded, approximate "
                  "and evaluated against the exact grid frontier.", ""]
        reports = {}
        for target in ("fpga", "asic"):
            for method in ("grid", "nsga2"):
                r = search_report(perf, out, method, target)
                reports[target, method] = r
                lines += [f"### {target.upper()} / {method}", "",
                          f"Evaluated {r['evaluated']}/{r['search_size']}; resource-feasible "
                          f"{r['feasible']}; nondominated points {len(r['pareto'])}.",
                          "", f"![{target} {method}]({target}_{method}.png)", ""]
                if r["pareto"]:
                    best = min(r["pareto"], key=lambda p: p["latency_ratio"])
                    lines += [f"Fastest modeled point: normalized latency {best['latency_ratio']:.4f}; "
                              f"area {best['area']:.4f}; power {best['power_w']:.3f} W.",
                              f"Hardware: `{json.dumps(best['hardware'], sort_keys=True)}`.",
                              f"Mapping: `{json.dumps(best['mapping'], sort_keys=True)}`.", ""]
            exact = {tuple(round(p[k], 9) for k in ("latency_ratio", "area", "power_w"))
                     for p in reports[target, "grid"]["pareto"]}
            found = {tuple(round(p[k], 9) for k in ("latency_ratio", "area", "power_w"))
                     for p in reports[target, "nsga2"]["pareto"]}
            lines += [f"NSGA-II recovered {len(exact & found)}/{len(exact)} distinct exact-grid "
                      f"objective tuples for {target}. Duplicate mappings may share objectives.", ""]
        fpga = reports["fpga", "grid"]["pareto"]
        safer = [p for p in fpga if p["area"] <= .95]
        if safer:
            best = min(fpga, key=lambda p: p["latency_ratio"])
            safe = min(safer, key=lambda p: p["latency_ratio"])
            lines += ["### Resource headroom matters", "",
                      f"The fastest estimated FPGA point uses {100*best['area']:.2f}% of LUTs. "
                      "It is not a robust implementation recommendation. "
                      f"For example, restricting estimated LUT use to <=95% gives normalized latency "
                      f"{safe['latency_ratio']:.4f} at {100*safe['area']:.2f}% LUTs "
                      f"({100*(safe['latency_ratio']/best['latency_ratio']-1):.2f}% more latency). "
                      "This small modeled difference is below baseline model error and far below "
                      "the uncertainty of new-hardware PPA. Synthesize the lower-resource point first.",
                      f"Lower-resource hardware: `{json.dumps(safe['hardware'], sort_keys=True)}`.", ""]
    lines += ["## Recommended next experiments", "",
              "1. Profile DeltaNet ELT broadcast/state updates with wider VPU issue and resident state tiles.",
              "2. Widen/decouple GEMV epilogue writes and measure prefill backpressure.",
              "3. Test FDOT request coalescing and KV reuse at long contexts; retain GEMV for decode.",
              "4. Skip LM-head computation on intermediate prefill chunks when only final logits are consumed.",
              "5. Synthesize selected Pareto points, close timing, measure board power and rerun exported "
              "golden logits before treating any prediction as a hardware improvement.", "",
              "Sources and provenance: [../data/SOURCES.md](../data/SOURCES.md)."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return dict(training_max_error_pct=max(train), gemma_max_error_pct=max(gemma),
                output=str(out / "report.md"))
