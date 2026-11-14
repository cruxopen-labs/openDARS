"""Print the decision margins behind docs/cycle_accuracy_assessment.md.

Run from a source checkout: ``python tools/decision_margins.py``.
Reads only checked-in reports and measurements; no fitting or plotting.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dars.hardware import BASE, LIMITS, Hardware, ppa  # noqa: E402
from dars.workload import ROOT  # noqa: E402

HARDWARE_FIELDS = ("gemv_bytes", "gemv_batch", "vpu_lanes", "fdot_lanes", "sfu_lanes",
                   "outstanding", "queue_depth", "hold_kib", "write_bytes")


def six_workload_cycles():
    files = [("qwen3_profile", "decode t=250"), ("qwen3_prefill20", "prefill 20 tokens"),
             ("qwen35_profile", "decode t=250"), ("qwen35_prefill20", "prefill 20 tokens"),
             ("gemma3_profile", "decode t=250"), ("gemma3_profile", "prefill 17 tokens")]
    total_cycles = total_descriptors = 0
    for stem, label in files:
        data = json.loads((ROOT / f"data/measurements/{stem}.json").read_text())
        sample = next(s for s in data["samples"] if s["label"] == label)
        total_cycles += sample["cycles"]
        total_descriptors += sample["descriptors"]
    return total_cycles, total_descriptors


def main():
    fpga = json.loads((ROOT / "reports/fpga_grid.json").read_text())
    front = sorted(fpga["pareto"], key=lambda row: row["latency_ratio"])
    print(f"FPGA grid: {fpga['evaluated']} evaluated, {fpga['feasible']} feasible, "
          f"{len(front)} nondominated")
    print(f"{'latency':>9} {'area':>8} {'power W':>8}")
    for row in front:
        print(f"{row['latency_ratio']:9.6f} {row['area']:8.6f} {row['power_w']:8.4f}")

    if len(front) < 2:
        return
    best, second = front[0], front[1]
    print(f"\nfastest vs second: latency {100*(second['latency_ratio']/best['latency_ratio']-1):+.2f}%, "
          f"area {100*(second['area']/best['area']-1):+.2f}%, "
          f"power {100*(second['power_w']/best['power_w']-1):+.2f}%")
    if len(front) > 2:
        print(f"fastest vs third:  latency {100*(front[2]['latency_ratio']/best['latency_ratio']-1):+.2f}%")

    validation = json.loads((ROOT / "reports/validation.json").read_text())
    train = [abs(r["error_pct"]) for r in validation if r["train"]]
    gemma = [abs(r["error_pct"]) for r in validation if r["model"] == "gemma3"]
    other = [abs(r["error_pct"]) for r in validation if not r["train"] and r["model"] != "gemma3"]
    print(f"\nmodel error: training MAPE {sum(train)/len(train):.2f}% (max {max(train):.2f}%), "
          f"Gemma MAPE {sum(gemma)/len(gemma):.2f}% (max {max(gemma):.2f}%), "
          f"other held-out max {max(other):.2f}%")

    hardware = {k: best["hardware"][k] for k in HARDWARE_FIELDS}
    h = Hardware(**best["hardware"])
    estimate = ppa(h, "fpga")
    headroom = LIMITS["lut"] - estimate["lut"]
    print(f"\nfastest FPGA point: {estimate['lut']:.0f} LUT / {LIMITS['lut']} "
          f"(headroom {headroom:.0f}, {100*headroom/LIMITS['lut']:.2f}%)")
    print("a slope increase alone that would make it infeasible:")
    slopes = dict(gemv_bytes=110, gemv_batch=700, vpu_lanes=1450, fdot_lanes=350,
                  sfu_lanes=1800, outstanding=20, queue_depth=15, hold_kib=80, write_bytes=40)
    for field, slope in slopes.items():
        delta = hardware[field] - getattr(BASE, field)
        if delta:
            extra = headroom / delta
            print(f"  {field:12s} {slope:6.1f} -> {slope + extra:7.1f} "
                  f"({100*extra/slope:+.1f}%)")

    cycles, descriptors = six_workload_cycles()
    print(f"\nsix default workloads: {cycles:,} cycles, {descriptors:,} descriptors")
    print(f"one 128-point grid: {cycles*128/1e9:.2f} Gcycles, {descriptors*128/1e6:.1f} M descriptors")
    for rate in (1e4, 1e5, 1e6, 1e7):
        print(f"  at {rate:.0e} cycles/s: {cycles*128/rate/3600:7.1f} h "
              f"({cycles*128/rate/32/3600:5.2f} h on 32 workers)")


if __name__ == "__main__":
    main()
