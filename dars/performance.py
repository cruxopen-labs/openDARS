"""Nonnegative per-class analytical regression, cycles at the requested clock.

Each descriptor executes serially. Transfer/compute overlap is conservatively
represented by additive fitted service terms, not summed hardware wait counters.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import re
import numpy as np
from scipy.optimize import lsq_linear
from .hardware import BASE
from .delta import block_schedule
from .workload import ROOT, config, workload

KINDS = ("overhead", "gemv", "elt", "fdot_nt", "fdot_nn", "rope", "quant", "norm", "softmax", "act")
FEATURES = ("launch", "bottleneck_service", "strided_or_state_service", "gemv_write_serialization", "row_setup", "reserved_sfu")


@dataclass(frozen=True)
class Mapping:
    chunk: int = 4
    fusion: bool = False
    gqa_reuse: bool = False
    delta_block: bool = False
    skip_head: bool = False

    def __post_init__(self):
        if self.chunk not in (1, 2, 4):
            raise ValueError("Mapping chunk must be 1, 2 or 4")


@lru_cache(maxsize=200000)
def features(op, h=BASE):
    m, n, k = op.m, op.n, op.k
    x = m * n
    read_rate = min(h.read_gbs * 1000 / h.mhz, h.gemv_bytes) * min(1, h.outstanding / 16)
    narrow = 4 * h.vpu_lanes
    compute, read, write, rows, sfu = 0., 0., x * 4 / h.write_bytes, m, 0.
    kind = op.kind
    if kind == "gemv":
        groups = np.ceil(m / h.gemv_batch)
        compute = groups * n * k / h.gemv_bytes
        # MXU tiles include fill/drain; INT8 MACs, no magical reuse across queues.
        if h.mxu_rows:
            mxu = np.ceil(m / h.mxu_rows) * np.ceil(n / h.mxu_cols) * (k + h.mxu_rows + h.mxu_cols - 2)
            compute = min(compute, mxu)
        read = (groups * n * k + m * k + 4 * n) / read_rate
        sfu = x / h.sfu_lanes if op.activation else 0
        # GEMV epilogue retirement, separate from its optional pipelined SFU.
        rows = x
    elif kind.startswith("fdot"):
        fast = (n % 64 == 0) if kind == "fdot_nn" else (k <= 512 and k % 32 == 0 and n % 4 == 0)
        compute = x * k / (h.fdot_lanes if fast else 2)
        read = 4 * m * n * k / min(read_rate, 64 if fast else 16)
        rows = x * (1 if kind == "fdot_nn" else 12)
    elif kind == "elt":
        compute = x / h.vpu_lanes * (3 if op.ctrl == 2 else 2 if op.hdr & 3 == 0 else 1)
        operands = 3 if op.ctrl == 2 else 2
        read = 4 * x * operands / min(read_rate, narrow)
        sfu = x / h.sfu_lanes if op.activation else 0
    elif kind == "rope":
        compute = m * (k / h.sfu_lanes + (n - k) / h.vpu_lanes)
        read = 4 * x / min(read_rate, narrow)
    elif kind == "quant":
        compute = 2 * x / h.vpu_lanes
        read = 8 * x / min(read_rate, narrow)
        write = x / h.write_bytes
        rows = m * 42
    elif kind == "norm":
        compute = 2 * x / h.vpu_lanes
        read = 4 * x * (1 if n <= min(512, h.hold_kib * 256) else 2) / min(read_rate, narrow)
        rows = m * 117
    elif kind == "softmax":
        compute = 2 * x / h.vpu_lanes
        read = 4 * x * (1 if n <= min(1024, h.hold_kib * 256) else 3) / min(read_rate, narrow)
        rows = m * 90
        sfu = x / h.sfu_lanes
    else:
        read = 4 * x / min(read_rate, narrow)
        sfu = x / h.sfu_lanes
        rows = m * (170 if op.activation == 6 else 122 if op.activation in (3, 4, 5) else 51)
    return np.array([1, compute, read, write, rows, sfu], dtype=float)


def aggregate(ops, h=BASE, mapping=Mapping(), c=None):
    if mapping.delta_block:
        ops = block_schedule(ops, c, h.hold_kib)
    out = {k: np.zeros(len(FEATURES)) for k in KINDS}
    out["overhead"][0] = len(ops)
    counts = Counter(replace(op, tag=op.tag.split("/")[-1]) for op in ops)
    for op, count in counts.items():
        f = features(op, h).copy()
        # Hypothetical mapping mechanisms, not numerical/RTL implementations.
        if mapping.gqa_reuse and c and op.tag == "attention" and op.kind.startswith("fdot"):
            f[2] *= c["num_key_value_heads"] / c["num_attention_heads"]
        if mapping.fusion and op.kind in ("quant", "elt"):
            # Bypass half of producer/consumer DDR traffic, keep arithmetic.
            f[2:4] *= 0.5
        # The service term is a roofline, so a collinear regression cannot
        # accidentally delete all bandwidth or SFU dependence.
        f[1] = max(f[1], f[2], f[3], f[5])
        f[2] = f[5] = 0
        # Broadcast matrix state updates and strided KV reads have different
        # issue/request paths. Classify from descriptor fields, never model ID.
        if ((op.kind == "elt" and op.m > 1 and op.hdr in (3, 6))
                or (op.kind == "fdot_nn" and op.group != 0)):
            f[2], f[1] = f[1], 0
        if op.kind != "gemv":
            f[3] = 0
        out[op.kind] += count * f
    return out


def samples(include_holdout=True):
    names = [("qwen3", "qwen3"), ("qwen3.5", "qwen35")]
    if include_holdout:
        names.append(("gemma3", "gemma3"))
    for model, stem in names:
        files = [ROOT / f"data/measurements/{stem}_profile.json"]
        if model != "gemma3":
            files.append(ROOT / f"data/measurements/{stem}_prefill20.json")
            if include_holdout:
                files.extend(ROOT / f"data/measurements/{stem}_{suffix}.json" for suffix in ("baseline", "repeat"))
        for path in files:
            data = json.loads(path.read_text())
            for sample in data["samples"]:
                label = sample["label"]
                if label.startswith("decode"):
                    position = int(re.search(r"t=(\d+)", label)[1])
                    # t8 Qwen3 is retained in validation but excluded from fitting.
                    kwargs = dict(mode="decode", position=position)
                elif label.startswith("prefill"):
                    kwargs = dict(mode="prefill", tokens=int(re.search(r"\d+", label)[0]))
                else:
                    continue
                yield dict(model=model, label=label, dataset=path.name, kwargs=kwargs, sample=sample,
                           train=model != "gemma3" and path.stem.endswith(("_profile", "_prefill20"))
                           and not (model == "qwen3" and label == "decode t=8"))


class Performance:
    def __init__(self, coefficients=None):
        self.coefficients = coefficients or {k: [1] * 6 for k in KINDS}

    def fit(self):
        datasets = defaultdict(list)
        for item in samples(False):
            if not item["train"]:
                continue
            c = config(item["model"])
            a = aggregate(workload(c, **item["kwargs"]))
            for op in item["sample"]["operators"]:
                datasets[op["name"]].append((a[op["name"]], op["cycles"]))
        for kind in KINDS:
            data = datasets[kind]
            X = np.array([x for x, y in data])
            y = np.array([y for x, y in data])
            # Relative-error fit, normalized columns. Nonidentifiable service
            # terms are explicitly not interpreted as physical unit latencies.
            scale = np.maximum(np.linalg.norm(X / np.maximum(y[:, None], 1), axis=0), 1e-12)
            A = X / np.maximum(y[:, None], 1) / scale
            active = np.any(X != 0, axis=0)
            lower = np.zeros(6)
            if kind != "overhead":
                lower[1] = 0.1 * scale[1]
                lower[2] = 0.1 * scale[2]
            result = lsq_linear(A[:, active], y / np.maximum(y, 1), bounds=(lower[active], np.inf),
                                tol=1e-12, max_iter=1000)
            if not result.success:
                raise RuntimeError(f"Calibration failed: {kind}: {result.message}")
            coef = np.zeros(6)
            coef[active] = result.x / scale[active]
            self.coefficients[kind] = coef.tolist()
        return self

    def predict_features(self, a, h=BASE):
        by_kind = {k: float(np.dot(a[k], self.coefficients[k])) for k in KINDS}
        cycles = sum(by_kind.values())
        return dict(cycles=cycles, latency_ms=cycles / (h.mhz * 1000), by_kind=by_kind)

    def predict(self, ops, h=BASE, mapping=Mapping(), c=None):
        return self.predict_features(aggregate(ops, h, mapping, c), h)

    def validate(self):
        rows = []
        for item in samples():
            c = config(item["model"])
            p = self.predict(workload(c, **item["kwargs"]))
            actual = item["sample"]["cycles"]
            class_errors = {op["name"]: 100 * (p["by_kind"][op["name"]] / op["cycles"] - 1)
                            for op in item["sample"].get("operators", []) if op["cycles"]}
            rows.append(dict(model=item["model"], label=item["label"], dataset=item["dataset"], train=item["train"],
                             measured_cycles=actual, predicted_cycles=p["cycles"],
                             error_pct=100 * (p["cycles"] / actual - 1), class_error_pct=class_errors))
        return rows

    def save(self, path):
        path.write_text(json.dumps(dict(features=FEATURES, coefficients=self.coefficients), indent=2) + "\n")
