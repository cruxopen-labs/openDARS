"""Dump openTARS's real descriptor queues as a compact JSON fixture for DARS's lowering test.

Run once in openTARS's environment (it needs torch and transformers for the model structure):

    source <openTARS>/env.sh
    python tools/dump_opentars_queue.py --opentars <openTARS> --out tests/fixtures/opentars_queues.json

Weights are random with the real shapes; descriptor shapes do not depend on weight values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CASES = (("decode", 16, 1), ("decode", 250, 1), ("prefill", 4, 4), ("prefill", 0, 4))
FIELDS = ("OPCODE", "CTRL", "EPI", "M", "N", "K", "HDR_RSV", "GROUP_SIZE", "LDA16", "LDY16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opentars", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", default="qwen3,qwen3.5,gemma3")
    args = ap.parse_args()
    sys.path.insert(0, str(args.opentars))
    from tars_mini import contract as C
    from tars_mini import qwen_ref as Q
    from tars_mini.llm import QwenLowering

    configs = {
        "qwen3": (Q.CFG_Q3_06B, ["full_attention"] * 28, Q.VOCAB_Q3_06B),
        "qwen3.5": (Q.CFG_08B, Q.LAYERS_08B, Q.VOCAB_08B),
        "gemma3": (Q.CFG_G3_1B, Q.LAYERS_G3_1B, Q.VOCAB_G3_1B),
    }
    out = {}
    for name in args.models.split(","):
        base, layers, vocab = configs[name]
        cfg = Q.make_config(base, layers, vocab=vocab)
        weights = Q.random_weights(cfg, seed=0, embed=True)
        low = QwenLowering(cfg, weights, max_tokens=256, ddr_bytes=1 << 31, head=True)
        del weights
        cases = {}
        for mode, t0, n in CASES:
            start = len(low.p.descs)
            low.chunk(t0, n)
            rows = [[C.decode(d)[f] for f in FIELDS] for d in low.p.descs[start:]]
            cases[f"{mode}:{t0}:{n}"] = rows
            print(f"{name} {mode} t0={t0} n={n}: {len(rows)} descriptors", flush=True)
        out[name] = cases
        del low
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(fields=FIELDS, models=out), separators=(",", ":")))


if __name__ == "__main__":
    main()
