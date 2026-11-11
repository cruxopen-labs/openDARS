"""Shape-only HF lowering following openTARS tars_mini/llm.py.

No checkpoints, tensors, torch, or openTARS import is required at runtime.
Addresses (including GEMM HDR_RSV row-scale addresses) are deliberately not IR.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"qwen3": "qwen3-0.6b", "qwen3.5": "qwen3.5-0.8b", "gemma3": "gemma-3-1b"}


def config(name):
    path = Path(name) if str(name).endswith(".json") else ROOT / "configs/models" / (MODELS[name] + ".json")
    raw = json.loads(path.read_text())
    c = raw.get("text_config", raw)
    if c["model_type"] not in {"qwen3", "qwen3_5_text", "gemma3_text"}:
        raise ValueError("Unsupported lowering family")
    return c


@dataclass(frozen=True)
class Op:
    opcode: int
    m: int
    n: int
    k: int = 0
    ctrl: int = 0
    epi: int = 0
    hdr: int = 0
    group: int = 0
    lda: int = 0
    ldy: int = 0
    tag: str = ""

    @property
    def kind(self):
        if self.opcode == 4:
            return "quant" if self.ctrl == 4 else "softmax" if self.ctrl == 0 else "norm"
        return {2: "gemv", 3: "act", 6: "elt", 7: "rope", 8: "fdot_nn" if self.ctrl & 1 else "fdot_nt"}[self.opcode]

    @property
    def activation(self):
        return (self.epi & 3) | ((self.epi >> 5 & 1) << 2)

    def signature(self):
        return [self.opcode, self.ctrl, self.epi, self.m, self.n, self.k,
                self.hdr if self.opcode != 2 else 0, self.group, self.lda, self.ldy]


def epi(act):
    return (act & 3) | ((act >> 2 & 1) << 5)


def lower(c, t0=16, batch=1, head=True):
    """One ordered queue. Context must fit Gemma's implemented sliding window."""
    if t0 < 0 or not 1 <= batch <= 4:
        raise ValueError("t0 >= 0 and 1 <= batch <= 4 required")
    gemma = c["model_type"] == "gemma3_text"
    if gemma and t0 + batch > c["sliding_window"]:
        raise ValueError("openTARS Gemma lowering requires context <= sliding_window")
    H, I, nh, nkv, hd = (c[k] for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim"))
    rot = hd if gemma else int(hd * c.get("rope_parameters", {}).get("partial_rotary_factor", 1))
    types = c.get("layer_types", ["full_attention"] * c["num_hidden_layers"])
    if len(types) != c["num_hidden_layers"]:
        raise ValueError("layer_types length mismatch")
    ops = []
    tag = ""

    def emit(code, m, n, k=0, **kw):
        ops.append(Op(code, m, n, k, tag=tag, **kw))

    def row(m, n, fn=2, k=0):
        emit(4, m, n, k, ctrl=fn, epi=8 if fn == 4 else 0)

    def gem(n, k, act=0, bias=False, m=batch, lda=0):
        emit(2, m, n, k, ctrl=283 | (4096 if act else 0), epi=epi(act) | (16 if bias else 0), lda=lda)

    def elt(m, n, fn=0, hdr=0, act=0, add=False):
        emit(6, m, n, ctrl=fn, hdr=hdr, epi=epi(act) | (16 if add else 0))

    def act(m, n, a):
        emit(3, m, n, epi=epi(a))

    for layer, typ in enumerate(types):
        tag = f"layer{layer}/" + ("delta" if typ == "linear_attention" else "attention")
        row(batch, H)
        row(batch, H, 4)
        if typ == "linear_attention":
            nk, nv, dk, dv = (c[k] for k in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
            if nk != nv:
                raise ValueError("openTARS requires equal linear key/value heads")
            kd, vd = nk * dk, nv * dv
            for n in (kd, kd, vd, vd):
                gem(n, H)
            gem(nv, H, 5)
            gem(nv, H, 6, True)
            elt(batch, nv, hdr=1)
            act(batch, nv, 7)
            for d in (kd, kd, vd):
                elt(batch, d, hdr=1)
                for _ in range(c.get("linear_conv_kernel_dim", 4) - 1):
                    elt(batch, d, 2, hdr=1, add=True)
                act(batch, d, 4)
            row(batch * nk, dk, 3)
            row(batch * nk, dk, 3)
            for _ in range(batch * nv):
                elt(dk, dv, hdr=3)
                emit(8, 1, dv, dk, ctrl=1)
                elt(1, dv, 3)
                elt(1, dv, hdr=3)
                elt(dk, dv, 2, hdr=6, add=True)
                emit(8, 1, dv, dk, ctrl=1)
            row(batch * nv, dv)
            elt(batch, vd, act=4)
            row(batch, vd, 4)
            gem(H, vd)
        else:
            if typ not in {"full_attention", "sliding_attention"}:
                raise ValueError(f"Unsupported layer type {typ}")
            gem(nh * hd, H)
            gate = c.get("attn_output_gate", c["model_type"] == "qwen3_5_text")
            if gate:
                gem(nh * hd, H)
            gem(nkv * hd, H)
            gem(nkv * hd, H)
            row(batch * nh, hd)
            row(batch * nkv, hd)
            emit(7, batch * nh, hd, rot, group=nh)
            emit(7, batch * nkv, hd, rot, group=nkv)
            p = math.ceil((t0 + batch) / 16) * 16
            for _ in range(nh):
                emit(8, batch, p, hd, group=nkv * hd // 4 if nkv > 1 else 0,
                     lda=nh * hd // 4 if nh > 1 else 0)
                row(batch, p, 0, t0 + 1)
                emit(8, batch, hd, p, ctrl=1, group=nkv * hd // 4 if nkv > 1 else 0,
                     ldy=nh * hd // 4 if nh > 1 else 0)
            if gate:
                elt(batch, nh * hd, act=5)
            row(batch, nh * hd, 4)
            gem(H, nh * hd)
            if gemma:
                row(batch, H)
        elt(batch, H, 1)
        tag = f"layer{layer}/mlp"
        row(batch, H)
        row(batch, H, 4)
        gem(I, H, 3 if gemma else 4)
        gem(I, H)
        elt(batch, I)
        row(batch, I, 4)
        parts = math.ceil(I / 4096)
        if I % parts or I // parts % 16:
            raise ValueError("Cannot evenly partition down projection as openTARS")
        for part in range(parts):
            gem(H, I // parts, lda=I // 16 if parts > 1 else 0)
            if part:
                elt(batch, H, 1)
        if gemma:
            row(batch, H)
        elt(batch, H, 1)
    tag = "head"
    row(batch, H)
    if head:
        row(1, H, 4)
        blocks = next(b for b in range(1, c["vocab_size"] + 1)
                      if c["vocab_size"] % b == 0 and c["vocab_size"] // b <= 65535 and c["vocab_size"] // b % 16 == 0)
        for _ in range(blocks):
            gem(c["vocab_size"] // blocks, H, m=1)
    return ops


def workload(c, mode="decode", position=16, tokens=20, chunk=4, skip_head=False):
    if mode == "decode":
        return lower(c, position)
    if mode != "prefill" or tokens < 1 or chunk not in (1, 2, 4):
        raise ValueError("prefill requires positive tokens and chunk in (1,2,4)")
    result, t = [], 0
    while t < tokens:
        b = chunk if t % chunk == 0 and tokens - t >= chunk else 1
        result.extend(lower(c, t, b, head=not skip_head or t + b == tokens))
        t += b
    return result
