"""Block DeltaNet mathematical prototype and conservative analytical schedule.

This is not an openTARS executable queue: triangular solve and causal masking
need new lowering/kernel support. Float64 prototype validates the algebra only.
"""
import numpy as np
from .workload import Op


def block_delta(q, k, v, beta, decay, state):
    """One head, B tokens; returns all outputs and final dk x dv state.

    L d = beta * (v - G K S0), with
    L[t,i] = beta[t] * (G[t]/G[i]) * dot(k[t], k[i]) for i<t.
    """
    q, k, v, beta, decay, state = (np.asarray(x, dtype=np.float64)
                                  for x in (q, k, v, beta, decay, state))
    b = len(q)
    if q.shape != k.shape or v.shape[0] != b or beta.shape != (b,) or decay.shape != (b,):
        raise ValueError("incompatible block shapes")
    if state.shape != (k.shape[1], v.shape[1]) or np.any(decay <= 0):
        raise ValueError("state shape or positive decay required")
    g = np.cumprod(decay)
    if np.any(g == 0):
        raise ValueError("decay underflow: use shorter blocks")
    ratios = g[:, None] / g[None, :]
    gram = k @ k.T
    lower = np.eye(b) + np.tril(beta[:, None] * ratios * gram, -1)
    rhs = beta[:, None] * (v - g[:, None] * (k @ state))
    delta = np.linalg.solve(lower, rhs)
    outputs = g[:, None] * (q @ state) + np.tril(ratios * (q @ k.T)) @ delta
    final = g[-1] * state + k.T @ ((g[-1] / g)[:, None] * delta)
    return outputs, final


def workspace_kib(c, batch):
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    return (dk * dv * 4 + batch * (3 * dk + 3 * dv) * 4 + 3 * batch * batch * 4) / 1024


def block_schedule(ops, c, hold_kib):
    """Replace each contiguous token/head recurrence with per-head block work.

    Costs for triangular solve use an FDOT-equivalent B²dv service proxy.
    The baseline (non-block) queue remains bit-for-bit unchanged.
    """
    if not c or not c.get("linear_num_value_heads"):
        return ops
    dk, dv, heads = (c[k] for k in ("linear_key_head_dim", "linear_value_head_dim", "linear_num_value_heads"))
    result, i = [], 0

    def start(op):
        return op.tag.endswith("/delta") and op.opcode == 6 and op.m == dk and op.n == dv and op.hdr == 3

    while i < len(ops):
        if not start(ops[i]):
            result.append(ops[i])
            i += 1
            continue
        end = i
        while end + 5 < len(ops) and start(ops[end]):
            if [o.kind for o in ops[end:end + 6]] != ["elt", "fdot_nn", "elt", "elt", "elt", "fdot_nn"]:
                raise ValueError("unexpected DeltaNet recurrence layout")
            end += 6
        batch = (end - i) // (6 * heads)
        if batch <= 1 or hold_kib < workspace_kib(c, batch):
            result.extend(ops[i:end])
            i = end
            continue
        tag = ops[i].tag + "_block"
        for _ in range(heads):
            def dot(m, n, k, nn=True):
                result.append(Op(8, m, n, k, ctrl=int(nn), tag=tag))
            def elt(m, n, fn=0):
                result.append(Op(6, m, n, ctrl=fn, tag=tag))
            # Gram, decay/beta weighting, masked unit-lower construction.
            dot(batch, batch, dk, False)
            for _ in range(3):
                elt(batch, batch)
            dot(batch, dv, dk)  # K @ S0
            for fn in (0, 3, 0):
                elt(batch, dv, fn)
            dot(batch, dv, batch)  # triangular solve cost proxy
            dot(batch, dv, dk)     # Q @ S0
            dot(batch, batch, dk, False)
            elt(batch, batch)
            dot(batch, dv, batch)
            elt(batch, dv, 1)
            elt(batch, dv)          # weighted deltas for state update
            dot(dk, dv, batch)      # K.T @ weighted deltas
            elt(dk, dv)             # decay S0
            elt(dk, dv, 1)
        i = end
    return result
