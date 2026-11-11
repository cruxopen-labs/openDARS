from dataclasses import replace
import numpy as np
import pytest
from dars.delta import block_delta, block_schedule, workspace_kib
from dars.workload import config, lower


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("dk,dv", [(8, 12), (128, 128)])
def test_block_matches_independent_token_recurrence(batch, dk, dv):
    rng = np.random.default_rng(42)
    q, k = rng.normal(size=(batch, dk)), rng.normal(size=(batch, dk))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    v = rng.normal(size=(batch, dv))
    beta = rng.uniform(0, 1, batch)
    decay = rng.uniform(.1, 1, batch)
    state = rng.normal(size=(dk, dv))
    expected = []
    s = state.copy()
    # Direct equations from the six openTARS per-token state-update operations.
    for t in range(batch):
        s = decay[t] * s
        delta = beta[t] * (v[t] - k[t] @ s)
        s = s + np.outer(k[t], delta)
        expected.append(q[t] @ s)
    actual, final = block_delta(q, k, v, beta, decay, state)
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(final, s, atol=1e-11, rtol=1e-11)


def test_block_schedule_capacity_and_queue_boundaries():
    c = config("qwen3.5")
    ops = lower(c, 0, 4)
    assert 64 < workspace_kib(c, 4) < 80
    assert block_schedule(ops, c, 64) == ops
    blocked = block_schedule(ops, c, 80)
    assert len(blocked) < len(ops)
    assert sum(o.tag.endswith("_block") for o in blocked) == 18 * 16 * 18
    single = lower(c, 0, 1)
    assert block_schedule(single, c, 80) == single
    fa = lower(config("qwen3"), 0, 4)
    assert block_schedule(fa, config("qwen3"), 80) == fa
