import json
from pathlib import Path
import pytest
from dars.workload import config, lower, workload


@pytest.mark.parametrize("name", ["qwen3", "qwen3.5", "gemma3"])
def test_real_opentars_queues(name):
    fixture = json.loads((Path(__file__).parent / "fixtures/opentars_queues.json").read_text())
    for case, rows in fixture["models"][name].items():
        _, t, b = case.split(":")
        expected = [list(r) for r in rows]
        for r in expected:
            if r[0] == 2:
                r[6] = 0  # address of row scales, not a semantic field
        assert [o.signature() for o in lower(config(name), int(t), int(b))] == expected, case


def test_prefill_and_validation():
    c = config("qwen3.5")
    assert len(workload(c, "prefill", tokens=5)) == 7914 + 2730
    assert sum(o.tag == "head" and o.kind == "gemv" for o in workload(c, "prefill", tokens=20, skip_head=True)) == 4
    with pytest.raises(ValueError):
        lower(config("gemma3"), 512)
