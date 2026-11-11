import json
from itertools import product
import pytest
from dars.search import Space, Evaluator, dominates, fronts, crowding, grid, nsga2, pareto
from dars.performance import Performance


def test_non_dominated_sort_independent_oracle():
    points = [(1, 3, 2), (2, 2, 2), (3, 1, 2), (3, 3, 3), (1, 3, 2)]
    expected = []
    for i, a in enumerate(points):
        if not any(all(b[k] <= a[k] for k in range(3)) and any(b[k] < a[k] for k in range(3))
                   for j, b in enumerate(points) if i != j):
            expected.append(i)
    assert fronts(points)[0] == expected == [0, 1, 2, 4]
    assert crowding([0], points)[0] == float("inf")
    assert not dominates(points[0], points[4])


def test_grid_nsga_reproducible_and_pruning(tmp_path):
    path = tmp_path / "space.json"
    path.write_text(json.dumps(dict(hardware=dict(vpu_lanes=[4, 8], mxu=[[0, 0], [16, 16]]),
                                    mappings=[dict(chunk=1), dict(chunk=4, skip_head=True)])))
    space = Space(path)
    e = Evaluator(Performance().fit(), space, tokens=5)
    rows = grid(e)
    assert len(rows) == 8
    assert sum(r["feasible"] for r in rows) == 4
    assert all(not r["latencies_ms"] for r in rows if not r["feasible"])
    assert all(len(r["latencies_ms"]) == 6 for r in rows if r["feasible"])
    front = pareto(rows)
    assert front
    assert not any(dominates(tuple(a[k] for k in ("latency_ratio", "area", "power_w")),
                             tuple(b[k] for k in ("latency_ratio", "area", "power_w")))
                   for a in rows if a["feasible"] for b in front)
    a = nsga2(e, population=12, generations=4, seed=5)
    b = nsga2(e, population=12, generations=4, seed=5)
    assert a == b
    assert {tuple(r["genes"]) for r in a} <= {tuple(r["genes"]) for r in rows}
    assert pareto(a)
    with pytest.raises(ValueError):
        nsga2(e, population=1)
