"""Discrete exhaustive search and seeded NSGA-II with elitism/crowding."""
from dataclasses import asdict, replace
from functools import lru_cache
from itertools import product
from pathlib import Path
import json
import random
import numpy as np
from .hardware import BASE, ppa
from .performance import Mapping
from .workload import ROOT, MODELS, config, workload

OBJECTIVES = ("latency_ratio", "area", "power_w")


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fronts(values):
    """Non-dominated sorting, including equal-objective points."""
    remaining = set(range(len(values)))
    result = []
    while remaining:
        front = [i for i in sorted(remaining)
                 if not any(dominates(values[j], values[i]) for j in remaining if j != i)]
        result.append(front)
        remaining.difference_update(front)
    return result


def crowding(front, values):
    d = {i: 0. for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    for k in range(len(values[0])):
        order = sorted(front, key=lambda i: values[i][k])
        lo, hi = values[order[0]][k], values[order[-1]][k]
        if hi == lo:
            continue
        d[order[0]] = d[order[-1]] = float("inf")
        for j in range(1, len(order) - 1):
            d[order[j]] += (values[order[j + 1]][k] - values[order[j - 1]][k]) / (hi - lo)
    return d


def pareto(rows):
    feasible = [r for r in rows if r["feasible"]]
    if not feasible:
        return []
    return [feasible[i] for i in fronts([tuple(r[k] for k in OBJECTIVES) for r in feasible])[0]]


class Space:
    def __init__(self, path=None):
        spec = json.loads((Path(path) if path else ROOT / "configs/search.json").read_text())
        self.names = list(spec["hardware"])
        self.domains = list(spec["hardware"].values()) + [spec["mappings"]]
        if not all(self.domains):
            raise ValueError("Search domains cannot be empty")
        self.size = int(np.prod([len(d) for d in self.domains]))

    def decode(self, genes):
        hw = {}
        for name, domain, i in zip(self.names, self.domains, genes):
            if name == "mxu":
                hw["mxu_rows"], hw["mxu_cols"] = domain[i]
            else:
                hw[name] = domain[i]
        return replace(BASE, **hw), Mapping(**self.domains[-1][genes[-1]])


class Evaluator:
    def __init__(self, perf, space, target="fpga", position=250, tokens=20):
        self.perf, self.space, self.target = perf, space, target
        self.position, self.tokens = position, tokens
        self.configs = {m: config(m) for m in MODELS}
        self.ops = {}
        self.reference = {}
        for model, c in self.configs.items():
            for mode in ("decode", "prefill"):
                ops = workload(c, mode, position, tokens)
                self.reference[model, mode] = perf.predict(ops)["latency_ms"]

    def get_ops(self, model, mode, mapping):
        key = (model, mode, mapping.chunk, mapping.skip_head)
        if key not in self.ops:
            self.ops[key] = workload(self.configs[model], mode, self.position, self.tokens,
                                     mapping.chunk, mapping.skip_head)
        return self.ops[key]

    @lru_cache(maxsize=None)
    def __call__(self, genes):
        h, mapping = self.space.decode(genes)
        area = ppa(h, self.target)
        row = dict(genes=list(genes), hardware=asdict(h), mapping=asdict(mapping),
                   feasible=area["feasible"], area=area["area"], power_w=area["power_w"],
                   ppa=area)
        if not row["feasible"]:
            row["latency_ratio"] = 1e6
            row["latencies_ms"] = {}
            return row
        ratios, latencies = [], {}
        for model, c in self.configs.items():
            for mode in ("decode", "prefill"):
                # Chunk only changes prefill recurrence residency.
                mp = mapping if mode == "prefill" else replace(mapping, chunk=1)
                pred = self.perf.predict(self.get_ops(model, mode, mapping), h, mp, c)
                latencies[f"{model}/{mode}"] = pred["latency_ms"]
                ratios.append(pred["latency_ms"] / self.reference[model, mode])
        row["latency_ratio"] = float(np.mean(ratios))
        row["latencies_ms"] = latencies
        row["energy_mj_mean"] = float(np.mean(list(latencies.values()))) * row["power_w"]
        return row


def grid(evaluator):
    return [evaluator(tuple(g)) for g in product(*(range(len(d)) for d in evaluator.space.domains))]


def nsga2(evaluator, population=32, generations=12, seed=7):
    if population < 4 or generations < 1:
        raise ValueError("population >= 4 and generations >= 1 required")
    rng = random.Random(seed)
    lengths = [len(d) for d in evaluator.space.domains]
    random_gene = lambda: tuple(rng.randrange(n) for n in lengths)
    pop = [random_gene() for _ in range(population)]
    seen = set(pop)

    def objectives(g):
        r = evaluator(g)
        if not r["feasible"]:
            # Constraint domination: every feasible individual wins.
            violation = max(0, r["area"] - 1)
            return (1e6 + violation,) * 3
        return tuple(r[k] for k in OBJECTIVES)

    for _ in range(generations):
        values = [objectives(g) for g in pop]
        ranks, distances = {}, {}
        for rank, f in enumerate(fronts(values)):
            ranks.update({i: rank for i in f})
            distances.update(crowding(f, values))

        def tournament():
            a, b = rng.sample(range(len(pop)), 2)
            return pop[min((a, b), key=lambda i: (ranks[i], -distances[i]))]

        children = []
        for _ in range(population):
            a, b = tournament(), tournament()
            child = tuple((rng.randrange(n) if rng.random() < 1 / len(lengths)
                           else a[j] if rng.random() < 0.5 else b[j]) for j, n in enumerate(lengths))
            children.append(child)
        seen.update(children)
        combined = pop + children
        values = [objectives(g) for g in combined]
        selected = []
        for f in fronts(values):
            dist = crowding(f, values)
            selected.extend(sorted(f, key=lambda i: -dist[i])[:population - len(selected)])
            if len(selected) == population:
                break
        pop = [combined[i] for i in selected]
    return [evaluator(g) for g in sorted(seen)]
