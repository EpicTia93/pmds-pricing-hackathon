"""Evolutionary search over ``ParamAgent`` strategies, driven by a YAML experiment."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .presets import resolve as resolve_presets
from .scenarios import DEFAULT_PERTURB
from .space import ParamSpace
from .tournament import (
    REPO,
    Record,
    ScenarioPool,
    builtin_entrant,
    bundle_entrant,
    leaderboard,
    run_matches,
    schedule,
    score,
)

DEFAULT_EXPERIMENT: dict[str, Any] = {
    "name": "experiment",
    "seed": 0,
    "workers": None,
    "output_dir": "outputs/swarm",
    "scenarios": [
        {"path": "config/default.yaml", "weight": 1.0},
        {"path": "config/scaled_market.yaml", "weight": 1.0},
    ],
    "perturb": dict(DEFAULT_PERTURB),   # null disables procedural variants
    "turns": None,                      # override every scenario's duration
    "population": 32,
    "generations": 10,
    "match_size": 6,                    # population members per match
    "rounds_per_generation": 4,         # matches each member plays per generation
    "opponents": {
        "builtin": ["PriceWarAgent", "DummyAgent"],
        "bundles": [],                  # extra agent bundle folders
        "hall_of_fame": 2,              # past champions sampled into each match
        "hall_of_fame_size": 10,
    },
    "selection": {
        "elite": 6,
        "tournament_k": 3,
        "crossover_rate": 0.6,
        "mutation_rate": 0.25,
        "mutation_scale": 0.15,
        "immigrants": 2,                # fresh random members per generation
        "fitness_memory": True,         # survivors keep earlier results
        "min_matches_for_elite": 0,
    },
    "init": {
        "presets": "all",               # names, "all" or "none"
        "seed_from": None,              # population/champion JSON of a previous run
        "random": None,                 # fill remaining slots (default: rest)
    },
    "space": {},                        # {param: {low:, high:, log:, choices:}}
    "fixed": {},                        # {param: value} pinned for everyone
    "playoff": {"top": 8, "matches": 60, "include_hall_of_fame": True},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if key not in base:
            raise KeyError(f"unknown experiment key: {key}")
        if isinstance(value, dict) and isinstance(base[key], dict) and key not in ("space", "fixed"):
            out[key] = deep_merge(base[key], value) if base[key] else dict(value)
        else:
            out[key] = value
    return out


def load_experiment(path: str | Path | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    exp = dict(DEFAULT_EXPERIMENT)
    if path:
        with open(path, encoding="utf-8") as stream:
            exp = deep_merge(exp, yaml.safe_load(stream) or {})
    if overrides:
        exp = deep_merge(exp, overrides)
    return exp


def scenario_pool(exp: dict[str, Any]) -> ScenarioPool:
    scenarios = []
    for item in exp["scenarios"]:
        if isinstance(item, str):
            item = {"path": item}
        path = Path(item["path"])
        if not path.is_absolute():
            path = (REPO / path) if (REPO / path).exists() else path.resolve()
        scenarios.append((str(path), float(item.get("weight", 1.0))))
    perturb = exp["perturb"]
    if perturb:
        perturb = {**DEFAULT_PERTURB, **perturb}
    return ScenarioPool(scenarios, perturb or None, exp["turns"])


def opponent_entrants(exp: dict[str, Any]) -> list[dict[str, Any]]:
    opp = exp["opponents"]
    return [builtin_entrant(name) for name in opp.get("builtin") or []] + [
        bundle_entrant(path) for path in opp.get("bundles") or []
    ]


@dataclass
class Individual:
    id: str
    params: dict[str, Any]
    origin: str
    born: int
    record: Record = field(default_factory=Record)

    def entrant(self) -> dict[str, Any]:
        return {"id": self.id, "kind": "param", "params": self.params}

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id, "origin": self.origin, "born": self.born,
            "params": self.params, **self.record.summary(),
        }


def load_param_sets(path: str | Path) -> list[dict[str, Any]]:
    """Accept a champion file, a generation file, or a list of param dicts."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "params" in data:
        return [data["params"]]
    if isinstance(data, dict) and "population" in data:
        return [item["params"] for item in data["population"]]
    if isinstance(data, list):
        return [item.get("params", item) for item in data]
    raise ValueError(f"unrecognised parameter file: {path}")


class Evolution:
    def __init__(self, exp: dict[str, Any]) -> None:
        self.exp = exp
        self.rng = random.Random(exp["seed"])
        self.space = ParamSpace(exp["space"], exp["fixed"])
        self.pool = scenario_pool(exp)
        self.opponents = opponent_entrants(exp)
        self.hall: list[Individual] = []
        self.counter = 0
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.out = Path(exp["output_dir"]) / f"{exp['name']}-{stamp}"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))

    def _new(self, params: dict[str, Any], origin: str, gen: int) -> Individual:
        self.counter += 1
        return Individual(f"g{gen:03d}_{self.counter:04d}", self.space.clip(params), origin, gen)

    def initial_population(self) -> list[Individual]:
        size = self.exp["population"]
        init = self.exp["init"]
        pop = [self._new(p, f"preset:{name}", 0) for name, p in resolve_presets(init["presets"]).items()]
        if init.get("seed_from"):
            pop += [self._new(p, "seeded", 0) for p in load_param_sets(init["seed_from"])]
        pop = pop[:size]
        n_random = size - len(pop) if init.get("random") is None else int(init["random"])
        pop += [self._new(self.space.sample(self.rng), "random", 0) for _ in range(n_random)]
        return pop

    def evaluate(self, pop: list[Individual], gen: int) -> None:
        opp = self.exp["opponents"]
        hall = [ind.entrant() for ind in self.hall if ind.id not in {p.id for p in pop}]
        jobs = schedule(
            [ind.entrant() for ind in pop],
            self.exp["match_size"],
            self.exp["rounds_per_generation"],
            self.pool,
            self.rng,
            fixed_opponents=self.opponents,
            sampled_opponents=hall,
            sampled_count=opp.get("hall_of_fame", 0),
        )
        results = run_matches(jobs, self.exp["workers"], desc=f"gen {gen}")
        if not self.exp["selection"]["fitness_memory"]:
            for ind in pop:
                ind.record = Record()
        by_id = {ind.id: ind for ind in pop}
        fresh = score(results)
        for team, rec in fresh.items():
            if team in by_id:
                target = by_id[team].record
                target.ranks += rec.ranks
                target.norm_ranks += rec.norm_ranks
                target.returns += rec.returns
                target.wins += rec.wins
                target.bankrupt += rec.bankrupt
                for k, v in rec.by_scenario.items():
                    target.by_scenario[k].extend(v)
        self.last_opponent_records = {t: r for t, r in fresh.items() if t not in by_id}

    def _pick(self, ranked: list[Individual]) -> Individual:
        k = self.exp["selection"]["tournament_k"]
        return min(self.rng.sample(ranked, min(k, len(ranked))), key=lambda i: i.record.fitness)

    def next_generation(self, pop: list[Individual], gen: int) -> list[Individual]:
        sel = self.exp["selection"]
        ranked = sorted(pop, key=lambda i: i.record.fitness)
        eligible = [i for i in ranked if i.record.matches >= sel["min_matches_for_elite"]]
        nxt = eligible[: sel["elite"]]
        nxt += [self._new(self.space.sample(self.rng), "immigrant", gen) for _ in range(sel["immigrants"])]
        while len(nxt) < self.exp["population"]:
            a = self._pick(ranked)
            if self.rng.random() < sel["crossover_rate"]:
                b = self._pick(ranked)
                child = self.space.crossover(a.params, b.params, self.rng)
                origin = f"cross:{a.id}+{b.id}"
            else:
                child, origin = dict(a.params), f"mut:{a.id}"
            child = self.space.mutate(child, self.rng, sel["mutation_rate"], sel["mutation_scale"])
            nxt.append(self._new(child, origin, gen))
        return nxt

    def update_hall(self, pop: list[Individual]) -> None:
        best = min(pop, key=lambda i: i.record.fitness)
        if all(best.id != h.id for h in self.hall):
            self.hall.append(best)
        size = self.exp["opponents"].get("hall_of_fame_size", 10)
        self.hall = sorted(self.hall, key=lambda i: i.record.fitness)[:size]

    def save_generation(self, pop: list[Individual], gen: int) -> None:
        ranked = sorted(pop, key=lambda i: i.record.fitness)
        data = {
            "generation": gen,
            "population": [ind.to_json() for ind in ranked],
            "opponents": {t: r.summary() for t, r in self.last_opponent_records.items()},
            "hall_of_fame": [ind.to_json() for ind in self.hall],
        }
        (self.out / f"generation_{gen:03d}.json").write_text(json.dumps(data, indent=1))

    def run(self) -> Path:
        pop = self.initial_population()
        gens = self.exp["generations"]
        for gen in range(1, gens + 1):
            started = time.time()
            self.evaluate(pop, gen)
            self.update_hall(pop)
            self.save_generation(pop, gen)
            records = {ind.id: ind.record for ind in pop}
            records.update(self.last_opponent_records)
            print(f"\n=== generation {gen}/{gens}  ({time.time() - started:.1f}s) ===")
            print(leaderboard(records, limit=12))
            if gen < gens:
                pop = self.next_generation(pop, gen)
        return self.playoff(pop)

    def playoff(self, pop: list[Individual]) -> Path:
        cfg = self.exp["playoff"]
        finalists = sorted(pop, key=lambda i: i.record.fitness)[: cfg["top"]]
        if cfg.get("include_hall_of_fame"):
            ids = {f.id for f in finalists}
            finalists += [h for h in self.hall if h.id not in ids]
        print(f"\n=== playoff: {len(finalists)} finalists, {cfg['matches']} matches each ===")
        entrants = [f.entrant() for f in finalists]
        jobs = schedule(entrants, len(entrants), cfg["matches"], self.pool, self.rng,
                        fixed_opponents=self.opponents)
        records = score(run_matches(jobs, self.exp["workers"], desc="playoff"))
        print(leaderboard(records))
        by_id = {f.id: f for f in finalists}
        ranked = sorted((t for t in records if t in by_id), key=lambda t: records[t].fitness)
        result = [
            {"id": t, "origin": by_id[t].origin, "params": by_id[t].params, **records[t].summary()}
            for t in ranked
        ]
        (self.out / "playoff.json").write_text(json.dumps(result, indent=1))
        champion = self.out / "champion.json"
        champion.write_text(json.dumps(result[0], indent=1))
        print(f"\nChampion {result[0]['id']} saved to {champion}")
        return champion
