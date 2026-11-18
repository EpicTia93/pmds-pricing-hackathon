"""Island-model evolution of several strategy families in parallel.

Each family (``swarm/families``) is an island with its own population, search
space and selection. Every generation, all islands are pooled into the same
matches ("mixed", co-evolution: families must beat each other) or kept apart
("separate"). All matches go through one parallel process pool. Selection,
crossover and mutation happen within each island, because parameter sets of
different families are not compatible. A global hall of fame and a final
cross-family playoff choose the champion.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .families import FAMILIES
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

SELECTION_DEFAULTS: dict[str, Any] = {
    "elite": 0.2,                   # survivors: count, or fraction of the island if < 1
    "tournament_k": 3,
    "crossover_rate": 0.6,
    "mutation_rate": 0.25,          # per-parameter mutation probability
    "mutation_scale": 0.15,         # gaussian sigma as a fraction of the range
    "immigrants": 0.08,             # fresh random members: count, or fraction if < 1
    "fitness_memory": True,         # survivors accumulate results across generations
    "min_matches_for_elite": 0,
}

ISLAND_DEFAULTS: dict[str, Any] = {
    "population": 24,
    "presets": "all",               # names, "all" or "none"
    "seed_from": None,              # JSON results file(s) of earlier runs
    "random": None,                 # random members (default: fill the island)
    "space": {},                    # {param: {low:, high:, log:, choices:}}
    "fixed": {},                    # {param: value}
    "selection": {},                # per-island overrides of `selection`
    # Seed with the best members of *other* families found in `seed_from`,
    # converted into this family's parameters: {family: how many}. Meant for the
    # engine families, which share one parameter vocabulary (e.g. seed tactex
    # with revman, auction and capacity champions so crossover combines them).
    "seed_families": {},
}

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
    "stratified": False,                # cycle through scenarios evenly
    "families": None,                   # {family: island settings}; null = adaptive only
    "mixing": "mixed",                  # mixed | separate
    "generations": 10,
    "match_size": 6,                    # members per match (plus opponents); int or list
    "rounds_per_generation": 4,         # matches each member plays per generation
    "opponents": {
        "builtin": ["PriceWarAgent", "DummyAgent"],
        "bundles": [],                  # extra agent bundle folders
        # Clones that fill matches up to a realistic field size, e.g.
        # [{bundle: src/dme_simulator/agents/test-agent, count: 10},
        #  {builtin: PatientAgent, count: 1}, {preset: "adaptive:volume", count: 2}]
        "crowd": [],
        "hall_of_fame": 2,              # past champions sampled into each match
        "hall_of_fame_size": 12,
    },
    "selection": dict(SELECTION_DEFAULTS),
    "playoff": {"per_family": 3, "matches": 60, "include_hall_of_fame": True,
                "match_size": None,  # None = all finalists in every match; int or list
                "top": None},  # legacy alias of per_family
    # Legacy single-family keys (used when `families` is null).
    "population": 32,
    "init": {"presets": "all", "seed_from": None, "random": None},
    "space": {},
    "fixed": {},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any], free: tuple[str, ...] = ()) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if key not in base and key not in free:
            raise KeyError(f"unknown experiment key: {key}")
        if isinstance(value, dict) and isinstance(base.get(key), dict) and base[key] \
                and key not in ("space", "fixed", "families"):
            out[key] = deep_merge(base[key], value)
        else:
            out[key] = value
    return out


def load_experiment(path: str | Path | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    exp = dict(DEFAULT_EXPERIMENT)
    if path:
        with open(path, encoding="utf-8") as stream:
            exp = deep_merge(exp, yaml.safe_load(stream) or {})
    if overrides:
        families = overrides.pop("families", None)
        exp = deep_merge(exp, overrides)
        if families:  # --set families.glut.population=40 merges into the file's islands
            merged = dict(exp.get("families") or {})
            for name, island in families.items():
                merged[name] = {**(merged.get(name) or {}), **(island or {})}
            exp["families"] = merged
    if exp["playoff"].get("top") is not None:   # legacy key
        exp["playoff"] = {**exp["playoff"], "per_family": exp["playoff"]["top"]}
    return exp


def islands(exp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve per-family island settings (legacy experiments become one island)."""
    if not exp.get("families"):
        init = exp.get("init") or {}
        raw = {"adaptive": {
            "population": exp["population"], "presets": init.get("presets", "all"),
            "seed_from": init.get("seed_from"), "random": init.get("random"),
            "space": exp.get("space") or {}, "fixed": exp.get("fixed") or {},
        }}
    else:
        raw = exp["families"]
    out = {}
    for name, settings in raw.items():
        if name not in FAMILIES:
            raise KeyError(f"unknown family {name!r}; available: {', '.join(FAMILIES)}")
        island = deep_merge(ISLAND_DEFAULTS, settings or {})
        island["selection"] = deep_merge(SELECTION_DEFAULTS, {**exp["selection"], **island["selection"]})
        out[name] = island
    return out


def scenario_pool(exp: dict[str, Any]) -> ScenarioPool:
    scenarios = []
    for item in exp["scenarios"]:
        if isinstance(item, str):
            item = {"path": item}
        pattern = str(item["path"])
        base = REPO if not Path(pattern).is_absolute() else Path("/")
        if Path(base / pattern).is_dir():
            pattern = str(Path(pattern) / "*.yaml")
        if any(ch in pattern for ch in "*?["):
            paths = sorted(base.glob(pattern) if base == REPO else Path("/").glob(pattern.lstrip("/")))
            if not paths:
                raise FileNotFoundError(f"no scenarios match {pattern}")
        else:
            paths = [base / pattern if (base / pattern).exists() else Path(pattern).resolve()]
        for path in paths:
            scenarios.append((str(path.resolve()), float(item.get("weight", 1.0))))
    perturb = exp["perturb"]
    if perturb:
        perturb = {**DEFAULT_PERTURB, **perturb}
    return ScenarioPool(scenarios, perturb or None, exp["turns"], exp["stratified"])


def opponent_entrants(exp: dict[str, Any]) -> list[dict[str, Any]]:
    """Fixed opponents of every match: built-ins, bundles and ``crowd`` clones.

    A crowd entry names one ``builtin``, ``bundle``, ``preset`` (``family:name``)
    or ``family`` + ``params``, and a ``count``; each clone gets the id
    ``<label>#<n>``. With ``probability`` < 1 the whole group joins only that
    share of the matches, so field harshness varies (e.g. aggressive bidders
    that drain lots in half the matches). The official tournament puts every
    submitted team in one market (about 30 on the portal), so a crowd of weak
    and average clones keeps evolution in that regime.
    """
    from .presets import preset as preset_params

    opp = exp["opponents"]
    out = [builtin_entrant(name) for name in opp.get("builtin") or []] + [
        bundle_entrant(path) for path in opp.get("bundles") or []
    ]
    for item in opp.get("crowd") or []:
        if "builtin" in item:
            base = builtin_entrant(item["builtin"])
        elif "bundle" in item:
            base = bundle_entrant(item["bundle"])
        elif "preset" in item:
            fam, _, name = str(item["preset"]).partition(":")
            base = {"id": item["preset"], "kind": "param", "family": fam,
                    "params": preset_params(fam, name)}
        elif "family" in item:
            fam = item["family"]
            base = {"id": item.get("id") or fam, "kind": "param", "family": fam,
                    "params": FAMILIES[fam].complete(item.get("params") or {})}
        else:
            raise KeyError(f"crowd entry needs builtin, bundle, preset or family: {item}")
        label = item.get("id") or base["id"]
        extra = {}
        if float(item.get("probability", 1.0)) < 1.0:
            extra = {"group": label, "p": float(item["probability"])}
        out += [{**base, **extra, "id": f"{label}#{i + 1}"} for i in range(int(item.get("count", 1)))]
    return out


def convert_params(source: str, target: str, params: dict[str, Any]) -> dict[str, Any]:
    """Map a member of one engine family onto another family's parameters.

    The source's pinned modules (``FIXED``) become explicit values, and keys the
    target does not evolve are dropped (the target's own defaults fill the rest).
    """
    full = {**getattr(FAMILIES[source].agent_class, "FIXED", {}), **FAMILIES[source].complete(params)}
    return {k: v for k, v in full.items() if k in FAMILIES[target].defaults}


def load_param_sets(path: str | Path, family: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Read (family, params) pairs from a champion, generation, playoff or ranking file.

    Entries without a ``family`` field are ``adaptive`` (older runs). With
    ``family`` given, only that family's entries are returned.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "params" in data:
        items = [data]
    elif isinstance(data, dict) and "population" in data:
        items = data["population"]
    elif isinstance(data, dict) and "islands" in data:
        items = [m for members in data["islands"].values() for m in members]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"unrecognised parameter file: {path}")
    out = []
    for item in items:
        params = item.get("params", item)
        fam = item.get("family", "adaptive")
        if family is None or fam == family:
            out.append((fam, params))
    return out


@dataclass
class Individual:
    id: str
    family: str
    params: dict[str, Any]
    origin: str
    born: int
    record: Record = field(default_factory=Record)

    def entrant(self) -> dict[str, Any]:
        return {"id": self.id, "kind": "param", "family": self.family, "params": self.params}

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id, "family": self.family, "origin": self.origin, "born": self.born,
            "params": self.params, **self.record.summary(),
        }


def _count(value: float, size: int) -> int:
    return int(round(value * size)) if 0 < value < 1 else int(value)


class Evolution:
    def __init__(self, exp: dict[str, Any]) -> None:
        self.exp = exp
        self.rng = random.Random(exp["seed"])
        self.islands = islands(exp)
        self.spaces = {
            name: ParamSpace(name, island["space"], island["fixed"])
            for name, island in self.islands.items()
        }
        self.pool = scenario_pool(exp)
        self.opponents = opponent_entrants(exp)
        self.hall: list[Individual] = []
        self.counter = 0
        self.last_opponent_records: dict[str, Record] = {}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.out = Path(exp["output_dir"]) / f"{exp['name']}-{stamp}"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))

    # ------------------------------------------------------------------
    def _new(self, family: str, params: dict[str, Any], origin: str, gen: int) -> Individual:
        self.counter += 1
        params = self.spaces[family].clip(FAMILIES[family].complete(params))
        return Individual(f"{family}-g{gen:03d}-{self.counter:04d}", family, params, origin, gen)

    def initial_population(self) -> dict[str, list[Individual]]:
        pops = {}
        for name, island in self.islands.items():
            pop = [self._new(name, p, f"preset:{n}", 0)
                   for n, p in resolve_presets(name, island["presets"]).items()]
            seeds = island["seed_from"] or []
            seeds = [seeds] if isinstance(seeds, str) else seeds
            wanted = dict(island.get("seed_families") or {})
            for path in seeds:  # converted champions of other families come first
                for fam, p in load_param_sets(path):
                    if fam != name and wanted.get(fam, 0) > 0:
                        wanted[fam] -= 1
                        pop.append(self._new(name, convert_params(fam, name, p),
                                             f"converted:{fam}:{Path(path).name}", 0))
            for path in seeds:
                pop += [self._new(name, p, f"seeded:{Path(path).name}", 0)
                        for _, p in load_param_sets(path, name)]
            size = island["population"]
            pop = pop[:size]
            n_random = size - len(pop) if island["random"] is None else int(island["random"])
            pop += [self._new(name, self.spaces[name].sample(self.rng), "random", 0) for _ in range(n_random)]
            pops[name] = pop
        return pops

    def _jobs(self, members: list[Individual], hall: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return schedule(
            [ind.entrant() for ind in members],
            self.exp["match_size"],
            self.exp["rounds_per_generation"],
            self.pool,
            self.rng,
            fixed_opponents=self.opponents,
            sampled_opponents=hall,
            sampled_count=self.exp["opponents"].get("hall_of_fame", 0),
        )

    def evaluate(self, pops: dict[str, list[Individual]], gen: int) -> None:
        members = [ind for pop in pops.values() for ind in pop]
        ids = {ind.id for ind in members}
        hall = [ind.entrant() for ind in self.hall if ind.id not in ids]
        if self.exp["mixing"] == "separate":
            jobs = [job for pop in pops.values() for job in self._jobs(pop, hall)]
        else:
            jobs = self._jobs(members, hall)
        results = run_matches(jobs, self.exp["workers"], desc=f"gen {gen}")
        by_id = {ind.id: ind for ind in members}
        for ind in members:
            if not self.islands[ind.family]["selection"]["fitness_memory"]:
                ind.record = Record()
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

    # ------------------------------------------------------------------
    def _pick(self, ranked: list[Individual], k: int) -> Individual:
        return min(self.rng.sample(ranked, min(k, len(ranked))), key=lambda i: i.record.fitness)

    def next_island(self, family: str, pop: list[Individual], gen: int) -> list[Individual]:
        sel = self.islands[family]["selection"]
        size = self.islands[family]["population"]
        space = self.spaces[family]
        ranked = sorted(pop, key=lambda i: i.record.fitness)
        eligible = [i for i in ranked if i.record.matches >= sel["min_matches_for_elite"]]
        nxt = eligible[: _count(sel["elite"], size)]
        nxt += [self._new(family, space.sample(self.rng), "immigrant", gen)
                for _ in range(_count(sel["immigrants"], size))]
        while len(nxt) < size:
            a = self._pick(ranked, sel["tournament_k"])
            if self.rng.random() < sel["crossover_rate"]:
                b = self._pick(ranked, sel["tournament_k"])
                child, origin = space.crossover(a.params, b.params, self.rng), f"cross:{a.id}+{b.id}"
            else:
                child, origin = dict(a.params), f"mut:{a.id}"
            child = space.mutate(child, self.rng, sel["mutation_rate"], sel["mutation_scale"])
            nxt.append(self._new(family, child, origin, gen))
        return nxt[:size]

    def update_hall(self, pops: dict[str, list[Individual]]) -> None:
        """Keep the best of every island, then trim to the best overall."""
        known = {h.id for h in self.hall}
        for pop in pops.values():
            best = min(pop, key=lambda i: i.record.fitness)
            if best.id not in known:
                self.hall.append(best)
        size = self.exp["opponents"].get("hall_of_fame_size", 12)
        self.hall = sorted(self.hall, key=lambda i: i.record.fitness)[:size]

    def save_generation(self, pops: dict[str, list[Individual]], gen: int) -> None:
        data = {
            "generation": gen,
            "islands": {
                name: [ind.to_json() for ind in sorted(pop, key=lambda i: i.record.fitness)]
                for name, pop in pops.items()
            },
            "opponents": {t: r.summary() for t, r in self.last_opponent_records.items()},
            "hall_of_fame": [ind.to_json() for ind in self.hall],
        }
        (self.out / f"generation_{gen:03d}.json").write_text(json.dumps(data, indent=1))

    def report(self, pops: dict[str, list[Individual]], gen: int, seconds: float) -> None:
        gens = self.exp["generations"]
        print(f"\n=== generation {gen}/{gens}  ({seconds:.1f}s) ===")
        print(f"{'family':<10} {'best':>6} {'median':>7} {'mean rank':>9}  best id")
        for name, pop in pops.items():
            fits = [i.record.fitness for i in pop]
            best = min(pop, key=lambda i: i.record.fitness)
            print(f"{name:<10} {min(fits):6.3f} {statistics.median(fits):7.3f} "
                  f"{statistics.fmean(best.record.ranks) if best.record.ranks else float('nan'):9.2f}  {best.id}")
        grouped: dict[str, Record] = {}
        for team, rec in self.last_opponent_records.items():
            if any(team == h.id for h in self.hall):
                continue
            label = team.split("#", 1)[0] + ("*" if "#" in team else "")
            agg = grouped.setdefault(label, Record())
            agg.ranks += rec.ranks
            agg.norm_ranks += rec.norm_ranks
        for label, rec in sorted(grouped.items(), key=lambda kv: kv[1].fitness):
            print(f"{'(opp)':<10} {rec.fitness:6.3f} {'':>7} {statistics.fmean(rec.ranks):9.2f}  {label}")

    # ------------------------------------------------------------------
    def run(self) -> Path:
        pops = self.initial_population()
        print("islands: " + ", ".join(f"{k}={len(v)}" for k, v in pops.items())
              + f" | mixing={self.exp['mixing']} | scenarios={len(self.pool.scenarios)}")
        gens = self.exp["generations"]
        for gen in range(1, gens + 1):
            started = time.time()
            self.evaluate(pops, gen)
            self.update_hall(pops)
            self.save_generation(pops, gen)
            self.report(pops, gen, time.time() - started)
            if gen < gens:
                pops = {name: self.next_island(name, pop, gen) for name, pop in pops.items()}
        return self.playoff(pops)

    def playoff(self, pops: dict[str, list[Individual]]) -> Path:
        cfg = self.exp["playoff"]
        finalists: list[Individual] = []
        for pop in pops.values():
            finalists += sorted(pop, key=lambda i: i.record.fitness)[: cfg["per_family"]]
        if cfg.get("include_hall_of_fame"):
            ids = {f.id for f in finalists}
            finalists += [h for h in self.hall if h.id not in ids]
        print(f"\n=== playoff: {len(finalists)} finalists from {len(pops)} families, "
              f"{cfg['matches']} rounds, match size {cfg.get('match_size') or 'all'} ===")
        entrants = [f.entrant() for f in finalists]
        jobs = schedule(entrants, cfg.get("match_size"), cfg["matches"], self.pool, self.rng,
                        fixed_opponents=self.opponents)
        records = score(run_matches(jobs, self.exp["workers"], desc="playoff"))
        print(leaderboard(records))
        by_id = {f.id: f for f in finalists}
        ranked = sorted((t for t in records if t in by_id), key=lambda t: records[t].fitness)
        result = [
            {"id": t, "family": by_id[t].family, "origin": by_id[t].origin,
             "params": by_id[t].params, **records[t].summary()}
            for t in ranked
        ]
        (self.out / "playoff.json").write_text(json.dumps(result, indent=1))
        best_of: dict[str, dict[str, Any]] = {}
        for item in result:
            best_of.setdefault(item["family"], item)
        (self.out / "family_champions.json").write_text(json.dumps(list(best_of.values()), indent=1))
        print("\nbest per family: " + ", ".join(
            f"{fam}={item['fitness']:.3f} ({item['id']})" for fam, item in best_of.items()))
        champion = self.out / "champion.json"
        champion.write_text(json.dumps(result[0], indent=1))
        print(f"Champion {result[0]['id']} saved to {champion}")
        return champion
