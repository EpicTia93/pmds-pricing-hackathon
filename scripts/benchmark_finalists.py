"""Evaluate submission bundles in the local simulator against reproducible rival fields.

Run from the repository root, for example
    .venv/bin/python scripts/benchmark_finalists.py --out outputs/finals --seeds 4 \
        --challengers src/dme_simulator/agents/demand-planner

Selected bundles can be compared together, or placed one at a time in an
identical seat via --substitute. The output JSONL records each individual game.
"""
import argparse
import io
import json
import logging
import multiprocessing
import random
import shutil
import sys
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AGENTS = ROOT / "src/dme_simulator/agents"

from dme_simulator.runner import run_simulation
from swarm.presets import preset
from swarm.scenarios import build_config
from swarm.tournament import bundle_entrant, make_agent, builtin_entrant
from generate_scenarios import hardware_scenario, synthetic_scenario


def markets(directory, count, seed):
    rng = np.random.default_rng(seed)
    base = yaml.safe_load((ROOT / "config/default.yaml").read_text())
    scaled = yaml.safe_load((ROOT / "config/scaled_market.yaml").read_text())
    paths = []
    for name, raw in (("default", base), ("extended", dict(base, turns=80)),
                      ("extended_scaled", dict(scaled, turns=80))):
        raw = {**raw, "initial_state": dict(raw["initial_state"])}
        if name != "default":
            raw["initial_state"]["starting_capital"] = 40000.
        path = directory / f"{name}.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        paths.append((name, path))
    for i in range(count):
        hardware = hardware_scenario(rng, (base, scaled)[i % 2],
                                     [["scarce"], ["glut"], ["squeeze","short"], ["fat","segments"],
                                      ["dead_good"], ["head_start","volatile"]][i % 6])
        synthetic, _ = synthetic_scenario(rng)
        for name, scenario in (("hardware", hardware), ("synthetic", synthetic)):
            path = directory / f"{name}_{i}.yaml"
            path.write_text(yaml.safe_dump(scenario, sort_keys=False))
            paths.append((name, path))
    for _, path in paths:
        build_config(path)
    return paths


def field(profile, field_size, candidates):
    roster = candidates[:]
    if profile != "team":
        roster.extend(builtin_entrant(n) for n in ["DummyAgent", "PatientAgent", "PriceWarAgent"])
    roster.extend(bundle_entrant(AGENTS / n) for n in
                  ["champion-generated-421", "swarm-robust-valuation", "swarm-robust", "swarm-islands"])
    if profile == "pressure":
        mix = ["whale"] * 5 + ["starter"] * 4 + ["passive"] * 2 + ["adaptive"] * 3
    elif profile == "team":
        mix = ["adaptive"] * 4 + ["starter"] * 2 + ["whale", "neutral"]
    else:
        mix = ["starter"] * 6 + ["adaptive"] * 2 + ["neutral", "passive"]
    names = ["premium", "volume", "aggressive_bidder", "bargain_hunter", "explorer", "undercutter"]
    for i in range(field_size - len(roster)):
        role = mix[i % len(mix)]
        if role == "adaptive":
            entry = {"kind":"param", "family":"adaptive", "params":preset("adaptive", names[i % len(names)])}
        elif role == "whale":
            level = (2., 3.5, 5., 1.5, 4.5)[i % 5]
            entry = {"kind":"param", "family":"adaptive", "params":{
                "bid_markup":level, "max_bid_markup":level + 1.5, "bid_adapt":.3,
                "fill_target":.9, "budget_share":.8, "init_spend_frac":.50,
                "init_price_mult":3., "stock_cover":1.5}}
        elif role == "passive":
            entry = builtin_entrant("PatientAgent")
        else:
            entry = bundle_entrant(AGENTS / ("neutral" if role == "neutral" else "test-agent"))
        roster.append(dict(entry, id=f"{role}-{i}"))
    return roster


def game(job):
    path, group, seed, profile, size, candidates, substitute = job
    cfg = build_config(path)
    entrants = [bundle_entrant(p, "test-seat" if substitute else p.name) for p in candidates]
    roster = field(profile, size, entrants)
    random.Random(seed).shuffle(roster)
    agents = {e["id"]:make_agent(e) for e in roster}
    warnings = io.StringIO()
    handler = logging.StreamHandler(warnings)
    logger = logging.getLogger("dme_simulator.runner")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        with redirect_stdout(io.StringIO()):
            sim = run_simulation(cfg, agents, seed_override=seed, quiet=False, profile=True)
    finally:
        logger.removeHandler(handler)
    cash = {team:s.capital for team,s in sim.teams_state.items()}
    rows = {}
    for path, entry in zip(candidates, entrants):
        team = entry["id"]
        rows[path.name] = {"rank":1 + sum(x > cash[team] for x in cash.values()),
            "cash":cash[team], "ms":round(max(sim.history.diagnostics.decision_duration_seconds[team]) * 1000, 3),
            "skipped":warnings.getvalue().count(f"Agent {team} "),
            "internal_fallbacks":getattr(agents[team], "errors", 0)}
    return {"scenario":Path(job[0]).name,"group":group,"seed":seed,"field":profile,"size":size,
            "start":cfg.initial_state.starting_capital,"turns":cfg.turns,"results":rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--challengers", nargs="+", required=True, type=Path)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--seed", type=int, default=715930)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--substitute", action="store_true")
    p.add_argument("--fields", nargs="+", choices=["mixed","pressure","team"], default=["mixed","pressure","team"])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    candidates = []
    for src in a.challengers:
        dest = a.out / "agents" / src.name
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        candidates.append(dest.resolve())
    directory = a.out / "scenarios"
    directory.mkdir()
    groups = markets(directory, a.seeds, a.seed)
    jobs = []
    for label, market in groups:
        for i in range(a.seeds) if label not in ("hardware","synthetic") else [int(market.stem.split("_")[-1])]:
            for j, profile in enumerate(a.fields):
                seed = a.seed + i * 1009 + j * 379
                size = {"mixed":30, "pressure":53, "team":25}[profile]
                for batch in [[c] for c in candidates] if a.substitute else [candidates]:
                    jobs.append((market, label, seed, profile, size, batch, a.substitute))
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        with (a.out / "matches.jsonl").open("w") as out:
            for result in pool.map(game, jobs):
                out.write(json.dumps(result) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
