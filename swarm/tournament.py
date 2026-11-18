"""Run matches between entrants in parallel and score them the official way.

An entrant is a dict ``{"id": str, "kind": "param", "params": {...}}`` for a
``ParamAgent`` configuration, or ``{"id": str, "kind": "bundle", "path": str}``
for any agent bundle folder (built-ins, exported champions, other experiments).
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import random
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dme_simulator.agents import BaseAgent
from dme_simulator.agents.loader import load_agent_bundle_class
from dme_simulator.runner import run_simulation

from .scenarios import build_config
from .strategy import ParamAgent

REPO = Path(__file__).resolve().parent.parent
BUILTIN_DIR = REPO / "src" / "dme_simulator" / "agents" / "builtin_bundles"

_bundle_cache: dict[str, type] = {}


def builtin_entrant(name: str) -> dict[str, Any]:
    return {"id": name, "kind": "bundle", "path": str(BUILTIN_DIR / name)}


def bundle_entrant(path: str | Path, entrant_id: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"id": entrant_id or path.name, "kind": "bundle", "path": str(path)}


def make_agent(entrant: dict[str, Any]) -> BaseAgent:
    if entrant["kind"] == "param":
        return ParamAgent(**entrant["params"])
    path = entrant["path"]
    if path not in _bundle_cache:
        _bundle_cache[path] = load_agent_bundle_class(path)
    return _bundle_cache[path]()


# ---------------------------------------------------------------------------
# match execution
# ---------------------------------------------------------------------------

def run_match(job: dict[str, Any]) -> dict[str, Any]:
    """Play one simulation. ``job`` holds scenario, seed, turns and roster."""
    config = build_config(job["scenario"], job.get("perturb_seed"), job.get("perturb"))
    agents: dict[str, BaseAgent] = {
        entrant["id"]: make_agent(entrant) for entrant in job["roster"]
    }
    logging.disable(logging.CRITICAL)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        simulator = run_simulation(
            config,
            agents,
            seed_override=job["seed"],
            turns_override=job.get("turns"),
            quiet=True,
        )
    start = config.initial_state.starting_capital
    capital = {team: float(s.capital) for team, s in simulator.teams_state.items()}
    return {
        "scenario": Path(job["scenario"]).stem,
        "perturbed": job.get("perturb_seed") is not None,
        "seed": job["seed"],
        "turns": job.get("turns") or config.turns,
        "start_capital": start,
        "capital": capital,
        "bankrupt": {team: s.bankrupt for team, s in simulator.teams_state.items()},
        "ranks": rank(capital),
    }


def rank(capital: dict[str, float]) -> dict[str, float]:
    """Rank by capital, highest first; ties share their average rank."""
    ordered = sorted(capital.items(), key=lambda item: -item[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        shared = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = shared
        i = j + 1
    return ranks


def run_matches(
    jobs: list[dict[str, Any]], workers: int | None = None, desc: str = "matches"
) -> list[dict[str, Any]]:
    workers = workers or multiprocessing.cpu_count()
    if workers <= 1 or len(jobs) <= 1:
        return [run_match(job) for job in tqdm(jobs, desc=desc, leave=False)]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(
            tqdm(
                pool.map(run_match, jobs, chunksize=max(1, len(jobs) // (workers * 8))),
                total=len(jobs),
                desc=desc,
                leave=False,
            )
        )


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------

@dataclass
class ScenarioPool:
    """Weighted scenario list plus optional procedural perturbation."""

    scenarios: list[tuple[str, float]]
    perturb: dict[str, Any] | None = None
    turns: int | None = None
    stratified: bool = False   # deal scenarios like a shuffled deck (ignores weights)
    _deck: list[str] = field(default_factory=list)

    def _pick(self, rng: random.Random) -> str:
        paths = [path for path, _ in self.scenarios]
        if not self.stratified:
            return rng.choices(paths, [weight for _, weight in self.scenarios])[0]
        if not self._deck:
            self._deck = paths[:]
            rng.shuffle(self._deck)
        return self._deck.pop()

    def draw(self, rng: random.Random) -> dict[str, Any]:
        job: dict[str, Any] = {
            "scenario": self._pick(rng),
            "seed": rng.randrange(1_000_000_000),
            "turns": self.turns,
        }
        if self.perturb and rng.random() < self.perturb.get("probability", 1.0):
            job["perturb_seed"] = rng.randrange(1_000_000_000)
            job["perturb"] = self.perturb
        return job


def schedule(
    entrants: list[dict[str, Any]],
    match_size: int,
    rounds: int,
    pool: ScenarioPool,
    rng: random.Random,
    fixed_opponents: list[dict[str, Any]] | None = None,
    sampled_opponents: list[dict[str, Any]] | None = None,
    sampled_count: int = 0,
) -> list[dict[str, Any]]:
    """Each round shuffles entrants into groups of ``match_size``.

    Every match also contains all ``fixed_opponents`` and ``sampled_count``
    entries drawn from ``sampled_opponents`` (e.g. the hall of fame).
    """
    jobs = []
    size = max(1, min(match_size, len(entrants)))
    for _ in range(rounds):
        order = entrants[:]
        rng.shuffle(order)
        for start in range(0, len(order), size):
            group = order[start:start + size]
            if len(group) < size:  # top up the last group with other entrants
                others = [e for e in entrants if e not in group]
                group += rng.sample(others, min(size - len(group), len(others)))
            roster = list(group) + list(fixed_opponents or [])
            ids = {e["id"] for e in roster}
            extra = [e for e in (sampled_opponents or []) if e["id"] not in ids]
            roster += rng.sample(extra, min(sampled_count, len(extra)))
            job = pool.draw(rng)
            job["roster"] = roster
            jobs.append(job)
    return jobs


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class Record:
    ranks: list[float] = field(default_factory=list)
    norm_ranks: list[float] = field(default_factory=list)   # 0 = best, 1 = worst
    returns: list[float] = field(default_factory=list)      # final / start capital
    wins: int = 0
    bankrupt: int = 0
    by_scenario: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, team: str, result: dict[str, Any]) -> None:
        r = result["ranks"][team]
        n = len(result["ranks"])
        self.ranks.append(r)
        self.norm_ranks.append((r - 1) / (n - 1) if n > 1 else 0.0)
        self.returns.append(result["capital"][team] / result["start_capital"])
        self.wins += r == 1
        self.bankrupt += bool(result["bankrupt"][team])
        key = result["scenario"] + ("*" if result["perturbed"] else "")
        self.by_scenario[key].append(self.norm_ranks[-1])

    @property
    def matches(self) -> int:
        return len(self.ranks)

    @property
    def fitness(self) -> float:
        """Mean normalised rank: lower is better (0 = won every match)."""
        return statistics.fmean(self.norm_ranks) if self.norm_ranks else 1.0

    @property
    def stderr(self) -> float:
        if len(self.norm_ranks) < 2:
            return float("nan")
        return statistics.stdev(self.norm_ranks) / len(self.norm_ranks) ** 0.5

    def summary(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "fitness": self.fitness,
            "stderr": self.stderr,
            "mean_rank": statistics.fmean(self.ranks) if self.ranks else None,
            "mean_return": statistics.fmean(self.returns) if self.returns else None,
            "win_rate": self.wins / self.matches if self.matches else 0.0,
            "bankrupt": self.bankrupt,
            "by_scenario": {k: statistics.fmean(v) for k, v in sorted(self.by_scenario.items())},
        }


def score(results: list[dict[str, Any]], records: dict[str, Record] | None = None) -> dict[str, Record]:
    records = records if records is not None else {}
    for result in results:
        for team in result["ranks"]:
            records.setdefault(team, Record()).add(team, result)
    return records


def leaderboard(records: dict[str, Record], limit: int | None = None) -> str:
    rows = sorted(records.items(), key=lambda item: item[1].fitness)[:limit]
    width = max([len(team) for team, _ in rows] + [10])
    lines = [
        f"{'#':>3}  {'entrant':<{width}}  {'fitness':>7} {'±':>5}  {'rank':>5}  "
        f"{'return':>6}  {'win%':>5}  {'n':>4}  per-scenario"
    ]
    for i, (team, rec) in enumerate(rows, 1):
        s = rec.summary()
        items = s["by_scenario"].items()
        if len(items) > 6:
            worst = sorted(items, key=lambda kv: -kv[1])[:3]
            scen = f"{len(items)} scen., worst: " + " ".join(f"{k}={v:.2f}" for k, v in worst)
        else:
            scen = " ".join(f"{k}={v:.2f}" for k, v in items)
        lines.append(
            f"{i:>3}  {team:<{width}}  {s['fitness']:7.3f} {s['stderr']:5.3f}  "
            f"{s['mean_rank']:5.2f}  {s['mean_return']:6.2f}  {100 * s['win_rate']:5.1f}  "
            f"{s['matches']:>4}  {scen}"
        )
    return "\n".join(lines)
