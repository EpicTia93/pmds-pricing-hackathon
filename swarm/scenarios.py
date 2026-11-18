"""Load scenarios and generate randomised variants of them.

Official scenarios are private, so strategies are trained on procedurally
perturbed copies of the supplied markets to avoid overfitting to one config.
"""

from __future__ import annotations

import copy
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from dme_simulator.config import EnvironmentConfig, _from_mapping

# Default jitter strengths (multiplicative, factor drawn from [1-s, 1+s]).
DEFAULT_PERTURB: dict[str, Any] = {
    "probability": 0.7,   # share of matches played on a perturbed scenario
    "capital": 0.4,
    "supply": 0.4,        # supplier production ranges and capacities
    "material_price": 0.4,
    "customers": 0.5,
    "wtp": 0.35,          # willingness-to-pay ranges, drawn per segment & good
    "shares": 0.4,
    "beta": 0.6,
    "turns": [20, 50],    # inclusive range, or null to keep the scenario's
    "observe_all": 0.3,   # probability the full transaction log is visible
    "capacity_prob": 0.2, # probability of a production cap per team and turn
    "capacity_range": [10, 80],
}


@lru_cache(maxsize=64)
def _raw(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_config(
    path: str | Path,
    perturb_seed: int | None = None,
    perturb: dict[str, Any] | None = None,
) -> EnvironmentConfig:
    """Return a validated config, optionally randomly perturbed by ``perturb_seed``."""
    raw = copy.deepcopy(_raw(str(Path(path).resolve())))
    if perturb_seed is not None:
        settings = dict(DEFAULT_PERTURB)
        settings.update(perturb or {})
        _perturb(raw, random.Random(perturb_seed), settings)
    raw.setdefault("output", {})
    raw["output"]["render_plots"] = False
    raw["output"]["dashboard"] = "none"
    raw.setdefault("agents", {})["include"] = []
    config = _from_mapping(EnvironmentConfig, raw)
    config.validate()
    return config


def _jitter(rng: random.Random, strength: float) -> float:
    return rng.uniform(max(0.05, 1.0 - strength), 1.0 + strength)


def _perturb(raw: dict[str, Any], rng: random.Random, s: dict[str, Any]) -> None:
    turns = s.get("turns")
    if turns:
        raw["turns"] = rng.randint(int(turns[0]), int(turns[1]))

    state = raw.setdefault("initial_state", {})
    if "starting_capital" in state:
        state["starting_capital"] = float(state["starting_capital"]) * _jitter(rng, s["capital"])

    for supplier in raw.get("suppliers", {}).values():
        f = _jitter(rng, s["supply"])
        lo = max(1, round(supplier["production_min"] * f))
        hi = max(lo, round(supplier["production_max"] * f))
        supplier["production_min"], supplier["production_max"] = lo, hi
        supplier["stock_capacity"] = max(hi, round(supplier["stock_capacity"] * _jitter(rng, s["supply"])))
        supplier["base_price"] = float(supplier["base_price"]) * _jitter(rng, s["material_price"])

    raw.setdefault("agents", {})["observe_all_transactions"] = rng.random() < s["observe_all"]
    if rng.random() < s["capacity_prob"]:
        low, high = s["capacity_range"]
        raw.setdefault("production", {})["capacity_per_turn"] = rng.randint(int(low), int(high))

    market = raw.get("consumer_market", {})
    f = _jitter(rng, s["customers"])
    market["base_customers"] = max(1, round(market["base_customers"] * f))
    market["amplitude"] = max(0, round(market.get("amplitude", 0) * f))
    market["price_sensitivity_beta"] = float(market["price_sensitivity_beta"]) * _jitter(rng, s["beta"])
    segments = market.get("segments", {})
    for segment in segments.values():
        segment["share"] = float(segment["share"]) * _jitter(rng, s["shares"])
    total = sum(segment["share"] for segment in segments.values())
    for segment in segments.values():
        segment["share"] /= total
    for segment in segments.values():
        for good, (low, high) in segment["willingness_to_pay"].items():
            g = _jitter(rng, s["wtp"])
            segment["willingness_to_pay"][good] = [float(low) * g, float(high) * g]
