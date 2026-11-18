"""Search space over ``ParamAgent`` parameters: sample, mutate, cross over.

Every parameter in each family's ``DEFAULTS`` must have a spec here. Experiments
can narrow ranges (``space:``) or pin values (``fixed:``) from YAML.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any

from .families import FAMILIES


@dataclass(frozen=True)
class Spec:
    kind: str                       # float | int | bool | choice
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    choices: tuple[Any, ...] = ()


F, I, B, C = "float", "int", "bool", "choice"

# Ranges for every parameter of the paper-based engine (families/econ.py). The
# revman, auction, capacity and tactex families evolve subsets of these; the
# rest of each family is pinned by its ``FIXED`` modules.
ENGINE_SPECS: dict[str, Spec] = {
    "pricing": Spec(C, choices=("dp", "myopic", "runout")),
    "listing": Spec(C, choices=("units", "rungs")),
    "quantity": Spec(C, choices=("value", "cover", "basestock")),
    "bidding": Spec(C, choices=("kit", "quantile", "shade")),
    "kits": Spec(C, choices=("independent", "saa")),
    "regime": Spec(B),
    "price_mult0": Spec(F, 1.5, 25.0, log=True),
    "accept0": Spec(F, 0.03, 0.9),
    "slope0": Spec(F, 0.3, 15.0, log=True),
    "prior_weight": Spec(F, 0.5, 40.0, log=True),
    "memory": Spec(F, 0.5, 0.99),
    "cannibal": Spec(F, 0.0, 1.2),
    "explore": Spec(F, 0.0, 0.6),
    "rungs": Spec(I, 1, 6),
    "horizon_median": Spec(F, 8.0, 150.0, log=True),
    "horizon_spread": Spec(F, 0.05, 1.5),
    "basis_floor": Spec(F, 0.0, 1.5),
    "runout_frac": Spec(F, 0.2, 2.0),
    "margin": Spec(F, -0.2, 2.0),
    "cover": Spec(F, 1.0, 15.0, log=True),
    "kappa": Spec(F, 0.0, 3.0),
    "budget_share": Spec(F, 0.05, 1.0),
    "stop_turns": Spec(F, 0.0, 8.0),
    "leftover_margin": Spec(F, 0.0, 1.5),
    "max_lot_share": Spec(F, 0.1, 1.0),
    "clear0": Spec(F, -0.2, 2.0),
    "clear_sd0": Spec(F, 0.05, 1.5, log=True),
    "clear_rate": Spec(F, 0.05, 0.9),
    "uncontested0": Spec(F, 0.0, 0.95),
    "aggression": Spec(F, 0.4, 2.0, log=True),
    "salvage": Spec(F, 0.0, 1.0),
    "bid_quantile": Spec(F, 0.2, 0.99),
    "shade": Spec(F, 0.0, 1.0),
    "pacing_gain": Spec(F, 0.0, 2.0),
    "spend_rate": Spec(F, 0.01, 0.6, log=True),
    "saa_samples": Spec(I, 4, 16),
    "clear_corr": Spec(F, 0.0, 0.95),
    "scarce_boost": Spec(F, 0.0, 1.5),
    "glut_ratio": Spec(F, 0.3, 1.0),
    "glut_cover": Spec(F, 0.0, 6.0),
}
ENGINE_FAMILIES = ("revman", "auction", "capacity", "tactex")


def _engine_space(name: str) -> dict[str, Spec]:
    """Engine specs for a family's parameters; a family may narrow module choices
    with a ``CHOICES`` mapping (e.g. capacity keeps only stock-target quantities)."""
    narrowed = getattr(FAMILIES[name].module, "CHOICES", {})
    out = {}
    for key in FAMILIES[name].defaults:
        spec = ENGINE_SPECS[key]
        if key in narrowed:
            spec = replace(spec, choices=tuple(narrowed[key]))
        out[key] = spec
    return out


SPACES: dict[str, dict[str, Spec]] = {
    "adaptive": {
        "init_price_mult": Spec(F, 1.5, 40.0, log=True),
        "price_up": Spec(F, 0.0, 0.5),
        "price_down": Spec(F, 0.0, 0.5),
        "zero_sale_down": Spec(F, 0.0, 0.5),
        "sellout_threshold": Spec(F, 0.6, 1.0),
        "slow_threshold": Spec(F, 0.0, 0.95),
        "floor_mult": Spec(F, 0.5, 3.0),
        "ladder_steps": Spec(I, 1, 6),
        "ladder_spread": Spec(F, 0.0, 0.6),
        "ladder_low_first": Spec(B),
        "list_fraction": Spec(F, 0.3, 1.0),
        "market_anchor": Spec(F, 0.0, 1.0),
        "undercut": Spec(F, -0.1, 0.3),
        "init_spend_frac": Spec(F, 0.02, 0.8, log=True),
        "demand_ema": Spec(F, 0.05, 0.95),
        "sellout_boost": Spec(F, 0.0, 2.0),
        "stock_cover": Spec(F, 0.5, 4.0),
        "probe_units": Spec(I, 0, 5),
        "min_margin": Spec(F, -0.2, 2.0),
        "allocation": Spec(C, choices=("greedy", "proportional")),
        "margin_power": Spec(F, 0.0, 3.0),
        "build_leftovers": Spec(B),
        "budget_share": Spec(F, 0.1, 1.0),
        "bid_markup": Spec(F, 0.0, 3.0),          # widened: last run hit 1.0
        "bid_adapt": Spec(F, 0.0, 0.5),
        "fill_target": Spec(F, 0.3, 1.0),
        "max_bid_markup": Spec(F, 0.0, 5.0),      # widened: last run hit 2.0
        "cheap_ratio": Spec(F, 0.4, 1.0),
        "cheap_extra": Spec(F, 0.0, 3.0),
        "raw_buffer": Spec(F, 0.0, 1.0),
        "max_lot_share": Spec(F, 0.1, 1.0),
        "horizon": Spec(I, 0, 60),
        "endgame_turns": Spec(I, 0, 10),
        "endgame_discount": Spec(F, 0.0, 0.5),
    },
    "valuation": {
        "bins": Spec(I, 3, 12),
        "width": Spec(F, 0.1, 1.5),
        "init_price_mult": Spec(F, 1.5, 40.0, log=True),
        "memory": Spec(F, 0.5, 0.99),
        "prior_accept": Spec(F, 0.0, 1.0),
        "prior_weight": Spec(F, 0.1, 10.0, log=True),
        "explore_bonus": Spec(F, 0.0, 1.5),
        "temperature": Spec(F, 0.01, 2.0, log=True),
        "recenter_rate": Spec(F, 0.0, 1.0),
        "upside_decay": Spec(F, 0.1, 1.0),
        "cost_floor": Spec(F, 0.3, 2.5),
        "demand_ema": Spec(F, 0.05, 0.95),
        "sellout_boost": Spec(F, 0.0, 2.0),
        "stock_cover": Spec(F, 0.5, 4.0),
        "probe_units": Spec(I, 0, 6),
        "init_spend_frac": Spec(F, 0.02, 0.8, log=True),
        "min_profit": Spec(F, -0.2, 2.0),
        "value_share": Spec(F, 0.2, 1.5),
        "aggression": Spec(F, 0.0, 1.0),
        "budget_share": Spec(F, 0.1, 1.0),
        "list_fraction": Spec(F, 0.3, 1.0),
        "build_leftovers": Spec(B),
    },
    "glut": {
        "buy_threshold": Spec(F, 0.4, 1.05),
        "bid_markup": Spec(F, 0.0, 0.5),
        "safety_turns": Spec(F, 0.0, 3.0),
        "panic_markup": Spec(F, 0.0, 3.0),
        "batch_turns": Spec(F, 0.5, 8.0, log=True),
        "max_lot_share": Spec(F, 0.1, 1.0),
        "budget_share": Spec(F, 0.1, 1.0),
        "production_rate": Spec(F, 0.3, 3.0, log=True),
        "stock_cover": Spec(F, 0.5, 4.0),
        "demand_ema": Spec(F, 0.05, 0.95),
        "sellout_boost": Spec(F, 0.0, 2.0),
        "init_spend_frac": Spec(F, 0.02, 0.8, log=True),
        "min_margin": Spec(F, -0.2, 2.0),
        "init_price_mult": Spec(F, 1.5, 40.0, log=True),
        "premium_share": Spec(F, 0.0, 1.0),
        "tier_ratio": Spec(F, 0.3, 1.0),
        "premium_first": Spec(B),
        "skim_bump": Spec(F, 0.0, 0.5),
        "skim_decay": Spec(F, 0.0, 0.4),
        "through_target": Spec(F, 0.1, 1.0),
        "value_cut": Spec(F, 0.0, 0.4),
        "floor_mult": Spec(F, 0.5, 3.0),
    },
    "rival": {
        "init_price_mult": Spec(F, 1.5, 40.0, log=True),
        "price_up": Spec(F, 0.0, 0.5),
        "price_down": Spec(F, 0.0, 0.5),
        "sellout_threshold": Spec(F, 0.6, 1.0),
        "slow_threshold": Spec(F, 0.0, 0.95),
        "floor_mult": Spec(F, 0.5, 3.0),
        "monopoly_premium": Spec(F, 0.0, 1.0),
        "crowd_undercut": Spec(F, 0.0, 0.5),
        "crowd_aversion": Spec(F, 0.0, 5.0),
        "anchor": Spec(F, 0.0, 1.0),
        "anchor_undercut": Spec(F, -0.1, 0.3),
        "base_markup": Spec(F, 0.0, 2.0),
        "pressure_gain": Spec(F, 0.0, 4.0),
        "pressure_ema": Spec(F, 0.05, 1.0),
        "capital_gain": Spec(F, -0.5, 1.0),
        "max_markup": Spec(F, 0.0, 5.0),
        "budget_share": Spec(F, 0.1, 1.0),
        "stock_cover": Spec(F, 0.5, 4.0),
        "demand_ema": Spec(F, 0.05, 0.95),
        "sellout_boost": Spec(F, 0.0, 2.0),
        "probe_units": Spec(I, 0, 5),
        "init_spend_frac": Spec(F, 0.02, 0.8, log=True),
        "min_margin": Spec(F, -0.2, 2.0),
        "ladder_steps": Spec(I, 1, 5),
        "ladder_spread": Spec(F, 0.0, 0.5),
        "build_leftovers": Spec(B),
    },
    "portfolio": {
        "init_price_mult": Spec(F, 1.5, 40.0, log=True),
        "target_through": Spec(F, 0.3, 1.0),
        "kp": Spec(F, 0.0, 1.5),
        "ki": Spec(F, 0.0, 0.5),
        "integral_clip": Spec(F, 0.5, 6.0),
        "floor_mult": Spec(F, 0.5, 3.0),
        "roi_ema": Spec(F, 0.05, 0.95),
        "roi_prior": Spec(F, -0.5, 3.0),
        "ucb_c": Spec(F, 0.0, 3.0),
        "temperature": Spec(F, 0.02, 3.0, log=True),
        "min_score": Spec(F, -1.0, 1.0),
        "budget_share": Spec(F, 0.1, 1.0),
        "stock_cover": Spec(F, 0.5, 4.0),
        "demand_ema": Spec(F, 0.05, 0.95),
        "sellout_boost": Spec(F, 0.0, 2.0),
        "probe_units": Spec(I, 0, 5),
        "init_spend_frac": Spec(F, 0.02, 0.8, log=True),
        "bid_markup": Spec(F, 0.0, 2.0),
        "fill_kp": Spec(F, 0.0, 1.5),
        "fill_ki": Spec(F, 0.0, 0.5),
        "fill_target": Spec(F, 0.3, 1.0),
        "max_markup": Spec(F, 0.0, 5.0),
        "build_leftovers": Spec(B),
    },
    "neutral": {},  # no parameters: never trades
    **{name: _engine_space(name) for name in ENGINE_FAMILIES},
}

from .families.econ import ENGINE_DEFAULTS  # noqa: E402

assert set(ENGINE_SPECS) == set(ENGINE_DEFAULTS), \
    f"engine space/defaults mismatch {sorted(set(ENGINE_SPECS) ^ set(ENGINE_DEFAULTS))}"

for _name, _specs in SPACES.items():
    _mismatch = set(FAMILIES[_name].defaults) ^ set(_specs)
    assert not _mismatch, f"{_name}: space/strategy parameter mismatch {sorted(_mismatch)}"
assert set(SPACES) == set(FAMILIES), "every family needs a search space"
SPECS = SPACES["adaptive"]  # backwards compatibility


class ParamSpace:
    def __init__(
        self,
        family: str = "adaptive",
        overrides: dict[str, dict[str, Any]] | None = None,
        fixed: dict[str, Any] | None = None,
    ) -> None:
        self.family = family
        self.defaults = FAMILIES[family].defaults
        self.specs = dict(SPACES[family])
        for name, change in (overrides or {}).items():
            if name not in self.specs:
                raise KeyError(f"unknown parameter in space override: {name}")
            change = dict(change)
            if "choices" in change:
                change["choices"] = tuple(change["choices"])
            self.specs[name] = replace(self.specs[name], **change)
        self.fixed = dict(fixed or {})
        unknown = set(self.fixed) - set(self.specs)
        if unknown:
            raise KeyError(f"unknown fixed parameters: {sorted(unknown)}")

    @property
    def free(self) -> list[str]:
        return [name for name in self.specs if name not in self.fixed]

    # -- primitives ------------------------------------------------------
    def _draw(self, spec: Spec, rng: random.Random) -> Any:
        if spec.kind == "bool":
            return rng.random() < 0.5
        if spec.kind == "choice":
            return rng.choice(spec.choices)
        if spec.log:
            value = math.exp(rng.uniform(math.log(spec.low), math.log(spec.high)))
        else:
            value = rng.uniform(spec.low, spec.high)
        if spec.kind == "int":
            return int(round(value))
        return value

    def _perturb(self, spec: Spec, value: Any, scale: float, rng: random.Random) -> Any:
        if spec.kind in ("bool", "choice"):
            return self._draw(spec, rng)
        if spec.log:
            lo, hi, x = math.log(spec.low), math.log(spec.high), math.log(max(value, spec.low))
        else:
            lo, hi, x = spec.low, spec.high, float(value)
        x += rng.gauss(0.0, scale * (hi - lo))
        # reflect at the bounds to avoid piling up on the edges
        span = hi - lo
        if span > 0:
            x = lo + (x - lo) % (2 * span)
            if x > hi:
                x = 2 * hi - x
        else:
            x = lo
        if spec.log:
            x = math.exp(x)
        if spec.kind == "int":
            return int(round(x))
        return x

    def clip(self, params: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for name, spec in self.specs.items():
            value = params.get(name, self.defaults[name])
            if spec.kind in ("float", "int"):
                value = min(max(value, spec.low), spec.high)
                value = int(round(value)) if spec.kind == "int" else float(value)
            out[name] = value
        out.update(self.fixed)
        return out

    # -- operators -------------------------------------------------------
    def sample(self, rng: random.Random) -> dict[str, Any]:
        return self.clip({name: self._draw(spec, rng) for name, spec in self.specs.items()})

    def mutate(
        self, params: dict[str, Any], rng: random.Random, rate: float, scale: float
    ) -> dict[str, Any]:
        child = dict(params)
        free = self.free
        changed = False
        for name in free:
            if rng.random() < rate:
                child[name] = self._perturb(self.specs[name], child[name], scale, rng)
                changed = True
        if not changed and free:
            name = rng.choice(free)
            child[name] = self._perturb(self.specs[name], child[name], scale, rng)
        return self.clip(child)

    def crossover(
        self, a: dict[str, Any], b: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        child = {}
        for name, spec in self.specs.items():
            if spec.kind == "float" and rng.random() < 0.3:
                w = rng.random()
                child[name] = w * a[name] + (1 - w) * b[name]
            else:
                child[name] = a[name] if rng.random() < 0.5 else b[name]
        return self.clip(child)
