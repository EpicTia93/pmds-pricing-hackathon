"""Search space over ``ParamAgent`` parameters: sample, mutate, cross over.

Every entry of ``strategy.DEFAULT_PARAMS`` must have a spec here. Experiments
can narrow ranges (``space:``) or pin values (``fixed:``) from YAML.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any

from .strategy import DEFAULT_PARAMS


@dataclass(frozen=True)
class Spec:
    kind: str                       # float | int | bool | choice
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    choices: tuple[Any, ...] = ()


SPECS: dict[str, Spec] = {
    "init_price_mult": Spec("float", 1.5, 40.0, log=True),
    "price_up": Spec("float", 0.0, 0.5),
    "price_down": Spec("float", 0.0, 0.5),
    "zero_sale_down": Spec("float", 0.0, 0.5),
    "sellout_threshold": Spec("float", 0.6, 1.0),
    "slow_threshold": Spec("float", 0.0, 0.95),
    "floor_mult": Spec("float", 0.5, 3.0),
    "ladder_steps": Spec("int", 1, 6),
    "ladder_spread": Spec("float", 0.0, 0.6),
    "ladder_low_first": Spec("bool"),
    "list_fraction": Spec("float", 0.3, 1.0),
    "market_anchor": Spec("float", 0.0, 1.0),
    "undercut": Spec("float", -0.1, 0.3),
    "init_spend_frac": Spec("float", 0.02, 0.8, log=True),
    "demand_ema": Spec("float", 0.05, 0.95),
    "sellout_boost": Spec("float", 0.0, 2.0),
    "stock_cover": Spec("float", 0.5, 3.0),
    "probe_units": Spec("int", 0, 5),
    "min_margin": Spec("float", -0.2, 2.0),
    "allocation": Spec("choice", choices=("greedy", "proportional")),
    "margin_power": Spec("float", 0.0, 3.0),
    "build_leftovers": Spec("bool"),
    "budget_share": Spec("float", 0.1, 1.0),
    "bid_markup": Spec("float", 0.0, 1.0),
    "bid_adapt": Spec("float", 0.0, 0.3),
    "fill_target": Spec("float", 0.3, 1.0),
    "max_bid_markup": Spec("float", 0.0, 2.0),
    "cheap_ratio": Spec("float", 0.4, 1.0),
    "cheap_extra": Spec("float", 0.0, 2.0),
    "raw_buffer": Spec("float", 0.0, 1.0),
    "max_lot_share": Spec("float", 0.1, 1.0),
    "horizon": Spec("int", 0, 60),
    "endgame_turns": Spec("int", 0, 10),
    "endgame_discount": Spec("float", 0.0, 0.5),
}

missing = set(DEFAULT_PARAMS) ^ set(SPECS)
assert not missing, f"space/strategy parameter mismatch: {sorted(missing)}"


class ParamSpace:
    def __init__(
        self,
        overrides: dict[str, dict[str, Any]] | None = None,
        fixed: dict[str, Any] | None = None,
    ) -> None:
        self.specs = dict(SPECS)
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
            value = params.get(name, DEFAULT_PARAMS[name])
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
