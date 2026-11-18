"""Generate many random, valid DME scenario YAMLs for testing agents.

Two kinds of scenario are produced:

* ``hw_*``   - the hardware market of ``config/default.yaml`` (or
  ``scaled_market.yaml``), with every number jittered plus one to three
  "stress" modifiers (scarce supply, glut, production cap, shifted segments,
  volatile demand, squeezed margins, ...).
* ``syn_*``  - fully synthetic markets with random material/good names,
  random recipes, segments and prices. These catch agents that hardcode the
  hardware goods, recipes or price levels.

Every file is validated with the simulator's own ``load_config`` before it is
kept. A ``manifest.csv`` summarising each scenario is written next to them.

Usage (from the repository root):

    uv run python scripts/generate_scenarios.py --count 40 --seed 0
    uv run dme-sim --config config/generated/hw_000_scarce.yaml ...
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dme_simulator.config import load_config  # noqa: E402

MATERIAL_NAMES = [
    "STEEL", "GLASS", "RESIN", "COPPER", "SILICON", "FABRIC", "WOOD",
    "LENS", "MOTOR", "SENSOR", "CELL", "ALLOY", "RUBBER", "LED",
]
GOOD_NAMES = [
    "DRONE", "SPEAKER", "CAMERA", "ROUTER", "HEADSET", "KEYBOARD", "MONITOR",
    "PRINTER", "ROBOT", "SCOOTER", "PROJECTOR", "EREADER", "TRACKER", "VRKIT",
]
SEGMENT_NAMES = ["BUDGET", "STUDENT", "MAINSTREAM", "PRO", "PREMIUM", "ENTERPRISE"]

HW_MODIFIERS = [
    "scarce", "glut", "capped", "segments", "volatile", "squeeze", "fat",
    "dead_good", "poor", "rich", "sensitive", "insensitive", "short", "long",
    "no_adapt", "fixed_reserve", "no_carryover", "open_book", "head_start",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def jitter(rng: np.random.Generator, value: float, spread: float) -> float:
    """Multiply by a log-uniform factor in [exp(-spread), exp(spread)]."""
    return value * math.exp(rng.uniform(-spread, spread))


def ijitter(rng: np.random.Generator, value: float, spread: float, low: int = 0) -> int:
    return max(low, int(round(jitter(rng, value, spread))))


def r2(x: float) -> float:
    return round(float(x), 2)


def shares(rng: np.random.Generator, n: int, concentration: float = 2.0) -> list[float]:
    """Random shares that are all > 0 and sum to exactly 1 (3 decimals)."""
    raw = rng.dirichlet([concentration] * n)
    raw = np.maximum(raw, 0.03)
    raw = raw / raw.sum()
    rounded = [round(float(x), 3) for x in raw[:-1]]
    rounded.append(round(1.0 - sum(rounded), 3))
    return rounded


def random_adaptation(rng: np.random.Generator, enabled: bool | None = None) -> dict:
    return {
        "enabled": bool(rng.random() < 0.75) if enabled is None else enabled,
        "low_stock_consecutive_turns": int(rng.integers(1, 5)),
        "high_stock_consecutive_turns": int(rng.integers(3, 8)),
        "empty_stock_percentage": r2(rng.uniform(0.02, 0.10)),
        "full_stock_percentage": r2(rng.uniform(0.85, 0.99)),
        "increase_percentage": r2(rng.uniform(0.05, 0.25)),
        "decrease_percentage": r2(rng.uniform(0.05, 0.20)),
    }


def random_supplier(
    rng: np.random.Generator, base_price: float, production_max: int
) -> dict:
    production_max = max(1, production_max)
    fixed = rng.random() < 0.15
    production_min = (
        production_max if fixed
        else max(0, int(round(production_max * rng.uniform(0.4, 0.9))))
    )
    factor = 0.0 if rng.random() < 0.15 else r2(rng.uniform(0.05, 0.40))
    floor = 1.0 if factor == 0.0 else r2(rng.uniform(0.30, 0.90))
    return {
        "production_min": production_min,
        "production_max": production_max,
        "base_price": r2(base_price),
        "stock_capacity": production_max * int(rng.integers(1, 9)),
        "price_stock_factor": factor,
        "price_floor_multiplier": floor,
        "adaptation": random_adaptation(rng),
    }


def agents_block(rng: np.random.Generator, observe_all: bool | None = None) -> dict:
    return {
        # Generated files live one level deeper than config/*.yaml; the
        # simulator resolves this path from the config file's grandparent.
        "directory": "../src/dme_simulator/agents",
        "decide_timeout_ms": 100.0,
        "observe_all_transactions": (
            bool(rng.random() < 0.3) if observe_all is None else observe_all
        ),
    }


def output_block(name: str) -> dict:
    return {
        "directory": f"../outputs/generated/{name}",
        "prefix": "market",
        "render_plots": True,
        "progress_interval_turns": 5,
        "dashboard": "none",
        "turn_delay_seconds": 0.0,
    }


def recipe_cost(recipe: dict, suppliers: dict) -> float:
    return sum(q * suppliers[m]["base_price"] for m, q in recipe["inputs"].items())


# ---------------------------------------------------------------------------
# hardware-market variants
# ---------------------------------------------------------------------------


def hardware_scenario(
    rng: np.random.Generator, base: dict, modifiers: list[str]
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(rng.integers(0, 1_000_000))
    cfg["turns"] = ijitter(rng, cfg["turns"], 0.3, low=5)
    init = cfg["initial_state"]
    init["starting_capital"] = r2(jitter(rng, init["starting_capital"], 0.3))
    init["initial_inventory"] = {}

    for sup in cfg["suppliers"].values():
        sup["base_price"] = r2(jitter(rng, sup["base_price"], 0.25))
        scale = math.exp(rng.uniform(-0.25, 0.25))
        sup["production_max"] = max(1, int(round(sup["production_max"] * scale)))
        sup["production_min"] = min(
            sup["production_max"], max(0, int(round(sup["production_min"] * scale)))
        )
        sup["stock_capacity"] = max(
            sup["production_max"], int(round(sup["stock_capacity"] * scale))
        )
        sup["price_stock_factor"] = r2(jitter(rng, sup["price_stock_factor"], 0.3))
        sup["price_floor_multiplier"] = r2(
            min(1.0, jitter(rng, sup["price_floor_multiplier"], 0.2))
        )

    market = cfg["consumer_market"]
    market["base_customers"] = ijitter(rng, market["base_customers"], 0.3, low=1)
    market["amplitude"] = min(
        market["base_customers"], ijitter(rng, market["amplitude"], 0.4)
    )
    market["price_sensitivity_beta"] = round(
        jitter(rng, market["price_sensitivity_beta"], 0.5), 4
    )
    for seg in market["segments"].values():
        for good, (lo, hi) in seg["willingness_to_pay"].items():
            f = math.exp(rng.uniform(-0.15, 0.15))
            seg["willingness_to_pay"][good] = [r2(lo * f), r2(hi * f)]
    cfg["agents"] = agents_block(rng)

    for mod in modifiers:
        apply_hw_modifier(rng, cfg, mod)
    return cfg


def apply_hw_modifier(rng: np.random.Generator, cfg: dict, mod: str) -> None:
    sups = cfg["suppliers"]
    market = cfg["consumer_market"]
    segs = market["segments"]

    def scale_supply(f: float) -> None:
        for sup in sups.values():
            sup["production_max"] = max(1, int(round(sup["production_max"] * f)))
            sup["production_min"] = min(
                sup["production_max"], max(0, int(round(sup["production_min"] * f)))
            )
            sup["stock_capacity"] = max(
                sup["production_max"], int(round(sup["stock_capacity"] * f))
            )

    def scale_wtp(f: float, goods: list[str] | None = None) -> None:
        for seg in segs.values():
            for good, (lo, hi) in seg["willingness_to_pay"].items():
                if goods is None or good in goods:
                    seg["willingness_to_pay"][good] = [r2(lo * f), r2(hi * f)]

    if mod == "scarce":  # materials are the bottleneck
        scale_supply(rng.uniform(0.25, 0.6))
    elif mod == "glut":  # too much supply, too few buyers
        scale_supply(rng.uniform(2.0, 4.0))
        market["base_customers"] = max(1, int(market["base_customers"] * rng.uniform(0.4, 0.8)))
        market["amplitude"] = min(market["amplitude"], market["base_customers"])
    elif mod == "capped":
        cfg["production"]["capacity_per_turn"] = int(
            max(1, market["base_customers"] * rng.uniform(0.2, 0.8))
        )
    elif mod == "segments":
        values = shares(rng, len(segs), concentration=1.0)
        for seg, share in zip(segs.values(), values):
            seg["share"] = share
            for good in seg["preferences"]:
                seg["preferences"][good] = int(rng.integers(1, 6))
    elif mod == "volatile":
        market["amplitude"] = int(market["base_customers"] * rng.uniform(0.6, 1.0))
        market["period_divisor"] = r2(rng.uniform(1.0, 4.0))
    elif mod == "squeeze":  # thin margins
        scale_wtp(rng.uniform(0.4, 0.7))
    elif mod == "fat":
        scale_wtp(rng.uniform(1.3, 2.0))
    elif mod == "dead_good":  # one or two goods sell below their material cost
        goods = list(cfg["goods"]["finished_goods"])
        victims = list(rng.choice(goods, size=int(rng.integers(1, 3)), replace=False))
        scale_wtp(rng.uniform(0.1, 0.3), victims)
    elif mod == "poor":
        cfg["initial_state"]["starting_capital"] = r2(
            cfg["initial_state"]["starting_capital"] * rng.uniform(0.15, 0.4)
        )
    elif mod == "rich":
        cfg["initial_state"]["starting_capital"] = r2(
            cfg["initial_state"]["starting_capital"] * rng.uniform(3.0, 8.0)
        )
    elif mod == "sensitive":
        market["price_sensitivity_beta"] = round(rng.uniform(0.04, 0.12), 4)
    elif mod == "insensitive":
        market["price_sensitivity_beta"] = 0.0
    elif mod == "short":
        cfg["turns"] = int(rng.integers(5, 15))
    elif mod == "long":
        cfg["turns"] = int(rng.integers(60, 120))
    elif mod == "no_adapt":
        for sup in sups.values():
            sup["adaptation"]["enabled"] = False
    elif mod == "fixed_reserve":
        for sup in sups.values():
            sup["price_stock_factor"] = 0.0
            sup["price_floor_multiplier"] = 1.0
    elif mod == "no_carryover":
        for sup in sups.values():
            sup["stock_capacity"] = sup["production_max"]
    elif mod == "open_book":
        cfg["agents"]["observe_all_transactions"] = True
    elif mod == "head_start":
        goods = cfg["goods"]["raw_materials"] + cfg["goods"]["finished_goods"]
        picks = rng.choice(goods, size=int(rng.integers(1, 4)), replace=False)
        cfg["initial_state"]["initial_inventory"] = {
            str(g): int(rng.integers(5, 60)) for g in picks
        }
    else:
        raise ValueError(f"unknown modifier {mod}")


# ---------------------------------------------------------------------------
# synthetic markets
# ---------------------------------------------------------------------------


def synthetic_scenario(rng: np.random.Generator) -> tuple[dict, str]:
    n_raw = int(rng.integers(1, 6))
    n_goods = int(rng.integers(1, 7))
    raws = [str(x) for x in rng.choice(MATERIAL_NAMES, size=n_raw, replace=False)]
    goods = [str(x) for x in rng.choice(GOOD_NAMES, size=n_goods, replace=False)]
    base_prices = {m: float(rng.uniform(1.0, 40.0)) for m in raws}

    # Recipes: random subset of materials with quantities 1-3.
    recipes: dict[str, dict] = {}
    for good in goods:
        k = int(rng.integers(1, n_raw + 1))
        chosen = rng.choice(raws, size=k, replace=False)
        recipes[f"RECIPE_{good}"] = {
            "output_good": good,
            "inputs": {str(m): int(rng.integers(1, 4)) for m in chosen},
        }
    costs = {
        r["output_good"]: sum(q * base_prices[m] for m, q in r["inputs"].items())
        for r in recipes.values()
    }

    # Segments: tiered markups over material cost; some goods/segments lose money.
    n_seg = int(rng.integers(1, min(4, len(SEGMENT_NAMES)) + 1))
    seg_names = sorted(rng.choice(SEGMENT_NAMES, size=n_seg, replace=False),
                       key=SEGMENT_NAMES.index)
    seg_shares = shares(rng, n_seg)
    segments: dict[str, dict] = {}
    uncovered = set(goods)
    for i, (name, share) in enumerate(zip(seg_names, seg_shares)):
        tier = (i + 1) / n_seg  # later segments pay more
        k = int(rng.integers(1, n_goods + 1))
        wanted = [str(g) for g in rng.choice(goods, size=k, replace=False)]
        if i == n_seg - 1:  # last segment picks up any good nobody wants
            wanted = sorted(set(wanted) | uncovered, key=goods.index)
        uncovered -= set(wanted)
        prefs, wtp = {}, {}
        for good in wanted:
            prefs[good] = int(rng.integers(1, 6))
            markup = math.exp(rng.uniform(math.log(0.6), math.log(2.0 + 6.0 * tier)))
            center = costs[good] * markup
            width = rng.uniform(0.1, 0.5)
            wtp[good] = [r2(center * (1 - width)), r2(center * (1 + width))]
        segments[str(name)] = {"share": share, "preferences": prefs,
                               "willingness_to_pay": wtp}

    base_customers = int(math.exp(rng.uniform(math.log(5), math.log(250))))
    amplitude = int(base_customers * rng.uniform(0.0, 0.6))

    # Size supply against expected material use per turn.
    usage = {m: 0.0 for m in raws}
    for seg in segments.values():
        total_w = sum(seg["preferences"].values())
        for good, w in seg["preferences"].items():
            recipe = recipes[f"RECIPE_{good}"]
            for m, q in recipe["inputs"].items():
                usage[m] += seg["share"] * (w / total_w) * q
    regime = str(rng.choice(["scarce", "balanced", "abundant"], p=[0.35, 0.4, 0.25]))
    ratio_range = {"scarce": (0.25, 0.7), "balanced": (0.7, 1.5), "abundant": (1.5, 5.0)}[regime]
    suppliers = {}
    for m in raws:
        expected = max(1.0, usage[m] * base_customers)
        prod_max = max(1, int(round(expected * rng.uniform(*ratio_range))))
        suppliers[m] = random_supplier(rng, base_prices[m], prod_max)

    per_turn_value = sum(
        s["production_max"] * s["base_price"] for s in suppliers.values()
    )
    capital = r2(max(500.0, per_turn_value * rng.uniform(3.0, 20.0)))

    capacity = None
    if rng.random() < 0.3:
        capacity = max(1, int(base_customers * rng.uniform(0.2, 0.9)))

    initial_inventory = {}
    if rng.random() < 0.2:
        for g in rng.choice(raws + goods, size=int(rng.integers(1, 3)), replace=False):
            initial_inventory[str(g)] = int(rng.integers(5, 50))

    cfg = {
        "turns": int(rng.integers(10, 61)),
        "seed": int(rng.integers(0, 1_000_000)),
        "initial_state": {"starting_capital": capital,
                          "initial_inventory": initial_inventory},
        "suppliers": suppliers,
        "production": {"capacity_per_turn": capacity},
        "consumer_market": {
            "base_customers": base_customers,
            "amplitude": amplitude,
            "period_divisor": r2(rng.uniform(1.5, 10.0)),
            "price_sensitivity_beta": round(float(rng.uniform(0.0, 0.08)), 4),
            "segments": segments,
        },
        "goods": {"raw_materials": raws, "finished_goods": goods,
                  "recipes": recipes},
        "agents": agents_block(rng),
    }
    return cfg, regime


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def summarize(name: str, kind: str, tags: str, cfg: dict) -> dict:
    sups = cfg["suppliers"]
    market = cfg["consumer_market"]
    supply = sum(s["production_max"] for s in sups.values())
    return {
        "file": f"{name}.yaml",
        "kind": kind,
        "tags": tags,
        "turns": cfg["turns"],
        "seed": cfg["seed"],
        "capital": cfg["initial_state"]["starting_capital"],
        "materials": len(cfg["goods"]["raw_materials"]),
        "goods": len(cfg["goods"]["finished_goods"]),
        "segments": len(market["segments"]),
        "customers": market["base_customers"],
        "amplitude": market["amplitude"],
        "beta": market["price_sensitivity_beta"],
        "supply_units_max": supply,
        "capacity": cfg["production"]["capacity_per_turn"],
        "observe_all": cfg["agents"]["observe_all_transactions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=40, help="scenarios to generate")
    parser.add_argument("--seed", type=int, default=0, help="generator seed")
    parser.add_argument("--synthetic-share", type=float, default=0.4,
                        help="fraction of fully synthetic markets (0-1)")
    parser.add_argument("--out", type=Path, default=ROOT / "config" / "generated")
    parser.add_argument("--clean", action="store_true",
                        help="delete the output directory first")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out: Path = args.out.resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    bases = {
        "default": yaml.safe_load((ROOT / "config/default.yaml").read_text()),
        "scaled": yaml.safe_load((ROOT / "config/scaled_market.yaml").read_text()),
    }

    rows = []
    for i in range(args.count):
        if rng.random() < args.synthetic_share:
            cfg, regime = synthetic_scenario(rng)
            name, kind, tags = f"syn_{i:03d}_{regime}", "synthetic", regime
        else:
            base_name = "scaled" if rng.random() < 0.25 else "default"
            n_mods = int(rng.integers(1, 4))
            mods = [str(m) for m in rng.choice(HW_MODIFIERS, size=n_mods, replace=False)]
            # Mutually exclusive pairs: keep the first one.
            for a, b in [("scarce", "glut"), ("squeeze", "fat"), ("poor", "rich"),
                         ("sensitive", "insensitive"), ("short", "long")]:
                if a in mods and b in mods:
                    mods.remove(b if mods.index(a) < mods.index(b) else a)
            cfg = hardware_scenario(rng, bases[base_name], mods)
            tags = "+".join(mods)
            name = f"hw_{i:03d}_{'-'.join(mods)}"
            kind = f"hardware/{base_name}"
        cfg["output"] = output_block(name)

        path = out / f"{name}.yaml"
        header = f"# Generated by scripts/generate_scenarios.py (seed={args.seed}, #{i})\n# kind={kind} tags={tags}\n"
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
        load_config(path)  # raises if the scenario is invalid
        rows.append(summarize(name, kind, tags, cfg))

    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} validated scenarios to {out}")


if __name__ == "__main__":
    main()
