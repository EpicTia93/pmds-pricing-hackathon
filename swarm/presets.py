"""Hand-designed archetypes per family, used to seed islands and as references.

Each preset lists only the parameters it changes from its family's ``DEFAULTS``.
Preset ids are ``family:name`` (e.g. ``glut:hoarder``).
"""

from __future__ import annotations

from typing import Any

from .families import FAMILIES

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "adaptive": {
        "baseline": {},
        "premium": {
            "init_price_mult": 8.0, "price_up": 0.15, "price_down": 0.05,
            "stock_cover": 0.8, "min_margin": 1.0, "floor_mult": 2.0,
            "ladder_steps": 4, "ladder_spread": 0.3,
        },
        "volume": {
            "init_price_mult": 3.0, "price_up": 0.05, "price_down": 0.2,
            "stock_cover": 2.0, "min_margin": 0.05, "floor_mult": 1.0,
            "budget_share": 0.9, "bid_markup": 0.2, "init_spend_frac": 0.4,
        },
        "aggressive_bidder": {
            "bid_markup": 0.5, "max_bid_markup": 1.5, "bid_adapt": 0.15,
            "fill_target": 1.0, "budget_share": 0.8,
        },
        "bargain_hunter": {
            "bid_markup": 0.0, "bid_adapt": 0.0, "max_bid_markup": 0.0,
            "cheap_ratio": 0.75, "cheap_extra": 1.0, "raw_buffer": 0.3,
        },
        "explorer": {
            "init_price_mult": 12.0, "price_down": 0.25, "zero_sale_down": 0.25,
            "price_up": 0.25, "ladder_steps": 5, "ladder_spread": 0.45,
            "demand_ema": 0.7,
        },
        "undercutter": {
            "market_anchor": 0.6, "undercut": 0.08, "price_down": 0.15,
            "allocation": "proportional", "margin_power": 1.5,
        },
        # Parameters of the champion of the config/generated run (g037_3172).
        "generated_champion": {
            "init_price_mult": 5.09, "price_up": 0.354, "price_down": 0.020,
            "zero_sale_down": 0.405, "sellout_threshold": 0.764, "slow_threshold": 0.288,
            "floor_mult": 1.616, "ladder_steps": 5, "ladder_spread": 0.230,
            "ladder_low_first": False, "list_fraction": 0.822, "market_anchor": 0.867,
            "undercut": 0.058, "init_spend_frac": 0.031, "demand_ema": 0.127,
            "sellout_boost": 0.536, "stock_cover": 2.594, "probe_units": 2,
            "min_margin": 0.102, "margin_power": 0.525, "build_leftovers": False,
            "budget_share": 0.290, "bid_markup": 0.999, "bid_adapt": 0.263,
            "fill_target": 0.775, "max_bid_markup": 1.992, "cheap_ratio": 0.510,
            "cheap_extra": 0.941, "raw_buffer": 0.241, "max_lot_share": 0.840,
            "horizon": 0, "endgame_turns": 2, "endgame_discount": 0.042,
        },
    },
    "valuation": {
        "baseline": {},
        "explorer": {"explore_bonus": 0.8, "temperature": 0.6, "bins": 9, "width": 1.0},
        "exploiter": {"explore_bonus": 0.05, "temperature": 0.05, "memory": 0.95,
                      "recenter_rate": 0.8},
        "value_bidder": {"value_share": 1.0, "aggression": 0.7, "budget_share": 0.8},
    },
    "glut": {
        "baseline": {},
        "hoarder": {"buy_threshold": 0.7, "batch_turns": 6.0, "budget_share": 0.9,
                    "safety_turns": 0.5},
        "steady": {"buy_threshold": 1.0, "batch_turns": 1.5, "safety_turns": 1.5,
                   "panic_markup": 0.6},
        "skimmer": {"premium_share": 0.7, "tier_ratio": 0.6, "skim_bump": 0.25,
                    "init_price_mult": 8.0, "premium_first": True},
    },
    "rival": {
        "baseline": {},
        "niche": {"crowd_aversion": 3.0, "monopoly_premium": 0.5, "crowd_undercut": 0.0},
        "fighter": {"crowd_aversion": 0.0, "crowd_undercut": 0.3, "pressure_gain": 2.0,
                    "base_markup": 0.3, "capital_gain": 0.3},
        "follower": {"anchor": 0.9, "anchor_undercut": 0.08},
    },
    "portfolio": {
        "baseline": {},
        "explorer": {"ucb_c": 1.5, "temperature": 1.0, "roi_prior": 1.5},
        "focused": {"ucb_c": 0.2, "temperature": 0.08, "budget_share": 0.8},
        "tight_control": {"kp": 0.8, "ki": 0.15, "target_through": 0.95, "fill_kp": 0.6},
    },
}

for _fam, _presets in PRESETS.items():
    for _name, _params in _presets.items():
        FAMILIES[_fam].complete(_params)


def preset(family: str, name: str) -> dict[str, Any]:
    return FAMILIES[family].complete(PRESETS[family][name])


def resolve(family: str, names: list[str] | str | None) -> dict[str, dict[str, Any]]:
    """Presets of one family: names, "all", or None/"none"."""
    if names in (None, [], "none"):
        return {}
    if names == "all" or names == ["all"]:
        names = list(PRESETS.get(family, {}))
    return {name: preset(family, name) for name in names}


def resolve_ids(ids: list[str] | None) -> dict[str, tuple[str, dict[str, Any]]]:
    """``family:name`` / ``family:all`` / ``all`` / bare adaptive names -> {id: (family, params)}."""
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for item in ids or []:
        fam, _, name = item.rpartition(":") if ":" in item else ("", "", item)
        families = list(PRESETS) if item == "all" else [fam or "adaptive"]
        for f in families:
            names = "all" if item == "all" or name == "all" else [name]
            for n, params in resolve(f, names).items():
                out[f"{f}:{n}"] = (f, params)
    return out
