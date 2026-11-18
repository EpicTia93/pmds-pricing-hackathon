"""Hand-designed archetypes used to seed populations and as reference opponents.

Each preset only lists the parameters it changes from ``DEFAULT_PARAMS``.
"""

from __future__ import annotations

from typing import Any

from .strategy import DEFAULT_PARAMS

PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "premium": {  # few units, high prices, slow to discount
        "init_price_mult": 8.0, "price_up": 0.15, "price_down": 0.05,
        "stock_cover": 0.8, "min_margin": 1.0, "floor_mult": 2.0,
        "ladder_steps": 4, "ladder_spread": 0.3,
    },
    "volume": {  # thin margins, buy a lot, clear stock fast
        "init_price_mult": 3.0, "price_up": 0.05, "price_down": 0.2,
        "stock_cover": 2.0, "min_margin": 0.05, "floor_mult": 1.0,
        "budget_share": 0.9, "bid_markup": 0.2, "init_spend_frac": 0.4,
    },
    "aggressive_bidder": {  # win materials at any reasonable price
        "bid_markup": 0.5, "max_bid_markup": 1.5, "bid_adapt": 0.15,
        "fill_target": 1.0, "budget_share": 0.8,
    },
    "bargain_hunter": {  # bid at reserve, stockpile when cheap
        "bid_markup": 0.0, "bid_adapt": 0.0, "max_bid_markup": 0.0,
        "cheap_ratio": 0.75, "cheap_extra": 1.0, "raw_buffer": 0.3,
    },
    "explorer": {  # wide price ladder, fast price discovery
        "init_price_mult": 12.0, "price_down": 0.25, "zero_sale_down": 0.25,
        "price_up": 0.25, "ladder_steps": 5, "ladder_spread": 0.45,
        "demand_ema": 0.7,
    },
    "lean": {  # minimal inventory, liquidate before the end of 30-40 turn games
        "stock_cover": 0.7, "probe_units": 0, "build_leftovers": False,
        "list_fraction": 1.0, "horizon": 30, "endgame_turns": 3,
        "endgame_discount": 0.25,
    },
    "undercutter": {  # follow and undercut the market when rivals are visible
        "market_anchor": 0.6, "undercut": 0.08, "price_down": 0.15,
        "allocation": "proportional", "margin_power": 1.5,
    },
}

for _name, _params in PRESETS.items():
    _unknown = set(_params) - set(DEFAULT_PARAMS)
    assert not _unknown, f"preset {_name} has unknown params {_unknown}"


def preset(name: str) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    params.update(PRESETS[name])
    return params


def resolve(names: list[str] | str | None) -> dict[str, dict[str, Any]]:
    if names in (None, [], "none"):
        return {}
    if names == "all" or names == ["all"]:
        names = list(PRESETS)
    return {name: preset(name) for name in names}
