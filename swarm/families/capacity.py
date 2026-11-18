"""Capacity family: commit to a restrained stock, sell it by the expected end.

Sources: Kreps & Scheinkman (1983), where firms that commit capacity before
competing on price end up with restrained, Cournot quantities; Kalymon (1971)
and Golabi (1985), where the optimal purchasing rule under random prices
orders up to a target that depends on the current price; and the fixed-price
run-out policy of Gallego & van Ryzin (1994), near-optimal for a stock and a
deadline.

The family targets a sustainable stock per good: only as many units as each
sell within ``cover`` turns on average at the planned price. With
``quantity: basestock``, the target is converted to materials and scaled by
(average reserve / current reserve) ** ``kappa``, so it buys more when
materials are cheap. Bids aim at a target win probability on the learned
clearing-price distribution. Listings use the price that maximises revenue
by the expected end of the game.
"""

from __future__ import annotations

from typing import Any

from .econ import Engine

HELPERS = ("econ.py",)
CHOICES = {"quantity": ("cover", "basestock")}   # stock targets only: the family's premise

DEFAULTS: dict[str, Any] = {
    "quantity": "basestock",
    "listing": "units",
    "cover": 8.0,
    "kappa": 1.0,
    "runout_frac": 0.5,
    "bid_quantile": 0.75,
    "margin": 0.2,
    "budget_share": 0.6,
    "max_lot_share": 0.9,
    "leftover_margin": 0.3,
    "stop_turns": 1.0,
    "price_mult0": 5.0,
    "accept0": 0.3,
    "slope0": 3.0,
    "prior_weight": 4.0,
    "memory": 0.85,
    "cannibal": 0.5,
    "explore": 0.12,
    "rungs": 3,
    "horizon_median": 40.0,
    "horizon_spread": 0.5,
    "clear0": 0.3,
    "clear_sd0": 0.4,
    "clear_rate": 0.35,
    "uncontested0": 0.4,
    "pacing_gain": 0.0,
    "spend_rate": 0.15,
}


class CapacityAgent(Engine):
    DEFAULTS = DEFAULTS
    FIXED = {"pricing": "runout", "bidding": "quantile", "kits": "independent", "regime": False}


AGENT = CapacityAgent
