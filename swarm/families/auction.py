"""Auction family: learn clearing prices from censored fills, best-respond with ladders.

Sources: first-price (pay-your-bid) best response and equilibrium shading;
Han, Zhou & Weissman (Operations Research) learning to bid from censored
first-price feedback; Ausubel, Cramton, Pycia, Rostek & Weretka (2014) shading
units differently in pay-as-bid auctions; Balseiro & Gur (2019) adaptive
pacing; Wellman, Osepayshvili, MacKie-Mason & Reeves (2008) price prediction
for complementary goods (the exposure problem); Greenwald, Lee & Naroditskiy
(2009) RoxyBot-06 sample average approximation.

The buying side is the core of this family. It keeps, per material, a
distribution of the clearing markup over the reserve plus the chance a lot is
uncontested, updated from full, partial and missed fills (or from the whole
log when it is visible). Bids for each good's kit come from one of:

* ``kit``: best response that maximises the expected margin of a complete kit,
  valuing stranded units at ``salvage``;
* ``quantile``: a target win probability;
* ``shade``: a fraction of the value.

Per-good kits form a descending bid ladder on each material. Optional
extras are a pacing multiplier on values (``pacing_gain``) and joint
per-material prices chosen by scenario sampling with correlated clearing
prices (``kits: saa``). Selling uses the static logit rule on a learned
demand curve.
"""

from __future__ import annotations

from typing import Any

from .econ import Engine

HELPERS = ("econ.py",)

DEFAULTS: dict[str, Any] = {
    "bidding": "kit",
    "kits": "independent",
    "listing": "units",
    "clear0": 0.3,
    "clear_sd0": 0.4,
    "clear_rate": 0.35,
    "uncontested0": 0.4,
    "aggression": 1.0,
    "salvage": 0.4,
    "bid_quantile": 0.75,
    "shade": 0.5,
    "pacing_gain": 0.0,
    "spend_rate": 0.15,
    "saa_samples": 12,
    "clear_corr": 0.5,
    "cover": 8.0,
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
    "basis_floor": 1.0,
    "horizon_median": 40.0,
    "horizon_spread": 0.5,
}


class AuctionAgent(Engine):
    DEFAULTS = DEFAULTS
    FIXED = {"pricing": "myopic", "quantity": "cover", "regime": False}


AGENT = AuctionAgent
