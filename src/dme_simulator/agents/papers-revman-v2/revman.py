"""Revenue-management family: learn demand, value stock by a finite-horizon DP.

Sources: Gallego & van Ryzin (1994) finite-horizon pricing of a stock with no
salvage value; Gallego & Hu (2014) the same under competition; Gallego, Huh,
Kang & Phillips (2006) logit price competition; Besbes & Zeevi (2009) and
den Boer (2015) pricing while learning a parametric demand model.

The selling side is the core of this family. Demand per good is a logistic
curve in log price, learned online. Each unit on the shelf is valued by a DP
over the remaining turns, where a lognormal prior on the unknown game length
gives the odds that each turn is played. The listing price comes from the DP
(``dp``), the static logit rule with the replacement cost as marginal cost
(``myopic``), or a fixed price that sells the stock by the expected end
(``runout``). It buys a unit when its marginal value beats the kit cost by
``margin``, and bids the kit-level best response on the learned clearing prices. Evolution tunes the priors,
forgetting, horizon prior, ladder and margins.
"""

from __future__ import annotations

from typing import Any

from .econ import Engine

HELPERS = ("econ.py",)

DEFAULTS: dict[str, Any] = {
    "pricing": "dp",
    "listing": "units",
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
    "basis_floor": 0.0,
    "runout_frac": 1.0,
    "margin": 0.3,
    "budget_share": 0.5,
    "stop_turns": 1.0,
    "leftover_margin": 0.3,
    "max_lot_share": 0.8,
    "clear0": 0.3,
    "clear_sd0": 0.4,
    "clear_rate": 0.35,
    "uncontested0": 0.4,
    "salvage": 0.4,
}


class RevmanAgent(Engine):
    DEFAULTS = DEFAULTS
    FIXED = {"quantity": "value", "bidding": "kit", "kits": "independent",
             "regime": False, "pacing_gain": 0.0}


AGENT = RevmanAgent
