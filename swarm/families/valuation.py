"""Valuation family: learn each good's demand curve, bid from the value of output.

Pricing: listings are spread over a grid of log-spaced price bins around a
moving centre. For every bin the agent keeps decayed counts of units listed and
sold, giving an acceptance rate with an optimistic prior and a UCB-style
exploration bonus. Expected profit per listed unit is (price - cost) x
acceptance. The stock is split over bins by a softmax of that profit, and the
centre drifts toward the best bin, so the ladder keeps learning.

Bidding: the expected realised price of each good (from the curve) sets a
ceiling on what its materials are worth, ``value_share`` x expected price.
Each material's bid lies between the reserve and that ceiling, at the fraction
set by ``aggression``. Goods whose ceiling is below the reserve cost are not
bought.
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Any

from dme_simulator.agents import AssemblyOrder, Bid, Decision, Observation
from dme_simulator.config import GoodsConfig

from .common import (
    BIG, ParamBase, ceil_pos, cheapest_recipes, listed_actual, plan_production, split_listings,
)

DEFAULTS: dict[str, Any] = {
    "bins": 7,                # price rungs on the grid
    "width": 0.6,             # grid half-width in log-price
    "init_price_mult": 5.0,   # first grid centre = mult x material cost
    "memory": 0.85,           # per-turn decay of listing statistics
    "prior_accept": 0.6,      # optimistic prior acceptance rate
    "prior_weight": 2.0,      # pseudo-counts behind the prior
    "explore_bonus": 0.3,     # UCB bonus on rarely tried bins
    "temperature": 0.2,       # softmax temperature (relative to best profit)
    "recenter_rate": 0.5,     # how fast the grid centre follows the best bin
    "upside_decay": 0.6,      # acceptance factor per bin above the highest price that sold
    "cost_floor": 1.0,        # no rung below cost_floor x reserve cost
    "demand_ema": 0.4,
    "sellout_boost": 0.3,
    "stock_cover": 1.5,
    "probe_units": 2,
    "init_spend_frac": 0.1,
    "min_profit": 0.1,        # produce if expected price >= cost x (1 + min_profit)
    "value_share": 0.7,       # materials may cost up to this share of expected price
    "aggression": 0.3,        # bid position between reserve (0) and ceiling (1)
    "budget_share": 0.5,
    "list_fraction": 1.0,
    "build_leftovers": True,
}


class ValuationAgent(ParamBase):
    DEFAULTS = DEFAULTS

    def __init__(self, **overrides: Any) -> None:
        super().__init__(**overrides)
        self.center: dict[str, float] = {}           # log price
        self.stats: dict[str, dict[int, list[float]]] = {}
        self.last_rungs: dict[str, list[tuple[int, float, int]]] = {}
        self.demand: dict[str, float] = {}
        self.top_sold: dict[str, int] = {}           # highest bin with a sale

    # ------------------------------------------------------------------
    def _step(self) -> float:
        return 2.0 * self.p["width"] / max(1, int(self.p["bins"]) - 1)

    def _learn(self, good: str, inventory: dict[str, int]) -> None:
        p = self.p
        stats = self.stats.setdefault(good, {})
        for k in list(stats):
            stats[k][0] *= p["memory"]
            stats[k][1] *= p["memory"]
            if stats[k][0] < 0.05:
                del stats[k]
        rungs = self.last_rungs.get(good)
        if not rungs:
            return
        sold_at = self.book.sold_at.get(good, {})
        n_sold = self.book.sold.get(good, 0)
        on_offer = listed_actual(sum(q for _, _, q in rungs), inventory.get(good, 0), n_sold)
        remaining = on_offer
        for k, price, qty in rungs:
            listed = min(qty, remaining)
            remaining -= listed
            if listed <= 0:
                continue
            cell = stats.setdefault(k, [0.0, 0.0])
            cell[0] += listed
            hits = min(listed, sold_at.get(price, 0))
            cell[1] += hits
            if hits > 0:
                self.top_sold[good] = max(self.top_sold.get(good, k), k)
        if on_offer > 0:
            observed = n_sold * (1 + p["sellout_boost"]) if n_sold >= on_offer else n_sold
            a = p["demand_ema"]
            self.demand[good] = (1 - a) * self.demand.get(good, observed) + a * observed

    def _curve(self, good: str, cost: float, floor: float):
        """Return grid rows (k, price, acceptance, expected profit per unit)."""
        p = self.p
        step = self._step()
        k_center = round(self.center[good] / step)
        half = int(p["bins"]) // 2
        stats = self.stats.get(good, {})

        def raw(k: int) -> float:
            listed, sold = stats.get(k, (0.0, 0.0))
            accept = (sold + p["prior_accept"] * p["prior_weight"]) / (listed + p["prior_weight"])
            return min(1.0, accept + p["explore_bonus"] / sqrt(1.0 + listed))

        # Demand can only fall with price: no bin may look better than any
        # cheaper bin that has real observations (prevents optimistic runaway).
        observed = sorted(k for k, cell in stats.items() if cell[0] >= 0.5)
        top = self.top_sold.get(good, round(self.center[good] / step) if good in self.center else 0)
        rows = []
        for k in range(k_center - half, k_center - half + int(p["bins"])):
            price = exp(k * step)
            if price < floor:
                continue
            accept = raw(k)
            for j in observed:
                if j > k:
                    break
                accept = min(accept, raw(j))
            if k > top:
                accept *= p["upside_decay"] ** (k - top)
            rows.append((k, price, accept, (price - cost) * accept))
        if not rows:  # whole grid below the floor: list at the floor
            k = round(log(max(floor, 1e-3)) / step) + 1
            rows.append((k, exp(k * step), p["prior_accept"], exp(k * step) - cost))
        return rows

    # ------------------------------------------------------------------
    def act(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        p = self.p
        inventory = dict(state.inventory)
        reserve = self.reserve
        recipes_res = cheapest_recipes(goods_config, reserve)
        goods = [g for g in goods_config.finished_goods if g in recipes_res]

        for good in goods:
            cost_res = recipes_res[good][1]
            if good not in self.center:
                self.center[good] = log(max(p["init_price_mult"] * cost_res, 1.0))
                self.top_sold[good] = round(self.center[good] / self._step())
                share = p["init_spend_frac"] * state.capital / max(1, len(goods))
                self.demand[good] = share / max(cost_res, 1e-6) / max(p["stock_cover"], 1e-6)
            else:
                self._learn(good, inventory)

        # Curves, expected prices and material ceilings.
        curves, expected, ceiling = {}, {}, {}
        for good in goods:
            cost_res = recipes_res[good][1]
            rows = self._curve(good, cost_res, p["cost_floor"] * cost_res)
            best = max(rows, key=lambda r: r[3])
            self.center[good] += p["recenter_rate"] * (best[0] * self._step() - self.center[good])
            scale = max(abs(best[3]), 1e-9) * max(p["temperature"], 1e-3)
            weights = [exp((r[3] - best[3]) / scale) for r in rows]
            curves[good] = (rows, weights)
            mass = sum(w * r[2] for w, r in zip(weights, rows)) or 1e-9
            expected[good] = sum(w * r[2] * r[1] for w, r in zip(weights, rows)) / mass
            ceiling[good] = p["value_share"] * expected[good] / max(cost_res, 1e-9)

        bid_price: dict[str, float] = {}
        for m, res in reserve.items():
            users = [
                ceiling[g] for g in goods
                if m in goods_config.recipes[recipes_res[g][0]].inputs
            ]
            top = max(users, default=0.0)
            if top > 1.0:
                bid_price[m] = res * (1.0 + p["aggression"] * (top - 1.0)) + 1e-9
        recipes_bid = cheapest_recipes(goods_config, {**reserve, **bid_price})

        candidates = [
            g for g in goods
            if g in recipes_bid
            and all(m in bid_price or inventory.get(m, 0) > 0
                    for m in goods_config.recipes[recipes_bid[g][0]].inputs)
            and expected[g] >= recipes_bid[g][1] * (1 + p["min_profit"])
        ]
        candidates.sort(key=lambda g: expected[g] / max(recipes_bid[g][1], 1e-9), reverse=True)
        wanted = {
            g: ceil_pos(p["stock_cover"] * self.demand[g] + p["probe_units"] - inventory.get(g, 0))
            for g in candidates
        }
        price_for_plan = {m: bid_price.get(m, float("inf")) for m in goods_config.raw_materials}
        plan, buy = plan_production(
            candidates, wanted, recipes_bid, goods_config, inventory, price_for_plan,
            state.capital * p["budget_share"],
        )

        lots = {lot.good: lot for lot in state.auction_lots}
        bids = [
            Bid(m, min(q, lots[m].quantity), bid_price[m])
            for m, q in buy.items() if q > 0 and m in lots and m in bid_price
        ]
        assembly = [AssemblyOrder(recipes_bid[g][0], plan[g]) for g in candidates if plan.get(g)]
        if p["build_leftovers"]:
            assembly += [AssemblyOrder(recipes_bid[g][0], BIG) for g in candidates]

        listings = []
        self.last_rungs = {}
        for good in goods:
            quantity = ceil_pos((inventory.get(good, 0) + plan.get(good, 0)) * p["list_fraction"])
            slack = BIG if p["build_leftovers"] and good in candidates else 0
            if quantity + slack <= 0:
                continue
            rows, weights = curves[good]
            rungs = split_listings(good, [r[1] for r in rows], weights, quantity, slack)
            listings += rungs
            by_price = {r[1]: r[0] for r in rows}
            self.last_rungs[good] = [(by_price.get(l.price, 0), l.price, l.quantity) for l in rungs]
        return Decision(bids=bids, assembly=assembly, listings=listings)


AGENT = ValuationAgent
