"""Portfolio family: goods as bandit arms, prices and markups by PI control.

Capital allocation: each good is an arm whose reward is its realised return on
capital, (revenue - material cost of the units sold) / cost of the units
offered. The rewards are smoothed by an EMA. Arms are scored with a UCB bonus,
and the auction budget is split by a softmax over scores, so capital flows to
the goods that actually pay.

Pricing: a proportional-integral controller on the log price of each good
steers the sell-through toward ``target_through``. Bidding: a second PI
controller per material steers the fill rate toward ``fill_target`` by
adjusting the bid markup.
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Any

from dme_simulator.agents import AssemblyOrder, Bid, Decision, Listing, Observation
from dme_simulator.config import GoodsConfig

from .common import BIG, ParamBase, ceil_pos, cheapest_recipes, listed_actual, plan_production

DEFAULTS: dict[str, Any] = {
    "init_price_mult": 4.0,
    "target_through": 0.85,   # desired share of offered units sold
    "kp": 0.4,                # proportional gain on log price
    "ki": 0.05,               # integral gain
    "integral_clip": 3.0,
    "floor_mult": 1.0,
    "roi_ema": 0.3,
    "roi_prior": 0.5,         # optimistic return assumed for untried goods
    "ucb_c": 0.5,
    "temperature": 0.3,       # softmax temperature over arm scores
    "min_score": -0.05,       # arms scoring below this get no budget
    "budget_share": 0.5,
    "stock_cover": 1.5,
    "demand_ema": 0.35,
    "sellout_boost": 0.3,
    "probe_units": 1,
    "init_spend_frac": 0.12,
    "bid_markup": 0.10,       # initial markup
    "fill_kp": 0.3,           # markup change per unit of fill-rate error
    "fill_ki": 0.05,
    "fill_target": 0.9,
    "max_markup": 2.0,
    "build_leftovers": True,
}


class PortfolioAgent(ParamBase):
    DEFAULTS = DEFAULTS

    def __init__(self, **overrides: Any) -> None:
        super().__init__(**overrides)
        self.log_price: dict[str, float] = {}
        self.integral: dict[str, float] = {}
        self.roi: dict[str, float] = {}
        self.pulls: dict[str, int] = {}
        self.demand: dict[str, float] = {}
        self.markup: dict[str, float] = {}
        self.fill_integral: dict[str, float] = {}
        self.last_listed: dict[str, int] = {}
        self.last_cost: dict[str, float] = {}
        self.last_bid: dict[str, int] = {}

    def act(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        p = self.p
        inventory = dict(state.inventory)

        # --- bid markup PI controller on fill rate --------------------------
        for m in self.reserve:
            mk = self.markup.get(m, p["bid_markup"])
            asked = self.last_bid.get(m, 0)
            if asked > 0:
                error = p["fill_target"] - self.book.won.get(m, 0) / asked
                self.fill_integral[m] = max(-3.0, min(3.0, self.fill_integral.get(m, 0.0) + error))
                mk += p["fill_kp"] * error + p["fill_ki"] * self.fill_integral[m]
            self.markup[m] = max(0.0, min(p["max_markup"], mk))
        bid_price = {m: r * (1 + self.markup[m]) + 1e-9 for m, r in self.reserve.items()}
        recipes = cheapest_recipes(goods_config, bid_price)
        recipes_res = cheapest_recipes(goods_config, self.reserve)
        goods = [g for g in goods_config.finished_goods if g in recipes]

        # --- price PI controller and arm rewards ----------------------------
        for good in goods:
            cost = recipes[good][1]
            if good not in self.log_price:
                self.log_price[good] = log(max(p["init_price_mult"] * cost, 1.0))
                share = p["init_spend_frac"] * state.capital / max(1, len(goods))
                self.demand[good] = share / max(cost, 1e-6) / max(p["stock_cover"], 1e-6)
                continue
            n_sold = self.book.sold.get(good, 0)
            offered = listed_actual(self.last_listed.get(good, 0), inventory.get(good, 0), n_sold)
            if offered <= 0:
                continue
            through = n_sold / offered
            error = through - p["target_through"]
            clip = p["integral_clip"]
            self.integral[good] = max(-clip, min(clip, self.integral.get(good, 0.0) + error))
            self.log_price[good] += p["kp"] * error + p["ki"] * self.integral[good]
            unit_cost = self.last_cost.get(good, cost)
            reward = (self.book.revenue.get(good, 0.0) - n_sold * unit_cost) / max(offered * unit_cost, 1e-9)
            a = p["roi_ema"]
            self.roi[good] = (1 - a) * self.roi.get(good, reward) + a * reward
            self.pulls[good] = self.pulls.get(good, 0) + 1
            observed = n_sold * (1 + p["sellout_boost"]) if through >= 1 else n_sold
            b = p["demand_ema"]
            self.demand[good] = (1 - b) * self.demand[good] + b * observed

        price = {}
        for good in goods:
            floor = p["floor_mult"] * recipes_res[good][1]
            self.log_price[good] = max(self.log_price[good], log(max(floor, 1e-3)))
            price[good] = exp(self.log_price[good])

        # --- bandit allocation ----------------------------------------------
        t = self.turns_seen
        score = {
            g: self.roi.get(g, p["roi_prior"])
            + p["ucb_c"] * sqrt(log(t + 1) / (self.pulls.get(g, 0) + 1))
            for g in goods
        }
        arms = [g for g in goods if score[g] >= p["min_score"] and price[g] > recipes[g][1]]
        arms.sort(key=lambda g: score[g], reverse=True)
        budget = state.capital * p["budget_share"]
        if arms:
            top = score[arms[0]]
            temp = max(p["temperature"], 1e-3)
            weights = {g: exp((score[g] - top) / temp) for g in arms}
            total = sum(weights.values())
            budgets = {g: budget * w / total for g, w in weights.items()}
        else:
            budgets = {}
        wanted = {
            g: ceil_pos(p["stock_cover"] * self.demand[g] + p["probe_units"] - inventory.get(g, 0))
            for g in arms
        }
        plan, buy = plan_production(arms, wanted, recipes, goods_config, inventory,
                                    bid_price, budget, budgets)
        lots = {lot.good: lot for lot in state.auction_lots}
        bids = [Bid(m, min(q, lots[m].quantity), bid_price[m])
                for m, q in buy.items() if q > 0 and m in lots]
        self.last_bid = {b.good: b.quantity for b in bids}
        assembly = [AssemblyOrder(recipes[g][0], plan[g]) for g in arms if plan.get(g)]
        if p["build_leftovers"]:
            assembly += [AssemblyOrder(recipes[g][0], BIG) for g in arms]

        listings = []
        self.last_listed = {}
        for good in goods:
            quantity = inventory.get(good, 0) + plan.get(good, 0)
            slack = BIG if p["build_leftovers"] and good in arms else 0
            if quantity + slack <= 0:
                continue
            listings.append(Listing(good, quantity + slack, price[good]))
            self.last_listed[good] = quantity + slack
            self.last_cost[good] = recipes[good][1]
        return Decision(bids=bids, assembly=assembly, listings=listings)


AGENT = PortfolioAgent
