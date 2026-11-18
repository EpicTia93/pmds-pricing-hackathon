"""Rival family: steer by what competitors hold and how hard they bid.

Every team's inventory and capital are public. This agent measures:

* crowding: rivals' finished stock of a good relative to the market's
  estimated per-turn sales. Crowded goods get lower production priority and a
  price undercut. When no rival holds or can build a good, it adds a
  monopoly premium.
* auction pressure: how much of each supplier lot vanished since last turn
  (other teams' buying). It is smoothed, and bid markups rise with it. Markups
  can also rise when the richest rival out-capitalises us.
* rival prices: when the full transaction log is visible, prices are pulled
  toward rivals' recent average sale price minus an undercut.

Underneath is a multiplicative sell-through price controller.
"""

from __future__ import annotations

from typing import Any

from dme_simulator.agents import AssemblyOrder, Bid, Decision, Observation
from dme_simulator.config import GoodsConfig

from .common import (
    BIG, ParamBase, ceil_pos, cheapest_recipes, listed_actual, plan_production, split_listings,
)

DEFAULTS: dict[str, Any] = {
    "init_price_mult": 4.0,
    "price_up": 0.12,
    "price_down": 0.10,
    "sellout_threshold": 0.9,
    "slow_threshold": 0.5,
    "floor_mult": 1.1,
    "monopoly_premium": 0.2,  # price uplift when no rival can supply the good
    "crowd_undercut": 0.10,   # price cut at full crowding
    "crowd_aversion": 1.0,    # production priority / (1 + aversion x crowding)
    "anchor": 0.4,            # pull toward rivals' recent sale price
    "anchor_undercut": 0.03,
    "base_markup": 0.10,
    "pressure_gain": 0.8,     # markup added per unit of auction pressure
    "pressure_ema": 0.5,
    "capital_gain": 0.0,      # markup added per unit of (richest rival / us - 1)
    "max_markup": 2.0,
    "budget_share": 0.5,
    "stock_cover": 1.3,
    "demand_ema": 0.4,
    "sellout_boost": 0.3,
    "probe_units": 1,
    "init_spend_frac": 0.12,
    "min_margin": 0.2,
    "ladder_steps": 2,
    "ladder_spread": 0.1,
    "build_leftovers": True,
}


class RivalAgent(ParamBase):
    DEFAULTS = DEFAULTS

    def __init__(self, **overrides: Any) -> None:
        super().__init__(**overrides)
        self.price: dict[str, float] = {}
        self.demand: dict[str, float] = {}
        self.pressure: dict[str, float] = {}
        self.replen: dict[str, float] = {}
        self.last_listed: dict[str, int] = {}

    def _rivals(self, state: Observation) -> list[str]:
        me = self.book.me
        teams = [t for t in state.public_inventories if t != me]
        if me is None:  # unknown identity: drop one team that looks exactly like us
            for t in teams:
                if state.public_inventories[t] == state.inventory:
                    teams.remove(t)
                    break
        return teams

    def act(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        p = self.p
        inventory = dict(state.inventory)
        rivals = self._rivals(state)
        n_teams = len(rivals) + 1

        # --- auction pressure from vanished lot stock -----------------------
        # lot_t = lot_{t-1} - ours - others + replenishment, so
        # growth = lot_t - lot_{t-1} + ours = replenishment - others. The largest
        # recent growth estimates replenishment; the shortfall is rivals' buying.
        for m, qty in self.lot_qty.items():
            before = self.prev_lot_qty.get(m)
            if not before:
                continue
            growth = qty - before + self.book.won.get(m, 0)
            self.replen[m] = max(growth, 0.9 * self.replen.get(m, growth))
            if self.book.full_log:
                others = sum(q for _, q in self.book.rival_buys.get(m, []))
            else:
                others = max(0.0, self.replen[m] - growth)
            a = p["pressure_ema"]
            signal = min(1.0, others / max(before, 1))
            self.pressure[m] = (1 - a) * self.pressure.get(m, 0.0) + a * signal
        richest = max((state.public_capital.get(t, 0.0) for t in rivals), default=0.0)
        cap_term = p["capital_gain"] * max(-1.0, min(3.0, richest / max(state.capital, 1.0) - 1.0))
        bid_price = {
            m: res * (1 + min(p["max_markup"], max(0.0, p["base_markup"]
                                                + p["pressure_gain"] * self.pressure.get(m, 0.0)
                                                + cap_term))) + 1e-9
            for m, res in self.reserve.items()
        }
        recipes = cheapest_recipes(goods_config, bid_price)
        recipes_res = cheapest_recipes(goods_config, self.reserve)
        goods = [g for g in goods_config.finished_goods if g in recipes]

        # --- crowding -------------------------------------------------------
        rival_stock = {g: sum(state.public_inventories[t].get(g, 0) for t in rivals) for g in goods}
        rival_can_build = {
            g: any(
                all(state.public_inventories[t].get(m, 0) >= a
                    for m, a in goods_config.recipes[recipes[g][0]].inputs.items())
                for t in rivals
            )
            for g in goods
        }

        for good in goods:
            cost_res = recipes_res[good][1]
            if good not in self.price:
                self.price[good] = max(p["init_price_mult"] * recipes[good][1], 1.0)
                share = p["init_spend_frac"] * state.capital / max(1, len(goods))
                self.demand[good] = share / max(recipes[good][1], 1e-6) / max(p["stock_cover"], 1e-6)
            else:
                n_sold = self.book.sold.get(good, 0)
                offered = listed_actual(self.last_listed.get(good, 0), inventory.get(good, 0), n_sold)
                if offered > 0:
                    through = n_sold / offered
                    if through >= p["sellout_threshold"]:
                        self.price[good] *= 1 + p["price_up"]
                    elif through < p["slow_threshold"]:
                        self.price[good] *= 1 - p["price_down"]
                    observed = n_sold * (1 + p["sellout_boost"]) if through >= 1 else n_sold
                    a = p["demand_ema"]
                    self.demand[good] = (1 - a) * self.demand[good] + a * observed
            anchor = self.book.rival_price(good)
            if anchor is not None and p["anchor"] > 0:
                target = anchor * (1 - p["anchor_undercut"])
                self.price[good] += p["anchor"] * (target - self.price[good])
            self.price[good] = max(self.price[good], p["floor_mult"] * cost_res, 1e-3)

        market_sales = {g: max(1.0, self.demand[g] * n_teams) for g in goods}
        crowding = {g: rival_stock[g] / market_sales[g] for g in goods}
        quote = {}
        for g in goods:
            q = self.price[g]
            if rival_stock[g] == 0 and not rival_can_build[g]:
                q *= 1 + p["monopoly_premium"]
            else:
                q *= 1 - p["crowd_undercut"] * min(1.0, crowding[g])
            quote[g] = max(q, p["floor_mult"] * recipes_res[g][1], 1e-3)

        # --- production -----------------------------------------------------
        margin = {g: self.price[g] / max(recipes[g][1], 1e-9) - 1 for g in goods}
        candidates = [g for g in goods if margin[g] >= p["min_margin"]]
        priority = {g: margin[g] / (1 + p["crowd_aversion"] * crowding[g]) for g in candidates}
        candidates.sort(key=lambda g: priority[g], reverse=True)
        wanted = {
            g: ceil_pos(p["stock_cover"] * self.demand[g] / (1 + p["crowd_aversion"] * crowding[g])
                        + p["probe_units"] - inventory.get(g, 0))
            for g in candidates
        }
        plan, buy = plan_production(candidates, wanted, recipes, goods_config, inventory,
                                    bid_price, state.capital * p["budget_share"])
        lots = {lot.good: lot for lot in state.auction_lots}
        bids = [Bid(m, min(q, lots[m].quantity), bid_price[m])
                for m, q in buy.items() if q > 0 and m in lots]
        assembly = [AssemblyOrder(recipes[g][0], plan[g]) for g in candidates if plan.get(g)]
        if p["build_leftovers"]:
            assembly += [AssemblyOrder(recipes[g][0], BIG) for g in candidates]

        # --- listings -------------------------------------------------------
        listings = []
        self.last_listed = {}
        steps = max(1, int(p["ladder_steps"]))
        for good in goods:
            quantity = inventory.get(good, 0) + plan.get(good, 0)
            slack = BIG if p["build_leftovers"] and good in candidates else 0
            if quantity + slack <= 0:
                continue
            center = quote[good]
            if steps == 1:
                prices = [center]
            else:
                prices = [center * (1 + p["ladder_spread"] * (2 * i / (steps - 1) - 1)) for i in range(steps)]
            floor = p["floor_mult"] * recipes_res[good][1]
            prices = [max(x, floor, 1e-3) for x in prices]
            listings += split_listings(good, prices, [1.0] * steps, quantity, slack)
            self.last_listed[good] = quantity + slack
        return Decision(bids=bids, assembly=assembly, listings=listings)


AGENT = RivalAgent
