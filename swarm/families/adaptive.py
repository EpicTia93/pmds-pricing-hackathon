"""Adaptive family: cost-plus pricing steered by sell-through (the original strategy).

This module is self-contained: it imports only the public agent interface, so
it can be copied verbatim into a submission bundle (see ``swarm export``).

Every behavioural knob lives in ``DEFAULT_PARAMS``; ``ParamAgent(**overrides)``
builds an agent with any subset overridden. The search space for each knob is
declared in ``swarm/space.py``.
"""

from __future__ import annotations

from math import ceil
from typing import Any

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig

DEFAULT_PARAMS: dict[str, Any] = {
    # --- pricing -----------------------------------------------------------
    "init_price_mult": 4.0,      # first price = mult x material cost
    "price_up": 0.10,            # raise after selling out
    "price_down": 0.10,          # lower after slow sales
    "zero_sale_down": 0.15,      # extra cut when nothing sold at all
    "sellout_threshold": 0.95,   # sell-through counted as "sold out"
    "slow_threshold": 0.50,      # sell-through below this lowers the price
    "floor_mult": 1.10,          # never list below floor_mult x current material cost
    "ladder_steps": 3,           # number of price rungs per good
    "ladder_spread": 0.15,       # rungs span price x (1 +- spread)
    "ladder_low_first": True,    # cheap rungs get inventory priority
    "list_fraction": 1.0,        # share of stock offered each turn
    "market_anchor": 0.0,        # blend toward rivals' recent prices (needs full log)
    "undercut": 0.05,            # discount applied to the rivals' anchor
    # --- demand & production ----------------------------------------------
    "init_spend_frac": 0.15,     # capital share planned for first-turn probing
    "demand_ema": 0.40,          # smoothing of observed demand
    "sellout_boost": 0.30,       # demand uplift when sold out (censored demand)
    "stock_cover": 1.20,         # target stock = cover x estimated demand
    "probe_units": 1,            # always keep this many units per profitable good
    "min_margin": 0.20,          # produce only if price >= cost x (1 + margin)
    "allocation": "greedy",      # greedy | proportional budget split across goods
    "margin_power": 1.0,         # proportional weights ~ margin ** power
    "build_leftovers": True,     # turn spare materials into profitable goods
    # --- bidding -----------------------------------------------------------
    "budget_share": 0.50,        # max capital committed per auction
    "bid_markup": 0.05,          # initial markup over reserve price
    "bid_adapt": 0.05,           # markup step when fill rate misses the target
    "fill_target": 0.90,         # desired share of bid quantity won
    "max_bid_markup": 0.60,      # markup cap
    "cheap_ratio": 0.80,         # reserve <= ratio x best-seen price counts as cheap
    "cheap_extra": 0.0,          # extra quantity bought when materials are cheap
    "raw_buffer": 0.0,           # extra raw safety stock relative to need
    "max_lot_share": 1.0,        # cap bid quantity at share of the lot
    # --- endgame -----------------------------------------------------------
    "horizon": 0,                # assumed final turn; 0 disables endgame logic
    "endgame_turns": 3,          # stop buying this many turns before horizon
    "endgame_discount": 0.15,    # per-turn price cut during the endgame
}

_BIG = 10**6


class ParamAgent(BaseAgent):
    """Adaptive cost-plus / sell-through strategy driven by ``DEFAULT_PARAMS``."""

    PARAMS: dict[str, Any] = {}

    def __init__(self, **overrides: Any) -> None:
        params = dict(DEFAULT_PARAMS)
        params.update(self.PARAMS)
        unknown = set(overrides) - set(DEFAULT_PARAMS)
        if unknown:
            raise TypeError(f"unknown strategy parameters: {sorted(unknown)}")
        params.update(overrides)
        self.p: dict[str, Any] = params
        self._me: str | None = None
        self._auction_seen = 0
        self._sales_seen = 0
        self._price: dict[str, float] = {}
        self._demand: dict[str, float] = {}
        self._markup: dict[str, float] = {}
        self._best_reserve: dict[str, float] = {}
        self._last_reserve: dict[str, float] = {}
        self._last_bid_qty: dict[str, int] = {}
        self._last_listed: dict[str, int] = {}

    # ------------------------------------------------------------------
    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        p = self.p
        self._identify(state)
        sold, won, rival_price = self._digest(state)
        inventory = dict(state.inventory)

        lots = {lot.good: lot for lot in state.auction_lots}
        for good, lot in lots.items():
            self._last_reserve[good] = lot.reserve_price
            self._best_reserve[good] = max(
                self._best_reserve.get(good, 0.0), lot.reserve_price
            )

        # Adapt bid markups to last turn's fill rate.
        for material in goods_config.raw_materials:
            markup = self._markup.get(material, float(p["bid_markup"]))
            asked = self._last_bid_qty.get(material, 0)
            if asked > 0:
                fill = won.get(material, 0) / asked
                if fill < p["fill_target"]:
                    markup += p["bid_adapt"]
                else:
                    markup -= 0.5 * p["bid_adapt"]
            self._markup[material] = min(max(markup, 0.0), p["max_bid_markup"])

        bid_price = {
            m: self._last_reserve[m] * (1.0 + self._markup[m]) + 1e-9
            for m in self._last_reserve
        }

        # Cheapest recipe per finished good, valued at bid and reserve prices.
        recipe_for: dict[str, str] = {}
        cost_bid: dict[str, float] = {}
        cost_res: dict[str, float] = {}
        for name, recipe in goods_config.recipes.items():
            try:
                cb = sum(bid_price[m] * a for m, a in recipe.inputs.items())
                cr = sum(self._last_reserve[m] * a for m, a in recipe.inputs.items())
            except KeyError:
                continue
            good = recipe.output_good
            if good not in cost_bid or cb < cost_bid[good]:
                recipe_for[good], cost_bid[good], cost_res[good] = name, cb, cr

        endgame = (
            p["horizon"] > 0 and state.turn > p["horizon"] - p["endgame_turns"]
        )
        self._update_prices(
            state, goods_config, inventory, sold, rival_price, cost_bid, cost_res,
            endgame,
        )

        # ---- production plan -------------------------------------------
        capital = state.capital
        goods = [g for g in goods_config.finished_goods if g in recipe_for]
        for good in goods:
            if good not in self._demand:
                share = p["init_spend_frac"] * capital / max(1, len(goods))
                self._demand[good] = share / max(cost_bid[good], 1e-6) / max(
                    p["stock_cover"], 1e-6
                )

        margin = {
            g: self._price[g] / max(cost_bid[g], 1e-9) - 1.0 for g in goods
        }
        candidates = [] if endgame else [
            g for g in goods if margin[g] >= p["min_margin"]
        ]
        candidates.sort(key=lambda g: margin[g], reverse=True)

        need = {
            g: max(
                0,
                ceil(p["stock_cover"] * self._demand[g]) + int(p["probe_units"])
                - inventory.get(g, 0),
            )
            for g in candidates
        }
        budget = capital * p["budget_share"]
        raw_left = {m: inventory.get(m, 0) for m in goods_config.raw_materials}
        buy: dict[str, int] = {m: 0 for m in goods_config.raw_materials}
        plan: dict[str, int] = {}

        if p["allocation"] == "proportional" and candidates:
            weights = {
                g: max(margin[g], 1e-6) ** p["margin_power"] for g in candidates
            }
            total = sum(weights.values())
            budgets = {g: budget * weights[g] / total for g in candidates}
        else:
            budgets = None

        for good in candidates:
            inputs = goods_config.recipes[recipe_for[good]].inputs
            limit = budget if budgets is None else min(budget, budgets[good])
            units = self._affordable(inputs, need[good], raw_left, bid_price, limit)
            if units <= 0:
                continue
            spent = 0.0
            for m, a in inputs.items():
                required = units * a
                from_stock = min(raw_left[m], required)
                raw_left[m] -= from_stock
                buy[m] += required - from_stock
                spent += (required - from_stock) * bid_price[m]
            budget -= spent
            plan[good] = units

        # ---- bids ------------------------------------------------------
        bids = []
        bid_qty: dict[str, int] = {}
        for material, lot in lots.items():
            qty = buy.get(material, 0)
            if qty > 0:
                qty = ceil(qty * (1.0 + p["raw_buffer"]))
            best = self._best_reserve.get(material, lot.reserve_price)
            if (
                not endgame
                and p["cheap_extra"] > 0
                and lot.reserve_price <= p["cheap_ratio"] * best
                and candidates
            ):
                qty += ceil(p["cheap_extra"] * max(qty, self._typical_use(
                    material, goods_config, candidates
                )))
            qty = min(qty, int(p["max_lot_share"] * lot.quantity))
            if qty > 0:
                bids.append(Bid(material, qty, bid_price[material]))
                bid_qty[material] = qty
        self._last_bid_qty = bid_qty

        # ---- assembly --------------------------------------------------
        assembly = [
            AssemblyOrder(recipe_for[g], plan[g]) for g in candidates if plan.get(g)
        ]
        if p["build_leftovers"] and not endgame:
            for g in candidates:
                assembly.append(AssemblyOrder(recipe_for[g], _BIG))

        # ---- listings --------------------------------------------------
        listings = []
        listed: dict[str, int] = {}
        for good in goods_config.finished_goods:
            expected = inventory.get(good, 0) + plan.get(good, 0)
            if good not in self._price:
                continue
            quantity = ceil(expected * p["list_fraction"])
            slack = _BIG if p["build_leftovers"] and p["list_fraction"] >= 1.0 else 0
            if quantity <= 0 and slack == 0:
                continue
            rungs = self._ladder(good, quantity, slack, cost_res.get(good, 0.0), endgame)
            listings.extend(rungs)
            listed[good] = quantity + slack
        self._last_listed = listed
        return Decision(bids=bids, assembly=assembly, listings=listings)

    # ------------------------------------------------------------------
    def _identify(self, state: Observation) -> None:
        if self._me is not None:
            return
        matches = [
            team
            for team, capital in state.public_capital.items()
            if capital == state.capital
            and state.public_inventories.get(team) == state.inventory
        ]
        if len(matches) == 1:
            self._me = matches[0]

    def _is_me(self, team: str) -> bool:
        return self._me is None or team == self._me

    def _digest(self, state: Observation):
        """Summarise transactions that appeared since the previous turn."""
        won: dict[str, int] = {}
        for tx in state.auction_transactions[self._auction_seen:]:
            if self._is_me(tx.buyer):
                won[tx.good] = won.get(tx.good, 0) + tx.quantity
        self._auction_seen = len(state.auction_transactions)

        sold: dict[str, int] = {}
        rival_value: dict[str, float] = {}
        rival_units: dict[str, int] = {}
        for tx in state.sales_transactions[self._sales_seen:]:
            if self._is_me(tx.seller):
                sold[tx.good] = sold.get(tx.good, 0) + tx.quantity
            else:
                rival_value[tx.good] = rival_value.get(tx.good, 0.0) + tx.price * tx.quantity
                rival_units[tx.good] = rival_units.get(tx.good, 0) + tx.quantity
        self._sales_seen = len(state.sales_transactions)
        rival_price = {g: rival_value[g] / rival_units[g] for g in rival_units}
        return sold, won, rival_price

    def _update_prices(
        self, state, goods_config, inventory, sold, rival_price, cost_bid, cost_res,
        endgame,
    ) -> None:
        p = self.p
        for good in goods_config.finished_goods:
            floor = p["floor_mult"] * cost_res.get(good, 0.0)
            if good not in self._price:
                if good not in cost_bid:
                    continue
                self._price[good] = max(p["init_price_mult"] * cost_bid[good], 1e-3)
                continue
            price = self._price[good]
            n_sold = sold.get(good, 0)
            planned = self._last_listed.get(good, 0)
            actually_listed = min(planned, inventory.get(good, 0) + n_sold)
            if actually_listed > 0:
                through = n_sold / actually_listed
                if through >= p["sellout_threshold"]:
                    price *= 1.0 + p["price_up"]
                    observed = n_sold * (1.0 + p["sellout_boost"])
                elif through < p["slow_threshold"]:
                    cut = p["price_down"] + (p["zero_sale_down"] if n_sold == 0 else 0.0)
                    price *= max(0.05, 1.0 - cut)
                    observed = n_sold
                else:
                    observed = n_sold
                alpha = p["demand_ema"]
                self._demand[good] = (1 - alpha) * self._demand.get(good, observed) + alpha * observed
            if p["market_anchor"] > 0 and good in rival_price:
                target = rival_price[good] * (1.0 - p["undercut"])
                price += p["market_anchor"] * (target - price)
            if endgame:
                price *= 1.0 - p["endgame_discount"]
            else:
                price = max(price, floor)
            self._price[good] = max(price, 1e-3)

    def _ladder(self, good, quantity, slack, unit_cost, endgame) -> list[Listing]:
        p = self.p
        steps = max(1, int(p["ladder_steps"]))
        center = self._price[good]
        spread = p["ladder_spread"]
        floor = 0.0 if endgame else p["floor_mult"] * unit_cost
        if steps == 1:
            prices = [center]
        else:
            prices = [
                center * (1.0 + spread * (2.0 * i / (steps - 1) - 1.0))
                for i in range(steps)
            ]
        prices = [max(price, floor, 1e-3) for price in prices]
        if not p["ladder_low_first"]:
            prices.reverse()
        base, extra = divmod(quantity, steps)
        listings = []
        for i, price in enumerate(prices):
            qty = base + (1 if i < extra else 0)
            if i == steps - 1:
                qty += slack
            if qty > 0:
                listings.append(Listing(good, qty, price))
        return listings

    @staticmethod
    def _affordable(inputs, wanted, raw_left, bid_price, budget) -> int:
        """Largest unit count <= wanted whose missing materials fit the budget."""
        def cost(units: int) -> float:
            return sum(
                max(0, units * a - raw_left.get(m, 0)) * bid_price[m]
                for m, a in inputs.items()
            )

        if wanted <= 0:
            return 0
        if cost(wanted) <= budget:
            return wanted
        low, high = 0, wanted
        while low < high:
            mid = (low + high + 1) // 2
            if cost(mid) <= budget:
                low = mid
            else:
                high = mid - 1
        return low

    def _typical_use(self, material, goods_config, candidates) -> int:
        total = 0.0
        for good in candidates:
            for recipe in goods_config.recipes.values():
                if recipe.output_good == good:
                    total += recipe.inputs.get(material, 0) * self._demand.get(good, 0.0)
                    break
        return ceil(total)


AGENT = ParamAgent
DEFAULTS = DEFAULT_PARAMS
