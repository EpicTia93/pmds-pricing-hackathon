"""Helpers shared by strategy families.

Like every family module this file only imports the public agent interface, so
it is copied verbatim into exported bundles next to the family module.
"""

from __future__ import annotations

from abc import abstractmethod
from math import ceil
from typing import Any

from dme_simulator.agents import BaseAgent, Decision, Listing, Observation
from dme_simulator.config import GoodsConfig

BIG = 10**6


class ParamBase(BaseAgent):
    """Merge ``DEFAULTS`` < class ``PARAMS`` < constructor overrides into ``self.p``."""

    DEFAULTS: dict[str, Any] = {}
    PARAMS: dict[str, Any] = {}

    def __init__(self, **overrides: Any) -> None:
        unknown = (set(self.PARAMS) | set(overrides)) - set(self.DEFAULTS)
        if unknown:
            raise TypeError(f"unknown strategy parameters: {sorted(unknown)}")
        self.p: dict[str, Any] = {**self.DEFAULTS, **self.PARAMS, **overrides}
        self.book = Book()
        self.reserve: dict[str, float] = {}
        self.lot_qty: dict[str, int] = {}
        self.prev_lot_qty: dict[str, int] = {}
        self.turns_seen = 0

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self.turns_seen += 1
        self.book.update(state)
        self.prev_lot_qty = dict(self.lot_qty)
        for lot in state.auction_lots:
            self.reserve[lot.good] = lot.reserve_price
            self.lot_qty[lot.good] = lot.quantity
        for good in self.reserve:
            if good not in {lot.good for lot in state.auction_lots}:
                self.lot_qty[good] = 0
        return self.act(state, goods_config)

    @abstractmethod
    def act(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        """Family-specific decision, called after the book and lots are updated."""


class Book:
    """Own identity plus per-turn deltas of the accumulated transaction logs."""

    def __init__(self) -> None:
        self.me: str | None = None
        self._a = 0
        self._s = 0
        self.full_log = False          # rival transactions are visible
        self.won: dict[str, int] = {}
        self.paid: dict[str, float] = {}
        self.sold: dict[str, int] = {}
        self.revenue: dict[str, float] = {}
        self.sold_at: dict[str, dict[float, int]] = {}
        self.rival_sales: dict[str, list[tuple[float, int]]] = {}
        self.rival_buys: dict[str, list[tuple[float, int]]] = {}

    def _mine(self, team: str) -> bool:
        return self.me is None or team == self.me

    def update(self, state: Observation) -> None:
        if self.me is None:
            matches = [
                team for team, cap in state.public_capital.items()
                if cap == state.capital and state.public_inventories.get(team) == state.inventory
            ]
            if len(matches) == 1:
                self.me = matches[0]
        self.won, self.paid, self.sold, self.revenue = {}, {}, {}, {}
        self.sold_at, self.rival_sales, self.rival_buys = {}, {}, {}
        for tx in state.auction_transactions[self._a:]:
            if self._mine(tx.buyer):
                self.won[tx.good] = self.won.get(tx.good, 0) + tx.quantity
                self.paid[tx.good] = self.paid.get(tx.good, 0.0) + tx.price * tx.quantity
            else:
                self.full_log = True
                self.rival_buys.setdefault(tx.good, []).append((tx.price, tx.quantity))
        for tx in state.sales_transactions[self._s:]:
            if self._mine(tx.seller):
                self.sold[tx.good] = self.sold.get(tx.good, 0) + tx.quantity
                self.revenue[tx.good] = self.revenue.get(tx.good, 0.0) + tx.price * tx.quantity
                at = self.sold_at.setdefault(tx.good, {})
                at[tx.price] = at.get(tx.price, 0) + tx.quantity
            else:
                self.full_log = True
                self.rival_sales.setdefault(tx.good, []).append((tx.price, tx.quantity))
        self._a = len(state.auction_transactions)
        self._s = len(state.sales_transactions)

    def rival_price(self, good: str) -> float | None:
        sales = self.rival_sales.get(good)
        if not sales:
            return None
        units = sum(q for _, q in sales)
        return sum(p * q for p, q in sales) / units


def cheapest_recipes(
    goods_config: GoodsConfig, unit_price: dict[str, float]
) -> dict[str, tuple[str, float]]:
    """Map each producible finished good to (recipe name, material cost)."""
    best: dict[str, tuple[str, float]] = {}
    for name, recipe in goods_config.recipes.items():
        if any(m not in unit_price for m in recipe.inputs):
            continue
        cost = sum(unit_price[m] * a for m, a in recipe.inputs.items())
        good = recipe.output_good
        if good not in best or cost < best[good][1]:
            best[good] = (name, cost)
    return best


def affordable(inputs, wanted: int, raw_left, price, budget: float) -> int:
    """Largest unit count <= wanted whose missing materials fit the budget."""
    def cost(units: int) -> float:
        return sum(max(0, units * a - raw_left.get(m, 0)) * price[m] for m, a in inputs.items())

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


def plan_production(
    order: list[str],
    wanted: dict[str, int],
    recipes: dict[str, tuple[str, float]],
    goods_config: GoodsConfig,
    inventory: dict[str, int],
    price: dict[str, float],
    budget: float,
    budgets: dict[str, float] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Greedy plan in priority ``order``: use stocked materials first, buy the rest.

    Returns (units planned per good, units to buy per material).
    """
    raw_left = {m: inventory.get(m, 0) for m in goods_config.raw_materials}
    buy = {m: 0 for m in goods_config.raw_materials}
    plan: dict[str, int] = {}
    for good in order:
        inputs = goods_config.recipes[recipes[good][0]].inputs
        limit = budget if budgets is None else min(budget, budgets.get(good, 0.0))
        units = affordable(inputs, wanted.get(good, 0), raw_left, price, limit)
        if units <= 0:
            continue
        for m, a in inputs.items():
            need = units * a
            used = min(raw_left[m], need)
            raw_left[m] -= used
            buy[m] += need - used
            budget -= (need - used) * price[m]
        plan[good] = units
    return plan, buy


def listed_actual(planned: int, inventory_now: int, sold: int) -> int:
    """Units that were really on offer last turn (listings are capped by stock)."""
    return min(planned, inventory_now + sold)


def split_listings(good: str, prices: list[float], weights: list[float], quantity: int,
                   slack: int = 0) -> list[Listing]:
    """Split ``quantity`` across price rungs by weight; the last rung absorbs ``slack``."""
    total = sum(weights) or 1.0
    counts = [int(quantity * w / total) for w in weights]
    remainder = quantity - sum(counts)
    for i in sorted(range(len(weights)), key=lambda i: -weights[i])[:remainder]:
        counts[i] += 1
    if counts:
        counts[-1] += slack
    return [Listing(good, c, max(p, 1e-3)) for p, c in zip(prices, counts) if c > 0]


def ceil_pos(x: float) -> int:
    return max(0, ceil(x))
