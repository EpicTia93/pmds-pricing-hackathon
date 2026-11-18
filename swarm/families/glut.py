"""Glut family: time the auction, buy in batches when materials are cheap.

Reserve prices fall as unsold supplier stock piles up. This agent tracks each
material's highest reserve seen and only buys when the current reserve is at
most ``buy_threshold`` x that reference, bidding barely over the reserve. Then
it buys ``batch_turns`` worth of expected usage in one go. If raw stock covers
fewer than ``safety_turns`` of usage it buys at any price, with
``panic_markup``.

Selling uses two tiers per good: a premium tranche whose price follows its own
sell-through (skimming), and a value tranche at ``tier_ratio`` x premium that
moves the bulk of the stock.
"""

from __future__ import annotations

from typing import Any

from dme_simulator.agents import AssemblyOrder, Bid, Decision, Listing, Observation
from dme_simulator.config import GoodsConfig

from .common import ParamBase, ceil_pos, cheapest_recipes

DEFAULTS: dict[str, Any] = {
    "buy_threshold": 0.85,    # buy when reserve <= threshold x highest reserve seen
    "bid_markup": 0.01,       # markup over reserve on opportunistic buys
    "safety_turns": 1.0,      # raw cover below this triggers a panic buy
    "panic_markup": 0.3,
    "batch_turns": 3.0,       # raw cover bought when cheap
    "max_lot_share": 0.8,
    "budget_share": 0.6,
    "production_rate": 1.0,   # finished units built per turn = rate x demand
    "stock_cover": 1.5,       # finished-goods cap = cover x demand
    "demand_ema": 0.35,
    "sellout_boost": 0.4,
    "init_spend_frac": 0.15,
    "min_margin": 0.15,
    "init_price_mult": 4.0,
    "premium_share": 0.4,     # share of listed stock in the premium tranche
    "tier_ratio": 0.75,       # value price = ratio x premium price
    "premium_first": False,   # which tranche gets inventory priority
    "skim_bump": 0.15,        # premium up after selling out its tranche
    "skim_decay": 0.08,       # premium down after weak premium sales
    "through_target": 0.6,    # premium sell-through below this decays the price
    "value_cut": 0.1,         # extra premium cut when even the value tier stalls
    "floor_mult": 1.0,
}


class GlutAgent(ParamBase):
    DEFAULTS = DEFAULTS

    def __init__(self, **overrides: Any) -> None:
        super().__init__(**overrides)
        self.ref: dict[str, float] = {}
        self.premium: dict[str, float] = {}
        self.demand: dict[str, float] = {}
        self.last: dict[str, tuple[float, int, float, int]] = {}  # premium p,q / value p,q

    def _reprice(self, good: str, inventory: dict[str, int]) -> None:
        p = self.p
        if good not in self.last:
            return
        prem_p, prem_q, val_p, val_q = self.last[good]
        sold_at = self.book.sold_at.get(good, {})
        n_sold = self.book.sold.get(good, 0)
        on_offer = min(prem_q + val_q, inventory.get(good, 0) + n_sold)
        if on_offer <= 0:
            return
        prem_sold = sold_at.get(prem_p, 0)
        val_sold = sold_at.get(val_p, 0)
        price = self.premium[good]
        if prem_q > 0 and prem_sold >= prem_q:
            price *= 1 + p["skim_bump"]
        elif prem_q == 0 or prem_sold < p["through_target"] * prem_q:
            price *= 1 - p["skim_decay"]
        if val_q > 0 and val_sold < 0.5 * min(val_q, on_offer):
            price *= 1 - p["value_cut"]
        self.premium[good] = price
        observed = n_sold * (1 + p["sellout_boost"]) if n_sold >= on_offer else n_sold
        a = p["demand_ema"]
        self.demand[good] = (1 - a) * self.demand.get(good, observed) + a * observed

    def act(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        p = self.p
        inventory = dict(state.inventory)
        for m, res in self.reserve.items():
            self.ref[m] = max(self.ref.get(m, 0.0), res)
        recipes = cheapest_recipes(goods_config, self.reserve)
        goods = [g for g in goods_config.finished_goods if g in recipes]

        for good in goods:
            cost = recipes[good][1]
            if good not in self.premium:
                self.premium[good] = max(p["init_price_mult"] * cost, 1.0)
                share = p["init_spend_frac"] * state.capital / max(1, len(goods))
                self.demand[good] = share / max(cost, 1e-6) / max(p["stock_cover"], 1e-6)
            else:
                self._reprice(good, inventory)
            self.premium[good] = max(self.premium[good], p["floor_mult"] * cost / max(p["tier_ratio"], 0.05))

        # Expected realised price ~ blend of the two tiers.
        blend = {
            g: self.premium[g] * (p["premium_share"] + (1 - p["premium_share"]) * p["tier_ratio"])
            for g in goods
        }
        profitable = [g for g in goods if blend[g] >= recipes[g][1] * (1 + p["min_margin"])]
        profitable.sort(key=lambda g: blend[g] / max(recipes[g][1], 1e-9), reverse=True)

        # Material usage rate implied by the production plan.
        usage = {m: 0.0 for m in goods_config.raw_materials}
        for g in profitable:
            for m, a in goods_config.recipes[recipes[g][0]].inputs.items():
                usage[m] += a * p["production_rate"] * self.demand[g]

        # Production this turn: rate x demand, capped by the finished-stock target.
        wanted = {
            g: min(
                ceil_pos(p["production_rate"] * self.demand[g]),
                ceil_pos(p["stock_cover"] * self.demand[g] + 1 - inventory.get(g, 0)),
            )
            for g in profitable
        }

        # Bids: opportunistic batches when cheap, panic buys when short.
        lots = {lot.good: lot for lot in state.auction_lots}
        budget = state.capital * p["budget_share"]
        bids = []
        for m in sorted(lots, key=lambda m: -usage.get(m, 0.0)):
            lot, use = lots[m], usage.get(m, 0.0)
            if use <= 0 or lot.quantity <= 0:
                continue
            stock = inventory.get(m, 0)
            cheap = lot.reserve_price <= p["buy_threshold"] * self.ref[m]
            if cheap:
                target, markup = p["batch_turns"] * use, p["bid_markup"]
            elif stock < p["safety_turns"] * use:
                target, markup = p["safety_turns"] * use, p["panic_markup"]
            else:
                continue
            price = lot.reserve_price * (1 + markup) + 1e-9
            qty = min(ceil_pos(target - stock), int(p["max_lot_share"] * lot.quantity),
                      int(budget / max(price, 1e-9)))
            if qty > 0:
                bids.append(Bid(m, qty, price))
                budget -= qty * price

        # Assembly in priority order; the simulator caps it by materials held.
        assembly = [AssemblyOrder(recipes[g][0], wanted[g]) for g in profitable if wanted[g] > 0]

        listings = []
        self.last = {}
        for good in goods:
            expected = inventory.get(good, 0) + wanted.get(good, 0)
            if expected <= 0:
                continue
            prem_q = int(round(expected * p["premium_share"]))
            val_q = expected - prem_q
            prem_p = self.premium[good]
            val_p = max(prem_p * p["tier_ratio"], 1e-3)
            tiers = [(prem_p, prem_q), (val_p, val_q)]
            if not p["premium_first"]:
                tiers.reverse()
            listings += [Listing(good, q, pr) for pr, q in tiers if q > 0]
            self.last[good] = (prem_p, prem_q, val_p, val_q)
        return Decision(bids=bids, assembly=assembly, listings=listings)


AGENT = GlutAgent
