"""Adaptive market maker for simultaneous first-price auctions and retail sales."""
from __future__ import annotations

from collections import defaultdict
from math import ceil, exp, sqrt

from dme_simulator.agents import AssemblyOrder, BaseAgent, Bid, Decision, Listing


# Retail prices and market sizes from the public example, used only as priors.
# Every good has an independently updated retail price and sales estimate.
HARDWARE = {
    "SMARTWATCH": (118, 5.6),
    "PHONE": (205, 7.8),
    "TABLET": (273, 4.4),
    "CONSOLE": (475, 2.9),
    "LAPTOP": (650, 3.8),
}


class ClearingBidder(BaseAgent):
    def __init__(self):
        self.identity = None
        self.buys_seen = self.sales_seen = 0
        self.markup = {}
        self.bid_ladder = {}
        self.shelf = {}
        self.price = {}
        self.demand = {}
        self.initial_price = {}
        self.initial_volume = {}
        self.no_sales = defaultdict(int)
        self.initial_capital = None
        self.concentrated = False
        self.lifetime_sales = defaultdict(int)
        self.rival_price = {}

    def observe_trades(self, state):
        matches = [t for t, cash in state.public_capital.items()
                   if cash == state.capital and state.public_inventories[t] == state.inventory]
        if len(matches) == 1:
            self.identity = matches[0]
        purchases = defaultdict(lambda: defaultdict(int))
        units_sold = defaultdict(int)
        for tx in state.auction_transactions[self.buys_seen:]:
            if tx.buyer == self.identity or (self.identity is None and tx.price in self.bid_ladder.get(tx.good, {})):
                purchases[tx.good][tx.price] += tx.quantity
        for tx in state.sales_transactions[self.sales_seen:]:
            if tx.seller == self.identity or (self.identity is None and tx.price in self.shelf.get(tx.good, ())):
                units_sold[tx.good] += tx.quantity
            elif self.identity is not None:
                self.rival_price[tx.good] = .9 * self.rival_price.get(tx.good, tx.price) + .1 * tx.price
        self.buys_seen = len(state.auction_transactions)
        self.sales_seen = len(state.sales_transactions)
        self.update_clearing_prices(purchases)
        self.update_retail_prices(state.inventory, units_sold)

    def update_clearing_prices(self, purchases):
        # A ladder gives several censored clearing-price observations in a
        # single auction. Update in log markup so reserve changes transfer.
        for material, ladder in self.bid_ladder.items():
            lower, upper = 0.0, float("inf")
            reserve = self.last_reserve[material]
            for bid, quantity in ladder.items():
                filled = purchases[material].get(bid, 0)
                if filled < quantity:
                    lower = max(lower, bid / reserve)
                if filled:
                    upper = min(upper, bid / reserve)
            if lower == 0:
                estimate = max(1.001, upper * .75)
            elif upper == float("inf"):
                estimate = lower * 1.32
            else:
                estimate = sqrt(lower * upper)
            self.markup[material] = max(1.003, min(12., estimate ** .72 * self.markup[material] ** .28))

    def update_retail_prices(self, inventory, units_sold):
        for good in self.shelf:
            sales = units_sold[good]
            self.lifetime_sales[good] += sales
            end = inventory.get(good, 0)
            on_offer = sales + end
            if not on_offer:
                continue
            self.no_sales[good] = 0 if sales else self.no_sales[good] + 1
            # With a nearly empty shelf, observed sales censor market demand.
            censored_sales = sales * (1.4 if not end else 1.)
            smoothing = .26 if censored_sales >= self.demand[good] else .14
            self.demand[good] += smoothing * (censored_sales - self.demand[good])
            fill = sales / on_offer
            target = 0.15
            adjustment = max(-.115, min(.16, .37 * (fill - target)))
            self.price[good] *= exp(adjustment)
            self.price[good] = min(self.initial_price[good] * 5., max(.08 * self.initial_price[good], self.price[good]))

    def decide(self, state, goods_config):
        self.observe_trades(state)
        hardware = set(goods_config.finished_goods) == set(HARDWARE) and set(goods_config.raw_materials) == {"CHIP", "BATTERY", "DISPLAY", "CASING"}
        lots = {lot.good: lot for lot in state.auction_lots}
        if not lots:
            return Decision(listings=[Listing(g, n, self.price.get(g, 1.))
                                      for g, n in state.inventory.items()
                                      if g in goods_config.finished_goods and n > 0])
        if self.initial_capital is None:
            self.initial_capital = state.capital
        if state.turn == 2:
            active = sum(abs(c - self.initial_capital) > .01 for c in state.public_capital.values())
            self.concentrated = active < .21 * len(state.public_capital)
        tight = self.concentrated and sum(lot.quantity for lot in lots.values()) / len(state.public_capital) < 8.0
        self.last_reserve = {m: max(.0001, lot.reserve_price) for m, lot in lots.items()}
        for m in lots:
            if m in self.markup and m not in self.bid_ladder:
                self.markup[m] = max(1.003, self.markup[m] * 0.86)
        self.bid_ladder = {}
        if tight:
            for m in self.markup:
                self.markup[m] = min(self.markup[m], 2.5)
        for m in lots:
            self.markup.setdefault(m, 2.0 if hardware else 1.25)
        estimate = {m: reserve * self.markup[m] * 1.08 for m, reserve in self.last_reserve.items()}
        recipes = self.cheapest_recipes(goods_config, estimate)
        self.initialize_markets(recipes, lots, hardware)
        plans = self.value_kits(state, hardware, tight, recipes, estimate)
        planned, raw, assembly = self.plan_orders(state, recipes, estimate, lots, plans, hardware, tight)
        bids = self.bid_for_materials(state.inventory, raw)
        listings = self.list_inventory(goods_config, state.inventory, planned, recipes, tight)
        return Decision(bids=bids, assembly=assembly, listings=listings)

    @staticmethod
    def cheapest_recipes(goods_config, estimate):
        recipes = {}
        for name, recipe in goods_config.recipes.items():
            if not all(m in estimate for m in recipe.inputs):
                continue
            cost = sum(estimate[m] * q for m, q in recipe.inputs.items())
            if recipe.output_good not in recipes or cost < recipes[recipe.output_good][2]:
                recipes[recipe.output_good] = (name, recipe.inputs, cost)
        return recipes

    def initialize_markets(self, recipes, lots, hardware):
        # Bootstrap shelf depth from published supply. Subsequent purchases
        # follow realized demand, including when the market differs in scale.
        scale = max(.25, min(8., sum(l.quantity for l in lots.values()) / 125))
        for good, (_, inputs, _) in recipes.items():
            if good not in self.price:
                if hardware:
                    prior, volume = HARDWARE[good]
                else:
                    reserve_cost = sum(self.last_reserve[m] * q for m, q in inputs.items())
                    prior, volume = 3.4 * reserve_cost, 1.5
                self.initial_price[good] = self.price[good] = prior * 0.9
                self.initial_volume[good] = volume * scale
                self.demand[good] = max(.8, volume * scale * .42)

    def value_kits(self, state, hardware, tight, recipes, estimate):
        # Order marginal kits by their incremental cash value over the
        # planning horizon. Stock on hand discounts the need for further kits.
        plans = []
        inventory = state.inventory
        for good, (_, inputs, _) in recipes.items():
            price = self.price[good] * (0.86 if state.turn <= 6 else 1.)
            stock = inventory.get(good, 0)
            demand = max(.18, self.demand[good])
            peer_stock = sum(inv.get(good, 0) for inv in state.public_inventories.values()) - stock
            # Opening inventory can be sold in subsequent turns as well. A
            # smaller weight on this option when strapped for cash protects
            # against unfinished kits and unexpectedly weak demand.
            cover = 7.0 if state.capital > .55 * self.initial_capital else 2.2
            if not hardware:
                cover = min(6., 2.5 + .35 * self.lifetime_sales[good])
                if good in self.rival_price:
                    price = .7 * price + .3 * self.rival_price[good]
            if tight:
                price *= (1 + peer_stock / (1. + self.initial_volume[good])) ** (-0.22)
                cover = min(cover, 3. + .1 * self.lifetime_sales[good])
            if self.no_sales[good] > 4:
                cover *= .6
            horizon = 80 if self.initial_capital >= 24000 else 40
            if state.turn <= horizon and state.turn > horizon - 5:
                cover = min(cover, 1 + (horizon - state.turn) * .65)
            for k in range(1, min(65, ceil(cover * (demand + .6)) + 2) + 1):
                if tight and k > 2 + self.lifetime_sales[good]:
                    break
                if not hardware and k > 2 + 1.5 * self.lifetime_sales[good]:
                    break
                depth = (stock + k - .5) / max(.5, demand)
                # Discount a unit by the time required to clear earlier stock.
                value = price * exp(-depth / cover)
                basis = sum(estimate[m] * q * (.25 if inventory.get(m, 0) >= k * q else 1.)
                            for m, q in inputs.items())
                surplus = value - basis
                if surplus <= 0:
                    break
                # Executing now is more valuable in a market with few stocked competitors.
                score = surplus / (basis ** .42) / (1 + .12 * peer_stock / (self.initial_volume[good] + peer_stock + 1))
                plans.append((score, good))
        plans.sort(reverse=True)
        return plans

    def plan_orders(self, state, recipes, estimate, lots, plans, hardware, tight):
        inventory = state.inventory
        planned = defaultdict(int)
        raw = defaultdict(int)
        commitments = 0.
        capital_limit = max(0., min(state.capital * .64, state.capital - .76 * self.initial_capital))
        if tight:
            # Most units cleared to one or two buyers. While their bids
            # are outstanding, limit exposure to unfinished kits.
            capital_limit = min(capital_limit, max(0., state.capital - self.initial_capital) + 0.05*self.initial_capital)
        if not hardware:
            # Finance exploratory trades from a fraction of cash. Profits
            # expand the budget once live demand confirms the opportunity.
            earned = max(0., state.capital - self.initial_capital)
            capital_limit = min(capital_limit, earned + self.initial_capital * 0.04)
        assembly = []
        for _, good in plans:
            name, inputs, _ = recipes[good]
            if any(raw[m] + q > inventory.get(m, 0) + lots[m].quantity for m, q in inputs.items()):
                continue
            incremental = sum(estimate[m] * max(0, min(q, raw[m] + q - inventory.get(m, 0))) for m, q in inputs.items())
            if commitments + incremental > capital_limit:
                continue
            commitments += incremental
            for m, q in inputs.items():
                raw[m] += q
            planned[good] += 1
            if assembly and assembly[-1].recipe_name == name:
                assembly[-1] = AssemblyOrder(name, assembly[-1].quantity + 1)
            else:
                assembly.append(AssemblyOrder(name, 1))
        return planned, raw, assembly

    def bid_for_materials(self, inventory, raw):
        # Ordered kit priorities provide recourse when only some materials
        # clear. If a recipe fails, the simulator continues down the list.
        bids = []
        for m, needed in raw.items():
            n = max(0, needed - inventory.get(m, 0))
            if not n:
                continue
            reserve = self.last_reserve[m]
            ladder = {}
            remaining = n
            for frac, factor in ((.22, 1.27), (.48, 1.08), (1., .92)):
                qty = min(remaining, max(1, round(n * frac)))
                if qty:
                    price = max(reserve * 1.0001, reserve * self.markup[m] * factor)
                    ladder[price] = ladder.get(price, 0) + qty
                    remaining -= qty
            self.bid_ladder[m] = ladder
            bids.extend(Bid(m, q, p) for p, q in ladder.items())
        return bids

    def list_inventory(self, goods_config, inventory, planned, recipes, tight):
        listings = []
        self.shelf = {}
        for good in goods_config.finished_goods:
            n = planned[good] + inventory.get(good, 0)
            if n <= 0:
                continue
            price = self.price.get(good, 1.) * (.84 if tight else 1.)
            if good in recipes:
                replacement_floor = min(self.initial_price[good], recipes[good][2] * .7)
                price = max(price, replacement_floor)
            price = max(.0001, price)
            if good not in self.price:
                listings.append(Listing(good, n, price))
                continue
            self.shelf[good] = tuple(price * x for x in (1., 1.12, .9))
            for k in range(min(n, 96)):
                qty = n - k if k == 95 else 1
                listings.append(Listing(good, qty, self.shelf[good][k % 3]))
        return listings
