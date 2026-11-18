"""Jointly choose a shelf depth and price under uncertain demand and rival supply."""

import math

import numpy as np

from dme_simulator.agents import AssemblyOrder, BaseAgent, Bid, Decision, Listing


# Public default market. Each row is the expected arrivals per 25 buyers and
# the endpoints of their uniform reservation-price interval.
SEGMENTS = {
    "PHONE": [(3.75, 150, 280), (3.75, 250, 420), (0.833, 400, 650)],
    "TABLET": [(1.875, 180, 320), (2.5, 300, 520)],
    "CONSOLE": [(1.25, 350, 600), (1.667, 600, 1000)],
    "LAPTOP": [(1.25, 450, 750), (2.5, 800, 1400)],
    "SMARTWATCH": [(5.625, 90, 180)],
}

# Deterministic quadrature across demand strength, WTP, elasticity and rival
# offer weights. Log-spaced quantiles retain support for unusual markets.
_INDEX = np.arange(1, 146)


def quantile(multiplier):
    return ((_INDEX * multiplier) % 149 + 0.5) / 149


class Market:
    def __init__(self, segments, scale, generic, rival_count):
        self.anchor = sum(n * low for n, low, _ in segments) / sum(n for n, _, _ in segments)
        self.beta = np.exp(np.log(0.01) + (quantile(31) - 0.5) * 2.8)
        self.competitor_price = self.anchor * np.exp((quantile(53) - 0.68) * 2.1)
        self.competitor_weight = 0.48 * np.exp((quantile(19) - 0.5) * 3)
        self.demand = scale * np.exp((quantile(41) - 0.5) * (3.6 if generic else 2.6))
        self.wtp = np.exp((quantile(83) - 0.5) * (3.4 if generic else 1.65))
        self.segments = segments
        self.log_weight = np.zeros(len(_INDEX))
        self.generic = generic
        self.count = rival_count
        self.prior = -0.5 * (np.log(self.wtp) / (0.85 if generic else 0.27)) ** 2
        self.prior -= 0.5 * (np.log(self.demand / scale) / 0.75) ** 2
        self.prices = self.anchor * np.geomspace(0.12 if generic else 0.16, 4 if generic else 2.5, 42)
        self.unit_ladder = np.array([1, 2, 3, 5, 8, 12, 18, 26, 38, 55, 80, 115, 165, 240, 320])
        self.last = None

    def expected(self, idx, prices, units, rivals):
        willingness = prices / self.wtp[idx]
        arrivals = sum(n * np.clip((hi - willingness) / (hi - lo), 0, 1) for n, lo, hi in self.segments)
        rivals_at_price = (rivals + 0.3) * self.competitor_weight[idx] * np.exp(
            np.clip(self.beta[idx] * (prices - self.competitor_price[idx]), -18, 18))
        # Use the midpoint of the stocks that remain as customers arrive.
        gross = arrivals * self.demand[idx]
        depth = np.maximum(0.65 * units, units - gross * 0.35)
        return np.minimum(units, gross * depth / (depth + rivals_at_price))

    def observe(self, price, units, sold, rivals):
        if units <= 0:
            return
        mu = self.expected(slice(None), price, units, rivals)
        # One-sided likelihood for a shelf cleared by the end of the turn.
        # Overdispersion allows for seasonality and unpredictable new offers.
        mu = np.maximum(0.015, mu)
        score = sold * np.log(mu) - (sold + 4) * np.log(4 + mu)
        self.log_weight = 0.94 * self.log_weight + score - max(score)

    def plan(self, owned, cost, rivals, horizon):
        logp = self.prior + self.log_weight
        weights = np.exp(logp - max(logp))
        cdf = np.cumsum(weights)
        idx = np.searchsorted(cdf, cdf[-1] * (np.arange(12) + 0.5) / 12)
        # Include owned units in this grid even when the target is zero.
        depths = np.unique(np.append(self.unit_ladder, max(1, owned)))
        shelf = depths[None, :, None]
        prices = self.prices[:, None, None]
        expected = self.expected(idx, prices, shelf, rivals).mean(axis=2)
        # Charge for outstanding stock over the useful planning horizon.
        # Existing stock is sunk, but receives a carry value while replenishment remains attractive.
        daily = expected * (self.prices[:, None] - cost)
        value = daily - depths * cost / horizon
        best_price, best_n = np.unravel_index(np.argmax(value), value.shape)
        target = int(depths[best_n]) if value[best_price, best_n] > 0 else 0
        urgency = max(0, value[best_price, best_n]) / max(1, target)
        if owned >= target:
            own_j = np.where(depths == max(1, owned))[0][0]
            marginal_cost = cost * min(1, target / max(1, owned))
            cash = expected[:, own_j] * (self.prices - marginal_cost)
            best_price = int(np.argmax(cash))
        return target, float(self.prices[best_price]), float(urgency)


class DemandPlanner(BaseAgent):
    def __init__(self):
        self.market = {}
        self.identity = None
        self.auction_cursor = 0
        self.sales_cursor = 0
        self.markup = {}
        self.last_orders = {}
        self.last_display = {}
        self.starting_capital = 0

    def decide(self, state, goods_config):
        inventory = state.inventory
        lots = {lot.good: lot for lot in state.auction_lots}
        goods = goods_config.finished_goods
        if state.turn == 1:
            self.starting_capital = state.capital
        if self.identity is None:
            candidates = [t for t, cap in state.public_capital.items() if cap == state.capital and
                          state.public_inventories.get(t) == inventory]
            if len(candidates) == 1:
                self.identity = candidates[0]

        won, sold = {}, {}
        for tx in state.auction_transactions[self.auction_cursor:]:
            if tx.buyer == self.identity or self.identity is None and tx.price in self.last_orders.get(tx.good, ()):
                won[tx.good] = won.get(tx.good, 0) + tx.quantity
        for tx in state.sales_transactions[self.sales_cursor:]:
            if tx.seller == self.identity or self.identity is None and self.last_display.get(tx.good, (0,))[0] == tx.price:
                sold[tx.good] = sold.get(tx.good, 0) + tx.quantity
        self.auction_cursor = len(state.auction_transactions)
        self.sales_cursor = len(state.sales_transactions)

        rivals = {}
        for g in goods:
            rivals[g] = (0.13 * len(state.public_capital)
                         + sum(min(190, row.get(g, 0)) ** 0.88 for row in state.public_inventories.values())
                         - min(190, inventory.get(g, 0)) ** 0.88)
            if g in self.last_display:
                price, pressure = self.last_display[g]
                units = inventory.get(g, 0) + sold.get(g, 0)
                self.market[g].observe(price, units, sold.get(g, 0), pressure)

        prices = {}
        for m, lot in lots.items():
            markup = self.markup.get(m, 1.55)
            if m in self.last_orders:
                _, wanted = self.last_orders[m]
                fill = min(1, won.get(m, 0) / wanted)
                markup = max(1.003, min(10, markup * math.exp(0.47 * (0.78 - fill))))
            self.markup[m] = markup
            prices[m] = max(0.001, lot.reserve_price) * markup
        recipes = {}
        for name, recipe in goods_config.recipes.items():
            if not all(m in prices for m in recipe.inputs):
                continue
            cost = sum(prices[m] * n for m, n in recipe.inputs.items())
            if recipe.output_good not in recipes or cost < recipes[recipe.output_good][0]:
                recipes[recipe.output_good] = (cost, name, recipe.inputs)

        base_scale = 1.0
        if "CHIP" in lots:
            base_scale = max(0.35, min(5.5, (lots["CHIP"].quantity / 43) ** 1.65))
        horizon = 19.0
        if self.starting_capital in (12000.0, 40000.0) and set(goods) == set(SEGMENTS):
            end = 40 if self.starting_capital == 12000 else 80
            if state.turn <= end:
                horizon = max(0.65, min(horizon, (end + 1 - state.turn) * 0.66))
        plans = {}
        self.last_display = {}
        for g in goods:
            owned = inventory.get(g, 0)
            if g not in recipes and not owned:
                continue
            cost = recipes[g][0] if g in recipes else 1
            if g not in self.market:
                prior_cost = sum(n * lots[m].reserve_price for m, n in recipes[g][2].items()) if g in recipes else cost
                base = max(2.0, prior_cost)
                segments = SEGMENTS.get(g, [(2.2, base * 1.3, base * 3.3), (1.5, base * 3.2, base * 6.5),
                                            (0.85, base * 6.5, base * 12)])
                self.market[g] = Market(segments, base_scale if g in SEGMENTS else 1, g not in SEGMENTS, len(state.public_capital))
            target, price, urgency = self.market[g].plan(owned, cost, rivals[g], horizon)
            plans[g] = (target, price, urgency)
            self.last_display[g] = (price, rivals[g])

        # Allocate the missing shelf units in descending daily contribution per
        # dollar of raw inputs. Keep an independent inventory account for each kit.
        remaining = {m: inventory.get(m, 0) for m in lots}
        wanted = {m: 0 for m in lots}
        budget = 0.73 * max(0, state.capital)
        assembly, listings = [], []
        for g in sorted(plans, key=lambda key: -plans[key][2] / max(1, recipes.get(key, (1,))[0])):
            target, price, urgency = plans[g]
            owned = inventory.get(g, 0)
            to_build = max(0, target - owned)
            if g in recipes:
                cost, name, inputs = recipes[g]
                for m, count in inputs.items():
                    cap = remaining[m] + max(0, lots[m].quantity - wanted[m])
                    to_build = min(to_build, cap // count)
                def bill(units):
                    return sum(max(0, units * n - remaining[m]) * prices[m] for m, n in inputs.items())
                if bill(to_build) > budget:
                    lo, hi = 0, to_build
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if bill(mid) <= budget:
                            lo = mid
                        else:
                            hi = mid - 1
                    to_build = lo
                budget -= bill(to_build)
                if to_build:
                    for m, n in inputs.items():
                        needed = max(0, n * to_build - remaining[m])
                        wanted[m] += needed
                        remaining[m] = max(0, remaining[m] - n * to_build)
                    assembly.append(AssemblyOrder(name, to_build))
            count = min(320, owned + to_build)
            listings.extend([Listing(g, 1, price)] * count)
            if owned + to_build > count:
                listings.append(Listing(g, owned + to_build - count, price))

        bids = [Bid(m, units, prices[m]) for m, units in wanted.items() if units]
        self.last_orders = {m: (prices[m], n) for m, n in wanted.items() if n}
        return Decision(bids=bids, assembly=assembly, listings=listings)
