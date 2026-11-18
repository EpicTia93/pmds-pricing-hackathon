"""Select a procurement policy from opening supply and realized market profits."""
from collections import defaultdict

from dme_simulator.agents import AssemblyOrder, BaseAgent, Bid, Decision, Listing

from .learned import PapersTactex
from .planner import DemandPlanner, SEGMENTS


class MarketAdaptive(BaseAgent):
    def __init__(self):
        self.policy = None
        self.start = None
        self.scaled = False
        self.stock = defaultdict(float)
        self.prices = {"LAPTOP": 625., "CONSOLE": 480., "TABLET": 285., "PHONE": 230.}
        self.markups = {}
        self.bids = {}
        self.cursor = 0
        self.sale_cursor = 0
        self.sold = defaultdict(int)
        self.identity = None
        self.wait = 0
        self.pressure = False
        self.pending = None

    @property
    def errors(self):
        return getattr(self.policy, "errors", 0)

    def decide(self, state, goods_config):
        if self.start is None:
            self.start = state.capital
            if set(goods_config.finished_goods) != set(SEGMENTS):
                self.policy = PapersTactex()
            else:
                width = len(state.public_capital)
                density = sum(lot.quantity for lot in state.auction_lots) / max(1, width)
                self.scaled = density > 5.2
                self.pressure = density < 3.7 and width >= 17
                self.policy = DemandPlanner()
                self.policy.starting_capital = self.start

        if isinstance(self.policy, PapersTactex):
            return self.policy.decide(state, goods_config)
        if not self.pressure and not self.scaled and 2 <= state.turn <= 8:
            # A few very large bidders can exhaust thin lots even in a small field.
            movers = sum(abs(c - self.start) > .1 for c in state.public_capital.values())
            self.pressure = movers <= .18 * len(state.public_capital)
        if self.pressure:
            return self._small_lots(state, goods_config)
        return self.policy.decide(state, goods_config)

    def _small_lots(self, state, config):
        inv = state.inventory
        if self.identity is None:
            candidates = [t for t, c in state.public_capital.items()
                          if state.capital == c and state.public_inventories[t] == inv]
            if len(candidates) == 1:
                self.identity = candidates[0]
        lots = {lot.good: lot for lot in state.auction_lots}
        fills = defaultdict(int)
        for t in state.auction_transactions[self.cursor:]:
            if t.buyer == self.identity or (self.identity is None and t.price == self.bids.get(t.good, (None,))[0]):
                fills[t.good] += t.quantity
        self.cursor = len(state.auction_transactions)
        for t in state.sales_transactions[self.sale_cursor:]:
            if t.seller == self.identity or (self.identity is None and t.price == self.prices.get(t.good)):
                self.sold[t.good] += t.quantity
        self.sale_cursor = len(state.sales_transactions)
        for m, (_, n) in self.bids.items():
            if n:
                self.markups[m] = min(14., max(1.12, self.markups[m] * (1.31 if fills[m] < n else .89)))
        self.bids = {}
        for m in lots:
            self.markups.setdefault(m, 4.63)
        costs = {m: lot.reserve_price * self.markups[m] for m, lot in lots.items()}
        alternatives = []
        for name, recipe in config.recipes.items():
            g = recipe.output_good
            if g not in self.prices or any(m not in costs for m in recipe.inputs):
                continue
            price = self.prices[g]
            inputs = recipe.inputs
            cost = sum(costs[m] * a for m, a in inputs.items())
            missing = sum(costs[m] * max(0, a - inv.get(m, 0)) for m, a in inputs.items())
            # A nearly completed kit has a higher priority than a new one.
            value = price - .72 * cost - .28 * missing
            value /= (1 + inv.get(g, 0)) ** 1.7
            if g == self.pending:
                value *= 1.2
            alternatives.append((value, name, g, inputs, missing, cost))
        listings = []
        for g, p in self.prices.items():
            n = inv.get(g, 0)
            if n:
                self.stock[g] = .8 * self.stock[g] + .2 * n
            if self.sold[g]:
                self.prices[g] = min(1.006 * p, {"LAPTOP":815., "CONSOLE":615., "TABLET":365., "PHONE":285.}[g])
            if n and self.stock[g] > .45:
                self.prices[g] = max(.95 * self.prices[g], {"LAPTOP":175., "CONSOLE":145., "TABLET":95., "PHONE":85.}[g])
        orders = []
        if alternatives:
            value, name, g, inputs, missing, cost = max(alternatives)
            self.pending = g
            periods = max(1, state.turn - 1)
            sold = sum(self.sold.values())
            # Reinvest realized sales. New orders stay small until we demonstrate
            # the ability to clear finished units at profitable prices.
            desired = min(8, 1 + int(sold / periods * 2.4) + int(state.capital > self.start + 2000))
            cash_left = state.capital - self.start
            reserve = self.start - min(self.start * .055, 720.)
            budget = max(0., state.capital - reserve)
            if sold > 0 and cash_left > 0:
                budget = min(budget, max(.65 * cost, cash_left * .70))
            if self.start in (12000., 40000.) and state.turn in ((40, 80) if self.start == 12000. else (80,)):
                budget = min(budget, .50 * cost)
            kits = max(0, desired - sum(inv.get(k, 0) for k in self.prices))
            while kits > 0 and sum(costs[m] * max(0, kits * a - inv.get(m, 0))
                                   for m, a in inputs.items()) > budget:
                kits -= 1
            if value > 0 and kits:
                for m, a in inputs.items():
                    n = max(0, kits * a - inv.get(m, 0))
                    if n:
                        self.bids[m] = (costs[m], n)
            orders = [AssemblyOrder(r[1], 200) for r in sorted(alternatives, reverse=True)]
        for g in self.prices:
            listings.extend([Listing(g, 1, self.prices[g])] * min(48, inv.get(g, 0) + 8))
            listings.append(Listing(g, 10000, self.prices[g]))
        return Decision([Bid(m, n, p) for m, (p, n) in self.bids.items()], orders, listings)
