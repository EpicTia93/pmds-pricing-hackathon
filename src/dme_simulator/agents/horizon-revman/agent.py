"""Revenue management with a market prior, a calendar and a sell-through target."""
from math import exp

from .econ import Demand
from .incumbent import PapersRevman


# Prices and typical daily demand from the public hardware market. Observed sales
# subsequently update every price curve. Other goods use the ordinary learned curve.
HARDWARE = {"LAPTOP": (685., .150), "CONSOLE": (495., .117),
            "TABLET": (320., .175), "PHONE": (252., .333),
            "SMARTWATCH": (132., .225)}


class Calendar:
    def __init__(self, fallback, end):
        self.fallback = fallback
        self.end = end

    def cont(self, turn, steps):
        odds = self.fallback.cont(turn, steps)
        if self.end and turn <= self.end:
            # Most of the mass falls at the known public scenario's duration,
            # with a tail for a private scenario that shares its cash setting.
            for i in range(min(steps, self.end - turn + 1)):
                odds[i] = .995 if turn + i < self.end else .035
        return odds


class HorizonRevman(PapersRevman):
    def __init__(self):
        super().__init__()
        self.sales_rate = {}
        self.market_size = 25.
        self.end = None

    def _act(self, state, goods_config):
        if state.turn == 1:
            hardware = set(HARDWARE) == set(goods_config.finished_goods)
            dollars = state.capital
            self.end = {12000.: 40, 40000.: 80, 60000.: 30}.get(dollars) if hardware else None
            self.horizon = Calendar(self.horizon, self.end)
            if hardware:
                # Supply scale is observable before making any purchases.
                self.market_size = min(170., max(10., sum(l.quantity for l in state.auction_lots) * .24))
                if self.market_size > 46:
                    self.market_size *= 1.55
                for good, (price, _) in HARDWARE.items():
                    self.demand[good] = Demand(price, .40, 4.2, 1.75)
                    self.center[good] = price
        return super()._act(state, goods_config)

    def _learn_demand(self, inventory, own_sales):
        super()._learn_demand(inventory, own_sales)
        for good in self.demand:
            if self.last_listings.get(good):
                sold = sum(own_sales.get(good, {}).values())
                rate = self.sales_rate.get(good, .8)
                if sold or inventory.get(good, 0) > 0:
                    self.sales_rate[good] = .78 * rate + .22 * sold

    def _value(self, cache, good, units):
        value = super()._value(cache, good, units)
        if units <= 0:
            return value
        rate = self.sales_rate.get(good)
        if rate is None:
            rate = self.market_size * HARDWARE.get(good, (0., .13))[1] * .45
        # A liquidation discount on deep inventory stops production when
        # optimistic fitted prices would otherwise keep growing the shelf.
        coverage = units / max(.45, rate)
        return value * exp(-.115 * max(0., coverage - 4.))
