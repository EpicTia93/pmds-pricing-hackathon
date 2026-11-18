"""Paper-derived engine shared by the revman, auction, capacity and tactex families.

Each turn the engine solves the turn backwards, from sales to bids, the way
TacTex did in the TAC supply chain game (Pardoe & Stone):

1. Learn.
   * Demand per good: the chance that one listed unit sells in a turn is
     ``sigmoid(a - b ln p + offset)``. a and b are fitted online by weighted
     logistic regression with a prior and forgetting, the parametric case of
     Besbes & Zeevi. The offset accounts for our own listings competing with
     each other (``cannibal``).
   * Clearing prices per material, as a log markup over the reserve, from
     censored pay-your-bid feedback. A full fill means the clearing price was at
     or below our bid, a partial fill means it was at our bid, no fill means it
     was above. With an open book, the whole log is used (Han, Zhou & Weissman
     learn first-price bids from censored feedback). ``pi0`` is the chance that a
     lot is not contested at all.
2. Value. One unit on the shelf is worth V, the value of a finite-horizon
   pricing dynamic program (Gallego & van Ryzin, 1994). The game length is
   unknown, so a lognormal prior on it gives the odds that each future turn is
   played.
3. Buy.
   * Quantity: units whose marginal value beats the kit cost (``value``), a
     target stock cover (``cover``), or a price-dependent base stock of
     materials (``basestock``, Kalymon 1971).
   * Bid price: a kit-level best response that maximises the expected margin of
     a complete kit, with partial wins valued at ``salvage`` (exposure problem,
     Wellman et al. 2008). Alternatives are a target win probability
     (``quantile``) or shading the value (``shade``).
   * Optional pacing multiplier on values (Balseiro & Gur), and sample average
     approximation over joint price scenarios (``kits: saa``, RoxyBot-06).
   * Optional regime switch: bid harder on drained lots and stock up on
     glutted ones (Ketter et al.).
4. Build. Assembly orders go in value order, so whatever the auction yields is
   built into the best products first.
5. Sell. Each unit can be its own listing (``listing: units``): the simulator
   gives every listing its own choice weight. Prices come from the DP, the
   static logit rule with the replacement cost as marginal cost (``myopic``),
   or a fixed run-out price over the expected remaining turns (``runout``).
   A small ladder (``explore``) keeps the demand curve identifiable.

Module choices, and every constant the theory cannot pin down (priors,
learning rates, risk attitude, the horizon prior), are parameters, so the swarm
evolves them. Only the public agent interface and ``.common`` are imported, so
the file is copied into exported bundles as-is.
"""

from __future__ import annotations

import os
import traceback
from math import erf, exp, gcd, log, sqrt
from statistics import NormalDist
from time import perf_counter
from typing import Any

from dme_simulator.agents import AssemblyOrder, Bid, Decision, Listing, Observation
from dme_simulator.config import GoodsConfig

from .common import BIG, ParamBase, cheapest_recipes

ENGINE_DEFAULTS: dict[str, Any] = {
    # --- module choices ----------------------------------------------------
    "pricing": "dp",          # dp | myopic | runout
    "listing": "units",       # units (one listing per unit) | rungs
    "quantity": "value",      # value | cover | basestock
    "bidding": "kit",         # kit (best response) | quantile | shade
    "kits": "independent",    # independent | saa (joint scenario refinement)
    "regime": False,          # scarcity / glut switch on the auction side
    # --- demand learning ---------------------------------------------------
    "price_mult0": 5.0,       # prior reference price = mult x kit cost at reserve
    "accept0": 0.3,           # prior chance a single listed unit sells at that price
    "slope0": 3.0,            # prior slope of the logit in log price
    "prior_weight": 4.0,      # pseudo-observations behind the prior
    "memory": 0.85,           # per-turn decay of demand observations
    "cannibal": 0.5,          # 0: our listings do not compete; 1: fixed demand pool
    "explore": 0.12,          # log-price half-width of the listing ladder
    "rungs": 3,               # distinct prices per good
    # --- horizon prior (the game length is not observable) -----------------
    "horizon_median": 40.0,
    "horizon_spread": 0.5,    # lognormal sigma
    # --- pricing extras ----------------------------------------------------
    "basis_floor": 0.0,       # list >= floor x replacement kit cost (and myopic marginal cost)
    "runout_frac": 1.0,       # run-out pricing sells the stock over this share of the expected rest
    # --- quantities --------------------------------------------------------
    "margin": 0.2,            # buy when value >= (1 + margin) x kit cost
    "cover": 4.0,             # expected turns to sell a unit (cover / basestock)
    "kappa": 1.0,             # base-stock elasticity to (average reserve / reserve)
    "budget_share": 0.6,      # capital committed per auction at most
    "stop_turns": 1.0,        # stop buying when fewer turns are expected
    "leftover_margin": 0.3,   # build spare stock when value >= margin x replacement cost
    "max_lot_share": 0.9,
    # --- clearing-price model and bidding ----------------------------------
    "clear0": 0.3,            # prior log markup of contested clearing prices
    "clear_sd0": 0.4,
    "clear_rate": 0.35,
    "uncontested0": 0.4,      # prior chance a lot clears at its reserve
    "aggression": 1.0,        # value multiplier in the best response
    "salvage": 0.4,           # value of a stranded material unit, x its reserve
    "bid_quantile": 0.75,     # target win probability (quantile bidding)
    "shade": 0.5,             # bid = reserve + shade x (value share - reserve)
    "pacing_gain": 0.0,       # Balseiro-Gur multiplier step; 0 disables pacing
    "spend_rate": 0.15,       # target spend per turn, share of capital
    "saa_samples": 12,
    "clear_corr": 0.5,        # correlation of clearing prices across materials
    # --- regime ------------------------------------------------------------
    "scarce_boost": 0.3,      # extra kit value when its lots are drained
    "glut_ratio": 0.8,        # reserve below ratio x its peak = glutted lot
    "glut_cover": 2.0,        # turns of extra material stock bought in a glut
}

GRID = 15            # price points per DP
GRID_SPAN = 1.0      # log-price half-width of the DP grid
DP_STEPS = 36        # look-ahead turns
MAX_UNIT_LISTINGS = 200   # one-unit listings per good; the rest share one listing
TIME_BUDGET = 0.035   # seconds per decision before optional work is skipped (deadline 100 ms)
_Y_GRID = (0.0005, 0.02, 0.05, 0.1, 0.18, 0.3, 0.45, 0.65, 0.9, 1.2, 1.6, 2.1)
_SQRT2 = sqrt(2.0)
_INV = NormalDist().inv_cdf
DEBUG = bool(os.environ.get("SWARM_DEBUG"))


def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / _SQRT2))


def _sig(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    e = exp(x)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class Horizon:
    """Lognormal prior on the game length, as survival odds for future turns."""

    def __init__(self, median: float, spread: float, cap: int = 1000) -> None:
        mu, s = log(max(median, 1.0)), max(spread, 0.02)
        self.cap = cap
        self.surv = [1.0, 1.0] + [
            1.0 - _phi((log(n - 0.5) - mu) / s) for n in range(2, cap + 2)
        ]

    def _s(self, n: int) -> float:
        return self.surv[min(max(n, 0), self.cap + 1)]

    def cont(self, turn: int, steps: int) -> list[float]:
        """P(the game goes on after turn ``turn + i`` | it reached that turn)."""
        out = []
        for i in range(steps):
            here, nxt = self._s(turn + i), self._s(turn + i + 1)
            c = nxt / here if here > 1e-12 else 0.5
            out.append(min(0.995, max(0.0, c)))
        return out

    @staticmethod
    def remaining(cont: list[float]) -> float:
        """Expected number of turns after the current one."""
        total, alive = 0.0, 1.0
        for c in cont:
            alive *= c
            total += alive
        return total


class Demand:
    """Per-unit, per-turn sale chance ``sigmoid(a - b (ln p - c0) + offset)``."""

    __slots__ = ("c0", "a", "b", "b0", "prior", "obs")

    def __init__(self, ref: float, accept0: float, slope0: float, weight: float) -> None:
        self.c0 = log(max(ref, 1e-6))
        self.a, self.b, self.b0 = _logit(accept0), slope0, slope0
        w = weight / 3.0
        self.prior = [(d, 0.0, w, w * _sig(self.a - slope0 * d)) for d in (-0.4, 0.0, 0.4)]
        self.obs: list[list[float]] = []

    def add(self, price: float, offset: float, n: float, s: float) -> None:
        self.obs.append([log(price) - self.c0, offset, float(n), float(s)])

    def decay(self, memory: float) -> None:
        for o in self.obs:
            o[2] *= memory
            o[3] *= memory
        self.obs = [o for o in self.obs if o[2] > 0.02]

    def fit(self, iterations: int = 4) -> None:
        a, b = self.a, self.b
        rows = self.prior + self.obs
        ridge = self.prior[0][2]
        for _ in range(iterations):
            ga = gb = iaa = iab = ibb = 0.0
            for x, off, n, s in rows:
                q = _sig(a - b * x + off)
                r = s - n * q
                w = n * q * (1.0 - q)
                ga += r
                gb -= r * x
                iaa += w
                iab -= w * x
                ibb += w * x * x
            gb -= ridge * (b - self.b0)
            ibb += ridge
            det = iaa * ibb - iab * iab
            if det <= 1e-12:
                break
            da = (ibb * ga - iab * gb) / det
            db = (iaa * gb - iab * ga) / det
            a += max(-2.0, min(2.0, da))
            b = min(max(b + max(-2.0, min(2.0, db)), 0.05), 40.0)
        self.a, self.b = max(-12.0, min(12.0, a)), b

    def prob(self, price: float, offset: float) -> float:
        return _sig(self.a - self.b * (log(max(price, 1e-9)) - self.c0) + offset)


class Clearing:
    """Contested clearing price = reserve x exp(N(x, sd)); uncontested with prob pi0."""

    __slots__ = ("x", "sd", "pi0")

    def __init__(self, x: float, sd: float, pi0: float) -> None:
        self.x, self.sd, self.pi0 = x, sd, pi0

    def win(self, y: float) -> float:
        """Chance that a bid at log markup ``y`` over the reserve is filled."""
        if y < 0.0:
            return 0.0
        return self.pi0 + (1.0 - self.pi0) * _phi((y - self.x) / self.sd)

    def quantile(self, target: float) -> float:
        if target <= self.pi0:
            return 0.0005
        z = _INV(min(0.999, (target - self.pi0) / max(1e-6, 1.0 - self.pi0)))
        return max(0.0005, self.x + self.sd * z)

    def update(self, eta: float, exact, low, high, contested) -> None:
        if contested is True:
            self.pi0 *= 1.0 - eta
        elif contested is False:
            self.pi0 += eta * (1.0 - self.pi0)
        surprised = False
        if exact is not None and exact > 0.0:
            err = exact - self.x
            self.x += eta * err
            self.sd = sqrt((1.0 - eta) * self.sd ** 2 + eta * err * err)
        else:
            if low is not None and self.x < low + 0.5 * self.sd:
                self.x += eta * (low + 0.5 * self.sd - self.x)
                surprised = True
            if high is not None and high > 0.03 and self.x > high:
                self.x += 0.5 * eta * (high - self.x)
                surprised = True
            self.sd *= (1.0 + 0.3 * eta) if surprised else (1.0 - 0.1 * eta)
        self.x = max(-0.5, min(4.0, self.x))
        self.sd = max(0.05, min(2.0, self.sd))
        self.pi0 = max(0.0, min(0.98, self.pi0))


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

class Engine(ParamBase):
    """Predict-then-optimise agent; families set ``DEFAULTS`` (evolved) and ``FIXED``."""

    FIXED: dict[str, Any] = {}

    def __init__(self, **overrides: Any) -> None:
        super().__init__(**overrides)
        self.q: dict[str, Any] = {**ENGINE_DEFAULTS, **self.FIXED, **self.p}
        q = self.q
        self.horizon = Horizon(q["horizon_median"], q["horizon_spread"])
        self.demand: dict[str, Demand] = {}
        self.center: dict[str, float] = {}
        self.clear: dict[str, Clearing] = {}
        self.me: str | None = None
        self.open_book = False
        self._a_seen = 0
        self._s_seen = 0
        self.last_bids: dict[str, list[tuple[float, int]]] = {}
        self.last_reserve: dict[str, float] = {}
        self.last_lot: dict[str, int] = {}
        self.last_listings: dict[str, list[tuple[float, int]]] = {}
        self.peak_res: dict[str, float] = {}
        self.avg_res: dict[str, float] = {}
        self.mu = 0.0              # pacing multiplier
        self.spent_last = 0.0
        self.target_spend = 0.0
        self.errors = 0
        self.last_error = ""
        self._t0 = perf_counter()

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def _identify(self, state: Observation) -> set[str]:
        """Teams whose public state equals ours; adopt the identity once unique."""
        matches = {
            team for team, cap in state.public_capital.items()
            if cap == state.capital and state.public_inventories.get(team) == state.inventory
        }
        if self.me is None and len(matches) == 1:
            self.me = next(iter(matches))
        return matches

    def _is_mine(self, team: str, price: float, ours: set[float], candidates: set[str]) -> bool:
        if self.me is not None:
            return team == self.me
        # Identity still ambiguous: a fill at exactly one of our prices, by a
        # team that looks like us, is ours.
        if price in ours and (not candidates or team in candidates):
            self.me = team
            return True
        return False

    def _digest(self, state: Observation, candidates: set[str]):
        """Split new transactions into ours and rivals' (by identity or our own prices)."""
        own_buys: dict[str, dict[float, int]] = {}
        own_sales: dict[str, dict[float, int]] = {}
        rival_buys: dict[str, list[tuple[float, int]]] = {}
        bid_prices = {m: {p for p, _ in t} for m, t in self.last_bids.items()}
        list_prices = {g: {p for p, _ in t} for g, t in self.last_listings.items()}
        spent = 0.0
        for tx in state.auction_transactions[self._a_seen:]:
            if self._is_mine(tx.buyer, tx.price, bid_prices.get(tx.good, set()), candidates):
                cell = own_buys.setdefault(tx.good, {})
                cell[tx.price] = cell.get(tx.price, 0) + tx.quantity
                spent += tx.price * tx.quantity
            else:
                self.open_book = True
                rival_buys.setdefault(tx.good, []).append((tx.price, tx.quantity))
        for tx in state.sales_transactions[self._s_seen:]:
            if self._is_mine(tx.seller, tx.price, list_prices.get(tx.good, set()), candidates):
                cell = own_sales.setdefault(tx.good, {})
                cell[tx.price] = cell.get(tx.price, 0) + tx.quantity
            else:
                self.open_book = True
        self._a_seen = len(state.auction_transactions)
        self._s_seen = len(state.sales_transactions)
        self.spent_last = spent
        return own_buys, own_sales, rival_buys

    def _learn_clearing(self, own_buys, rival_buys) -> None:
        eta = self.q["clear_rate"]
        for m, model in self.clear.items():
            r = self.last_reserve.get(m)
            if not r:
                continue
            got = own_buys.get(m, {})
            exact = low = high = contested = None
            if self.open_book:
                rivals = rival_buys.get(m, [])
                units = sum(got.values()) + sum(u for _, u in rivals)
                lot = self.last_lot.get(m, 0)
                prices = list(got) + [p for p, _ in rivals]
                if lot > 0:
                    if units >= lot and prices:
                        exact, contested = log(min(prices) / r), True
                    else:
                        contested = False
            for price, qty in self.last_bids.get(m, []):
                y = log(price / r)
                won = got.get(price, 0)
                if won >= qty:
                    high = y if high is None else min(high, y)
                    if y <= 0.03 and contested is None:
                        contested = False
                elif won > 0:
                    exact = y if exact is None else exact
                    contested = True
                else:
                    low = y if low is None else max(low, y)
                    contested = True
            model.update(eta, exact, low, high, contested)

    def _learn_demand(self, inventory, own_sales) -> None:
        q = self.q
        c = q["cannibal"]
        for good, model in self.demand.items():
            model.decay(q["memory"])
            rows = self.last_listings.get(good)
            if not rows:
                continue
            sold = own_sales.get(good, {})
            n_sold = sum(sold.values())
            remaining = min(sum(u for _, u in rows), inventory.get(good, 0) + n_sold)
            per_price: dict[float, int] = {}
            listings = 0
            for price, units in rows:
                if remaining <= 0:
                    break
                active = min(units, remaining)
                remaining -= active
                per_price[price] = per_price.get(price, 0) + active
                listings += 1
            k = sum(per_price.values())
            if k <= 0:
                continue
            offset = (1.0 - c) * log(listings) - log(k)
            for price, n in per_price.items():
                model.add(price, offset, n, min(n, sold.get(price, 0)))
            model.fit()

    # ------------------------------------------------------------------
    # valuation
    # ------------------------------------------------------------------
    def _offset(self, units: float) -> float:
        k = max(units, 1.0)
        c = self.q["cannibal"]
        if self.q["listing"] == "units":
            listings = min(k, MAX_UNIT_LISTINGS) + (1.0 if k > MAX_UNIT_LISTINGS else 0.0)
        else:
            listings = min(k, max(1, int(self.q["rungs"])))
        return (1.0 - c) * log(listings) - log(k)

    def _grid(self, good: str) -> list[float]:
        lc = log(max(self.center[good], 1e-6))
        step = 2.0 * GRID_SPAN / (GRID - 1)
        return [exp(lc - GRID_SPAN + i * step) for i in range(GRID)]

    def _dp(self, good: str, units: float, cont: list[float]) -> tuple[float, float, float]:
        """Value of one unit on the shelf now, the best price now, and the hold value."""
        model = self.demand[good]
        off = self._offset(units)
        grid = self._grid(good)
        probs = [model.prob(p, off) for p in grid]
        value = 0.0
        for i in range(len(cont) - 1, 0, -1):
            lam = cont[i] * value
            best = 0.0
            for p, pr in zip(grid, probs):
                gain = pr * (p - lam)
                if gain > best:
                    best = gain
            value = lam + best
        lam0 = cont[0] * value
        best, best_p = 0.0, grid[-1]
        for p, pr in zip(grid, probs):
            gain = pr * (p - lam0)
            if gain > best:
                best, best_p = gain, p
        return lam0 + best, best_p, lam0

    def _policy(self, good: str, units: float, cont: list[float], remaining: float,
                replacement: float) -> tuple[float, float, float]:
        """(unit value, list price, hold value) under the chosen pricing module.

        The value is what one unit earns under the policy actually used to sell
        it, so buying decisions stay consistent with pricing:

        * dp: the DP value; the hold value is the continuation value.
        * myopic: the static logit price, relisted every turn until sold.
        * runout: the fixed price maximising revenue by the expected end.
        """
        q = self.q
        mode = q["pricing"]
        if mode == "dp":
            return self._dp(good, units, cont)
        model = self.demand[good]
        off = self._offset(units)
        grid = self._grid(good)
        probs = [model.prob(p, off) for p in grid]
        if mode == "myopic":
            cost = q["basis_floor"] * replacement
            i = max(range(len(grid)), key=lambda j: probs[j] * (grid[j] - cost))
            price, pr = grid[i], probs[i]
            value, alive = 0.0, 1.0
            for c in [1.0] + cont[:-1]:
                alive *= c
                value += alive * pr * price
                alive *= 1.0 - pr
            return value, price, 0.0
        horizon = max(1.0, q["runout_frac"] * (1.0 + remaining))
        scores = [p * (1.0 - (1.0 - pr) ** horizon) for p, pr in zip(grid, probs)]
        i = max(range(len(grid)), key=scores.__getitem__)
        return scores[i], grid[i], 0.0

    def _eval(self, cache: dict, good: str, units: float) -> tuple[float, float]:
        """(unit value, list price) at ``units``, interpolated between powers of two.

        Policies are cached per turn, so scanning many candidate quantities
        costs a handful of solves per good.
        """
        units = max(units, 1.0)
        l2 = log(units, 2)
        lo = int(l2)
        out = []
        for e in (lo, lo + 1):
            key = (good, e)
            if key not in cache:
                cache[key] = self._policy(good, float(2 ** e), self._cont, self._remaining,
                                          self._repl.get(good, 0.0))
            out.append(cache[key])
        w = l2 - lo
        value = (1.0 - w) * out[0][0] + w * out[1][0]
        price = out[0][1] if w < 0.5 else out[1][1]
        return value, price

    def _value(self, cache: dict, good: str, units: float) -> float:
        return self._eval(cache, good, units)[0]

    # ------------------------------------------------------------------
    # bidding
    # ------------------------------------------------------------------
    def _material_tables(self, lots) -> dict[str, list[tuple[float, float, float]]]:
        """Per material: candidate (log markup, win chance, price) rows."""
        tables = {}
        for m, lot in lots.items():
            model = self.clear[m]
            ys = set(_Y_GRID)
            for z in (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0):
                ys.add(round(max(0.0005, model.x + z * model.sd), 4))
            r = max(lot.reserve_price, 1e-3)
            tables[m] = [(y, model.win(y), r * exp(y)) for y in sorted(ys)]
        return tables

    def _kit_bid(self, value: float, inputs: dict[str, int], tables, salvage) -> tuple[dict, float, float]:
        """Best response for one complete kit: bid prices, cost if all won, P(all won)."""
        q = self.q
        mode = q["bidding"]
        choice: dict[str, tuple[float, float, float]] = {}
        if mode == "kit":
            for m in inputs:  # start at the expected clearing markup
                x = self.clear[m].x
                choice[m] = min(tables[m], key=lambda row: abs(row[0] - max(x, 0.0005)))
            zeta = q["salvage"]
            # Objective = V P - sum a b G + zeta sum a s (G - P), with P = prod G.
            # With every material but m fixed it is affine in (G_m, b_m G_m), so
            # each candidate row costs O(1) (coordinate ascent, two passes).
            for _ in range(2):
                for m, a in inputs.items():
                    p_o, pay_o, s_o, t_o = 1.0, 0.0, 0.0, 0.0
                    for j, aj in inputs.items():
                        if j == m:
                            continue
                        _, gj, bj = choice[j]
                        p_o *= gj
                        pay_o += aj * bj * gj
                        s_o += aj * salvage[j] * gj
                        t_o += aj * salvage[j]
                    s_m = a * salvage[m]
                    cap = value / a
                    best_row, best_val = choice[m], None
                    for row in tables[m]:
                        _, g, b = row
                        if b > cap and row is not choice[m]:
                            continue
                        p = g * p_o
                        val = value * p - pay_o - a * b * g + zeta * (s_o + s_m * g - p * (t_o + s_m))
                        if best_val is None or val > best_val:
                            best_row, best_val = row, val
                    choice[m] = best_row
        else:
            exp_cost = {m: self._expected_price(m, tables) for m in inputs}
            kit_cost = sum(a * exp_cost[m] for m, a in inputs.items()) or 1e-9
            for m, a in inputs.items():
                model = self.clear[m]
                r = tables[m][0][2] / exp(tables[m][0][0])
                share = value * exp_cost[m] / kit_cost  # value of one unit of m
                if mode == "quantile":
                    y = model.quantile(q["bid_quantile"])
                    price = min(r * exp(y), share)
                else:  # shade
                    price = r + q["shade"] * (share - r)
                price = max(price, r * 1.0005)
                y = log(price / r)
                choice[m] = (y, model.win(y), price)
        prices = {m: row[2] for m, row in choice.items()}
        cost = sum(a * prices[m] for m, a in inputs.items())
        p_all = 1.0
        for m in inputs:
            p_all *= choice[m][1]
        return prices, cost, p_all

    def _expected_price(self, m: str, tables) -> float:
        model = self.clear[m]
        r = tables[m][0][2] / exp(tables[m][0][0])
        return r * (model.pi0 + (1.0 - model.pi0) * exp(max(model.x, 0.0) + 0.5 * model.sd ** 2))

    def _saa(self, plan, inputs_of, values, stock, lots, tables) -> dict[str, float]:
        """Joint per-material bid prices by sample average approximation."""
        q = self.q
        need: dict[str, int] = {}
        for g, units in plan.items():
            for m, a in inputs_of[g].items():
                need[m] = need.get(m, 0) + a * units
        need = {m: max(0, n - stock.get(m, 0)) for m, n in need.items() if m in lots}
        need = {m: n for m, n in need.items() if n > 0}
        if not need:
            return {}
        mats = sorted(need)
        n_s = max(2, int(q["saa_samples"]))
        rho = q["clear_corr"]
        # Stratified, deterministic draws: each material walks the strata with
        # its own stride (coprime with n_s), so the samples are a permutation.
        strides = [st for st in (7, 5, 11, 13, 3, 17, 19, 23) if gcd(st, n_s) == 1] or [1]
        scen = []
        for s in range(n_s):
            zc = _INV((s + 0.5) / n_s)
            row = {}
            for j, m in enumerate(mats):
                st = strides[j % len(strides)]
                k = (s * st + 3 * (j + 1)) % n_s
                zi = _INV((k + 0.5) / n_s)
                u = ((s * strides[(j + 1) % len(strides)] + 2 * j + 1) % n_s + 0.5) / n_s
                model = self.clear[m]
                if u < model.pi0:
                    row[m] = -1.0  # uncontested: any bid at the reserve wins
                else:
                    row[m] = model.x + model.sd * (sqrt(rho) * zc + sqrt(1.0 - rho) * zi)
            scen.append(row)
        order = sorted(plan, key=lambda g: -values[g])
        salvage = {m: q["salvage"] * lots[m].reserve_price for m in mats}

        def payoff(sel: dict[str, tuple[float, float, float]]) -> float:
            total = 0.0
            for row in scen:
                avail = dict(stock)
                pay = 0.0
                for m in mats:
                    y, _, price = sel[m]
                    if y >= row[m]:
                        avail[m] = avail.get(m, 0) + need[m]
                        pay += price * need[m]
                gain = 0.0
                for g in order:
                    ins = inputs_of[g]
                    units = min([plan[g]] + [avail.get(m, 0) // a for m, a in ins.items()])
                    if units > 0:
                        gain += units * values[g]
                        for m, a in ins.items():
                            avail[m] -= a * units
                gain += sum(avail.get(m, 0) * salvage[m] for m in mats)
                total += gain - pay
            return total / n_s

        sel = {m: min(tables[m], key=lambda row: abs(row[0] - max(self.clear[m].x, 0.0005))) for m in mats}
        # Only rows near the plausible clearing range are worth testing.
        near = {}
        for m in mats:
            model = self.clear[m]
            lo, hi = model.x - 1.5 * model.sd, model.x + 2.5 * model.sd
            near[m] = [r for r in tables[m] if r[0] <= 0.001 or lo <= r[0] <= hi] or tables[m]
        best = payoff(sel)
        for _ in range(2):
            for m in mats:
                if self._late():
                    return {m: sel[m][2] for m in mats}
                for row in near[m]:
                    if row is sel[m]:
                        continue
                    trial = dict(sel)
                    trial[m] = row
                    val = payoff(trial)
                    if val > best:
                        best, sel = val, trial
        return {m: sel[m][2] for m in mats}

    def _late(self) -> bool:
        """True once this turn has used its compute budget (deadlines are wall clock)."""
        return perf_counter() - self._t0 > TIME_BUDGET

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------
    def act(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        try:
            return self._act(state, goods_config)
        except Exception as exc:  # never hand the runner an exception: fall back
            if DEBUG:
                raise
            self.errors += 1
            self.last_error = f"turn {state.turn}: {traceback.format_exc()}"
            return self._fallback(state, goods_config)

    def _fallback(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        listings = [
            Listing(g, int(n), max(self.center.get(g, 1.0), 1e-3))
            for g, n in state.inventory.items()
            if g in goods_config.finished_goods and n > 0
        ]
        self.last_bids, self.last_listings = {}, {g.good: [(g.price, g.quantity)] for g in listings}
        return Decision(listings=listings)

    def _act(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self._t0 = perf_counter()
        q = self.q
        inv = dict(state.inventory)
        lots = {lot.good: lot for lot in state.auction_lots}
        candidates = self._identify(state)
        own_buys, own_sales, rival_buys = self._digest(state, candidates)

        for m, lot in lots.items():
            if m not in self.clear:
                self.clear[m] = Clearing(q["clear0"], q["clear_sd0"], q["uncontested0"])
            r = max(lot.reserve_price, 1e-3)
            self.peak_res[m] = max(self.peak_res.get(m, 0.0), r)
            self.avg_res[m] = 0.8 * self.avg_res.get(m, r) + 0.2 * r
        self._learn_clearing(own_buys, rival_buys)

        # Pacing (Balseiro & Gur): raise the multiplier when spend ran ahead of plan.
        if q["pacing_gain"] > 0 and self.target_spend > 0:
            ratio = self.spent_last / self.target_spend - 1.0
            self.mu = max(0.0, min(4.0, self.mu + q["pacing_gain"] * 0.5 * ratio))
        self.target_spend = q["spend_rate"] * max(state.capital, 0.0)

        tables = self._material_tables(lots)
        exp_price = {m: self._expected_price(m, tables) for m in lots}
        recipes = cheapest_recipes(goods_config, exp_price)
        reserve_cost = cheapest_recipes(goods_config, {m: l.reserve_price for m, l in lots.items()})
        goods = list(goods_config.finished_goods)
        inputs_of = {g: goods_config.recipes[recipes[g][0]].inputs for g in recipes}
        known_refs = [q["price_mult0"] * c for _, c in reserve_cost.values() if c > 0]
        fallback_ref = max(known_refs) if known_refs else 10.0
        for g in goods:
            if g not in self.demand:
                ref = q["price_mult0"] * reserve_cost[g][1] if g in reserve_cost else fallback_ref
                ref = max(ref, 0.5)
                self.demand[g] = Demand(ref, q["accept0"], q["slope0"], q["prior_weight"])
                self.center[g] = ref
        self._learn_demand(inv, own_sales)

        cont = self.horizon.cont(state.turn, DP_STEPS)
        remaining = Horizon.remaining(cont)
        self._cont, self._remaining = cont, remaining
        self._repl = {g: c for g, (_, c) in recipes.items()}
        cache: dict = {}
        pace = 1.0 / (1.0 + self.mu)

        # --- regime signals (Ketter): drained lots vs glutted lots -----------
        scarce = {m: 1.0 if l.reserve_price >= 0.97 * self.peak_res[m] else 0.0 for m, l in lots.items()}
        glut = {m: l.reserve_price <= q["glut_ratio"] * self.peak_res[m] for m, l in lots.items()}

        # --- values, kit bids and quantities --------------------------------
        stock = {m: inv.get(m, 0) for m in goods_config.raw_materials}
        salvage = {m: lots[m].reserve_price for m in lots}
        buying = remaining >= q["stop_turns"] and state.capital > 0
        plan: dict[str, int] = {}
        kit_prices: dict[str, dict[str, float]] = {}
        unit_value: dict[str, float] = {}
        targets: dict[str, int] = {}
        priority: dict[str, float] = {}
        for g in goods:
            if g not in recipes:
                continue
            ins = inputs_of[g]
            shelf = inv.get(g, 0)
            v_now = self._value(cache, g, shelf)
            unit_value[g] = v_now
            if not buying or any(m not in lots for m in ins):
                continue
            boost = 1.0 + (q["scarce_boost"] * max(scarce[m] for m in ins) if q["regime"] else 0.0)
            v_bid = v_now * pace * q["aggression"] * boost
            prices, kit_cost, p_all = self._kit_bid(v_bid, ins, tables, salvage)
            if kit_cost <= 0:
                continue
            kit_prices[g] = prices
            cap_lot = min(
                int(q["max_lot_share"] * (lots[m].quantity + stock.get(m, 0)) / a) for m, a in ins.items()
            )
            hurdle = (1.0 + q["margin"]) * kit_cost
            best = 0
            # Cover / base stock: a unit must sell within `cover` turns at least
            # half the time at the price the pricing module would set.
            need_sell = 1.0 - 0.5 ** (1.0 / max(q["cover"], 1.0))
            for delta in _candidates(cap_lot):
                if self._late():
                    break
                total = shelf + delta
                if q["quantity"] == "value":
                    mv = (total * self._value(cache, g, total) - shelf * v_now) / delta
                    ok = mv * pace >= hurdle
                else:
                    v_tot, p_unit = self._eval(cache, g, total)
                    sell = self.demand[g].prob(p_unit, self._offset(total))
                    ok = v_tot * pace >= hurdle and sell >= need_sell
                if not ok:
                    break
                best = delta
            targets[g] = best
            priority[g] = (v_now * pace - kit_cost) / kit_cost

        order = sorted(targets, key=lambda g: -priority[g])

        # --- turn targets into material tranches, budget and lot caps -------
        tranches: dict[str, dict[float, int]] = {}
        left = dict(stock)
        budget = q["budget_share"] * max(state.capital, 0.0)
        if q["quantity"] == "basestock":
            need: dict[str, float] = {}
            best_price: dict[str, float] = {}
            for g in order:
                for m, a in inputs_of[g].items():
                    need[m] = need.get(m, 0.0) + a * targets[g]
                    best_price[m] = max(best_price.get(m, 0.0), kit_prices[g][m])
            for m in sorted(need, key=lambda m: -best_price[m]):
                factor = (self.avg_res[m] / max(lots[m].reserve_price, 1e-3)) ** q["kappa"]
                units = int(need[m] * factor) - left.get(m, 0)
                units = min(units, int(q["max_lot_share"] * lots[m].quantity),
                            int(budget / max(best_price[m], 1e-9)))
                if units > 0:
                    tranches.setdefault(m, {})[best_price[m]] = units
                    budget -= units * best_price[m]
            plan = {g: targets[g] for g in order}
        else:
            for g in order:
                ins = inputs_of[g]
                units = targets[g]
                while units > 0:
                    cost = sum(max(0, a * units - left.get(m, 0)) * kit_prices[g][m] for m, a in ins.items())
                    if cost <= budget:
                        break
                    units = units * 3 // 4 if units > 4 else units - 1
                if units <= 0:
                    continue
                plan[g] = units
                for m, a in ins.items():
                    need = a * units
                    used = min(left.get(m, 0), need)
                    left[m] = left.get(m, 0) - used
                    buy = need - used
                    if buy > 0:
                        cell = tranches.setdefault(m, {})
                        price = kit_prices[g][m]
                        cell[price] = cell.get(price, 0) + buy
                        budget -= buy * price

        # Glut regime: stock up on materials whose reserve collapsed.
        if q["regime"] and buying and q["glut_cover"] > 0:
            for g in order:
                if not targets[g]:
                    continue
                for m, a in inputs_of[g].items():
                    if glut.get(m) and m in lots:
                        extra = int(q["glut_cover"] * a * max(1.0, targets[g] / max(q["cover"], 1.0)))
                        price = lots[m].reserve_price * 1.0005
                        extra = min(extra, int(budget / max(price, 1e-9)))
                        if extra > 0:
                            cell = tranches.setdefault(m, {})
                            cell[price] = cell.get(price, 0) + extra
                            budget -= extra * price

        # Joint refinement of material prices by sample average approximation.
        if q["kits"] == "saa" and plan and tranches and not self._late():
            joint = self._saa(plan, inputs_of, {g: unit_value[g] * pace for g in plan}, stock, lots, tables)
            for m, price in joint.items():
                if m in tranches:
                    tranches[m] = {price: sum(tranches[m].values())}

        bids = []
        self.last_bids = {}
        for m, cells in tranches.items():
            lot = lots[m]
            cap = int(q["max_lot_share"] * lot.quantity)
            rows = []
            for price, units in sorted(cells.items(), reverse=True):
                units = min(units, cap)
                if units <= 0 or price < lot.reserve_price:
                    continue
                cap -= units
                bids.append(Bid(m, int(units), float(price)))
                rows.append((float(price), int(units)))
            if rows:
                self.last_bids[m] = rows
        self.last_reserve = {m: max(l.reserve_price, 1e-3) for m, l in lots.items()}
        self.last_lot = {m: l.quantity for m, l in lots.items()}

        # --- assembly in value order, then spare stock ----------------------
        build_order = sorted(unit_value, key=lambda g: -unit_value[g])
        assembly = [AssemblyOrder(recipes[g][0], int(plan[g])) for g in build_order if plan.get(g)]
        for g in build_order:
            repl = sum(a * exp_price[m] for m, a in inputs_of[g].items())
            if unit_value[g] >= q["leftover_margin"] * repl:
                assembly.append(AssemblyOrder(recipes[g][0], BIG))

        # --- listings ---------------------------------------------------------
        # Raw stock not claimed by the plan may still be built (leftover orders).
        spare_raw = dict(stock)
        for g, units in plan.items():
            for m, a in inputs_of[g].items():
                spare_raw[m] = max(0, spare_raw.get(m, 0) - a * units)
        listings = []
        self.last_listings = {}
        for g in goods:
            spare = 0
            if g in recipes:
                spare = min(spare_raw.get(m, 0) // a for m, a in inputs_of[g].items())
            expected = inv.get(g, 0) + plan.get(g, 0) + spare
            if expected <= 0:
                continue
            repl = recipes[g][1] if g in recipes else 0.0
            _, price, hold = self._policy(g, expected, cont, remaining, repl)
            self.center[g] = price
            floor = hold
            if q["basis_floor"] > 0 and remaining >= 2.0:
                floor = max(floor, q["basis_floor"] * repl)
            listings += self._ladder(g, expected, price, floor)
        return Decision(bids=bids, assembly=assembly, listings=listings)

    def _ladder(self, good: str, units: int, price: float, floor: float) -> list[Listing]:
        q = self.q
        n_rungs = max(1, int(q["rungs"]))
        if n_rungs == 1:
            zs = [0.0]
        else:
            zs = sorted((-1.0 + 2.0 * i / (n_rungs - 1) for i in range(n_rungs)), key=abs)
        prices = [max(price * exp(q["explore"] * z), floor, 1e-3) for z in zs]
        rows: list[tuple[float, int]] = []
        if q["listing"] == "units":
            for i in range(min(units, MAX_UNIT_LISTINGS)):
                rows.append((prices[i % len(prices)], 1))
            rows.append((prices[0], BIG))   # absorbs anything built beyond the plan
        else:
            base, extra = divmod(units, len(prices))
            for i, p in enumerate(prices):
                n = base + (1 if i < extra else 0)
                if n > 0:
                    rows.append((p, n))
            rows.append((prices[0], BIG))
        self.last_listings[good] = rows
        return [Listing(good, n, float(p)) for p, n in rows]


def _candidates(cap: int) -> list[int]:
    """Geometric ladder of extra units to test: 1, 2, 3, 4, 6, 8, 12, ..."""
    out, n = [], 1
    while n <= cap:
        out.append(n)
        n = n + 1 if n < 4 else int(n * 1.5)
    return out
