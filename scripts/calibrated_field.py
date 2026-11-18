"""A 53-team auction stress field calibrated on the public tournament 277 cash curves.

With no challengers, default.yaml and seed 42, the synthetic field had two
opening cash movers, four profitable teams, and about $85,500 aggregate loss.
The leaderboard showed two movers, three profitable teams, and $72,013 loss.
The public capital histories do not identify competitors' bids or decisions.
All 34 synthetic bidders here are illustrative. Use other fields and fresh
seeds as well. These policies use only the ordinary agent observation.
"""

import random
from pathlib import Path

from dme_simulator.agents import AssemblyOrder, BaseAgent, Bid, Decision, Listing
from swarm.tournament import builtin_entrant, bundle_entrant, make_agent

AGENTS = Path(__file__).resolve().parents[1] / "src" / "dme_simulator" / "agents"


class KitBuyer(BaseAgent):
    def __init__(self, units, bid_multiple, retail_multiple, rotation, singles):
        self.units = units
        self.bid_multiple = bid_multiple
        self.retail_multiple = retail_multiple
        self.rotation = rotation
        self.singles = singles

    def decide(self, state, goods_config):
        materials = goods_config.raw_materials
        lots = {lot.good: lot for lot in state.auction_lots}
        need = {m: 0 for m in materials}
        recipes = list(goods_config.recipes.items())
        recipes = recipes[self.rotation:] + recipes[:self.rotation]
        assembly = []
        listings = []
        for index, (name, recipe) in enumerate(recipes):
            if any(m not in lots for m in recipe.inputs):
                continue
            good = recipe.output_good
            goal = max(0, round(self.units * (1.0 if index < 2 else 0.7) - state.inventory[good]))
            for material, amount in recipe.inputs.items():
                need[material] += goal * amount
            assembly.append(AssemblyOrder(name, max(1, goal)))
            cost = sum(lots[m].reserve_price * a for m, a in recipe.inputs.items())
            price = max(0.001, round(cost * self.retail_multiple, 3))
            listings.extend(Listing(good, 1, price) for _ in range(self.singles))
            listings.append(Listing(good, 1000, price))
        bids = []
        for i, material in enumerate(materials):
            if material not in lots:
                continue
            premium = 1.25 if self.units > 5 and i == self.rotation % len(materials) else 1.0
            quantity = max(0, need[material] - state.inventory[material])
            bids.append(Bid(material, quantity, lots[material].reserve_price * (self.bid_multiple * premium)))
        return Decision(bids, assembly, listings)


def roster(candidates=()):
    """Add swarm entrant specs, replacing matching named agents or passive seats."""
    fixed = ["papers-revman", "papers-tactex", "swarm-robust-valuation", "swarm-islands"]
    agents = {name: make_agent(bundle_entrant(AGENTS / name)) for name in fixed}
    for name in ["PriceWarAgent", "DummyAgent", "PatientAgent"]:
        agents[name] = make_agent(builtin_entrant(name))
    additions = {spec["id"]: make_agent(spec) for spec in candidates}
    passive_count = 12 - len(additions.keys() - agents.keys())
    if passive_count < 0:
        raise ValueError("The calibrated field has at most 12 available challenger seats.")
    agents.update(additions)
    for i in range(passive_count):
        agents[f"starter{i}"] = make_agent(builtin_entrant("PatientAgent"))
    rng = random.Random(541)
    for i in range(34):
        if i < 2:
            units, markup = 12.0, 4.15 + i * 0.01
            price = rng.uniform(4, 7)
            rng.random()  # Keep subsequent agents' seeded parameters stable.
            singles = 1
        elif i < 18:
            units = rng.uniform(1.3, 4)
            markup = rng.uniform(1.5, 3.5)
            price = rng.uniform(3, 6)
            singles = 1 + int(rng.random() * 4)
        else:
            units = rng.uniform(0.4, 1.15)
            markup = rng.uniform(1.05, 1.6)
            price = rng.uniform(3.2, 5.5)
            singles = 1 + int(rng.random() * 4)
        agents[f"crowd{i}"] = KitBuyer(units, markup, price, i % 5, singles)
    return agents
