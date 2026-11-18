"""Neutral family: never bids, builds or sells. Capital stays at its starting value.

A zero-risk benchmark. Any strategy that ranks below it would have done better
by sitting the game out. Because final capital equals starting capital, it
wins whenever every other team loses money. That happens in glutted or
crowded markets, so it is a meaningful opponent, not just a placeholder.
It has no parameters, so its island needs only one member.
"""

from __future__ import annotations

from typing import Any

from dme_simulator.agents import Decision, Observation
from dme_simulator.config import GoodsConfig

from .common import ParamBase

DEFAULTS: dict[str, Any] = {}


class NeutralAgent(ParamBase):
    DEFAULTS = DEFAULTS

    def act(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        return Decision()


AGENT = NeutralAgent
