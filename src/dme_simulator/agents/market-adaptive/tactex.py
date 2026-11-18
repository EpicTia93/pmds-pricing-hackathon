"""TacTex family: every module of the engine open to evolution, plus regime switching.

Sources: Pardoe & Stone (TacTex, TAC SCM 2005 and 2006 winner), predict then
optimise; Kiekintveld, Miller, Jordan & Wellman (2007), Deep Maize forecasting
from every available signal; Ketter, Collins, Gini, Gupta & Schrater (2009),
scarcity and oversupply regimes; plus every building block of the revman,
auction and capacity families.

This is the combination family. The pricing module (dp, myopic, runout),
listing (units, rungs), quantity rule (value, cover, basestock), bid rule
(kit best response, quantile, shade), kit refinement (independent, saa),
pacing and the regime switch are all parameters. Evolution searches over
which strategies to combine as well as their constants. With ``regime`` on,
kits whose lots were drained get ``scarce_boost`` more value, and materials
whose reserve collapsed (a glut) are stocked ``glut_cover`` turns ahead at
the reserve.
"""

from __future__ import annotations

from typing import Any

from .econ import ENGINE_DEFAULTS, Engine

HELPERS = ("econ.py",)

DEFAULTS: dict[str, Any] = dict(ENGINE_DEFAULTS)


class TactexAgent(Engine):
    DEFAULTS = DEFAULTS
    FIXED: dict[str, Any] = {}


AGENT = TactexAgent
