"""Registry of strategy families.

Each family module exposes ``DEFAULTS`` (all parameters with defaults) and
``AGENT`` (a ``BaseAgent`` subclass that accepts parameter overrides as keyword
arguments). Family modules import only the public agent interface and, at most,
``.common``. That keeps them copyable into an upload bundle as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from . import adaptive, glut, portfolio, rival, valuation


@dataclass(frozen=True)
class Family:
    name: str
    module: ModuleType

    @property
    def defaults(self) -> dict[str, Any]:
        return self.module.DEFAULTS

    @property
    def agent_class(self) -> type:
        return self.module.AGENT

    @property
    def source(self) -> Path:
        return Path(self.module.__file__)

    def build(self, params: dict[str, Any]):
        return self.agent_class(**params)

    def complete(self, params: dict[str, Any]) -> dict[str, Any]:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        return {**self.defaults, **params}


FAMILIES: dict[str, Family] = {
    m.__name__.rsplit(".", 1)[-1]: Family(m.__name__.rsplit(".", 1)[-1], m)
    for m in (adaptive, valuation, glut, rival, portfolio)
}
COMMON = Path(__file__).with_name("common.py")


def family(name: str | None) -> Family:
    name = name or "adaptive"
    if name not in FAMILIES:
        raise KeyError(f"unknown strategy family {name!r}; available: {', '.join(FAMILIES)}")
    return FAMILIES[name]
