"""SO101Backend: the Backend protocol on the SO-101 arm. A stub until S5."""

from __future__ import annotations

from ..schemas import Observation
from .backend import ActResult, Command


class SO101Backend:
    """Will bridge to LeRobot in S5 (how is decided then)."""

    def __init__(self, *args, **kwargs):
        pass

    def reset(self, seed: int) -> Observation:
        raise NotImplementedError("SO101Backend arrives in S5")

    def observe(self) -> Observation:
        raise NotImplementedError("SO101Backend arrives in S5")

    def act(self, command: Command) -> ActResult:
        raise NotImplementedError("SO101Backend arrives in S5")

    def stop(self) -> None:
        raise NotImplementedError("SO101Backend arrives in S5")
