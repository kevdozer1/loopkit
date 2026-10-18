"""The Backend protocol: the one surface skills use to move a robot.

``observe() -> Observation``, ``act(command) -> ActResult``, ``stop()`` and
``reset(seed)``. Commands are small typed records. Poses are in the robot base
frame: a TCP position plus the approach yaw (and an optional pitch); the backend
keeps the tool in its canonical family (approach axis horizontal, finger closing
axis vertical), so a skill never has to reason about the wrist roll.
"""

from __future__ import annotations

from typing import Literal, Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import Observation

Vec3 = tuple[float, float, float]


class Pose(BaseModel):
    """TCP pose in the base frame. ``yaw_deg`` is the approach direction about world z."""

    model_config = ConfigDict(extra="forbid")

    position: Vec3
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


class MoveTo(BaseModel):
    """Straight-line TCP move.

    ``guard_force``: stop early (ok, ``stopped_on_force``) when the estimated contact force
    exceeds it. ``compliant``: soften the arm for contact work (insertion, sliding).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["move_to"] = "move_to"
    pose: Pose
    speed: float = Field(default=0.10, gt=0, description="m/s, capped by the task's max_speed")
    guard_force: float | None = Field(default=None, gt=0)
    compliant: bool = False
    timeout_s: float | None = Field(default=None, gt=0)


class MoveJoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["move_joints"] = "move_joints"
    q: list[float] = Field(min_length=7, max_length=7)
    speed: float = Field(default=0.6, gt=0, description="rad/s for the fastest joint")


class Grip(BaseModel):
    """Close with ``force`` (N per finger) until blocked, or open to ``width`` (m)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["grip"] = "grip"
    width: float | None = Field(default=None, ge=0.0, le=0.083, description="open to this width; None closes")
    force: float = Field(default=40.0, gt=0, le=140.0)


class Wait(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["wait"] = "wait"
    seconds: float = Field(gt=0, le=10.0)


Command = Union[MoveTo, MoveJoints, Grip, Wait]


class ActResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    reason: str = ""
    sim_seconds: float = 0.0
    tcp_position: Vec3 = (0.0, 0.0, 0.0)
    tcp_yaw_deg: float = 0.0
    max_force: float = 0.0
    stopped_on_force: bool = False
    gripper_width: float = 0.0
    gripper_force: float = 0.0
    safety_violation: str | None = None


class SafetyViolation(RuntimeError):
    """A command would leave the workspace, exceed the speed limit or the force limit."""


@runtime_checkable
class Backend(Protocol):
    def reset(self, seed: int) -> Observation: ...

    def observe(self) -> Observation: ...

    def act(self, command: Command) -> ActResult: ...

    def stop(self) -> None: ...
