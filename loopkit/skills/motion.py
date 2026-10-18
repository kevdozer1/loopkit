"""Motion skills: move_to, approach, retreat."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .base import Pose, SkillContext, skill
from .common import TRANSIT_Z, direction, move, pose_out, rise, transit, xyz


class MotionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tcp: Pose = Field(description="where the tool centre point ended up")


@skill("motion", post=("the tool centre point is at the pose",))
def move_to(ctx: SkillContext,
            pose: Annotated[Pose, Field(description="target tool centre point; yaw_deg is the approach direction")],
            speed_mm_s: Annotated[float, Field(ge=10, le=250)] = 120.0,
            safe_path: Annotated[bool, Field(description="go up to the transit height and come down behind the "
                                                         "target instead of a straight line")] = True) -> MotionResult:
    """Move the tool centre point to a pose. With safe_path the hand rises to the transit height, flies over,
    descends behind the target and moves in; otherwise it moves in a straight line."""
    if safe_path:
        transit(ctx, xyz(pose), pose.yaw_deg, speed=speed_mm_s / 1000.0)
    else:
        move(ctx, xyz(pose), pose.yaw_deg, speed_mm_s / 1000.0, what="move_to")
    p, yaw = ctx.tcp()
    return MotionResult(tcp=pose_out(p, yaw))


@skill("motion", post=("the hand faces the target from offset_mm away",))
def approach(ctx: SkillContext,
             target: Annotated[Pose, Field(description="a part, slot or port pose")],
             offset_mm: Annotated[float, Field(ge=10, le=300, description="standoff in front of the target")] = 60.0,
             ) -> MotionResult:
    """Bring the tool centre point to offset_mm in front of the target along its approach direction, at the
    target's height, by the safe path."""
    pre = xyz(target) - (offset_mm / 1000.0) * direction(target.yaw_deg)
    transit(ctx, pre, target.yaw_deg)
    p, yaw = ctx.tcp()
    return MotionResult(tcp=pose_out(p, yaw))


@skill("motion", post=("the hand is clear of the fixtures at the transit height",))
def retreat(ctx: SkillContext,
            distance_mm: Annotated[float, Field(ge=10, le=200, description="how far to back away")] = 80.0) -> MotionResult:
    """Back the hand straight away from what it faces by distance_mm, then rise to the transit height."""
    p, yaw = ctx.tcp()
    back = np.asarray(p) - (distance_mm / 1000.0) * direction(yaw)
    move(ctx, back, yaw, 0.06, what="retreat")
    if back[2] < TRANSIT_Z - 0.005:
        rise(ctx)
    p, yaw = ctx.tcp()
    return MotionResult(tcp=pose_out(p, yaw))
