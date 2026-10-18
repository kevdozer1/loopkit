"""Manipulation skills: grasp, release, slide_into, insert."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .base import Pose, SkillContext, SkillFailure, skill
from .common import (GRIP_FORCE, OPEN_WIDTH, direction, grasp_offset, grip_close, grip_open, held_point_in_tool,
                     lateral, move, pose_out, rise, tool_rotation, transit, xyz)

EXPECTED_WIDTH = {"box": (0.040, 0.050), "plug": (0.002, 0.009)}


class GraspResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holding: bool
    width_mm: float = Field(description="finger gap after closing")
    grasp_pose: Pose


class ReleaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width_mm: float


class SlideResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seated: bool = Field(description="the box reached the back stop")
    travel_mm: float
    max_force_n: float


class InsertResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seated: bool = Field(description="the tip went at least 8 mm past the faceplate")
    depth_mm: float = Field(description="estimated tip depth past the faceplate front")
    attempts: int
    max_force_n: float


@skill("manipulation", pre=("the gripper is empty",), post=("the part is held and lifted to the transit height",))
def grasp(ctx: SkillContext,
          part: Annotated[Literal["box", "plug"], Field(description="which part")],
          pose: Annotated[Pose, Field(description="the part pose, usually from locate")],
          approach_mm: Annotated[float, Field(ge=20, le=150, description="standoff before moving in")] = 60.0,
          force_n: Annotated[float | None, Field(ge=5, le=100, description="grip force per finger; default box 70, plug 30")] = None,
          ) -> GraspResult:
    """Pick a part from its stand: open, approach from the robot side along the part's yaw, close on the
    box's top and bottom faces near its front edge (or on the plug's boot), check the grip by finger width,
    lift 30 mm, back off and rise to the transit height."""
    if ctx.held is not None:
        raise SkillFailure(f"the gripper already holds the {ctx.held['part']}; release it first")
    yaw = pose.yaw_deg
    d = direction(yaw)
    g = xyz(pose) + grasp_offset(ctx, part) * d
    pre = g - (approach_mm / 1000.0) * d
    grip_open(ctx, OPEN_WIDTH[part])
    transit(ctx, pre, yaw, stage_back=0.0)  # parts stand in the open: descend straight onto the pre-grasp
    move(ctx, g, yaw, 0.04, what="move in to grasp")
    r = grip_close(ctx, force_n or GRIP_FORCE[part])
    lo, hi = EXPECTED_WIDTH[part]
    width = r.gripper_width
    if not lo <= width <= hi:
        grip_open(ctx, OPEN_WIDTH[part])
        move(ctx, pre, yaw, 0.06, what="back off after a missed grasp")
        raise SkillFailure(f"grasp missed: the fingers closed to {width * 1000:.1f} mm, expected "
                           f"{lo * 1000:.0f} to {hi * 1000:.0f} mm for the {part}",
                           outputs={"holding": False, "width_mm": round(width * 1000, 2)}, failure_class="grasp")
    move(ctx, g + np.array([0.0, 0.0, 0.03]), yaw, 0.05, what="lift")
    move(ctx, g + np.array([0.0, 0.0, 0.03]) - 0.06 * d, yaw, 0.08, what="back off")
    rise(ctx)
    ctx.held = {"part": part, "point_in_tool": held_point_in_tool(ctx, part).tolist(), "grasp_yaw_deg": yaw}
    return GraspResult(holding=True, width_mm=round(width * 1000, 2), grasp_pose=pose_out(g, yaw))


@skill("manipulation", post=("the gripper is open and empty",))
def release(ctx: SkillContext,
            width_mm: Annotated[float | None, Field(ge=10, le=80, description="opening; default 75 for the box, "
                                                                              "30 for the plug")] = None,
            ) -> ReleaseResult:
    """Open the gripper to let go of the held part. The hand does not move."""
    part = ctx.held["part"] if ctx.held else "box"
    w = (width_mm / 1000.0) if width_mm else OPEN_WIDTH[part]
    r = grip_open(ctx, w)
    ctx.held = None
    return ReleaseResult(width_mm=round(r.gripper_width * 1000, 2))


def _tcp_for(ctx: SkillContext, point, yaw_deg: float) -> np.ndarray:
    """TCP position that puts the held part's reference point at ``point``."""
    if ctx.held is None:
        raise SkillFailure("nothing is held: grasp a part first")
    return xyz(point) - tool_rotation(yaw_deg) @ np.asarray(ctx.held["point_in_tool"], float)


# Lateral offsets (mm) tried when the box catches on a rail: the rails leave 1 mm per side.
SLIDE_OFFSETS_MM = (0.0, 0.7, -0.7, 1.4, -1.4)
RAIL_ENTRY_GUARD_N = 25.0


@skill("manipulation", pre=("the box is held",), post=("the box rests on the rails against the back stop, still held",))
def slide_into(ctx: SkillContext,
               target: Annotated[Pose, Field(description="the slot pose from locate(name='slot')")],
               push_force_n: Annotated[float, Field(ge=5, le=50, description="stop pushing at this contact force")] = 30.0,
               speed_mm_s: Annotated[float, Field(ge=5, le=60)] = 30.0) -> SlideResult:
    """Slide the held box into a rack slot: line it up 10 mm in front of the rails, slide it onto the rails
    4 mm high, set it down, then push it compliantly to the back stop. If the box catches on a rail on the
    way in, or stops short of the back stop, back off and retry at small sideways offsets (up to 1.4 mm).
    Fails if the box does not reach the back stop. The box stays in the gripper; call release afterwards."""
    if ctx.held is None or ctx.held["part"] != "box":
        raise SkillFailure("slide_into needs the box in the gripper")
    yaw = target.yaw_deg
    d, left = direction(yaw), lateral(yaw)
    seat = xyz(target)
    depth = float(ctx.geom["box"]["size"]["depth_x"])
    up = np.array([0.0, 0.0, 0.004])
    pre_center = seat - (depth + 0.010) * d + up
    on_rails = seat - 0.085 * d
    transit(ctx, _tcp_for(ctx, pre_center, yaw), yaw)
    # onto the rails, guarded: a corner catching a rail stops the move well below the safety limit
    entered, max_f, shift = False, 0.0, np.zeros(3)
    for dl in SLIDE_OFFSETS_MM:
        shift = (dl / 1000.0) * left
        r = move(ctx, _tcp_for(ctx, on_rails + up + shift, yaw), yaw, 0.03, guard_force=RAIL_ENTRY_GUARD_N,
                 what="slide onto the rails")
        max_f = max(max_f, r.max_force)
        if not r.stopped_on_force:
            entered = True
            break
        move(ctx, _tcp_for(ctx, pre_center + shift, yaw), yaw, 0.03, what="back out of the rails")
    if not entered:
        raise SkillFailure(f"the box caught on the rails at every sideways offset up to {SLIDE_OFFSETS_MM[-2]} mm "
                           f"(max force {max_f:.1f} N)", outputs={"seated": False, "travel_mm": 0.0,
                                                                  "max_force_n": round(max_f, 2)},
                           failure_class="alignment")
    move(ctx, _tcp_for(ctx, on_rails + shift, yaw), yaw, 0.02, what="set down on the rails")
    start, _ = ctx.tcp()
    start = np.asarray(start)
    expected = 0.087  # from 85 mm short of the seat to 2 mm past it
    tcp_seat = _tcp_for(ctx, seat, yaw)  # the TCP when the box sits at the seat
    travel, seated = 0.0, False
    for dl in SLIDE_OFFSETS_MM:
        lat = float(shift @ left) + dl / 1000.0  # sideways offset from the seat line, m
        if dl != 0.0:
            # stopped short: back off 4 mm to the next sideways offset, then push again
            cur, _y = ctx.tcp()
            back = np.asarray(cur) - 0.004 * d
            back = back + (lat - float((back - tcp_seat) @ left)) * left
            move(ctx, back, yaw, 0.01, what="back off and shift sideways")
        r = move(ctx, tcp_seat + 0.002 * d + lat * left, yaw, speed_mm_s / 1000.0, compliant=True,
                 guard_force=push_force_n, what="push to the back stop")
        max_f = max(max_f, r.max_force)
        end, _ = ctx.tcp()
        travel = float(np.dot(np.asarray(end) - start, d))
        seated = (r.stopped_on_force and travel > expected - 0.008) or abs(travel - expected) < 0.003
        if seated:
            break
    if not seated:
        raise SkillFailure(f"the box stopped {1000 * (expected - 0.002 - travel):.0f} mm short of the back stop "
                           f"after {len(SLIDE_OFFSETS_MM)} pushes (max force {max_f:.1f} N)",
                           outputs={"seated": False, "travel_mm": round(travel * 1000, 2), "max_force_n": round(max_f, 2)},
                           failure_class="alignment")
    return SlideResult(seated=True, travel_mm=round(travel * 1000, 2), max_force_n=round(max_f, 2))


SPIRAL = [(0.0, 0.0), (0.6, 0.0), (-0.6, 0.0), (0.0, 0.6), (0.0, -0.6), (0.9, 0.9), (-0.9, 0.9), (0.9, -0.9),
          (-0.9, -0.9), (1.4, 0.0), (-1.4, 0.0)]


def insert_at(ctx: SkillContext, mouth, yaw: float, depth_mm: float, speed_mm_s: float, max_force_n: float,
              search: bool) -> InsertResult:
    if ctx.held is None or ctx.held["part"] != "plug":
        raise SkillFailure("insert needs the plug in the gripper")
    d = direction(yaw)
    left = lateral(yaw)
    up = np.array([0.0, 0.0, 1.0])
    mouth = xyz(mouth)
    transit(ctx, _tcp_for(ctx, mouth - 0.025 * d, yaw), yaw)
    max_f = 0.0
    tries = SPIRAL if search else SPIRAL[:1]
    for i, (dl, dv) in enumerate(tries, 1):
        aim = mouth + (dl * left + dv * up) / 1000.0
        move(ctx, _tcp_for(ctx, aim - 0.004 * d, yaw), yaw, 0.03, what="move to the port mouth")
        start, _ = ctx.tcp()
        r = move(ctx, _tcp_for(ctx, aim + (depth_mm / 1000.0) * d, yaw), yaw, speed_mm_s / 1000.0, compliant=True,
                 guard_force=max_force_n, what="insert")
        max_f = max(max_f, r.max_force)
        end, _ = ctx.tcp()
        tip_depth = float(np.dot(np.asarray(end) - np.asarray(start), d)) - 0.004
        if tip_depth >= 0.008:
            return InsertResult(seated=True, depth_mm=round(tip_depth * 1000, 2), attempts=i, max_force_n=round(max_f, 2))
        # blocked: back out a little before the next try
        move(ctx, _tcp_for(ctx, aim - 0.006 * d, yaw), yaw, 0.02, what="back out")
    raise SkillFailure(f"insertion blocked after {len(tries)} tries (max force {max_f:.1f} N)",
                       outputs={"seated": False, "attempts": len(tries), "max_force_n": round(max_f, 2)},
                       failure_class="alignment")


@skill("manipulation", pre=("the plug is held",), post=("the plug is in the port, still held",))
def insert(ctx: SkillContext,
           target: Annotated[Pose, Field(description="the port pose from locate_port")],
           depth_mm: Annotated[float, Field(ge=4, le=14, description="how far past the faceplate to push the tip")] = 12.0,
           speed_mm_s: Annotated[float, Field(ge=2, le=30)] = 10.0,
           max_force_n: Annotated[float, Field(ge=5, le=40, description="stop pushing at this contact force")] = 20.0,
           search: Annotated[bool, Field(description="if blocked at the mouth, retry at small offsets in a spiral")] = True,
           ) -> InsertResult:
    """Insert the held plug into a port: line the tip up 25 mm in front of the opening, move to the mouth,
    then push in compliantly along the port axis. If the plug is blocked before 8 mm and search is on,
    retry at small lateral and vertical offsets. The plug stays in the gripper; call release afterwards."""
    return insert_at(ctx, target, target.yaw_deg, depth_mm, speed_mm_s, max_force_n, search)
