"""Helpers shared by the skills: guarded moves, the collision-safe transit, part geometry.

All geometry comes from the TaskSpec's nominal geometry. The constants here are the
skill library's own tuning (approach heights, standoffs, grip widths and forces), the
same numbers a cell integrator would set once.
"""

from __future__ import annotations

import math

import numpy as np

from ..runtime.backend import ActResult, Grip, MoveTo
from ..runtime.backend import Pose as BackendPose
from .base import Pose, SkillContext, SkillFailure

TRANSIT_Z = 0.42          # safe height for moves between the tray and the rack
STAGE_BACK = 0.12         # staging point this far behind a pre-pose, clear of fixtures
BOX_GRASP_DEPTH = 0.012   # TCP this far inside the box's front edge
OPEN_WIDTH = {"box": 0.075, "plug": 0.030}
GRIP_FORCE = {"box": 70.0, "plug": 30.0}


def direction(yaw_deg: float) -> np.ndarray:
    a = math.radians(yaw_deg)
    return np.array([math.cos(a), math.sin(a), 0.0])


def lateral(yaw_deg: float) -> np.ndarray:
    """Unit vector to the left of the approach direction (horizontal)."""
    a = math.radians(yaw_deg)
    return np.array([-math.sin(a), math.cos(a), 0.0])


def tool_rotation(yaw_deg: float) -> np.ndarray:
    """The canonical tool frame (columns x, y, z) at approach yaw: z forward, y down, x to the right."""
    d = direction(yaw_deg)
    left = lateral(yaw_deg)
    return np.column_stack([-left, np.array([0.0, 0.0, -1.0]), d])


def xyz(p) -> np.ndarray:
    if isinstance(p, Pose):
        return np.array([p.x, p.y, p.z])
    return np.asarray(p, float)


def pose_out(p, yaw_deg: float) -> Pose:
    p = xyz(p)
    return Pose(x=round(float(p[0]), 5), y=round(float(p[1]), 5), z=round(float(p[2]), 5),
                yaw_deg=round(float(yaw_deg), 3))


def move(ctx: SkillContext, p, yaw_deg: float, speed: float = 0.12, *, compliant: bool = False,
         guard_force: float | None = None, what: str = "move") -> ActResult:
    p = xyz(p)
    try:
        r = ctx.act(MoveTo(pose=BackendPose(position=tuple(float(v) for v in p), yaw_deg=float(yaw_deg)),
                           speed=float(speed), compliant=compliant, guard_force=guard_force))
    except SkillFailure as exc:
        if exc.failure_class == "safety":  # say which move it was: the failure summary is all the writer sees
            raise SkillFailure(f"{exc.reason} (during '{what}')", outputs=exc.outputs, failure_class="safety") from None
        raise
    if not r.ok:
        raise SkillFailure(f"{what} failed: {r.reason}")
    return r


def grip_open(ctx: SkillContext, width: float) -> ActResult:
    return ctx.act(Grip(width=float(width)))


def grip_close(ctx: SkillContext, force: float) -> ActResult:
    return ctx.act(Grip(force=float(force)))


def transit(ctx: SkillContext, pre, yaw_deg: float, speed: float = 0.15, stage_back: float = STAGE_BACK) -> None:
    """Collision-safe path to ``pre``: rise to TRANSIT_Z, fly above a staging point behind ``pre``,
    descend there, then move in. Skips the detour when already close."""
    pre = xyz(pre)
    (x, y, z), cur_yaw = ctx.tcp()
    if np.linalg.norm(np.array([x, y, z]) - pre) < 0.03 and abs(cur_yaw - yaw_deg) < 3.0:
        move(ctx, pre, yaw_deg, 0.05, what="approach")
        return
    stage = pre - stage_back * direction(yaw_deg)
    if z < TRANSIT_Z - 0.01:
        move(ctx, (x, y, TRANSIT_Z), cur_yaw, speed, what="rise to transit height")
    try:
        move(ctx, (stage[0], stage[1], TRANSIT_Z), yaw_deg, speed, what="transit")
    except SkillFailure as exc:
        if "unreachable" not in str(exc):
            raise
        # close to the base the arm cannot hold the forward tool that high: fly over 6 cm lower
        move(ctx, (stage[0], stage[1], TRANSIT_Z - 0.06), yaw_deg, speed, what="transit (lower)")
    if np.linalg.norm(stage - pre) > 1e-6:
        move(ctx, stage, yaw_deg, 0.10, what="descend to the staging point")
    move(ctx, pre, yaw_deg, 0.06, what="move to the pre-pose")


def rise(ctx: SkillContext, speed: float = 0.10) -> None:
    (x, y, z), yaw = ctx.tcp()
    if z < TRANSIT_Z - 0.005:
        move(ctx, (x, y, TRANSIT_Z), yaw, speed, what="rise")


# ------------------------------------------------------------------------- part geometry
def box_dims(ctx: SkillContext) -> tuple[float, float, float]:
    s = ctx.geom["box"]["size"]
    return float(s["depth_x"]), float(s["width_y"]), float(s["height_z"])


def plug_dims(ctx: SkillContext) -> dict:
    p = ctx.geom["plug"]
    return {k: (float(v["length_x"]), float(v["width_y"]), float(v["height_z"])) for k, v in p.items() if isinstance(v, dict)}


def grasp_offset(ctx: SkillContext, part: str) -> float:
    """Signed distance along the part's x axis from its centre to the grasp point."""
    if part == "box":
        return -box_dims(ctx)[0] / 2 + BOX_GRASP_DEPTH
    if part == "plug":
        d = plug_dims(ctx)
        return -d["body"][0] / 2 - d["boot"][0] / 2
    raise SkillFailure(f"unknown part {part!r}; known: box, plug")


def held_point_in_tool(ctx: SkillContext, part: str) -> np.ndarray:
    """Where the held part's reference point sits in the tool frame for a nominal grasp:
    the box centre, or the plug tip."""
    off = grasp_offset(ctx, part)
    if part == "box":
        return np.array([0.0, 0.0, -off])
    d = plug_dims(ctx)
    return np.array([0.0, 0.0, d["body"][0] / 2 - off])
