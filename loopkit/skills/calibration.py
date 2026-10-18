"""Calibration skills: dance and aligned_insert.

dance follows the "little dance" idea: a few small offset moves near the target, the
target's image position measured in the wrist camera at each, and a linear regression
from image error to pose correction. aligned_insert applies the correction and inserts.
"""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from . import perception as P
from .base import Pose, SkillContext, SkillFailure, skill
from .common import direction, lateral, move, pose_out, transit, xyz
from .manipulation import InsertResult, _tcp_for, insert_at

MAX_CORRECTION_MM = 2.5


class DanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correction_mm: dict[str, float] = Field(description="lateral (left of the approach) and vertical correction "
                                                        "to apply to the target")
    refined: Pose = Field(description="the target with the correction applied")
    residual_px: float = Field(description="RMS misfit of the regression")
    points: int
    applied: bool = Field(description="false when the fit was rejected and the correction set to zero")


def _offsets(n: int, step: float) -> list[tuple[float, float]]:
    base = [(0.0, 0.0), (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
            (step, step), (-step, -step), (step, -step), (-step, step)]
    return base[:max(3, min(n, len(base)))]


@skill("calibration", post=("stores a correction for this target that aligned_insert uses",))
def dance(ctx: SkillContext,
          target: Annotated[Pose, Field(description="a port pose, usually from locate_port")],
          n: Annotated[int, Field(ge=3, le=9, description="number of offset moves")] = 5,
          step_mm: Annotated[float, Field(ge=0.5, le=5.0, description="size of each offset move")] = 1.5,
          standoff_mm: Annotated[float, Field(ge=15, le=80, description="distance of the tip (or the tool centre "
                                                                        "point if nothing is held) from the target")] = 40.0,
          ) -> DanceResult:
    """Refine a port position with the wrist camera: from a standoff in front of the target, make n small
    lateral and vertical offset moves, find the port opening in the wrist image at each, fit a linear map from
    image error to pose offset, and solve for the correction that centres the port where the camera model
    expects it. Rejects the fit (correction 0) when it is inconsistent."""
    yaw = target.yaw_deg
    d, left, up = direction(yaw), lateral(yaw), np.array([0.0, 0.0, 1.0])
    mouth = xyz(target)
    if ctx.held is not None:
        base_tcp = _tcp_for(ctx, mouth - (standoff_mm / 1000.0) * d, yaw)
    else:
        base_tcp = mouth - (standoff_mm / 1000.0) * d
    transit(ctx, base_tcp, yaw)
    offs, errs, pts = [], [], []
    desired = None
    for dl, dv in _offsets(n, step_mm / 1000.0):
        move(ctx, base_tcp + dl * left + dv * up, yaw, 0.02, what="dance move")
        obs = ctx.observe()
        if desired is None:
            # the pixel where the port appears from the centre pose if the estimate is exact; fixed for all views
            desired = P.project(obs.cameras["wrist"], mouth)[0][0]
        try:
            uv, p3, _ = P.locate_port_in_wrist(obs, mouth, math.radians(yaw), ctx.geom)
        except P.PerceptionError:
            continue
        offs.append((dl, dv))
        errs.append(np.asarray(uv) - desired)
        pts.append(p3)
    move(ctx, base_tcp, yaw, 0.02, what="return to the dance centre")
    if len(offs) < 3:
        raise SkillFailure(f"dance saw the port in only {len(offs)} of {n} views", failure_class="perception")
    O = np.array(offs)
    E = np.array(errs)
    # regression: image error e = A @ offset + e0. At the corrected target the error is zero.
    X = np.c_[O, np.ones(len(O))]
    coef, *_ = np.linalg.lstsq(X, E, rcond=None)
    A, e0 = coef[:2].T, coef[2]
    resid = E - X @ coef
    rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
    applied = True
    try:
        # the TCP offset that zeroes the image error; the target sits that far from its estimate
        target_shift = -np.linalg.solve(A, e0)
    except np.linalg.LinAlgError:
        target_shift, applied = np.zeros(2), False
    if not np.all(np.isfinite(target_shift)) or np.max(np.abs(target_shift)) * 1000 > MAX_CORRECTION_MM or rms > 4.0:
        target_shift, applied = np.zeros(2), False
    refined = mouth + target_shift[0] * left + target_shift[1] * up
    key = f"{round(mouth[0], 4)},{round(mouth[1], 4)},{round(mouth[2], 4)}"
    ctx.calibration[key] = refined.tolist()
    ctx.calibration["last"] = refined.tolist()
    return DanceResult(correction_mm={"lateral": round(float(target_shift[0]) * 1000, 3),
                                      "vertical": round(float(target_shift[1]) * 1000, 3)},
                       refined=pose_out(refined, yaw), residual_px=round(rms, 3), points=len(offs), applied=applied)


@skill("calibration", pre=("the plug is held", "dance ran on this target"), post=("the plug is in the port, still held",))
def aligned_insert(ctx: SkillContext,
                   target: Annotated[Pose, Field(description="the port pose; the correction from the last dance on "
                                                             "this target is applied")],
                   depth_mm: Annotated[float, Field(ge=4, le=14)] = 12.0,
                   speed_mm_s: Annotated[float, Field(ge=2, le=30)] = 10.0,
                   max_force_n: Annotated[float, Field(ge=5, le=40)] = 20.0,
                   search: bool = True) -> InsertResult:
    """Insert the held plug into the port using the correction from dance (run dance on the same target
    first). Otherwise the same as insert."""
    mouth = xyz(target)
    key = f"{round(mouth[0], 4)},{round(mouth[1], 4)},{round(mouth[2], 4)}"
    refined = ctx.calibration.get(key) or ctx.calibration.get("last")
    if refined is None:
        raise SkillFailure("aligned_insert needs a dance on this target first")
    return insert_at(ctx, np.asarray(refined, float), target.yaw_deg, depth_mm, speed_mm_s, max_force_n, search)
