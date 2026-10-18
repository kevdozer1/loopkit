"""Perception skills: locate, locate_port, read_panel."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from . import perception as P
from .base import Pose, SkillContext, SkillFailure, skill
from .common import pose_out


class LocateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pose: Pose = Field(description="for box and plug: the part centre; for slot: where the box centre sits when "
                                   "seated; yaw_deg is the direction to push or insert")
    confidence: float = Field(description="0 to 1; below 0.5 the estimate is doubtful")
    camera: str


class PortResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pose: Pose = Field(description="centre of the port opening on the faceplate front; yaw_deg is the insertion direction")
    occupied: bool = Field(description="a plug is already in this port")
    confidence: float


class PanelReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: dict[str, dict[str, object]] = Field(
        description='{"1": {"occupied": bool, "led": "green|amber|red|off"}, ..., "12": {...}}')


def rack_estimate(ctx: SkillContext, refresh: bool = False) -> P.RackEstimate:
    """The rack pose from the front camera, cached for the run."""
    if refresh or "rack" not in ctx.cache:
        obs = ctx.observe()
        try:
            ctx.cache["rack"] = P.estimate_rack_pose(obs, ctx.geom)
        except P.PerceptionError as exc:
            raise SkillFailure(f"could not localize the rack: {exc}", failure_class="perception") from None
    return ctx.cache["rack"]


def _confidence(residual_px: float, n: int, n_expected: int) -> float:
    return round(max(0.0, min(1.0, (1.0 - residual_px / 3.0) * n / n_expected)), 3)


@skill("perception", post=("returns the pose of the named part or slot",))
def locate(ctx: SkillContext,
           name: Annotated[Literal["box", "plug", "slot", "rack"], Field(description="what to find")],
           index: Annotated[int | None, Field(ge=1, le=2, description="slot number (1 lower, 2 upper); "
                                                                      "required for slot")] = None) -> LocateResult:
    """Find a part on the tray or a fixture on the rack with the cameras. box and plug use the overhead
    camera; slot and rack use the front camera's view of the patch panel. The returned yaw_deg is the
    approach direction (away from the robot)."""
    if name in ("box", "plug"):
        obs = ctx.observe()
        try:
            est = P.locate_box(obs, ctx.geom) if name == "box" else P.locate_plug(obs, ctx.geom)
        except P.PerceptionError as exc:
            raise SkillFailure(f"could not find the {name}: {exc}", failure_class="perception") from None
        ctx.cache[f"part_{name}"] = est
        return LocateResult(pose=pose_out((est.x, est.y, est.z), math.degrees(est.yaw)), confidence=1.0,
                            camera="overhead")
    est = rack_estimate(ctx, refresh=True)
    conf = _confidence(est.residual_px, est.n_ports, int(ctx.geom["ports"]["count"]))
    yaw = math.degrees(est.yaw)
    if name == "rack":
        return LocateResult(pose=pose_out((est.x, est.y, 0.0), yaw), confidence=conf, camera="front")
    if index is None:
        raise SkillFailure("locate slot needs index (1 or 2)")
    seated = ctx.geom["slots"]["seated_box_center"].get(str(index))
    if seated is None:
        raise SkillFailure(f"there is no slot {index}")
    return LocateResult(pose=pose_out(est.to_world(seated), yaw), confidence=conf, camera="front")


@skill("perception", post=("returns the pose of the port opening",))
def locate_port(ctx: SkillContext,
                n: Annotated[int, Field(ge=1, le=12, description="port number on the label strip")]) -> PortResult:
    """Find patch-panel port n with the front camera (the rack pose fitted to all 12 ports). Returns the
    centre of the port opening on the faceplate front and the insertion direction."""
    est = rack_estimate(ctx, refresh=True)
    geom = ctx.geom
    p_rack = (float(geom["rack"]["faceplate_front_x"]), float(geom["ports"]["center_y"][str(n)]),
              float(geom["ports"]["center_z"]))
    conf = _confidence(est.residual_px, est.n_ports, int(geom["ports"]["count"]))
    return PortResult(pose=pose_out(est.to_world(p_rack), math.degrees(est.yaw)), occupied=n in est.occupied_seen,
                      confidence=conf)


READER_SYSTEM = (
    "You inspect a 12-port patch panel in a robot cell from a camera image. Ports are numbered 1 to 12 on the "
    "white label strip under the ports. Above each port is a small LED: green, amber, red, or off (dark grey). "
    "A port is occupied when a plug (yellow with a black boot) is in it. Answer with JSON only."
)


@skill("perception", big_model=True, post=("returns the panel report",))
def read_panel(ctx: SkillContext) -> PanelReport:
    """Read the patch panel: for each of the 12 ports, whether a plug is in it and the LED colour.
    Calls the vision model configured as this run's reader once, with the front camera image."""
    if ctx.chat is None or ctx.reader_model is None:
        raise SkillFailure("no reader model is configured for read_panel")
    obs = ctx.observe()
    image = obs.image("front")
    schema = ctx.task.output_schema or {"type": "object"}
    question = ("Report every port 1 to 12 as JSON: {\"1\": {\"occupied\": true|false, \"led\": \"green|amber|red|off\"}, "
                "..., \"12\": {...}}. Read the port numbers from the label strip.")
    ctx.images["read_panel_front"] = image
    kwargs = dict(ctx.chat_kwargs)  # the run's reader settings (bakeoff.yaml reader:) win over these defaults
    max_tokens = int(kwargs.pop("max_tokens", 1500))
    reasoning = kwargs.pop("reasoning", "disabled")
    resp = ctx.chat(ctx.reader_model, [{"role": "system", "content": READER_SYSTEM},
                                       {"role": "user", "content": question}],
                    images=[image], json_schema=schema, role="reader", max_tokens=max_tokens, reasoning=reasoning,
                    schema_name="panel_report", **kwargs)
    ctx.model_calls += 1
    ctx.model_usd += float(getattr(resp, "usd", 0.0) or 0.0)
    data = getattr(resp, "json", None)
    if not isinstance(data, dict):
        try:
            data = json.loads(getattr(resp, "text", "") or "")
        except (ValueError, TypeError):
            data = None
    if not isinstance(data, dict):
        raise SkillFailure(f"the reader's answer is not a JSON object: {getattr(resp, 'json_error', '') or 'empty'}",
                           failure_class="perception")
    report = {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return PanelReport(report=report)
