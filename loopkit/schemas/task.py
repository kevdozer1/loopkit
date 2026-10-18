"""TaskSpec: what the program writer is asked to do, and what it is allowed to know."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .jsonschema_lite import check

Vec3 = tuple[float, float, float]


class AABB(BaseModel):
    """Axis-aligned box in the robot base frame, metres. ``min`` is below ``max`` on every axis."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    min: Vec3
    max: Vec3

    @model_validator(mode="after")
    def _ordered(self) -> "AABB":
        if any(lo >= hi for lo, hi in zip(self.min, self.max)):
            raise ValueError(f"AABB min {self.min} must be below max {self.max} on every axis")
        return self

    def contains(self, point: Any, margin: float = 0.0) -> bool:
        p = [float(v) for v in point]
        return all(lo - margin <= v <= hi + margin for v, lo, hi in zip(p, self.min, self.max))


class Budgets(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    max_big_model_calls: int = Field(ge=0, description="Model calls allowed while the program runs (runtime, not setup)")
    max_usd: float = Field(ge=0, description="USD allowed for runtime model calls in one program run")
    max_seconds: float = Field(gt=0, description="Sim seconds allowed for one program run")


class Safety(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    workspace_bounds: AABB = Field(description="The tool centre point must stay inside this box (base frame)")
    max_speed: float = Field(gt=0, description="Maximum commanded tool speed, m/s")
    max_force: float = Field(gt=0, description="Maximum estimated contact force at the hand, N")


class TaskSpec(BaseModel):
    """One task. ``params`` holds this instance's values (they vary per seed);
    ``params_schema`` declares them. Every key of ``params`` is declared in
    ``params_schema`` and every value satisfies its schema. A graph reads them as
    ``{"ref": "task.<name>"}``. Budget and safety numbers are finite (no inf or NaN),
    and the workspace box's min is below its max on every axis.

    ``nominal_geometry`` is the fixture geometry the robot may know (the "BIM"):
    nominal poses and dimensions. The real poses differ by the per-seed
    randomization, so skills still have to localize.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    goal_text: str
    end_condition: str
    checker_id: str
    scene_id: str
    allowed_skills: list[str]
    nominal_geometry: dict[str, Any]
    budgets: Budgets
    safety: Safety
    params_schema: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="JSON Schema for each task param, by name")
    params: dict[str, Any] = Field(
        default_factory=dict, description="This instance's param values: declared in params_schema and valid against it")
    output_schema: dict[str, Any] | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _params_declared(self) -> "TaskSpec":
        unknown = sorted(set(self.params) - set(self.params_schema))
        if unknown:
            raise ValueError(f"task params {unknown} are not declared in params_schema")
        problems = []
        for name, value in self.params.items():
            problems += check(value, self.params_schema[name], path=name)
        if problems:
            raise ValueError("task params do not satisfy params_schema: " + "; ".join(problems))
        return self

    def for_writer(self) -> dict[str, Any]:
        """The spec as the writer sees it: no seed and no per-seed param values."""
        data = self.model_dump(mode="json", exclude={"seed", "params"})
        return data
