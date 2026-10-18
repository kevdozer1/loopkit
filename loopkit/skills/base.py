"""The skill decorator, the skill context and the registry.

A skill is a typed function ``fn(ctx, **params) -> ReturnsModel`` with a docstring.
The decorator builds its SkillSpec from the signature and the docstring (params as a
closed JSON Schema, returns from the return annotation). The catalog the writer sees is
built from these specs only.

Skills see the robot through ``ctx.observe()`` and ``ctx.act()`` (the Backend protocol),
plus the TaskSpec's nominal geometry. They never touch simulator state.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from ..runtime.backend import ActResult, Command
from ..schemas import Observation, SkillSpec, TaskSpec


class SkillFailure(RuntimeError):
    """A skill could not do its job. ``failure_class`` hints at the cause for reports."""

    def __init__(self, reason: str, outputs: dict | None = None, failure_class: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.outputs = outputs or {}
        self.failure_class = failure_class


class Pose(BaseModel):
    """A pose in the robot base frame (metres, degrees)."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float
    yaw_deg: float = Field(description="approach direction about world z: the fixture's or part's x axis")


class NoOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class SkillContext:
    """What a running skill may use: the backend, the task and its own run state."""

    backend: Any
    task: TaskSpec
    trace: Any = None
    reader_model: str | None = None
    chat: Callable | None = None          # router.chat, injectable for tests
    chat_kwargs: dict = field(default_factory=dict)
    held: dict | None = None              # set by grasp, cleared by release
    calibration: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    last_tcp: tuple[float, float, float] | None = None
    last_yaw_deg: float | None = None
    model_calls: int = 0
    model_usd: float = 0.0
    images: dict = field(default_factory=dict)

    @property
    def geom(self) -> dict:
        return self.task.nominal_geometry

    def observe(self) -> Observation:
        obs = self.backend.observe()
        self.last_tcp = tuple(obs.tcp_position)
        R = obs.tcp_rotation
        self.last_yaw_deg = math.degrees(math.atan2(R[1][2], R[0][2]))
        return obs

    def act(self, command: Command) -> ActResult:
        result = self.backend.act(command)
        self.last_tcp = tuple(result.tcp_position)
        self.last_yaw_deg = result.tcp_yaw_deg
        if result.safety_violation:
            raise SkillFailure(f"safety stop: {result.safety_violation}", failure_class="safety")
        return result

    def tcp(self) -> tuple[tuple[float, float, float], float]:
        if self.last_tcp is None or self.last_yaw_deg is None:
            self.observe()
        return self.last_tcp, self.last_yaw_deg


@dataclass
class Skill:
    name: str
    fn: Callable
    params_model: type[BaseModel]
    returns_model: type[BaseModel]
    spec: SkillSpec


REGISTRY: dict[str, Skill] = {}


def _closed(schema: dict) -> dict:
    schema = dict(schema)
    schema["additionalProperties"] = False
    schema.setdefault("properties", {})
    schema.setdefault("type", "object")
    return schema


def skill(category: str, *, terminal: str | None = None, big_model: bool = False,
          pre: tuple[str, ...] = (), post: tuple[str, ...] = ()) -> Callable:
    """Register a typed skill function and generate its SkillSpec."""

    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True)
        fields: dict[str, Any] = {}
        for name, p in list(sig.parameters.items())[1:]:
            default = ... if p.default is inspect.Parameter.empty else p.default
            fields[name] = (hints[name], default)
        params_model = create_model(f"{fn.__name__}_params", __config__=ConfigDict(extra="forbid"), **fields)
        returns_model = hints.get("return")
        if returns_model is None or returns_model is type(None):  # "-> None" means no outputs
            returns_model = NoOutput
        params_schema = _closed(params_model.model_json_schema())
        params_schema.pop("title", None)
        returns_schema = _closed(returns_model.model_json_schema())
        returns_schema.pop("title", None)
        doc = inspect.getdoc(fn) or fn.__name__
        spec = SkillSpec(name=fn.__name__, docstring=doc, params=params_schema, returns=returns_schema,
                         preconditions=list(pre), postconditions=list(post), category=category,
                         terminal=terminal, big_model=big_model)
        REGISTRY[fn.__name__] = Skill(fn.__name__, fn, params_model, returns_model, spec)
        return fn

    return deco


def call_skill(name: str, ctx: SkillContext, params: dict) -> dict:
    """Validate ``params`` against the skill's model, run it, return its outputs as JSON-safe data."""
    sk = REGISTRY[name]
    try:
        parsed = sk.params_model.model_validate(params)
    except ValidationError as exc:
        msgs = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise SkillFailure(f"bad params for {name}: {msgs}") from None
    result = sk.fn(ctx, **{k: getattr(parsed, k) for k in sk.params_model.model_fields})
    if result is None:
        return {}
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    return dict(result)


def catalog(names: list[str] | None = None) -> dict[str, SkillSpec]:
    from . import library  # noqa: F401  (registers every skill)

    specs = {n: s.spec for n, s in REGISTRY.items()}
    if names is None:
        return specs
    return {n: specs[n] for n in names if n in specs}
