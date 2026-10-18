"""TraceEvent: one append-only line in a run's ``events.jsonl``.

Every number in a report traces back to one of these. The payload is checked
against the model for its ``kind``; extra payload fields are kept. Declared float
fields are finite (a NaN or inf would be written as null and could not be read
back); free dicts such as ``metrics`` may hold anything JSON-safe.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .verdict import FailureClass, verdict_rule_errors

EVENT_KINDS = ("model_call", "node_start", "node_end", "verdict", "patch", "intervention", "budget_stop")
EventKind = Literal["model_call", "node_start", "node_end", "verdict", "patch", "intervention", "budget_stop"]
ROLES = ("writer", "reader", "system1")
Role = Literal["writer", "reader", "system1"]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class ModelCallPayload(_Payload):
    provider: str
    model: str
    role: Role
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    reasoning_tokens: int | None = Field(ge=0)
    usd: float = Field(ge=0)
    usd_source: Literal["provider", "computed"]
    latency_ms: float = Field(ge=0)
    cache_hit: bool
    request_hash: str


class NodeStartPayload(_Payload):
    node_id: str
    skill: str
    params: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0


class NodeEndPayload(_Payload):
    node_id: str
    skill: str
    ok: bool
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    sim_seconds: float = Field(ge=0)


class VerdictPayload(_Payload):
    """A Verdict, flattened: evidence.metrics and evidence.image_paths sit at the top level (a
    Verdict-shaped payload is flattened on the way in; top-level keys win). The Verdict rules hold:
    a passed verdict has score 1, no failure_class and no failing_node; a failed one has a
    failure_class and a score below 1."""

    passed: bool
    score: float = Field(ge=0, le=1)
    failure_class: FailureClass | None = None
    failing_node: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    image_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _flatten_evidence(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "evidence" not in data:
            return data
        out = dict(data)
        evidence = out.pop("evidence")
        if isinstance(evidence, dict):
            out.setdefault("metrics", evidence.get("metrics", {}))
            out.setdefault("image_paths", evidence.get("image_paths", []))
            rest = {k: v for k, v in evidence.items() if k not in ("metrics", "image_paths")}
            if rest:
                out["evidence"] = rest
        elif evidence is not None:
            out["evidence"] = evidence
        return out

    @model_validator(mode="after")
    def _consistent(self) -> "VerdictPayload":
        problems = verdict_rule_errors(self.passed, self.score, self.failure_class, self.failing_node)
        if problems:
            raise ValueError("; ".join(problems))
        return self


class PatchPayload(_Payload):
    round: int = Field(ge=0)
    graph_before_hash: str | None
    graph_after_hash: str | None
    failure_summary: str


class InterventionPayload(_Payload):
    reason: str
    actor: str = "human"


class BudgetStopPayload(_Payload):
    scope: str
    cap_usd: float
    spent_usd: float
    reason: str


PAYLOAD_MODELS: dict[str, type[_Payload]] = {
    "model_call": ModelCallPayload,
    "node_start": NodeStartPayload,
    "node_end": NodeEndPayload,
    "verdict": VerdictPayload,
    "patch": PatchPayload,
    "intervention": InterventionPayload,
    "budget_stop": BudgetStopPayload,
}


class TraceEvent(BaseModel):
    """One trace line. ``payload`` follows the model for ``kind`` (the JSON Schema lists one
    branch per kind); extra payload fields are kept."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    run_id: str
    ts: float = Field(description="Unix time, seconds")
    seq: int = Field(ge=0)
    kind: EventKind
    payload: dict[str, Any]

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> "TraceEvent":
        model = PAYLOAD_MODELS[self.kind]
        self.payload = model.model_validate(self.payload).model_dump(mode="json")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        """The model's schema plus a oneOf with one branch per kind: that kind and its payload's schema."""
        json_schema = handler(core_schema)
        target = handler.resolve_ref_schema(json_schema)
        target["oneOf"] = [
            {"properties": {"kind": {"const": kind}, "payload": handler(model.__pydantic_core_schema__)},
             "required": ["kind", "payload"]}
            for kind, model in PAYLOAD_MODELS.items()
        ]
        return json_schema
