"""Verdict: a checker's score for one program run."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FAILURE_CLASSES = ("perception", "grasp", "alignment", "stopping", "timeout", "safety", "other")
FailureClass = Literal["perception", "grasp", "alignment", "stopping", "timeout", "safety", "other"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, Any] = Field(default_factory=dict)
    image_paths: list[str] = Field(default_factory=list)


def verdict_rule_errors(passed: bool, score: float, failure_class: str | None, failing_node: str | None) -> list[str]:
    """The cross-field rules shared by Verdict and the verdict trace payload."""
    if passed:
        out = []
        if score < 1.0:
            out.append("a passed verdict has score 1")
        if failure_class is not None:
            out.append("a passed verdict has no failure_class")
        if failing_node is not None:
            out.append("a passed verdict has no failing_node")
        return out
    out = []
    if failure_class is None:
        out.append("a failed verdict needs a failure_class")
    if score >= 1.0:
        out.append("a failed verdict has a partial score, below 1")
    return out


class Verdict(BaseModel):
    """A checker's score for one run. A passed verdict has score 1, no failure_class and no
    failing_node. A failed verdict names its failure_class and has a partial score below 1."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="1 when passed, else a partial score below 1")
    evidence: Evidence = Field(default_factory=Evidence)
    failure_class: FailureClass | None = Field(default=None, description="Required when failed, absent when passed")
    failing_node: str | None = Field(default=None, description="The node that failed; absent when passed")
    task_id: str | None = None
    seed: int | None = None
    checker_id: str | None = None
    summary: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "Verdict":
        problems = verdict_rule_errors(self.passed, self.score, self.failure_class, self.failing_node)
        if problems:
            raise ValueError("; ".join(problems))
        return self
