"""Terminal skills: done and abort end the program."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .base import NoOutput, SkillContext, skill


class AbortResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


@skill("terminal", terminal="done")
def done(ctx: SkillContext) -> NoOutput:
    """End the program: the task is complete. The robot stops where it is."""
    return NoOutput()


@skill("terminal", terminal="abort")
def abort(ctx: SkillContext,
          reason: Annotated[str, Field(max_length=200, description="why the program gives up")] = "") -> AbortResult:
    """End the program without finishing the task, with a short reason."""
    return AbortResult(reason=reason)
