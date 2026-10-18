"""SkillSpec: one typed skill as the program writer sees it.

Specs are generated from the typed skill functions and their docstrings (S3).
The catalog text sent to the writer is built from these fields only, with the
same schema descriptions (``describe.py``) the validator's fix texts use.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .describe import describe

Category = Literal["perception", "motion", "manipulation", "calibration", "terminal", "other"]


class SkillSpec(BaseModel):
    """A skill's name, docstring and JSON Schemas. ``params`` is closed: a param it does not declare is an
    error unless it sets additionalProperties to true or a schema. A ``returns`` object with
    additionalProperties is a map output: any field name can be read."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")
    docstring: str
    params: dict[str, Any] = Field(description="JSON Schema (type object) for the skill's params")
    returns: dict[str, Any] = Field(description="JSON Schema (type object) for the skill's outputs")
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    category: Category = "other"
    terminal: Literal["done", "abort"] | None = None
    big_model: bool = Field(default=False, description="Calls a model through the router when it runs")

    def catalog_entry(self) -> str:
        """Plain-text entry for the writer prompt."""
        lines = [f"### {self.name}  ({self.category}{', terminal' if self.terminal else ''}"
                 f"{', calls a model' if self.big_model else ''})", self.docstring.strip()]
        props = self.params.get("properties") if isinstance(self.params.get("properties"), dict) else {}
        required = set(self.params.get("required") or [])
        extra = self.params.get("additionalProperties")
        if props:
            lines.append("params:")
            for name, sub in props.items():
                flag = "required" if name in required else "optional"
                lines.append(f"  - {name} ({flag}): {describe(sub, self.params, descriptions=True)}")
            if isinstance(extra, dict):
                lines.append(f"  - any other name: {describe(extra, self.params, descriptions=True)}")
        elif isinstance(extra, dict):
            lines.append(f"params: any names, each {describe(extra, self.params, descriptions=True)}")
        else:
            lines.append("params: none")
        rprops = self.returns.get("properties") if isinstance(self.returns.get("properties"), dict) else {}
        rextra = self.returns.get("additionalProperties")
        if rprops:
            lines.append("returns:")
            for name, sub in rprops.items():
                lines.append(f"  - {name}: {describe(sub, self.returns, descriptions=True)}")
            if isinstance(rextra, dict):
                lines.append(f"  - any other name: {describe(rextra, self.returns, descriptions=True)}")
        elif isinstance(rextra, dict):
            lines.append(f"returns: {describe(self.returns, self.returns, descriptions=True)}")
        if self.preconditions:
            lines.append("preconditions: " + "; ".join(self.preconditions))
        if self.postconditions:
            lines.append("postconditions: " + "; ".join(self.postconditions))
        return "\n".join(lines)


def as_catalog(catalog: Mapping[str, SkillSpec] | Iterable[SkillSpec]) -> dict[str, SkillSpec]:
    """A name-to-spec dict from a mapping or an iterable of specs."""
    if isinstance(catalog, Mapping):
        return {str(k): v for k, v in catalog.items()}
    return {spec.name: spec for spec in catalog}


def catalog_text(catalog: Mapping[str, SkillSpec] | Iterable[SkillSpec], allowed: Iterable[str] | None = None) -> str:
    """The catalog entries of ``allowed`` (default: every skill), in that order.

    Raises ValueError when ``allowed`` names a skill the catalog does not have:
    the writer must never be told it may use a skill it cannot see.
    """
    specs = as_catalog(catalog)
    if allowed is None:
        names = list(specs)
    else:
        names = list(dict.fromkeys(allowed))
        missing = [n for n in names if n not in specs]
        if missing:
            raise ValueError(f"allowed skills not in the catalog: {', '.join(missing)}")
    return "\n\n".join(specs[n].catalog_entry() for n in names)
