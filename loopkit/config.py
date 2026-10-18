"""Environment, secrets and the two config files.

``.env`` is parsed here with a CRLF-safe reader (``splitlines`` drops ``\\r``; a
bash loader on this machine once produced paths ending in ``\\r``). Values in the
process environment win over ``.env``. Secret values are returned to the caller
and never printed, logged or put in an exception message: a missing key raises
:class:`MissingCredentialError`, which names the variable to set and nothing else.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .paths import REPO_ROOT

ENV_FILE = REPO_ROOT / ".env"
MODELS_YAML = REPO_ROOT / "config" / "models.yaml"
BAKEOFF_YAML = REPO_ROOT / "config" / "bakeoff.yaml"

ADAPTERS = ("openrouter", "elorian", "jev", "fake", "anthropic", "openai", "gemini")
Role = Literal["writer", "reader", "system1"]


# --------------------------------------------------------------------------- .env
def parse_dotenv(text: str) -> dict[str, str]:
    """``KEY=value`` lines to a dict. Blank lines and ``#`` comments are skipped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


_DOTENV: dict[str, str] | None = None


def dotenv_values(path: Path | None = None, reload: bool = False) -> dict[str, str]:
    """The repo ``.env`` as a dict (cached). Missing file gives an empty dict."""
    global _DOTENV
    if path is not None:
        p = Path(path)
        return parse_dotenv(p.read_text(encoding="utf-8-sig")) if p.is_file() else {}
    if _DOTENV is None or reload:
        _DOTENV = parse_dotenv(ENV_FILE.read_text(encoding="utf-8-sig")) if ENV_FILE.is_file() else {}
    return _DOTENV


def env(name: str, default: str | None = None) -> str | None:
    """A setting from the process environment, then ``.env``. Empty counts as unset."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        value = dotenv_values().get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


class MissingCredentialError(RuntimeError):
    """A model's API key is absent. The message names the variable, never a value."""


def has_secret(names: list[str] | tuple[str, ...]) -> bool:
    return any(env(n) for n in names)


def load_secret(names: list[str] | tuple[str, ...], label: str) -> str:
    """The first set variable among ``names``. Raises MissingCredentialError naming ``names[0]``."""
    for name in names:
        value = env(name)
        if value:
            return value
    others = f" (or {', '.join(names[1:])})" if len(names) > 1 else ""
    raise MissingCredentialError(
        f"{label} credential not found. Set {names[0]}{others} in the environment "
        f"or add {names[0]}=... to {ENV_FILE}."
    )


# --------------------------------------------------------------------- models.yaml
class DirectRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: str
    id: str | None


class ModelEntry(BaseModel):
    """One row of ``config/models.yaml``.

    Fields without a default are required to be present (they may be null where
    the type allows it), which is what the "every entry has the required fields"
    test checks.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = ""
    display: str
    adapter: str
    id: str | None
    price_in: float | None
    price_out: float | None
    vision: bool
    roles: list[Role]
    verified: bool
    source: str | None
    accessed: str | None
    # optional, adapter-specific
    cached_in: float | None = None
    direct: DirectRoute | None = None
    alt_route: DirectRoute | None = None
    fallback_id: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    env_key: str | None = None
    json_mode: str | None = None
    reasoning: dict[str, Any] | None = None
    image_tokens: int | None = None
    notes: str | None = None

    @field_validator("adapter")
    @classmethod
    def _known_adapter(cls, value: str) -> str:
        if value not in ADAPTERS:
            raise ValueError(f"adapter {value!r} is not one of {', '.join(ADAPTERS)}")
        return value

    @field_validator("price_in", "price_out", "cached_in")
    @classmethod
    def _non_negative(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("prices are finite non-negative USD per million tokens (no NaN or infinity)")
        return value

    @model_validator(mode="after")
    def _zero_prices_marked_free(self) -> "ModelEntry":
        # a paid adapter at 0 / 0 is usually a typo or a promotion; the router still treats it as paid
        if self.adapter != "fake" and self.price_in == 0 and self.price_out == 0 \
                and "free" not in (self.notes or "").lower():
            raise ValueError("price_in and price_out are both 0 for a non-fake adapter. If the model really is "
                             "free, say so with notes: free (and why); otherwise set its prices")
        return self

    @field_validator("accessed", mode="before")
    @classmethod
    def _date_text(cls, value: Any) -> Any:
        return value.isoformat() if hasattr(value, "isoformat") else value

    @model_validator(mode="after")
    def _verified_is_complete(self) -> "ModelEntry":
        if self.verified and (self.id is None or self.price_in is None or self.price_out is None):
            raise ValueError("verified: true needs id, price_in and price_out")
        if not self.roles:
            raise ValueError("roles must list at least one of writer, reader, system1")
        return self

    @property
    def priced(self) -> bool:
        return self.price_in is not None and self.price_out is not None

    @property
    def runnable(self) -> bool:
        """True when the entry has a model id to call."""
        return self.adapter == "fake" or self.id is not None


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    currency: Literal["USD"]
    unit: Literal["usd_per_million_tokens"]
    models: dict[str, ModelEntry]

    @field_validator("version", mode="before")
    @classmethod
    def _version_text(cls, value: Any) -> Any:
        return value.isoformat() if hasattr(value, "isoformat") else value

    @model_validator(mode="after")
    def _keys(self) -> "ModelsConfig":
        for key, entry in self.models.items():
            entry.key = key
        return self

    def get(self, key: str) -> ModelEntry:
        if key not in self.models:
            raise KeyError(f"unknown model key {key!r}; known: {', '.join(sorted(self.models))}")
        return self.models[key]


def load_models(path: Path | None = None) -> ModelsConfig:
    data = yaml.safe_load(Path(path or MODELS_YAML).read_text(encoding="utf-8"))
    return ModelsConfig.model_validate(data)


# -------------------------------------------------------------------- bakeoff.yaml
class SeedRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    count: int = Field(gt=0)

    def seeds(self) -> list[int]:
        return list(range(self.start, self.start + self.count))


class PatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(gt=0)
    stop_after_no_improvement: int = Field(gt=0)


class CallConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    max_tokens: int = Field(gt=0)


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_repeat_in_hash: bool


class BakeoffConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    models: list[str]
    optional_models: list[str] = Field(default_factory=list)
    tasks: list[str]
    repeats: int = Field(gt=0)
    train_seeds: SeedRange
    heldout_seeds: SeedRange
    patch: PatchConfig
    writer: CallConfig
    reader: CallConfig
    caps_usd: dict[str, float]
    total_cap_usd: float = Field(gt=0)
    cache: CacheConfig
    checkpoint_every_unit: bool
    output_dir: str

    @model_validator(mode="after")
    def _consistent(self) -> "BakeoffConfig":
        missing = [m for m in self.models + self.optional_models if m not in self.caps_usd]
        if missing:
            raise ValueError(f"caps_usd has no cap for {', '.join(missing)}")
        return self


def load_bakeoff(path: Path | None = None) -> BakeoffConfig:
    data = yaml.safe_load(Path(path or BAKEOFF_YAML).read_text(encoding="utf-8"))
    return BakeoffConfig.model_validate(data)
