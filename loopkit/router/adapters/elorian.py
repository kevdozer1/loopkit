"""Elorian (Glimpse): a standard chat-completions adapter at the entry's ``base_url``.

``POST <entry.base_url>/chat/completions`` with a Bearer key from ``ELORIAN_API_KEY``
(or the entry's ``env_key``). Messages and images use the same format as the
OpenRouter adapter, and the same JSON-mode fallbacks apply (without OpenRouter's
``provider`` routing field). There is no ``usage.include`` and no cost in the
response: the cost is computed from the returned tokens at the models.yaml prices.

API facts this follows (from the provider's docs as recorded in the glimpse entry's
``source`` in config/models.yaml, read 2026-10-17): ``reasoning_effort`` is one of
``low``, ``medium`` and ``xhigh``, and ``xhigh`` is the default when it is omitted;
the output budget is ``max_completion_tokens``. So:

* ``disabled`` sends ``low``: the API has no off switch, and omitting the field
  would give the most reasoning, not none.
* ``low`` and ``medium`` are sent as they are; ``high`` sends ``xhigh`` (there is no
  ``high``); ``highest`` sends the entry's models.yaml ``reasoning`` effort when set,
  else ``xhigh``.

The request body is built in one function, :func:`build_elorian_payload`, and the
reasoning mapping in :func:`normalize_elorian_reasoning`: the two places to adjust
if the provider's API changes.

Reasoning tokens: nothing read so far says whether Elorian's ``completion_tokens``
include them, so the adapter's ``reasoning_in_output`` is None (unknown) and the
computed cost adds them on top of the output tokens. That is certainly right when
``reasoning_tokens`` exceeds ``completion_tokens``, and the conservative choice
otherwise. Set it to True here once the provider documents that they are included.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ModelNotConfigured
from .base import AdapterRequest, check_url, reasoning_level
from .openrouter import OpenRouterAdapter, chat_messages, response_format

if TYPE_CHECKING:  # pragma: no cover
    from ...config import ModelEntry

ELORIAN_EFFORTS = ("low", "medium", "xhigh")
_LEVEL_TO_EFFORT = {"disabled": "low", "low": "low", "medium": "medium", "high": "xhigh"}


def build_elorian_payload(request: AdapterRequest, mode: str | None = None) -> dict[str, Any]:
    """The Elorian chat-completions body for one JSON mode (None: a plain call)."""
    body: dict[str, Any] = {
        "model": request.model_id,
        "messages": chat_messages(request, mode),
        "max_completion_tokens": int(request.max_tokens),
    }
    if request.reasoning:
        body["reasoning_effort"] = request.reasoning
    if request.temperature is not None:
        body["temperature"] = request.temperature
    fmt = response_format(request, mode)
    if fmt is not None:
        body["response_format"] = fmt
    return body


def _effort_from(d: dict[str, Any] | None) -> str | None:
    if not d:
        return None
    for name in ("reasoning_effort", "effort"):
        if isinstance(d.get(name), str):
            return d[name]
    return None


def normalize_elorian_reasoning(reasoning: Any, entry: "ModelEntry") -> str:
    """The ``reasoning_effort`` sent for the caller's setting (always one value; see the module docstring)."""
    level = reasoning_level(reasoning)
    if isinstance(level, dict):
        effort = _effort_from(level)
        if effort is not None:
            return effort
        if level.get("enabled") is False:
            return _LEVEL_TO_EFFORT["disabled"]
        raise ValueError("an Elorian reasoning dict needs 'reasoning_effort' or 'effort' "
                         f"(or enabled: false), got {level!r}")
    if level == "highest":
        return _effort_from(entry.reasoning) or "xhigh"
    return _LEVEL_TO_EFFORT[level]


class ElorianAdapter(OpenRouterAdapter):
    name = "elorian"
    label = "Elorian"
    default_env_vars = ("ELORIAN_API_KEY",)
    reports_cost = False
    #: unknown whether completion_tokens include the reasoning tokens, so they are added (conservative)
    reasoning_in_output: bool | None = None

    def check_entry(self, entry: "ModelEntry") -> None:
        if not entry.base_url:
            raise ModelNotConfigured(
                f"model {entry.key!r} uses the elorian adapter but config/models.yaml has no base_url for it. "
                "Set base_url to the provider's chat-completions API root (from its API docs).")
        check_url(entry, "base_url", entry.base_url)

    def url(self, entry: "ModelEntry") -> str:
        base = str(entry.base_url).rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def headers(self) -> dict[str, str]:
        return {}

    def normalize_reasoning(self, reasoning: Any, entry: "ModelEntry") -> Any:
        return normalize_elorian_reasoning(reasoning, entry)

    def build_payload(self, request: AdapterRequest, mode: str | None = None) -> dict[str, Any]:
        return build_elorian_payload(request, mode)
