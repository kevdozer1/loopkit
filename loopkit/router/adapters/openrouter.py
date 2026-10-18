"""OpenRouter chat completions (patterns: statebench ``backends/llm.py``, robolabel ``providers/openrouter.py``).

``POST https://openrouter.ai/api/v1/chat/completions`` with ``OPENROUTER_API_KEY``.

Body: ``model``, ``messages`` (user content as parts: text, then the images as
PNG/JPEG data URLs on the last user message), ``max_tokens``, ``usage: {"include":
true}`` so ``usage.cost`` comes back, the normalized ``reasoning`` and
``temperature`` when given.

Reasoning: ``disabled`` sends ``{"enabled": false}`` (small JSON answers truncate
when a model reasons first), ``low``/``medium``/``high`` send ``{"effort": X}``,
``highest`` sends the models.yaml ``reasoning`` dict for the entry when it has one,
else ``{"effort": "high"}``. A raw dict is sent as it is. OpenRouter's
``completion_tokens`` include the reasoning tokens.

Cost: ``usage.cost``. On an account that brings its own provider key (BYOK),
``usage.cost`` is only OpenRouter's fee and the provider's charge comes back as
``usage.cost_details.upstream_inference_cost`` (with ``usage.is_byok``); the call
then costs the sum of the two.

Structured output, like robolabel: ``response_format`` ``json_schema`` (strict
false, because a program graph's ``params`` objects are free-form) is tried first.
A 400 that names the schema, ``response_format`` or structured output, or a 404/503
saying no endpoint supports the request, moves to ``json_object`` with the schema
appended to the prompt, then to plain text with the schema in the prompt. The
structured modes set ``provider.require_parameters`` so OpenRouter routes only to
providers that honor ``response_format`` instead of silently dropping it. The mode
that answered is recorded as ``json_mode``. Every POST, in every mode, is reserved
in the ledger before it is sent (``request.attempt_hooks``); a rejected mode
settles at $0 there.

A 200 whose generation failed upstream (``finish_reason`` "error", or an error
object in the body or in ``choices``) raises ModelCallError carrying the reported
cost; its partial answer is never returned or cached.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from ..errors import ModelCallError
from ..redact import redact, scrub
from .base import (
    Adapter,
    AdapterRequest,
    AdapterResult,
    as_cost,
    check_url,
    error_text,
    reasoning_level,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...config import ModelEntry
    from ..http import HttpClient, HttpResult

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODES = ("json_schema", "json_object_schema_in_prompt", "plain_schema_in_prompt")
STRUCTURED_MODES = ("json_schema", "json_object_schema_in_prompt")
SCHEMA_PROMPT = "Return only JSON that matches this JSON Schema:\n"
_MODE_400_WORDS = ("schema", "response_format", "response format", "structured", "json_object", "json mode")
_MODE_404_WORDS = ("no endpoints", "routing requirements", "no available", "require_parameters",
                   "requested parameters")
USAGE_TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens")


def schema_prompt(schema: dict[str, Any]) -> str:
    return SCHEMA_PROMPT + json.dumps(schema, separators=(",", ":"), ensure_ascii=False)


def safe_schema_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(name or "response"))[:64]
    return cleaned or "response"


def chat_messages(request: AdapterRequest, mode: str | None) -> list[dict[str, Any]]:
    """Chat-completions messages: user content as parts, images on the last user message.

    In the schema-in-prompt modes the schema is appended to the last user message as
    one more text part.
    """
    last_user = request.last_user_index
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(request.messages):
        if msg["role"] != "user":
            out.append({"role": msg["role"], "content": msg["content"]})
            continue
        parts: list[dict[str, Any]] = []
        if msg["content"]:
            parts.append({"type": "text", "text": msg["content"]})
        if i == request.image_message_index:
            parts.extend({"type": "image_url", "image_url": {"url": img.data_url()}} for img in request.images)
        if i == last_user and mode in ("json_object_schema_in_prompt", "plain_schema_in_prompt") \
                and request.json_schema is not None:
            parts.append({"type": "text", "text": schema_prompt(request.json_schema)})
        if not parts:
            parts.append({"type": "text", "text": ""})
        out.append({"role": "user", "content": parts})
    if last_user is None and mode in ("json_object_schema_in_prompt", "plain_schema_in_prompt") \
            and request.json_schema is not None:
        out.append({"role": "user", "content": [{"type": "text", "text": schema_prompt(request.json_schema)}]})
    return out


def response_format(request: AdapterRequest, mode: str | None) -> dict[str, Any] | None:
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": safe_schema_name(request.schema_name),
                                                       "strict": False, "schema": request.json_schema}}
    if mode == "json_object_schema_in_prompt":
        return {"type": "json_object"}
    return None


def build_payload(request: AdapterRequest, mode: str | None = None) -> dict[str, Any]:
    """The OpenRouter request body for one mode (None: a plain call without a schema)."""
    body: dict[str, Any] = {
        "model": request.model_id,
        "messages": chat_messages(request, mode),
        "max_tokens": int(request.max_tokens),
        "usage": {"include": True},
    }
    if request.reasoning is not None:
        body["reasoning"] = request.reasoning
    if request.temperature is not None:
        body["temperature"] = request.temperature
    fmt = response_format(request, mode)
    if fmt is not None:
        body["response_format"] = fmt
    if mode in STRUCTURED_MODES:
        body["provider"] = {"require_parameters": True}
    return body


def normalize_reasoning(reasoning: Any, entry: "ModelEntry") -> dict[str, Any]:
    """OpenRouter's ``reasoning`` object for the caller's setting."""
    level = reasoning_level(reasoning)
    if isinstance(level, dict):
        return level
    if level == "disabled":
        return {"enabled": False}
    if level == "highest":
        return dict(entry.reasoning) if entry.reasoning else {"effort": "high"}
    return {"effort": level}


def message_text(content: Any) -> str:
    """Assistant content as text: a string, or a list of parts whose text is joined."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    pieces.append(part["text"])
            elif isinstance(part, str):
                pieces.append(part)
        return "".join(pieces)
    return str(content)


def parse_usage(usage: Any) -> dict[str, Any]:
    """Token counts and cost from a chat-completions ``usage`` object.

    The token counts are returned as reported (None when absent): chat() drops a value
    that is not a count, with a note that names it. ``cost`` is what the call cost:
    ``usage.cost`` plus, when it is reported, ``usage.cost_details.upstream_inference_cost``
    (a BYOK account's provider charge, which ``usage.cost`` leaves out). ``usage_cost`` is
    ``usage.cost`` as reported.
    """
    usage = usage if isinstance(usage, dict) else {}
    out_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    in_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cost_details = usage.get("cost_details") if isinstance(usage.get("cost_details"), dict) else {}
    reported = as_cost(usage.get("cost"))
    upstream = as_cost(cost_details.get("upstream_inference_cost"))
    is_byok = usage.get("is_byok") if isinstance(usage.get("is_byok"), bool) else None
    total = reported
    if upstream is not None:
        total = (reported or 0.0) + upstream
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": out_details.get("reasoning_tokens"),
        "cached_tokens": in_details.get("cached_tokens"),
        "cost": total,
        "usage_cost": reported,
        "upstream_inference_cost": upstream,
        "is_byok": is_byok,
    }


def cost_breakdown(usage: dict[str, Any]) -> dict[str, Any] | None:
    """How ``cost`` was put together, when an upstream (BYOK) cost or ``is_byok`` was reported."""
    if usage.get("upstream_inference_cost") is None and usage.get("is_byok") is None:
        return None
    out = {"usage_cost": usage.get("usage_cost"), "upstream_inference_cost": usage.get("upstream_inference_cost"),
           "is_byok": usage.get("is_byok"), "cost": usage.get("cost")}
    if usage.get("is_byok") and usage.get("upstream_inference_cost") is None:
        out["note"] = ("is_byok is true but no upstream_inference_cost was reported, so usage.cost is only "
                       "OpenRouter's fee; the router checks it against the token-computed cost")
    return out


def is_mode_rejection(status: int | None, message: str) -> bool:
    """A refusal that a less demanding JSON mode may get past."""
    msg = message.lower()
    if status in (400, 422):
        return any(w in msg for w in _MODE_400_WORDS)
    if status in (404, 503):
        return any(w in msg for w in _MODE_404_WORDS)
    return False


def _error_message(err: Any) -> str:
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err.get("type") or "")
    if isinstance(err, str):
        return err
    return ""


class OpenRouterAdapter(Adapter):
    name = "openrouter"
    label = "OpenRouter"
    default_env_vars = ("OPENROUTER_API_KEY",)
    #: providers that return ``usage.cost`` (Elorian does not; its cost is computed from tokens)
    reports_cost = True
    #: OpenRouter's completion_tokens include the reasoning tokens
    reasoning_in_output: bool | None = True

    # ------------------------------------------------------------------ request
    def url(self, entry: "ModelEntry") -> str:
        if entry.base_url:
            base = entry.base_url.rstrip("/")
            return base if base.endswith("/chat/completions") else base + "/chat/completions"
        return API_URL

    def headers(self) -> dict[str, str]:
        return {"X-Title": "loopkit"}

    def check_entry(self, entry: "ModelEntry") -> None:
        if entry.base_url is not None:  # the value as written, and the URL built from it exactly as it is sent
            check_url(entry, "base_url", entry.base_url, sent=self.url(entry))

    def normalize_reasoning(self, reasoning: Any, entry: "ModelEntry") -> Any:
        return normalize_reasoning(reasoning, entry)

    def build_payload(self, request: AdapterRequest, mode: str | None = None) -> dict[str, Any]:
        return build_payload(request, mode)

    def modes(self, request: AdapterRequest) -> list[str | None]:
        if request.json_schema is None:
            return [None]
        start = request.entry.json_mode if request.entry.json_mode in MODES else MODES[0]
        return list(MODES[MODES.index(start):])

    # ------------------------------------------------------------------ call
    def send(self, request: AdapterRequest, key: str, http: "HttpClient") -> AdapterResult:
        modes = self.modes(request)
        attempts = 0
        log: list[dict[str, Any]] = []
        tried: list[dict[str, Any]] = []
        url = self.url(request.entry)
        for n, mode in enumerate(modes):
            try:
                body = self.build_payload(request, mode)
            except Exception as exc:  # noqa: BLE001 - nothing was sent for this mode
                raise self.build_error(request, exc, key, attempts, log) from None
            res = http.post_json(url, body, key=key, timeout_s=request.timeout_s, headers=self.headers(),
                                 hooks=request.attempt_hooks)
            attempts += res.attempts
            log.extend({**a, "mode": mode} for a in res.attempt_log)
            if res.ok:
                return self._success(request, res, key, mode, attempts, log, tried)
            message = error_text(res, key)
            tried.append({"mode": mode, "status": res.status, "error": message[:300]})
            if res.sent and mode is not None and n < len(modes) - 1 and is_mode_rejection(res.status, message):
                continue
            err = self.call_error(request, res, key, attempts, log=log)
            err.raw.update(attempt_log=log, modes_tried=tried)
            raise err
        raise ModelCallError(f"{request.model_key}: no JSON mode was accepted", attempts=attempts)  # pragma: no cover

    def _success(self, request: AdapterRequest, res: "HttpResult", key: str, mode: str | None, attempts: int,
                 log: list[dict[str, Any]], tried: list[dict[str, Any]]) -> AdapterResult:
        data = res.data
        usage = parse_usage(data.get("usage") if isinstance(data, dict) else None)
        cost = usage["cost"] if self.reports_cost else None
        tokens = {k: usage[k] for k in USAGE_TOKEN_FIELDS}
        where = f"{request.model_key} ({request.model_id}) via {self.label}"
        if not isinstance(data, dict):
            raise ModelCallError(
                f"{where}: HTTP {res.status} but the body is not JSON: {redact(res.text or '', key)[:200]}",
                status=res.status, attempts=attempts, raw={"attempt_log": log})
        gen_id = data.get("id") if isinstance(data.get("id"), str) else None
        if data.get("error"):
            # an upstream failure after the headers went out: generation may have happened
            raise ModelCallError(
                f"{where}: HTTP {res.status} with an error: {error_text(res, key)}", status=res.status,
                attempts=attempts, generation_id=gen_id, usd=cost,
                raw={"attempt_log": log, "response": scrub(data, key), "usage": tokens})
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelCallError(
                f"{where}: HTTP {res.status} without choices", status=res.status, attempts=attempts, usd=cost,
                generation_id=gen_id, raw={"attempt_log": log, "response": scrub(data, key), "usage": tokens})
        choice = choices[0]
        finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        choice_error = choice.get("error")
        if finish == "error" or (choice_error and isinstance(choice_error, (dict, str))):
            detail = redact(_error_message(choice_error), key)[:300]
            raise ModelCallError(
                redact(f"{where}: the generation failed upstream (finish_reason {finish or 'missing'}"
                       f"{': ' + detail if detail else ''}); its partial answer is not used or cached", key),
                status=res.status, attempts=attempts, generation_id=gen_id, usd=cost,
                raw={"attempt_log": log, "response": scrub(data, key), "usage": tokens,
                     "finish_reason": finish or "error"})
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        text = message_text(message.get("content"))
        raw = scrub(data, key)
        raw["attempt_log"] = log
        if tried:
            raw["modes_tried"] = tried
        if request.reasoning is not None:
            raw["reasoning_sent"] = request.reasoning
        breakdown = cost_breakdown(usage) if self.reports_cost else None
        if breakdown is not None:
            raw["cost_breakdown"] = breakdown
        return AdapterResult(
            text=text,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            cached_tokens=usage["cached_tokens"],
            usd=cost,
            model_returned=data.get("model") if isinstance(data.get("model"), str) else None,
            finish_reason=finish,
            raw=raw,
            http_status=res.status,
            attempts=attempts,
            json_mode=mode,
            generation_id=gen_id,
            upstream_provider=data.get("provider") if isinstance(data.get("provider"), str) else None,
            reasoning_in_output=self.reasoning_in_output,
        )
