"""The program writer: prompt a model for a ProgramGraph, validate it, repair up to 3 times."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..router.errors import SpendCapExceeded
from ..schemas import ProgramGraph, SkillSpec, TaskSpec, format_errors, normalize_params, parse_graph, validate_graph
from .prompt import build_messages, repair_message

MAX_REPAIRS = 3
WRITER_MAX_TOKENS = 16000
GRAPH_SCHEMA = ProgramGraph.model_json_schema()


@dataclass
class CallRecord:
    usd: float
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    cache_hit: bool
    ok: bool
    note: str = ""


@dataclass
class WriteResult:
    graph: ProgramGraph | None
    errors: list[str]
    attempts: int
    calls: list[CallRecord] = field(default_factory=list)
    last_text: str = ""

    @property
    def usd(self) -> float:
        return round(sum(c.usd for c in self.calls), 6)

    @property
    def ok(self) -> bool:
        return self.graph is not None


def _default_chat() -> Callable:
    from ..router import chat

    return chat


def write_graph(model_key: str, task: TaskSpec, catalog: dict[str, SkillSpec], images: list[np.ndarray] | None = None,
                *, image_names: tuple[str, ...] = (), previous: ProgramGraph | None = None,
                failure_summary: str | None = None, chat: Callable | None = None, trace=None,
                cache_salt: str | None = None, reasoning: str | dict = "highest", max_tokens: int = WRITER_MAX_TOKENS,
                max_repairs: int = MAX_REPAIRS, chat_kwargs: dict | None = None) -> WriteResult:
    """Ask ``model_key`` for a graph; send validator errors back until it validates or repairs run out."""
    chat = chat or _default_chat()
    messages = build_messages(task, catalog, previous=previous, failure_summary=failure_summary,
                              image_names=image_names)
    result = WriteResult(graph=None, errors=[], attempts=0)
    for attempt in range(1 + max_repairs):
        result.attempts = attempt + 1
        salt = f"{cache_salt or ''}|attempt{attempt}"
        try:
            resp = chat(model_key, messages, images=images, json_schema=GRAPH_SCHEMA, role="writer",
                        max_tokens=max_tokens, reasoning=reasoning, trace=trace, cache_salt=salt, phase="setup",
                        schema_name="program_graph", **(chat_kwargs or {}))
        except SpendCapExceeded:
            raise  # a cap stops the model, not just this write: the bake-off records a budget_stop
        except Exception as exc:  # noqa: BLE001  (a refused or failed call ends the write, with the reason)
            result.errors = [f"{type(exc).__name__}: {exc}"]
            result.calls.append(CallRecord(0.0, None, None, 0.0, False, False, f"{type(exc).__name__}"))
            return result
        result.calls.append(CallRecord(float(getattr(resp, "usd", 0.0) or 0.0), getattr(resp, "input_tokens", None),
                                       getattr(resp, "output_tokens", None), float(getattr(resp, "latency_ms", 0.0) or 0.0),
                                       bool(getattr(resp, "cache_hit", False)), True))
        text = getattr(resp, "text", "") or ""
        result.last_text = text
        parsed = getattr(resp, "json", None)
        # the router parses the first JSON object in the answer; when that is not a graph (an example in prose
        # before it), the validator searches the text itself
        source: Any = parsed if isinstance(parsed, dict) and "nodes" in parsed else (text or parsed)
        errors = validate_graph(source, catalog, task)
        if not errors:
            graph, _ = parse_graph(source)
            result.graph = normalize_params(graph, catalog)
            result.errors = []
            return result
        result.errors = [str(e) for e in errors]
        if attempt < max_repairs:
            reply = text if text.strip() else json.dumps(source) if isinstance(source, dict) else ""
            messages = messages + [{"role": "assistant", "content": reply[:30000]}, repair_message(format_errors(errors))]
    return result
