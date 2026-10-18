"""FakeModel: offline, deterministic, $0. For tests and dry runs only.

Its answers are placeholders that parse; they describe nothing about the input.
Never score a model or a program from FakeModel output and believe it.

Token counts are made up the same way every time: input is ``ceil(chars / 4)`` plus
765 per image, output is ``ceil(len(reply) / 4)``. The cost is computed from those
counts at the entry's models.yaml prices, which are 0 for ``fake``.

Scripts make the FakeModel say particular things::

    set_fake_script(["not json", {"nodes": [...]}])   # replies consumed in order
    set_fake_script(lambda req: {"green": 5})          # a function of the FakeRequest
    with fake_script([RuntimeError("boom")]):          # an Exception instance is raised
        ...

A reply may be a str, a dict (sent as ``json.dumps``) or an Exception instance.
With no script the FakeModel answers ``"fake response"``, or, when a JSON schema is
given, the smallest instance that satisfies it (the robolabel mock's rule, extended
to bounds, ``required`` and ``$ref``).

Fake replies are cached like any other response. While a script is active, a
token for that script is folded into the request hash, so a cached reply from one
script never answers for another; within one script an identical request still
hits the cache.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...schemas.jsonschema_lite import check, resolve
from ..errors import RouterError
from .base import Adapter, AdapterRequest, AdapterResult
from .openrouter import normalize_reasoning as _openrouter_reasoning

if TYPE_CHECKING:  # pragma: no cover
    from ...config import ModelEntry
    from ..http import HttpClient

TOKENS_PER_IMAGE = 765
DEFAULT_REPLY = "fake response"


@dataclass(frozen=True)
class FakeRequest:
    model_key: str
    role: str
    messages: list[dict[str, str]]
    images: tuple[str, ...]  # sha256 of each image, in order
    json_schema: dict[str, Any] | None
    max_tokens: int
    reasoning: Any
    extra: dict[str, Any] = field(default_factory=dict)
    call_index: int = 0

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def text(self) -> str:
        """The last user message's text (what most scripts look at)."""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""


Script = Callable[[FakeRequest], Any] | list[Any] | tuple[Any, ...] | Iterator[Any]

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"kind": None, "fn": None, "iter": None, "token": None, "calls": 0, "replies": 0}


def set_fake_script(script: Script | None) -> None:
    """Script the FakeModel's replies (None clears the script)."""
    if script is None:
        clear_fake_script()
        return
    with _LOCK:
        if isinstance(script, (list, tuple)):
            state = {"kind": "list", "fn": None, "iter": iter(list(script))}
        elif isinstance(script, Iterator):
            state = {"kind": "iter", "fn": None, "iter": script}
        elif callable(script):
            state = {"kind": "callable", "fn": script, "iter": None}
        else:
            raise TypeError("a fake script is a callable(FakeRequest) or a list/iterator of replies")
        _STATE.update(state, token=uuid.uuid4().hex, calls=0, replies=0)


def clear_fake_script() -> None:
    with _LOCK:
        _STATE.update(kind=None, fn=None, iter=None, token=None, calls=0, replies=0)


@contextmanager
def fake_script(script: Script | None):
    """``with fake_script(...)``: script the FakeModel inside the block, restore the previous script after."""
    with _LOCK:
        saved = dict(_STATE)
    set_fake_script(script)
    try:
        yield
    finally:
        with _LOCK:
            _STATE.clear()
            _STATE.update(saved)


def fake_calls() -> int:
    """FakeModel calls sent since the current script was set (cache hits are not calls)."""
    with _LOCK:
        return int(_STATE["calls"])


# --------------------------------------------------------------------------- schema
def from_schema(schema: dict[str, Any] | None, root: dict[str, Any] | None = None, _depth: int = 0) -> Any:
    """The smallest value that satisfies ``schema``: first enum value, one array item, zeros, ``"fake"``.

    Objects get their required properties (all properties when the schema lists no
    ``required``). Numeric bounds, string lengths, ``const``, ``anyOf``/``oneOf``,
    ``allOf`` and local ``$ref`` are honored.
    """
    if not isinstance(schema, dict):
        return "fake"
    root = root if root is not None else schema
    schema = resolve(schema, root)
    if _depth > 32:
        return None
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return copy.deepcopy(schema["enum"][0])
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            ordered = sorted(branches, key=lambda b: 1 if resolve(b, root).get("type") == "null" else 0)
            first = None
            for branch in ordered:
                value = from_schema(branch, root, _depth + 1)
                if first is None:
                    first = value
                if not check(value, branch, root=root):
                    return value
            return first
    if isinstance(schema.get("allOf"), list) and schema["allOf"]:
        merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
        for branch in schema["allOf"]:
            b = resolve(branch, root)
            for k, v in b.items():
                if k == "properties" and isinstance(v, dict):
                    merged.setdefault("properties", {}).update(v)
                elif k == "required" and isinstance(v, list):
                    merged["required"] = list(dict.fromkeys(list(merged.get("required", [])) + v))
                else:
                    merged.setdefault(k, v)
        return from_schema(merged, root, _depth + 1)
    t = schema.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else "null"
    if t is None:
        if "properties" in schema or "required" in schema:
            t = "object"
        elif "items" in schema:
            t = "array"
        else:
            return "fake"
    if t == "object":
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        names = schema["required"] if isinstance(schema.get("required"), list) else list(props)
        return {name: from_schema(props.get(name, {}), root, _depth + 1) for name in names}
    if t == "array":
        lo = int(schema.get("minItems", 0) or 0)
        hi = schema.get("maxItems")
        n = max(1, lo)
        if isinstance(hi, int):
            n = min(n, hi)
        items = schema.get("items")
        prefix = schema.get("prefixItems")
        if isinstance(prefix, list) and prefix:
            out = [from_schema(p, root, _depth + 1) for p in prefix]
            return out[:n] if isinstance(hi, int) else out
        return [from_schema(items if isinstance(items, dict) else {}, root, _depth + 1) for _ in range(n)]
    if t == "integer":
        return int(_number_in_bounds(schema, 0, integer=True))
    if t == "number":
        return _number_in_bounds(schema, 0.5, integer=False)
    if t == "boolean":
        return True
    if t == "null":
        return None
    # string
    value = "fake"
    lo = schema.get("minLength")
    hi = schema.get("maxLength")
    if isinstance(lo, int) and len(value) < lo:
        value = value + "x" * (lo - len(value))
    if isinstance(hi, int) and len(value) > hi:
        value = value[:hi]
    return value


def _number_in_bounds(schema: dict[str, Any], start: float, *, integer: bool) -> float:
    lo, hi = schema.get("minimum"), schema.get("maximum")
    xlo, xhi = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
    v = float(start)
    if isinstance(lo, (int, float)) and v < lo:
        v = math.ceil(lo) if integer else float(lo)
    if isinstance(xlo, (int, float)) and v <= xlo:
        if integer:
            v = math.floor(xlo) + 1
        else:
            top = hi if isinstance(hi, (int, float)) else (xhi if isinstance(xhi, (int, float)) else None)
            v = (xlo + top) / 2 if top is not None else xlo + 1.0
    if isinstance(hi, (int, float)) and v > hi:
        v = math.floor(hi) if integer else float(hi)
    if isinstance(xhi, (int, float)) and v >= xhi:
        if integer:
            v = math.ceil(xhi) - 1
        else:
            bottom = lo if isinstance(lo, (int, float)) else (xlo if isinstance(xlo, (int, float)) else None)
            v = (bottom + xhi) / 2 if bottom is not None else xhi - 1.0
    return v


# --------------------------------------------------------------------------- adapter
class FakeAdapter(Adapter):
    name = "fake"
    label = "FakeModel"
    default_env_vars = ()

    def env_vars(self, entry: "ModelEntry") -> list[str]:
        return []

    def normalize_reasoning(self, reasoning: Any, entry: "ModelEntry") -> Any:
        return _openrouter_reasoning(reasoning, entry)

    def cache_token(self, request: AdapterRequest) -> str | None:
        with _LOCK:
            return _STATE["token"]

    def _next_reply(self, freq_args: dict[str, Any], json_schema: dict[str, Any] | None) -> tuple[Any, int, bool]:
        with _LOCK:
            index = int(_STATE["calls"])
            _STATE["calls"] = index + 1
            kind = _STATE["kind"]
            fn = _STATE["fn"]
            it = _STATE["iter"]
            if kind in ("list", "iter"):
                try:
                    reply = next(it)
                except StopIteration:
                    raise RouterError(f"fake script exhausted after {_STATE['replies']} replies "
                                      f"(call {index} wanted one more)") from None
                _STATE["replies"] += 1
                return reply, index, True
        if kind == "callable":
            return fn(FakeRequest(call_index=index, **freq_args)), index, True
        if json_schema is not None:
            return from_schema(json_schema), index, False
        return DEFAULT_REPLY, index, False

    def send(self, request: AdapterRequest, key: str, http: "HttpClient") -> AdapterResult:
        freq_args = {
            "model_key": request.model_key,
            "role": request.role,
            "messages": copy.deepcopy(request.messages),
            "images": tuple(img.sha256 for img in request.images),
            "json_schema": copy.deepcopy(request.json_schema),
            "max_tokens": request.max_tokens,
            "reasoning": copy.deepcopy(request.reasoning),
            "extra": copy.deepcopy(request.extra),
        }
        reply, index, scripted = self._next_reply(freq_args, request.json_schema)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, (dict, list)):
            text = json.dumps(reply)
        elif isinstance(reply, str):
            text = reply
        else:
            raise TypeError(f"a fake reply must be a str, a dict or an Exception instance, got {type(reply).__name__}")
        chars = sum(len(m["content"]) for m in request.messages)
        input_tokens = math.ceil(chars / 4) + TOKENS_PER_IMAGE * len(request.images)
        output_tokens = math.ceil(len(text) / 4)
        return AdapterResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=0,
            cached_tokens=0,
            usd=None,  # computed from tokens at the entry's prices ($0 for the fake entry)
            model_returned=request.model_id,
            finish_reason="stop",
            raw={"fake": True, "scripted": scripted, "call_index": index,
                 "warning": "FakeModel output: placeholder, not derived from the input"},
            http_status=None,
            attempts=1,
            json_mode="fake" if request.json_schema is not None else None,
        )
