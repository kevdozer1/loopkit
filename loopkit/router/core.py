"""``chat()``: the one door every model call goes through.

Order of operations for one call:

1. Look up the models.yaml entry. A non-fake entry without an id is not callable yet
   (ModelNotConfigured); so is an elorian entry without a base_url, or a URL httpx
   cannot parse.
2. Normalize messages and images. ``json_schema``, ``extra`` and the reasoning
   setting become plain JSON (numpy scalars and arrays turn into numbers and lists),
   so the request body always serializes. Hash the canonical request: adapter, model
   id, destination URL, messages, image sha256s, the schema as order-preserving text
   (structured output follows property order), the JSON modes tried, schema name,
   max_tokens, the reasoning setting actually sent, temperature, cache_salt, extra.
3. Cache: a hit returns at once with ``cache_hit`` true and $0 (the original cost is
   kept as ``raw["original_usd"]``). No credential, no ledger write.
4. Credentials through the adapter (MissingCredentialError names the variable). A
   key holding a character that cannot go in an HTTP header (a smart quote, a
   zero-width space) is refused the same way, by name and never by value.
5. Projection: ``ceil(text chars / 3.5) + images * image_tokens`` input tokens plus
   ``max_tokens`` of output, plus the reasoning budget when the setting carries one
   (``max_tokens`` or ``budget_tokens``), at the entry's prices (Astra prices when a
   price is unknown).
6. Every HTTP attempt is reserved at the projection before it is sent: the first
   here, each retry and JSON-mode fallback through ``AdapterRequest.attempt_hooks``
   from the HTTP layer. ``Ledger.reserve`` refuses an attempt that would cross a cap
   (SpendCapExceeded). Every call except the FakeModel at a $0 projection is paid:
   it needs an active session, and a reached cap refuses it even at $0.
7. The adapter sends it. Per attempt: a refusal before generation (HTTP
   400/401/402/403/404/429 with no generation id) settles at $0; a lost attempt
   (timeout, deadline, dropped connection or read error, 5xx, a 200 without usage)
   stays reserved at its projection, because the provider may have generated and
   billed it; the attempt that answered settles at its cost.
8. The answer's cost is the provider's ``usage.cost`` (plus
   ``cost_details.upstream_inference_cost`` on a BYOK account) when given
   (``usd_source: provider``), unless that is below half the cost computed from its
   token counts at the models.yaml prices, which is then used instead (``computed``,
   with a note in ``raw``); else it is computed from tokens (``computed``), adding
   reasoning tokens unless the adapter counts them inside the output. A cost that is
   not a finite number settles at the projection.
9. With a JSON schema, the answer is parsed (code fences and leading prose dropped)
   and validated. Any parse or validation failure becomes ``Response.json_error``,
   never an exception. No repair here.
10. A successful answer is cached; a generation that failed upstream (finish_reason
    "error") raises ModelCallError and is never cached. One ``model_call`` trace
    event is emitted per call, also when something fails after the provider was
    paid; the ledger is settled (conservatively) before any exception is re-raised.

Trace and Response accounting: ``usd`` is the answer's cost on success (the
ledger's count for the call on failure). ``ledger_usd`` is everything the ledger
counts for the call, including lost attempts still reserved at their projection
(``unreconciled_attempts``, ``unreconciled_reserved_usd``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .. import config
from ..config import MissingCredentialError, ModelEntry, ModelsConfig, load_secret
from ..schemas.jsonschema_lite import check
from .adapters import get_adapter
from .adapters.base import Adapter, AdapterRequest, AdapterResult, as_cost, as_int, reasoning_level
from .cache import DiskCache
from .errors import ModelCallError, ModelNotConfigured, RouterError
from .http import HttpClient, default_client
from .ledger import Ledger, check_label
from .media import normalize_images
from .redact import redact, scrub, strip_secret

ROLES = ("writer", "reader", "system1")
MESSAGE_ROLES = ("system", "user", "assistant")
CHARS_PER_TOKEN = 3.5
DEFAULT_IMAGE_TOKENS = 1600
ASTRA_DEFAULT_PRICES = (10.00, 50.00)
#: per-attempt deadline when the caller gives none: writers run long generations
WRITER_TIMEOUT_S = 900.0
DEFAULT_TIMEOUT_S = 300.0
#: a provider total below this fraction of the token-computed cost is not trusted
PROVIDER_COST_FLOOR = 0.5
_JSON_ERRORS_SHOWN = 12


@dataclass
class Response:
    text: str
    json: Any | None
    json_error: str | None
    model_key: str
    provider: str
    model: str
    role: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_tokens: int | None
    usd: float
    usd_source: str
    latency_ms: float
    cache_hit: bool
    request_hash: str
    finish_reason: str | None
    reasoning_setting: Any
    json_mode: str | None
    image_count: int
    est_input_tokens: int
    projected_usd: float
    raw: dict  # scrubbed provider response (no key, no base64)

    @property
    def json_ok(self) -> bool:
        """True when a JSON answer parsed and validated."""
        return self.json is not None and self.json_error is None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- helpers
def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    item = getattr(obj, "item", None)
    if callable(item):  # numpy scalar
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy array
        return tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable (the request must be plain JSON data)")


def canonical_json(obj: Any) -> str:
    """Sorted keys, compact separators: the same request always gives the same text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
                      default=_json_default)


def request_hash(doc: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def plain_json(value: Any, what: str) -> Any:
    """``value`` as plain JSON data (numpy scalars and arrays become numbers and lists), key order kept.

    Raises ValueError or TypeError (a bad argument, before anything is spent).
    """
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default))
    except RecursionError:
        raise ValueError(f"{what} is nested too deeply to send") from None
    except TypeError as exc:
        raise TypeError(f"{what} must be plain JSON data: {exc}") from None
    except ValueError as exc:
        raise ValueError(f"{what} must be plain JSON data (no NaN or infinity): {exc}") from None


def _ordered_json(value: Any) -> str | None:
    """Compact JSON text that keeps key order (a JSON schema's property order changes the request)."""
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def normalize_messages(messages: list[dict[str, Any]] | str) -> list[dict[str, str]]:
    """A string becomes one user message; a list must hold {role, content} dicts with string content."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a string or a non-empty list of {'role', 'content'} dicts")
    out: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be a dict with 'role' and 'content'")
        role = msg.get("role")
        content = msg.get("content")
        if role not in MESSAGE_ROLES:
            raise ValueError(f"messages[{i}].role must be one of {', '.join(MESSAGE_ROLES)}, got {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"messages[{i}].content must be a string (attach images with images=, not in content)")
        out.append({"role": role, "content": content})
    return out


_MODELS_LOCK = threading.Lock()
_MODELS_CACHE: dict[str, Any] = {"stamp": None, "cfg": None}


def default_models() -> ModelsConfig:
    """config/models.yaml, reloaded when the file changes."""
    path = Path(config.MODELS_YAML)
    try:
        st = path.stat()
        stamp: tuple[Any, ...] | None = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    with _MODELS_LOCK:
        if stamp is not None and _MODELS_CACHE["stamp"] == stamp and _MODELS_CACHE["cfg"] is not None:
            return _MODELS_CACHE["cfg"]
        cfg = config.load_models(path)
        _MODELS_CACHE.update(stamp=stamp, cfg=cfg)
        return cfg


def resolve_route(entry: ModelEntry) -> tuple[ModelEntry, str | None]:
    """The entry to call: itself, or its models.yaml ``alt_route`` when only the alternate key is set.

    The alternate route is taken only when the primary adapter's key is missing and
    the alternate adapter's key is present (for example glimpse through OpenRouter
    when ``ELORIAN_API_KEY`` is empty). The returned copy keeps the key, prices and
    reasoning of the entry and swaps in the alternate adapter and id. The second
    value says why the route changed (None when it did not).
    """
    alt = getattr(entry, "alt_route", None)
    if alt is None or not getattr(alt, "id", None) or not getattr(alt, "adapter", None):
        return entry, None
    try:
        primary = get_adapter(entry.adapter)
        names = primary.env_vars(entry)
        if not names or config.has_secret(names):
            return entry, None
        alt_adapter = get_adapter(alt.adapter)
    except ModelNotConfigured:
        return entry, None
    routed = entry.model_copy(update={"adapter": alt.adapter, "id": alt.id, "base_url": None, "endpoint": None,
                                      "env_key": None, "alt_route": None})
    alt_names = alt_adapter.env_vars(routed)
    if alt_names and not config.has_secret(alt_names):
        return entry, None
    return routed, (f"{names[0]} is not set, so {entry.key} goes through its alt_route "
                    f"({alt.adapter}, {alt.id})")


def prices(entry: ModelEntry, models: ModelsConfig) -> tuple[float, float, float | None, str | None]:
    """(price_in, price_out, cached_in, fallback) in USD per million tokens.

    When either of the entry's prices is unknown, both come from the models.yaml
    ``astra`` entry (or 10.00 / 50.00 if that is missing), and ``fallback`` says so.
    """
    if entry.price_in is not None and entry.price_out is not None:
        return float(entry.price_in), float(entry.price_out), entry.cached_in, None
    astra = models.models.get("astra")
    if astra is not None and astra.price_in is not None and astra.price_out is not None:
        return float(astra.price_in), float(astra.price_out), None, "astra"
    return ASTRA_DEFAULT_PRICES[0], ASTRA_DEFAULT_PRICES[1], None, "astra_default"


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return int(math.ceil(value))
    return None


def reasoning_budget(setting: Any) -> int:
    """Reasoning tokens a setting allows beyond max_tokens: its ``max_tokens`` or ``budget_tokens``, else 0.

    When both are given the larger counts. Only a dict setting carries a budget.
    """
    if not isinstance(setting, dict):
        return 0
    found = [_token_count(setting.get(name)) for name in ("max_tokens", "budget_tokens")]
    return max((n for n in found if n is not None), default=0)


def project(entry: ModelEntry, models: ModelsConfig, text_chars: int, image_count: int,
            max_tokens: int) -> tuple[int, float, str | None]:
    """(estimated input tokens, worst-case USD, price fallback) for one attempt.

    ``max_tokens`` is the output allowance: the call's max_tokens plus any reasoning budget.
    """
    image_tokens = entry.image_tokens or DEFAULT_IMAGE_TOKENS
    est_in = math.ceil(text_chars / CHARS_PER_TOKEN) + image_count * image_tokens
    pin, pout, _cached, fallback = prices(entry, models)
    return est_in, est_in * pin / 1e6 + max_tokens * pout / 1e6, fallback


def _token_cost(result: AdapterResult, entry: ModelEntry, models: ModelsConfig, est_in: int,
                max_tokens: int) -> tuple[float | None, list[str]]:
    """The cost computed from the result's token counts at models.yaml prices (None with no counts)."""
    n_in, n_out = result.input_tokens, result.output_tokens
    if n_in is None and n_out is None:
        return None, []
    notes: list[str] = []
    pin, pout, cached_in, fallback = prices(entry, models)
    if fallback:
        notes.append(f"models.yaml has no price for {entry.key}: cost computed at Astra prices "
                     f"({pin:.2f} / {pout:.2f} per million tokens)")
    if n_in is None:
        n_in = est_in
        notes.append("input tokens not reported: the projection's estimate was used")
    if n_out is None:
        n_out = max_tokens
        notes.append("output tokens not reported: the output allowance (max_tokens plus any reasoning budget) "
                     "was used")
    reasoning = int(result.reasoning_tokens or 0)
    inside = getattr(result, "reasoning_in_output", True)
    if reasoning > 0 and inside is not True:
        n_out += reasoning
        if inside is False:
            notes.append("reasoning tokens are reported separately and were added to the output tokens")
        elif result.output_tokens is not None and reasoning > result.output_tokens:
            notes.append("reasoning tokens exceed the output tokens, so they are reported separately and were added")
        else:
            notes.append("whether this provider's output tokens include reasoning tokens is unknown, so they were "
                         "added (conservative)")
    n_cached = min(max(int(result.cached_tokens or 0), 0), int(n_in))
    rate_cached = cached_in if (cached_in is not None and fallback is None) else pin
    usd = ((n_in - n_cached) * pin + n_cached * rate_cached + n_out * pout) / 1e6
    return usd, notes


def cost_known(result: AdapterResult) -> bool:
    """False when the result carries neither a cost nor a token count (a 200 without usage)."""
    return (as_cost(result.usd) is not None or result.input_tokens is not None
            or result.output_tokens is not None)


def settled_cost(result: AdapterResult, entry: ModelEntry, models: ModelsConfig, projected: float,
                 est_in: int, max_tokens: int) -> tuple[float, str, list[str]]:
    """(usd, usd_source, notes) for a call that returned. ``max_tokens`` is the output allowance."""
    notes: list[str] = []
    computed, computed_notes = _token_cost(result, entry, models, est_in, max_tokens)
    reported = as_cost(result.usd)
    if result.usd is not None and reported is None:
        notes.append(f"the provider's cost {result.usd!r} is not a finite non-negative number and was ignored")
    if reported is not None:
        both = result.input_tokens is not None and result.output_tokens is not None
        if both and computed is not None and math.isfinite(computed) and reported < PROVIDER_COST_FLOOR * computed:
            notes.append(f"the provider reported ${reported:.6f}, below half the ${computed:.6f} computed from its "
                         "token counts at models.yaml prices, so the call is settled at the computed cost")
            notes.extend(computed_notes)
            usd, source = computed, "computed"
        else:
            return float(reported), "provider", notes
    elif computed is None:
        notes.append("the provider reported no token counts, so the call is settled at its projected worst case")
        usd, source = float(projected), "computed"
    else:
        notes.extend(computed_notes)
        usd, source = computed, "computed"
    if not math.isfinite(usd) or usd < 0:
        notes.append(f"the computed cost {usd!r} is not a finite non-negative number, so the call is settled at its "
                     "projection")
        usd, source = float(projected), "computed"
    return float(usd), source, notes


# --------------------------------------------------------------------------- JSON answers
_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _clip(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _decode_error(exc: BaseException) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"not JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    if isinstance(exc, RecursionError):
        return "not JSON: nested too deeply to parse"
    if isinstance(exc, ValueError):  # for example an integer literal longer than Python reads
        return f"not JSON: {_clip(str(exc))}"
    return f"not JSON: {type(exc).__name__}: {_clip(str(exc))}"


def _parse_candidate(text: str) -> tuple[Any, str | None]:
    s = text.strip()
    if not s:
        return None, "empty answer"
    try:
        return json.loads(s), None
    except Exception as exc:  # noqa: BLE001 - JSONDecodeError, ValueError, RecursionError: all a json_error
        first: BaseException = exc
    starts = sorted(i for i in (s.find("{"), s.find("[")) if i >= 0)
    if not starts:
        return None, _decode_error(first)
    last: BaseException = first
    decoder = json.JSONDecoder()
    for start in starts:
        body = s[start:]
        try:
            value, _end = decoder.raw_decode(body)
            return value, None
        except Exception as exc:  # noqa: BLE001
            last = exc
        end = max(body.rfind("}"), body.rfind("]"))
        if end > 0:
            try:
                return json.loads(body[: end + 1]), None
            except Exception:  # noqa: BLE001
                pass
    return None, _decode_error(last)


def parse_json_text(text: str | None) -> tuple[Any, str | None]:
    """A model's JSON answer: (value, None), or (None, why it is not JSON). Never raises.

    Code fences are stripped, and prose before the first ``{`` or ``[`` is dropped.
    """
    try:
        if text is None or not str(text).strip():
            return None, "empty answer"
        s = str(text).strip()
        candidates = [m.group(1) for m in _FENCE_RE.finditer(s)]
        if s.startswith("```") and not candidates:  # an opening fence the answer never closed (truncated)
            body = s[3:]
            newline = body.find("\n")
            candidates.append(body[newline + 1:] if newline >= 0 else body)
        candidates.append(s)
        error = "not JSON"
        for candidate in candidates:
            value, err = _parse_candidate(candidate)
            if err is None:
                return value, None
            error = err
        return None, error
    except Exception as exc:  # noqa: BLE001 - a degenerate answer is a json_error, never an exception
        return None, _decode_error(exc)


def validate_json(value: Any, schema: dict[str, Any]) -> str | None:
    """None when ``value`` satisfies ``schema``, else the errors joined into one line. Never raises."""
    try:
        errors = check(value, schema)
    except RecursionError:
        return "schema check failed: the answer is nested too deeply to check"
    except Exception as exc:  # noqa: BLE001 - e.g. OverflowError for a 400-digit number under "type": "number"
        return f"schema check failed: {type(exc).__name__}: {_clip(str(exc))}"
    if not errors:
        return None
    shown = [str(e) for e in errors[:_JSON_ERRORS_SHOWN]]
    more = f" (and {len(errors) - len(shown)} more)" if len(errors) > len(shown) else ""
    return "; ".join(shown) + more


def _json_result(text: str, json_value: Any, json_schema: dict[str, Any] | None) -> tuple[Any, str | None]:
    try:
        if json_schema is not None:
            value, err = parse_json_text(text)
            if err is not None:
                return None, err
            return value, validate_json(value, json_schema)
        if json_value is not None:
            return json_value, None
        return None, None
    except Exception as exc:  # noqa: BLE001 - never raise after a paid call
        return None, f"could not read the answer as JSON: {type(exc).__name__}: {_clip(str(exc))}"


def _text_chars(msgs: list[dict[str, str]], json_schema: dict[str, Any] | None, extra: dict[str, Any]) -> int:
    chars = sum(len(m["content"]) for m in msgs)
    if json_schema is not None:
        chars += len(canonical_json(json_schema))
    if extra:
        chars += len(canonical_json(extra))
    return chars


def _check_key(key: str, names: list[str], model_key: str, adapter: str) -> None:
    """A key must be printable ASCII without spaces: it goes into an HTTP header. Names the variable only."""
    if not key:
        return
    if key.isascii() and all(32 < ord(c) < 127 for c in key):
        return
    var = next((n for n in names if config.env(n)), names[0] if names else "the API key variable")
    what = ("a character that is not ASCII (a smart quote or a zero-width space, for example)" if not key.isascii()
            else "a space or a control character")
    raise MissingCredentialError(
        f"{model_key} ({adapter}): {var} holds {what}, so it cannot be sent in an HTTP header. Re-enter the key "
        f"in {config.ENV_FILE} using plain ASCII characters only.") from None


# --------------------------------------------------------------------------- spend per attempt
class _CallSpend:
    """The ledger side of one chat() call: one reservation per HTTP attempt.

    The first attempt is reserved by chat() before the adapter runs; the HTTP layer
    calls :meth:`before_attempt` and :meth:`after_attempt` around every POST
    (``loopkit.router.http.AttemptHooks``), so each retry and JSON-mode fallback is
    reserved before it is sent and refused when a cap would be crossed.
    """

    def __init__(self, ledger: Ledger, projected: float, model_key: str, rhash: str, tags: dict[str, Any],
                 paid: bool):
        self.ledger = ledger
        self.projected = float(projected)
        self.model_key = model_key
        self.rhash = rhash
        self.tags = dict(tags)
        self.paid = bool(paid)
        self.attempts: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ reservations
    def _reserve(self) -> dict[str, Any]:
        n = len(self.attempts) + 1
        rid = self.ledger.reserve(self.projected, self.model_key, self.rhash, tags={**self.tags, "attempt": n},
                                  paid=self.paid)
        rec = {"rid": rid, "n": n, "state": "reserved", "status": None, "usd": self.projected, "settled": False}
        self.attempts.append(rec)
        return rec

    def reserve_first(self) -> None:
        self._reserve()

    def before_attempt(self, *, url: str | None = None, attempt: int | None = None) -> str:
        last = self.attempts[-1] if self.attempts else None
        rec = last if last is not None and last["state"] == "reserved" else self._reserve()
        rec["state"] = "sent"
        return rec["rid"]

    def after_attempt(self, token: Any, *, outcome: str, status: int | None = None,
                      generation_id: str | None = None) -> None:
        rec = next((a for a in self.attempts if a["rid"] == token), None)
        if rec is None:
            return
        rec["state"], rec["status"] = outcome, status
        if outcome == "rejected":
            self._settle(rec, 0.0, "computed", f"HTTP {status} before generation, recorded as 0", quiet=True)

    # ------------------------------------------------------------------ settling
    def _settle(self, rec: dict[str, Any], usd: float, source: str, note: str | None, *, quiet: bool) -> None:
        try:
            self.ledger.settle(rec["rid"], usd, source, note=note)
        except Exception:
            if quiet:  # unsettled is the conservative outcome: it stays counted at the projection
                return
            raise
        rec["usd"], rec["settled"] = float(usd), True

    def _answer_record(self) -> dict[str, Any] | None:
        for rec in reversed(self.attempts):
            if rec["state"] == "answered":
                return rec
        return self.attempts[-1] if self.attempts else None

    def settle_answer(self, usd: float, source: str, note: str | None, *, known: bool) -> None:
        """Settle the attempt that answered at ``usd``; with ``known`` false it stays reserved."""
        rec = self._answer_record()
        if rec is None:
            return
        if rec["state"] == "reserved":  # a call that sends no HTTP (the FakeModel): its one reservation
            rec["state"] = "answered"
        if known:
            self._settle(rec, usd, source, note, quiet=False)

    def settle_failure(self, exc: BaseException, cost_of: Any) -> str:
        """Settle what a failed call's error tells; everything else stays reserved. Returns the usd_source."""
        if not isinstance(exc, ModelCallError):
            return "computed"
        if exc.rejected_before_generation:
            for rec in self.attempts:  # sent attempts were settled as they were refused
                if rec["state"] == "reserved" and not rec["settled"]:
                    self._settle(rec, 0.0, "computed", "nothing was sent for this reservation (the request was "
                                 "refused before generation or could not be built), recorded as 0", quiet=True)
            return "computed"
        if exc.usd is not None:
            usd, source, notes = cost_of(exc)
            rec = self._answer_record()
            if rec is not None:
                if rec["state"] == "reserved":
                    rec["state"] = "answered"
                self._settle(rec, usd, source, "failed call: " + ("; ".join(notes) or "cost reported with the error"),
                             quiet=True)
                return source if rec["settled"] else "computed"
        return "computed"

    # ------------------------------------------------------------------ accounting
    @property
    def sent(self) -> int:
        return sum(1 for a in self.attempts if a["state"] != "reserved")

    def counted(self) -> float:
        """What the ledger counts for this call: settled amounts, and projections for the rest."""
        return sum(a["usd"] for a in self.attempts)

    def fields(self) -> dict[str, Any]:
        open_ = [a for a in self.attempts if not a["settled"]]
        return {"ledger_usd": round(self.counted(), 10), "reserved_attempts": len(self.attempts),
                "unreconciled_attempts": len(open_),
                "unreconciled_reserved_usd": round(sum(a["usd"] for a in open_), 10)}


def _failed_call_cost(exc: ModelCallError, adapter: Adapter, entry: ModelEntry, models: ModelsConfig,
                      projected: float, est_in: int, max_out: int) -> tuple[float, str, list[str]]:
    """The cost of a failed call that reported one, checked against its token counts like any other."""
    usage = exc.raw.get("usage") if isinstance(exc.raw, dict) and isinstance(exc.raw.get("usage"), dict) else {}
    probe = AdapterResult(text="", input_tokens=as_int(usage.get("input_tokens")),
                          output_tokens=as_int(usage.get("output_tokens")),
                          reasoning_tokens=as_int(usage.get("reasoning_tokens")),
                          cached_tokens=as_int(usage.get("cached_tokens")), usd=exc.usd,
                          reasoning_in_output=adapter.reasoning_in_output)
    return settled_cost(probe, entry, models, projected, est_in, max_out)


# --------------------------------------------------------------------------- trace
def _payload(*, provider: str, model: str, role: str, input_tokens: int | None, output_tokens: int | None,
             reasoning_tokens: int | None, cached_tokens: int | None, usd: float, usd_source: str,
             latency_ms: float, cache_hit: bool, request_hash: str, model_key: str, adapter: str, ok: bool,
             error: str | None, finish_reason: str | None, reasoning_setting: Any, image_count: int,
             est_input_tokens: int, projected_usd: float, phase: str | None, json_mode: str | None,
             http_status: int | None, attempts: int | None, max_tokens: int, json_error: str | None,
             model_requested: str, upstream_provider: str | None, route: str = "primary",
             ledger_usd: float = 0.0, reserved_attempts: int = 0, unreconciled_attempts: int = 0,
             unreconciled_reserved_usd: float = 0.0, timeout_s: float | None = None,
             reasoning_budget: int = 0) -> dict[str, Any]:
    """A ``model_call`` payload: every ModelCallPayload field plus the router's own."""
    return {
        "provider": provider, "model": model, "role": role,
        "input_tokens": input_tokens, "output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens,
        "usd": round(float(usd), 10), "usd_source": usd_source, "latency_ms": round(float(latency_ms), 3),
        "cache_hit": bool(cache_hit), "request_hash": request_hash,
        "model_key": model_key, "adapter": adapter, "ok": bool(ok), "error": error,
        "finish_reason": finish_reason, "reasoning_setting": reasoning_setting, "image_count": int(image_count),
        "est_input_tokens": int(est_input_tokens), "projected_usd": round(float(projected_usd), 10),
        "phase": phase, "json_mode": json_mode, "http_status": http_status, "attempts": attempts,
        "cached_tokens": cached_tokens, "max_tokens": int(max_tokens), "json_error": json_error,
        "model_requested": model_requested, "upstream_provider": upstream_provider, "route": route,
        "ledger_usd": round(float(ledger_usd), 10), "reserved_attempts": int(reserved_attempts),
        "unreconciled_attempts": int(unreconciled_attempts),
        "unreconciled_reserved_usd": round(float(unreconciled_reserved_usd), 10), "timeout_s": timeout_s,
        "reasoning_budget": int(reasoning_budget),
    }


def _emit(trace: Any, payload: dict[str, Any]) -> None:
    if trace is not None:
        trace.emit("model_call", payload)


# --------------------------------------------------------------------------- chat
def chat(model_key: str, messages: list[dict] | str, images: list | None = None, json_schema: dict | None = None,
         role: str = "writer", max_tokens: int = 4096, reasoning: str | dict | None = "high", *,
         temperature: float | None = None, schema_name: str = "response", use_cache: bool = True,
         cache_salt: str | None = None, trace: Any = None, phase: str | None = None,
         timeout_s: float | None = None, models: ModelsConfig | None = None, ledger: Ledger | None = None,
         extra: dict | None = None, cache: DiskCache | None = None, http: HttpClient | None = None) -> Response:
    """Call ``model_key`` (a key in config/models.yaml) through its adapter. See the module docstring.

    ``timeout_s`` is the per-attempt deadline; None means 900 s for a writer call and
    300 s otherwise. ``cache`` and ``http`` replace the default disk cache and HTTP
    client (tests, or a bake-off that keeps its own cache). Raises ModelNotConfigured,
    MissingCredentialError, SpendCapExceeded, ModelCallError or RouterError; a bad
    argument raises ValueError or TypeError before anything is spent.
    """
    # 1. the entry (and its alt_route when only the alternate key is set)
    if not isinstance(model_key, str):
        raise TypeError(f"model_key must be a string, got {type(model_key).__name__}")
    check_label(model_key, "model_key")
    cfg = models if models is not None else default_models()
    configured = cfg.get(model_key)
    entry, route_note = resolve_route(configured)
    adapter: Adapter = get_adapter(entry.adapter)
    if entry.adapter != "fake" and not entry.id:
        raise ModelNotConfigured(
            f"model {model_key!r} has no id in config/models.yaml: the entry is marked unverified "
            f"(verified: {str(entry.verified).lower()}). Look up its id and price, set them, then call it.")
    adapter.check_entry(entry)
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}, got {role!r}")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}")
    if timeout_s is None:
        timeout_s = WRITER_TIMEOUT_S if role == "writer" else DEFAULT_TIMEOUT_S
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) \
            or timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive number of seconds, got {timeout_s!r}")
    timeout = float(timeout_s)
    if json_schema is not None and not isinstance(json_schema, dict):
        raise TypeError("json_schema must be a dict (a JSON Schema) or None")
    if extra is not None and not isinstance(extra, dict):
        raise TypeError("extra must be a dict or None")
    if phase is not None:
        if not isinstance(phase, str):
            raise TypeError(f"phase must be a string or None, got {type(phase).__name__}")
        check_label(phase, "phase", allow_empty=True)
    if cache_salt is not None and not isinstance(cache_salt, str):
        cache_salt = str(cache_salt)
    reasoning_level(reasoning)  # a bad setting fails here for every adapter, even one that ignores it

    # 2. normalize and hash (plain JSON first, so the body always serializes once money is reserved)
    msgs = normalize_messages(messages)
    imgs = normalize_images(images)
    target: int | None = None
    if imgs:
        if not entry.vision:
            raise RouterError(f"model {model_key!r} is text-only in config/models.yaml (vision: false), "
                              f"so it cannot take the {len(imgs)} image(s) given")
        target = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "user"), None)
        if target is None:
            raise ValueError("images attach to the last user message, but messages has no user message")
    schema_d: dict[str, Any] | None = plain_json(json_schema, "json_schema")
    extra_d: dict[str, Any] = plain_json(extra, "extra") or {}
    reasoning_setting = plain_json(adapter.normalize_reasoning(reasoning, entry), "reasoning")
    temperature = plain_json(temperature, "temperature")
    model_id = entry.id or "fake"
    request = AdapterRequest(
        model_key=model_key, model_id=model_id, entry=entry, role=role, messages=msgs, images=imgs,
        image_message_index=target, json_schema=schema_d, schema_name=schema_name, max_tokens=max_tokens,
        reasoning=reasoning_setting, reasoning_requested=reasoning, temperature=temperature,
        timeout_s=timeout, extra=extra_d)
    adapter.check_request(request)
    doc: dict[str, Any] = {
        "adapter": entry.adapter, "model": model_id, "url": adapter.url(entry), "messages": msgs,
        "images": [i.sha256 for i in imgs],
        "json_schema": _ordered_json(schema_d),  # property order is part of the request
        "json_modes": adapter.modes(request) if schema_d is not None else None,
        "schema_name": schema_name, "max_tokens": max_tokens, "reasoning": reasoning_setting,
        "temperature": temperature, "cache_salt": cache_salt, "extra": extra_d,
    }
    token = adapter.cache_token(request)
    if token:
        doc["fake_script"] = token
    rhash = request_hash(doc)
    request.request_hash = rhash
    budget = reasoning_budget(reasoning_setting)
    max_out = max_tokens + budget
    est_in, projected, price_fallback = project(entry, cfg, _text_chars(msgs, schema_d, extra_d), len(imgs),
                                                max_out)

    def payload(**kw: Any) -> dict[str, Any]:
        base = dict(provider=adapter.name, model=model_id, role=role, input_tokens=None, output_tokens=None,
                    reasoning_tokens=None, cached_tokens=None, usd=0.0, usd_source="computed", latency_ms=0.0,
                    cache_hit=False, request_hash=rhash, model_key=model_key, adapter=entry.adapter, ok=True,
                    error=None, finish_reason=None, reasoning_setting=reasoning_setting, image_count=len(imgs),
                    est_input_tokens=est_in, projected_usd=projected, phase=phase, json_mode=None,
                    http_status=None, attempts=0, max_tokens=max_tokens, json_error=None,
                    model_requested=model_id, upstream_provider=None,
                    route="alt_route" if route_note else "primary", timeout_s=timeout, reasoning_budget=budget)
        base.update(kw)
        return _payload(**base)

    # 3. cache
    store = cache if cache is not None else DiskCache()
    if use_cache:
        t0 = time.perf_counter()
        hit = store.get(rhash)
        if hit is not None and hit.get("finish_reason") == "error":
            hit = None  # a generation that failed upstream is never an answer
        if hit is not None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            value, json_error = _json_result(hit["text"], hit.get("json_value"), schema_d)
            raw = dict(hit.get("raw") or {})
            raw.update(original_usd=hit.get("usd"), original_usd_source=hit.get("usd_source"),
                       original_latency_ms=hit.get("latency_ms"), cached_at=hit.get("created"))
            if hit.get("ledger_usd") is not None:
                raw["original_ledger_usd"] = hit.get("ledger_usd")
            response = Response(
                text=hit["text"], json=value, json_error=json_error, model_key=model_key, provider=adapter.name,
                model=hit.get("model") or model_id, role=role, input_tokens=hit.get("input_tokens"),
                output_tokens=hit.get("output_tokens"), reasoning_tokens=hit.get("reasoning_tokens"),
                cached_tokens=hit.get("cached_tokens"), usd=0.0, usd_source="computed", latency_ms=latency_ms,
                cache_hit=True, request_hash=rhash, finish_reason=hit.get("finish_reason"),
                reasoning_setting=hit.get("reasoning_setting", reasoning_setting), json_mode=hit.get("json_mode"),
                image_count=len(imgs), est_input_tokens=est_in, projected_usd=projected, raw=raw)
            _emit(trace, payload(model=response.model, input_tokens=response.input_tokens,
                                 output_tokens=response.output_tokens, reasoning_tokens=response.reasoning_tokens,
                                 cached_tokens=response.cached_tokens, latency_ms=latency_ms, cache_hit=True,
                                 finish_reason=response.finish_reason, json_mode=response.json_mode,
                                 reasoning_setting=response.reasoning_setting, json_error=json_error,
                                 upstream_provider=hit.get("upstream_provider")))
            return response

    # 4. credentials
    names = adapter.env_vars(entry)
    key = ""
    if names:
        try:
            key = load_secret(names, adapter.label)
        except MissingCredentialError as exc:
            hint = ""
            alt = getattr(configured, "alt_route", None)
            if alt is not None and getattr(alt, "id", None):
                try:
                    alt_vars = get_adapter(alt.adapter).default_env_vars
                except ModelNotConfigured:
                    alt_vars = ()
                if alt_vars:
                    hint = f" Or set {alt_vars[0]} to call it through its alt_route ({alt.adapter}, {alt.id})."
            raise MissingCredentialError(f"{model_key} ({entry.adapter}): {exc}{hint}") from None
        _check_key(key, names, model_key, entry.adapter)

    # 5-6. projection (above) and the first attempt's reservation; later attempts reserve through the hooks
    led = ledger if ledger is not None else Ledger()
    spend = _CallSpend(led, projected, model_key, rhash,
                       tags={"role": role, "phase": phase, "adapter": entry.adapter, "model": model_id,
                             "call": uuid.uuid4().hex[:16]},
                       paid=entry.adapter != "fake" or projected > 0)
    spend.reserve_first()  # SpendCapExceeded here: nothing was attempted, so nothing is traced
    request.attempt_hooks = spend

    # 7. send
    client = http if http is not None else default_client()
    t_send = time.perf_counter()
    try:
        result = adapter.send(request, key, client)
    except BaseException as exc:
        latency_ms = (time.perf_counter() - t_send) * 1000.0
        source = "computed"
        try:
            source = spend.settle_failure(exc, lambda e: _failed_call_cost(e, adapter, entry, cfg, projected, est_in,
                                                                             max_out))
        except Exception:  # noqa: BLE001 - unsettled is the conservative outcome; the call's error matters more
            pass
        if trace is not None:
            try:
                raw_err = getattr(exc, "raw", None)
                attempts = getattr(exc, "attempts", None)
                _emit(trace, payload(ok=False, error=redact(f"{type(exc).__name__}: {exc}", key),
                                     usd=spend.counted(), usd_source=source, latency_ms=latency_ms,
                                     http_status=getattr(exc, "status", None),
                                     attempts=attempts if attempts is not None else (spend.sent or None),
                                     finish_reason=raw_err.get("finish_reason") if isinstance(raw_err, dict) else None,
                                     **spend.fields()))
            except Exception:  # noqa: BLE001 - the call's own error matters more
                pass
        raise
    latency_ms = (time.perf_counter() - t_send) * 1000.0

    # 8. cost and settle. From here on nothing escapes before the ledger is settled and the event is written.
    failure: BaseException | None = None
    usd, usd_source, notes, known = float(projected), "computed", [], False
    try:
        usd, usd_source, notes = settled_cost(result, entry, cfg, projected, est_in, max_out)
        known = cost_known(result)
    except BaseException as exc:  # noqa: BLE001
        failure = exc  # the cost is unknown: the reservation stays counted at the projection
    try:
        spend.settle_answer(usd, usd_source, "; ".join(notes) if notes else None, known=known and failure is None)
    except BaseException as exc:  # noqa: BLE001
        failure = failure or exc
    spent_fields = spend.fields()

    # 9-10. JSON, the response and the cache
    text, json_error, response = "", None, None
    model_name = getattr(result, "model_returned", None) or model_id
    if failure is None:
        try:
            raw = scrub(result.raw or {}, key)
            if notes:
                raw["usd_note"] = "; ".join(notes)
            if route_note:
                raw["route_note"] = route_note
            if price_fallback and usd_source == "computed":
                raw["price_fallback"] = price_fallback
            if budget:
                raw["reasoning_budget_projected"] = budget
            cached_raw = dict(raw)  # this call's ledger figures belong to this call, not to a later cache hit
            if spent_fields["unreconciled_attempts"]:
                raw["unreconciled_attempts"] = spent_fields["unreconciled_attempts"]
                raw["unreconciled_reserved_usd"] = spent_fields["unreconciled_reserved_usd"]
            raw["ledger_usd"] = spent_fields["ledger_usd"]
            text = result.text or ""
            if key and len(key) >= 6 and key in text:  # only the exact key; the answer is otherwise kept as written
                text = text.replace(key, "[redacted]")
            json_value = strip_secret(result.json_value, key)
            value, json_error = _json_result(text, json_value, schema_d)
            response = Response(
                text=text, json=value, json_error=json_error, model_key=model_key, provider=adapter.name,
                model=model_name, role=role, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                reasoning_tokens=result.reasoning_tokens, cached_tokens=result.cached_tokens, usd=usd,
                usd_source=usd_source, latency_ms=latency_ms, cache_hit=False, request_hash=rhash,
                finish_reason=result.finish_reason, reasoning_setting=reasoning_setting, json_mode=result.json_mode,
                image_count=len(imgs), est_input_tokens=est_in, projected_usd=projected, raw=raw)
            if use_cache and text.strip() and result.finish_reason != "error":
                record = {
                    "request_hash": rhash, "created": time.time(), "model_key": model_key, "provider": adapter.name,
                    "adapter": entry.adapter, "model": model_name, "model_requested": model_id, "text": text,
                    "json_value": json_value, "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens, "reasoning_tokens": result.reasoning_tokens,
                    "cached_tokens": result.cached_tokens, "usd": usd, "usd_source": usd_source,
                    "latency_ms": round(latency_ms, 3), "finish_reason": result.finish_reason,
                    "json_mode": result.json_mode, "reasoning_setting": reasoning_setting,
                    "http_status": result.http_status, "attempts": result.attempts,
                    "generation_id": result.generation_id, "upstream_provider": result.upstream_provider,
                    "ledger_usd": spent_fields["ledger_usd"], "raw": cached_raw,
                }
                store.put(rhash, record)
        except BaseException as exc:  # noqa: BLE001
            failure = exc

    # the trace event, then the result or the error
    event = payload(model=model_name, input_tokens=getattr(result, "input_tokens", None),
                    output_tokens=getattr(result, "output_tokens", None),
                    reasoning_tokens=getattr(result, "reasoning_tokens", None),
                    cached_tokens=getattr(result, "cached_tokens", None),
                    usd=usd if failure is None else spend.counted(), usd_source=usd_source, latency_ms=latency_ms,
                    finish_reason=getattr(result, "finish_reason", None), json_mode=getattr(result, "json_mode", None),
                    http_status=getattr(result, "http_status", None), attempts=getattr(result, "attempts", None),
                    json_error=json_error, upstream_provider=getattr(result, "upstream_provider", None),
                    ok=failure is None,
                    error=None if failure is None else redact(f"{type(failure).__name__}: {failure}", key),
                    **spent_fields)
    if failure is not None:
        try:
            _emit(trace, event)
        except Exception:  # noqa: BLE001 - the original error matters more
            pass
        raise failure
    _emit(trace, event)
    return response
