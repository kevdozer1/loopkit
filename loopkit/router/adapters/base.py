"""Adapter interface (pattern: statebench ``backends/base.py`` and robolabel ``providers/base.py``).

An adapter turns one normalized :class:`AdapterRequest` into one provider call and
returns an :class:`AdapterResult` with measured tokens and, when the provider
reports it, the cost. Everything else (cache, credentials, projection, ledger,
trace, JSON validation) lives in :mod:`loopkit.router.core`, so adding a provider
is one file plus a registry line.

Every POST an adapter sends goes through :meth:`HttpClient.post_json` with
``hooks=request.attempt_hooks``, so the router reserves each attempt (retries and
JSON-mode fallbacks included) before it is sent.

An adapter raises :class:`~loopkit.router.errors.ModelCallError` when the call
fails. ``rejected_before_generation`` is set only when no attempt of the call, in
any JSON mode, was lost or answered: every one was refused before generation (HTTP
400/401/402/403/404/429 with no generation id) or never sent (the request could not
be built, or no connection could be made). A cost in a non-2xx body is never put on
the error: that attempt's outcome is unknown, so it stays counted at its projection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import ModelCallError
from ..http import REJECTED_BEFORE_GENERATION as _REJECTED
from ..http import attempt_outcome, generation_id_of
from ..redact import redact

if TYPE_CHECKING:  # pragma: no cover
    from ...config import ModelEntry
    from ..http import HttpClient, HttpResult
    from ..media import ImageData

REASONING_LEVELS = ("low", "medium", "high")
_DISABLED_WORDS = ("disabled", "disable", "none", "off")
# HTTP statuses a provider returns before any generation (no generation id): settled at $0.
REJECTED_BEFORE_GENERATION = tuple(sorted(_REJECTED))
_UNSENT_OUTCOMES = ("rejected", "not_sent")
#: a token count this large is not a count (the router treats it as not reported)
MAX_TOKEN_COUNT = 10 ** 12
# a host name as it goes on the wire (IDNA already applied) or an IPv4 address
_HOST_RE = re.compile(r"[A-Za-z0-9._-]+")


@dataclass
class AdapterRequest:
    """One normalized call. ``images`` attach to ``messages[image_message_index]`` (the last user message)."""

    model_key: str
    model_id: str
    entry: "ModelEntry"
    role: str
    messages: list[dict[str, str]]
    images: list["ImageData"] = field(default_factory=list)
    image_message_index: int | None = None
    json_schema: dict[str, Any] | None = None
    schema_name: str = "response"
    max_tokens: int = 4096
    reasoning: Any = None  # the adapter-normalized setting that is sent
    reasoning_requested: Any = None  # what the caller asked for
    temperature: float | None = None
    timeout_s: float = 300.0
    extra: dict[str, Any] = field(default_factory=dict)
    request_hash: str = ""
    #: set by chat(): reserves every HTTP attempt in the ledger (see loopkit.router.http.AttemptHooks)
    attempt_hooks: Any = field(default=None, repr=False, compare=False)

    @property
    def last_user_index(self) -> int | None:
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]["role"] == "user":
                return i
        return None

    def last_user_text(self) -> str:
        i = self.last_user_index
        return self.messages[i]["content"] if i is not None else ""


@dataclass
class AdapterResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    usd: float | None = None  # provider-reported cost; None means compute it from tokens
    model_returned: str | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)  # scrubbed provider response
    http_status: int | None = None
    attempts: int = 1
    json_mode: str | None = None
    generation_id: str | None = None
    rejected_before_generation: bool = False
    #: a parsed answer the API returns as data (Jev's answers). chat() cuts the key from it and
    #: rebuilds ``text`` as ``json.dumps`` of the scrubbed value.
    json_value: Any = None
    upstream_provider: str | None = None
    #: True: output_tokens already include the reasoning tokens. False: they are reported
    #: separately. None: unknown, so they are added (the conservative choice).
    reasoning_in_output: bool | None = True


def reasoning_level(reasoning: Any) -> str | dict[str, Any]:
    """The caller's reasoning argument as ``disabled``, ``low``, ``medium``, ``high``, ``highest`` or a raw dict."""
    if reasoning is None or reasoning is False:
        return "disabled"
    if reasoning is True:
        return "high"
    if isinstance(reasoning, dict):
        return dict(reasoning)
    if isinstance(reasoning, str):
        word = reasoning.strip().lower()
        if word in _DISABLED_WORDS:
            return "disabled"
        if word in REASONING_LEVELS or word == "highest":
            return word
    raise ValueError(f"reasoning must be 'disabled', 'low', 'medium', 'high', 'highest', None, False or a dict, "
                     f"not {reasoning!r}")


def _host_ok(raw_host: bytes) -> bool:
    """True for a host a server can answer to: a DNS name (IDNA applied), an IPv4 or an IPv6 address."""
    import ipaddress

    try:
        host = raw_host.decode("ascii")
    except UnicodeDecodeError:
        return False
    if ":" in host:  # an IPv6 literal, given in brackets
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            return False
        return True
    return bool(_HOST_RE.fullmatch(host))


def check_url(entry: "ModelEntry", field_name: str, url: str | None, sent: str | None = None) -> None:
    """Raise ModelNotConfigured unless the models.yaml ``url`` is usable. Checked before any spend.

    ``sent`` is the exact URL the adapter will send for it (``url`` itself when not
    given). The configured value is checked as written, never stripped: leading or
    trailing whitespace (a YAML ``|`` or ``>`` block scalar leaves a trailing
    newline), any space and any control character are refused. The URL that will be
    sent must be one httpx parses, http or https, with a host a server can answer to
    (httpx accepts ``https://[api.x/v1`` and sends it to the host ``%5Bapi.x``).
    """
    import httpx

    from ..errors import ModelNotConfigured

    if url is None:
        return
    text = url if isinstance(url, str) else str(url)
    target = text if sent is None else (sent if isinstance(sent, str) else str(sent))
    problem = None
    if text != text.strip():
        problem = "has leading or trailing whitespace (a YAML '|' or '>' block scalar leaves a trailing newline)"
    elif any(c.isspace() or not c.isprintable() for c in text + target):
        problem = "holds a space or a control character"
    elif not (text.startswith("https://") or text.startswith("http://")) or len(text) < 10:
        problem = "is not an http(s) URL"
    else:
        try:
            parsed = httpx.URL(target)
        except Exception as exc:  # noqa: BLE001 - httpx.InvalidURL and friends
            problem = f"is not a valid URL ({exc})"
        else:
            if parsed.scheme not in ("http", "https") or not parsed.host:
                problem = "is not an http(s) URL with a host"
            elif not _host_ok(parsed.raw_host):
                problem = f"is not a valid URL (its host {parsed.host!r} is not a host name or an IP address)"
    if problem:
        raise ModelNotConfigured(f"model {entry.key!r}: {field_name} {text!r} in config/models.yaml {problem}")


def as_int(value: Any) -> int | None:
    """A token count from a provider response, or None (negative or non-integral counts are not counts).

    An absurdly large count is returned as it is, so the router can say so: chat()
    treats a count of :data:`MAX_TOKEN_COUNT` or more as not reported, with a note.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def as_cost(value: Any) -> float | None:
    """A provider-reported cost in USD, or None unless it is a finite non-negative number.

    Never raises: an integer too large for a float (``1`` followed by 400 zeros) is
    not a cost either.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        v = float(value)
    except (OverflowError, ValueError):
        return None
    return v if v >= 0 and v == v and v != float("inf") else None


def entry_outcome(entry: dict[str, Any]) -> str:
    """One attempt-log entry's outcome (``answered``, ``rejected``, ``lost`` or ``not_sent``)."""
    outcome = entry.get("outcome")
    if outcome in ("answered", "rejected", "lost", "not_sent"):
        return outcome
    return attempt_outcome(entry.get("status"), entry.get("generation_id"))


def all_rejected_before_generation(log: Iterable[dict[str, Any]]) -> bool:
    """True when no attempt in ``log`` (every JSON mode of one call) was lost or answered."""
    return all(entry_outcome(e) in _UNSENT_OUTCOMES for e in log)


def error_text(res: "HttpResult", key: str | None = None) -> str:
    """The provider's error message from a failed response, scrubbed and cut to a readable length."""
    if not res.sent:
        return redact(res.error or "the request could not be built", key)[:600]
    data = res.data
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        text = str(err.get("message") or err.get("type") or "")
        meta = err.get("metadata")
        if isinstance(meta, dict):
            if meta.get("raw"):
                text += " | " + str(meta.get("raw"))
            if meta.get("provider_name"):
                text += f" (upstream {meta.get('provider_name')})"
        return redact(text, key)[:600]
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return redact(data["error"], key)[:600]
    if isinstance(data, dict) and isinstance(data.get("detail"), (str, list, dict)):
        return redact(json.dumps(data["detail"]) if not isinstance(data["detail"], str) else data["detail"], key)[:600]
    if res.error:
        return redact(res.error, key)[:600]
    return redact(res.text or "", key)[:400]


class Adapter:
    """Base adapter. Subclasses set ``name``, ``label`` and ``default_env_vars`` and implement ``send``."""

    name = "base"
    label = "provider"
    default_env_vars: tuple[str, ...] = ()
    #: whether this provider's output token count already includes the reasoning tokens
    #: (True), reports them separately (False), or nobody knows (None: they are added)
    reasoning_in_output: bool | None = True

    # ------------------------------------------------------------------ config
    def env_vars(self, entry: "ModelEntry") -> list[str]:
        """Variables to look up for the key, first one named in the error.

        A models.yaml ``env_key`` overrides the adapter's default. When it names the
        adapter's own primary variable, the adapter's fallbacks are kept.
        """
        defaults = list(self.default_env_vars)
        if entry.env_key:
            if defaults and entry.env_key == defaults[0]:
                return defaults
            return [entry.env_key]
        return defaults

    def check_entry(self, entry: "ModelEntry") -> None:
        """Raise ModelNotConfigured when the entry lacks something this adapter needs."""

    def check_request(self, request: AdapterRequest) -> None:
        """Raise RouterError for a request this adapter cannot send (before any spend)."""

    def normalize_reasoning(self, reasoning: Any, entry: "ModelEntry") -> Any:
        """The exact reasoning setting this adapter sends (recorded as ``reasoning_setting``)."""
        return None

    def cache_token(self, request: AdapterRequest) -> str | None:
        """Extra text folded into the request hash (only the FakeModel uses it, for its scripts)."""
        return None

    def url(self, entry: "ModelEntry") -> str | None:
        """Where the request goes (part of the request hash); None for an adapter that sends nothing."""
        return None

    def modes(self, request: AdapterRequest) -> list[str | None]:
        """The JSON modes tried in order (part of the request hash when a schema is given)."""
        return [None]

    # ------------------------------------------------------------------ call
    def send(self, request: AdapterRequest, key: str, http: "HttpClient") -> AdapterResult:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def call_error(self, request: AdapterRequest, res: "HttpResult", key: str | None, attempts: int,
                   prefix: str = "", log: list[dict[str, Any]] | None = None) -> ModelCallError:
        """A ModelCallError for a failed HTTP result (its last attempt was not a 2xx).

        ``log`` is the call's whole attempt log (every JSON mode); without it the
        result's own log is used. The before-generation flag is set only when no
        attempt in it was lost or answered and the last one was refused or not sent.
        The error carries no ``usd``: a cost a non-2xx body reports belongs to an
        attempt whose outcome is unknown (a 5xx) or that was refused (settled at $0),
        and never settles anything.
        """
        entries = list(log) if log is not None else list(res.attempt_log)
        gen_id = generation_id_of(res.data, res.headers) if res.sent else None
        if not res.sent:
            what = f"{res.error or 'the request could not be built'}; nothing was sent"
            last_rejected = True
        elif res.status is None:
            what = res.error or "no response"
            last = res.attempt_log[-1] if res.attempt_log else None
            last_rejected = last is not None and entry_outcome(last) == "not_sent"  # it could not connect
        else:
            what = f"HTTP {res.status}: {error_text(res, key)}"
            last_rejected = res.status in REJECTED_BEFORE_GENERATION and not gen_id
        rejected = last_rejected and all_rejected_before_generation(entries)
        hint = ""
        if res.status == 402:
            hint = " (out of credits, or the key's credit limit was reached)"
        elif res.status == 401:
            hint = " (the key was refused; check it in .env)"
        message = (f"{prefix}{request.model_key} ({request.model_id}) via {self.label}: {what}{hint} "
                   f"after {attempts} attempt(s)")
        return ModelCallError(redact(message, key), status=res.status, attempts=attempts,
                              rejected_before_generation=rejected, generation_id=gen_id,
                              raw={"attempt_log": entries, "status": res.status})

    def build_error(self, request: AdapterRequest, exc: BaseException, key: str | None, attempts: int = 0,
                    log: list[dict[str, Any]] | None = None) -> ModelCallError:
        """A ModelCallError for a request body that could not be built (nothing was sent for it)."""
        entries = list(log or [])
        detail = redact(f"{type(exc).__name__}: {exc}", key)[:300]
        return ModelCallError(
            f"{request.model_key} ({request.model_id}) via {self.label}: the request body could not be built "
            f"({detail}); nothing was sent", status=None, attempts=attempts,
            rejected_before_generation=all_rejected_before_generation(entries), raw={"attempt_log": entries})
