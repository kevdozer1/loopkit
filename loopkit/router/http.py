"""JSON POST with loopkit's retry policy (pattern: statebench ``backends/http.py``).

Policy: up to 3 attempts. Retry on HTTP 429, 500, 502, 503, 504, 520, 522, 524 and
on timeouts and connection errors, with exponential backoff (1.5 s doubling,
capped at 20 s, with jitter) that honors ``Retry-After``. Any other 4xx is a
request defect and is returned at once. After the last attempt the result is
returned, never raised: the adapter decides what the failure means.

Spend hooks. Every POST that reaches a provider may be generated and billed, so a
caller can pass ``hooks`` (see :class:`AttemptHooks`). ``before_attempt`` runs right
before each POST is sent: the router reserves the call's projected cost there, and
a SpendCapExceeded it raises propagates and stops the call. ``after_attempt``
reports each sent POST's outcome:

* ``answered``: a 2xx response arrived. The router settles it at its measured cost.
* ``rejected``: HTTP 400, 401, 402, 403, 404 or 429 with no generation id. The
  provider refused it before generating anything, so it is settled at $0.
* ``lost``: anything else (a timeout, the per-attempt deadline, a connection error, a
  response that dropped or failed to read, a 5xx, any other status). Its cost is
  unknown, so its reservation stays counted at the projection.

The request (body, URL and headers) is built once, before anything is reserved or
sent. If it cannot be built (a body that is not plain JSON, a URL httpx cannot
parse, a header value that is not ASCII), nothing is sent: the result has
``sent=False`` and ``error`` says why, without the key.

The key is used in exactly one place, the ``Authorization`` header built in
:meth:`HttpClient.post_json`. It never reaches a log line, an exception message
or a result: error text is scrubbed of the key and of data URLs and base64 runs.

``timeout_s`` is a per-attempt deadline on the whole response, not only on the
gap between bytes. The body is read as a stream and abandoned once the deadline
passes, so a server that keeps a long call alive with filler bytes still times out.
A body that has fully arrived by then (its Content-Length reached, or a complete
JSON document) is kept: it was generated and billed, and buying it again would
cost twice.

Tests inject an ``httpx`` transport (``httpx.MockTransport``) and a ``sleep``
function, so nothing touches the network and nothing really sleeps.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import json
import math
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .redact import redact

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.5
BACKOFF_CAP_S = 20.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 520, 522, 524})
#: statuses a provider returns before generating anything (when there is no generation id): settled at $0
REJECTED_BEFORE_GENERATION = frozenset({400, 401, 402, 403, 404, 429})
OUTCOMES = ("answered", "rejected", "lost")
_KEEP_HEADERS = ("retry-after", "content-type", "x-request-id", "x-generation-id")
_ERROR_TEXT_CHARS = 4000


class _DeadlineExceeded(Exception):
    pass


class AttemptHooks(Protocol):
    """What :meth:`HttpClient.post_json` calls around every POST it sends."""

    def before_attempt(self, *, url: str, attempt: int) -> Any:
        """Called right before a POST is sent. May raise (SpendCapExceeded) to stop the call."""

    def after_attempt(self, token: Any, *, outcome: str, status: int | None,
                      generation_id: str | None) -> None:
        """Called once for every POST that was sent, with its outcome (see the module docstring)."""


def generation_id_of(data: Any, headers: Mapping[str, str] | None = None) -> str | None:
    """The generation id a response carries (body ``id`` or the ``x-generation-id`` header), if any."""
    if isinstance(data, dict):
        gid = data.get("id")
        if isinstance(gid, str) and gid.strip():
            return gid
    if headers:
        gid = headers.get("x-generation-id")
        if isinstance(gid, str) and gid.strip():
            return gid
    return None


def attempt_outcome(status: int | None, generation_id: str | None = None) -> str:
    """``answered``, ``rejected`` or ``lost`` for one sent attempt (see the module docstring)."""
    if status is None:
        return "lost"
    if 200 <= status < 300:
        return "answered"
    if status in REJECTED_BEFORE_GENERATION and not generation_id:
        return "rejected"
    return "lost"


@dataclass
class HttpResult:
    """The last attempt's outcome. ``status`` is None when no HTTP response arrived.

    ``sent`` is False when the request could not be built, so nothing was sent
    (``attempts`` is then 0 and ``error`` says why).
    """

    status: int | None
    data: Any = None
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    attempts: int = 0
    latency_ms: float = 0.0
    attempt_log: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    sent: bool = True

    @property
    def ok(self) -> bool:
        return self.sent and self.status is not None and 200 <= self.status < 300


def retry_after_seconds(value: Any) -> float | None:
    """Seconds from a ``Retry-After`` header (a number or an HTTP date), or None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        wait = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        wait = (when - dt.datetime.now(dt.timezone.utc)).total_seconds()
    if not math.isfinite(wait):
        return None
    return max(0.0, wait)


def build_error_text(exc: BaseException, key: str | None, stage: str) -> str:
    """Why a request could not be built, without the key (``stage`` is ``body`` or ``request``)."""
    if stage == "body":
        return redact(f"the request body is not plain JSON ({type(exc).__name__}: {exc})", key)[:300]
    if isinstance(exc, UnicodeError):
        # the exception's args hold the whole header value, so none of it is quoted here
        return ("a request header holds a character that is not ASCII (check the API key in .env for a "
                "smart quote or a zero-width space)")
    if isinstance(exc, httpx.InvalidURL):
        return redact(f"the URL is not valid ({exc})", key)[:300]
    return redact(f"the request could not be built ({type(exc).__name__}: {exc})", key)[:300]


def _content_length(headers: Mapping[str, str]) -> int | None:
    value = headers.get("content-length")
    try:
        n = int(str(value).strip()) if value is not None else None
    except ValueError:
        return None
    return n if n is not None and n >= 0 else None


def _arrived_complete(resp: httpx.Response, chunks: list[bytes], expected: int | None) -> bool:
    """True when the whole body is already here (so a deadline passing now must not discard it)."""
    received = sum(len(c) for c in chunks)
    if expected is not None and (resp.num_bytes_downloaded >= expected
                                 or ("content-encoding" not in resp.headers and received >= expected)):
        return True
    body = b"".join(chunks).strip()
    if not body or body[:1] not in (b"{", b"["):
        return False
    try:
        json.loads(body)
    except (ValueError, RecursionError):
        return False
    return True


class HttpClient:
    """A reusable ``httpx.Client`` plus the retry policy. Thread safe."""

    def __init__(self, transport: httpx.BaseTransport | None = None, *,
                 sleep: Callable[[float], None] = time.sleep, max_attempts: int = MAX_ATTEMPTS,
                 backoff_base_s: float = BACKOFF_BASE_S, backoff_cap_s: float = BACKOFF_CAP_S,
                 jitter: Callable[[], float] = random.random):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._transport = transport
        self._sleep = sleep
        self.max_attempts = int(max_attempts)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_cap_s = float(backoff_cap_s)
        self._jitter = jitter
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing
    def _http(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(transport=self._transport, follow_redirects=False)
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def backoff_s(self, attempt: int, retry_after: Any = None) -> float:
        """Wait before attempt ``attempt + 1``: Retry-After when given, else 1.5 s doubling with jitter."""
        hinted = retry_after_seconds(retry_after)
        if hinted is not None:
            return min(self.backoff_cap_s, hinted)
        base = self.backoff_base_s * (2 ** (attempt - 1))
        return min(self.backoff_cap_s, base * (0.7 + 0.6 * float(self._jitter())))

    def _build_request(self, url: str, content: bytes, *, key: str | None, timeout_s: float,
                       headers: dict[str, str] | None) -> httpx.Request:
        """The POST request with its URL parsed and its headers encoded. May raise."""
        send_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            send_headers.update(headers)
        if key:
            send_headers["Authorization"] = "Bearer " + key
        return self._http().build_request("POST", url, content=content, headers=send_headers,
                                          timeout=httpx.Timeout(timeout_s))

    # ------------------------------------------------------------------ the call
    def post_json(self, url: str, body: dict[str, Any], *, key: str | None = None, timeout_s: float = 300.0,
                  headers: dict[str, str] | None = None, hooks: AttemptHooks | None = None) -> HttpResult:
        """POST ``body`` as JSON with the retry policy. Never raises for HTTP or network failures.

        ``hooks.before_attempt`` may raise (the router's SpendCapExceeded); that propagates.
        """
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) \
                or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number of seconds")
        started = time.perf_counter()
        # 1. build everything before any attempt is reserved or sent
        try:
            content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except Exception as exc:  # noqa: BLE001 - nothing was sent; say why
            return self._not_sent(build_error_text(exc, key, "body"), started)
        try:
            request = self._build_request(url, content, key=key, timeout_s=float(timeout_s), headers=headers)
        except Exception as exc:  # noqa: BLE001 - nothing was sent; the text never quotes a header value
            return self._not_sent(build_error_text(exc, key, "request"), started)
        # 2. the attempts
        log: list[dict[str, Any]] = []
        status: int | None = None
        text = ""
        data: Any = None
        kept: dict[str, str] = {}
        error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            token = hooks.before_attempt(url=url, attempt=attempt) if hooks is not None else None
            t0 = time.perf_counter()
            entry: dict[str, Any] = {"attempt": attempt}
            text, data, kept = "", None, {}
            try:
                status, raw_bytes, resp_headers = self._once(request, float(timeout_s))
            except (_DeadlineExceeded, httpx.TimeoutException) as exc:
                status, error = None, f"timeout after {timeout_s:g} s ({type(exc).__name__.lstrip('_')})"
                entry.update(status=None, error=error)
            except httpx.RequestError as exc:
                status, error = None, redact(f"connection error: {type(exc).__name__}: {exc}", key)
                entry.update(status=None, error=error)
            else:
                error = None
                text = raw_bytes.decode("utf-8", errors="replace")
                try:
                    data = json.loads(text) if text.strip() else None
                except (ValueError, RecursionError):
                    data = None
                kept = {k: v for k, v in resp_headers.items() if k in _KEEP_HEADERS}
                entry["status"] = status
                if not (200 <= status < 300):
                    text = redact(text, key)[:_ERROR_TEXT_CHARS]
                    entry["error"] = text[:300]
            gen_id = generation_id_of(data, kept) if status is not None else None
            outcome = attempt_outcome(status, gen_id)
            entry["outcome"] = outcome
            if gen_id:
                entry["generation_id"] = gen_id
            entry["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            log.append(entry)
            if hooks is not None:
                hooks.after_attempt(token, outcome=outcome, status=status, generation_id=gen_id)
            retryable = status is None or status in RETRY_STATUSES
            if not retryable or attempt == self.max_attempts:
                break
            wait = self.backoff_s(attempt, kept.get("retry-after") if status is not None else None)
            entry["wait_s"] = round(wait, 3)
            self._sleep(wait)
        return HttpResult(status=status, data=data, text=text, headers=kept, attempts=len(log),
                          latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
                          attempt_log=log, error=error)

    @staticmethod
    def _not_sent(error: str, started: float) -> HttpResult:
        return HttpResult(status=None, attempts=0, sent=False, error=error,
                          latency_ms=round((time.perf_counter() - started) * 1000.0, 1))

    def _once(self, request: httpx.Request, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        resp = self._http().send(request, stream=True)
        try:
            expected = _content_length(resp.headers)
            chunks: list[bytes] = []
            for chunk in resp.iter_bytes():
                chunks.append(chunk)
                if time.monotonic() > deadline:
                    if _arrived_complete(resp, chunks, expected):
                        break  # the whole body is here: keep it rather than buy it again
                    raise _DeadlineExceeded()
            return int(resp.status_code), b"".join(chunks), {k.lower(): v for k, v in resp.headers.items()}
        finally:
            resp.close()


_DEFAULT_CLIENT: HttpClient | None = None
_DEFAULT_LOCK = threading.Lock()


def default_client() -> HttpClient:
    """The process-wide client chat() uses when it is not handed one."""
    global _DEFAULT_CLIENT
    with _DEFAULT_LOCK:
        if _DEFAULT_CLIENT is None:
            _DEFAULT_CLIENT = HttpClient()
        return _DEFAULT_CLIENT


def set_default_client(client: HttpClient | None) -> None:
    """Replace the process-wide client (None restores a fresh real one on next use)."""
    global _DEFAULT_CLIENT
    with _DEFAULT_LOCK:
        _DEFAULT_CLIENT = client
