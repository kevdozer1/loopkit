"""Scrubbing: no key, header, data URL or base64 blob leaves the router.

Error text from a provider can echo the request body (escaped JSON, URL-encoded),
so a base64 character may also appear as an escaped ``/`` or ``+``, and padding as
an escaped ``=``. The patterns follow that. Credentials are cut from every string;
media is cut from error text and from stored provider responses, but never from a
model's answer text (an answer is kept as the model wrote it).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Keys that would carry request headers or credentials. Compared lowercased, "_" read as "-".
FORBIDDEN_KEYS = frozenset({
    "authorization", "proxy-authorization", "headers", "request-headers", "response-headers",
    "api-key", "apikey", "x-api-key", "cookie", "set-cookie",
})

_B64_CHAR = r"(?:[A-Za-z0-9+/_-]|\\/|\\u002[bf]|%2[bf])"
_B64_PAD = r"(?:=|\\u003d|%3d)"
_DATA_URL_RE = re.compile(
    r"data(?::|%3a)[\w.+-]+(?:/|\\/|\\u002f|%2f)[\w.+-]+(?:;[\w.+=-]+)*(?:;|%3b)base64(?:,|%2c)"
    + _B64_CHAR + "*" + _B64_PAD + "*",
    re.IGNORECASE,
)
_MEDIA_MARKER_RE = re.compile(r"data:image[^\s\"',;]*|base64,", re.IGNORECASE)
_BASE64_RUN_RE = re.compile(_B64_CHAR + "{64,}" + _B64_PAD + "{0,2}", re.IGNORECASE)
_SECRET_RES = (
    re.compile(r"sk-or-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
)

REDACTED = "[redacted]"


def _secrets(secrets: str | Iterable[str | None] | None) -> list[str]:
    if secrets is None:
        return []
    if isinstance(secrets, str):
        secrets = [secrets]
    return [s for s in secrets if s and len(s) >= 6]


def redact(text: str, secrets: str | Iterable[str | None] | None = None, *, media: bool = True) -> str:
    """``text`` without credentials and (with ``media``) without data URLs or long base64 runs."""
    if not isinstance(text, str):
        text = str(text)
    for secret in _secrets(secrets):
        text = text.replace(secret, REDACTED)
    for pattern in _SECRET_RES:
        text = pattern.sub(REDACTED, text)
    if media:
        text = _DATA_URL_RE.sub("[data URL removed]", text)
        text = _BASE64_RUN_RE.sub("[base64 removed]", text)
        text = _MEDIA_MARKER_RE.sub("[media removed]", text)
    return text


def _forbidden(key: Any) -> bool:
    return str(key).lower().replace("_", "-") in FORBIDDEN_KEYS


def scrub(obj: Any, secrets: str | Iterable[str | None] | None = None, *, media: bool = True) -> Any:
    """A JSON-safe copy of ``obj`` without header or credential keys and with every string redacted."""
    keys = _secrets(secrets)
    return _scrub(obj, keys, media, 0)


def _scrub(obj: Any, keys: list[str], media: bool, depth: int) -> Any:
    if depth > 64:
        return "[too deep]"
    if isinstance(obj, dict):  # keys are text too: a server can echo a credential as a key
        return {redact(str(k), keys, media=media): _scrub(v, keys, media, depth + 1)
                for k, v in obj.items() if not _forbidden(k)}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v, keys, media, depth + 1) for v in obj]
    if isinstance(obj, str):
        return redact(obj, keys, media=media)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"[{len(obj)} bytes removed]"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return redact(str(obj), keys, media=media)


def strip_secret(obj: Any, secret: str | None) -> Any:
    """A copy of ``obj`` with every exact occurrence of ``secret`` in its strings (keys too) replaced.

    For a model's parsed answer (Jev's answers dict): only the exact key is cut, the
    answer is otherwise kept as the model wrote it. Lists, tuples and dicts are
    walked without recursion, so a deeply nested answer cannot raise here.
    """
    if not secret or len(secret) < 6:
        return obj

    def fix(value: Any) -> Any:
        return value.replace(secret, REDACTED) if isinstance(value, str) else value

    if not isinstance(obj, (dict, list, tuple)):
        return fix(obj)
    root: Any = {} if isinstance(obj, dict) else []
    copies: dict[int, Any] = {id(obj): root}  # a shared or cyclic part is copied once and shared again
    stack: list[tuple[Any, Any]] = [(obj, root)]
    while stack:
        src, dst = stack.pop()
        items = src.items() if isinstance(src, dict) else enumerate(src)
        for k, v in items:
            if isinstance(v, (dict, list, tuple)):
                new = copies.get(id(v))
                if new is None:
                    new = {} if isinstance(v, dict) else []
                    copies[id(v)] = new
                    stack.append((v, new))
            else:
                new = fix(v)
            if isinstance(dst, dict):
                dst[fix(k)] = new
            else:
                dst.append(new)
    return root


def has_secret(obj: Any, secret: str) -> bool:
    """True when ``secret`` appears anywhere in ``obj`` (used by tests and the smoke's self-check)."""
    if not secret:
        return False
    if isinstance(obj, dict):
        return any(has_secret(k, secret) or has_secret(v, secret) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(has_secret(v, secret) for v in obj)
    return secret in str(obj)
