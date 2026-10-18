"""Scrubbing: no key, header, data URL or base64 blob leaves the router.

Error text from a provider can echo the request body (escaped JSON, URL-encoded),
so a base64 character may also appear as an escaped ``/`` or ``+``, and padding as
an escaped ``=``. The patterns follow that. Credentials are cut from every string;
media is cut from error text and from stored provider responses, but never from a
model's answer text (an answer is kept as the model wrote it).

A key is cut in every form it can take in text: as written, and as JSON text holds
it (``json.dumps(key)[1:-1]``, so a key with a quote or a backslash is also caught
escaped; see :func:`secret_forms`).
"""

from __future__ import annotations

import json
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
_MIN_SECRET_CHARS = 6


def secret_forms(secret: str) -> list[str]:
    """``secret`` in every form text can hold it, longest first.

    As written, and as JSON text holds it: ``json.dumps(secret)[1:-1]`` (a quote or
    a backslash escaped), with and without non-ASCII escaped, and with ``/`` also
    written as ``\\/`` (some JSON encoders do that). Most keys have one form only.
    """
    forms = {secret}
    for ascii_only in (True, False):
        escaped = json.dumps(secret, ensure_ascii=ascii_only)[1:-1]
        forms.add(escaped)
        forms.add(escaped.replace("/", "\\/"))
    return sorted(forms, key=lambda s: (-len(s), s))


def _secrets(secrets: str | Iterable[str | None] | None) -> list[str]:
    """Every form of every secret of 6 or more characters, longest first (so a longer form is cut whole)."""
    if secrets is None:
        return []
    if isinstance(secrets, str):
        secrets = [secrets]
    forms: set[str] = set()
    for secret in secrets:
        if isinstance(secret, str) and len(secret) >= _MIN_SECRET_CHARS:
            forms.update(secret_forms(secret))
    return sorted(forms, key=lambda s: (-len(s), s))


def _redact(text: str, forms: list[str], media: bool) -> str:
    for form in forms:
        text = text.replace(form, REDACTED)
    for pattern in _SECRET_RES:
        text = pattern.sub(REDACTED, text)
    if media:
        text = _DATA_URL_RE.sub("[data URL removed]", text)
        text = _BASE64_RUN_RE.sub("[base64 removed]", text)
        text = _MEDIA_MARKER_RE.sub("[media removed]", text)
    return text


def redact(text: str, secrets: str | Iterable[str | None] | None = None, *, media: bool = True) -> str:
    """``text`` without credentials and (with ``media``) without data URLs or long base64 runs."""
    if not isinstance(text, str):
        text = str(text)
    return _redact(text, _secrets(secrets), media)


def _forbidden(key: Any) -> bool:
    return str(key).lower().replace("_", "-") in FORBIDDEN_KEYS


def scrub(obj: Any, secrets: str | Iterable[str | None] | None = None, *, media: bool = True) -> Any:
    """A JSON-safe copy of ``obj`` without header or credential keys and with every string redacted."""
    return _scrub(obj, _secrets(secrets), media, 0)


def _scrub(obj: Any, forms: list[str], media: bool, depth: int) -> Any:
    if depth > 64:
        return "[too deep]"
    if isinstance(obj, dict):  # keys are text too: a server can echo a credential as a key
        return {_redact(str(k), forms, media): _scrub(v, forms, media, depth + 1)
                for k, v in obj.items() if not _forbidden(k)}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v, forms, media, depth + 1) for v in obj]
    if isinstance(obj, str):
        return _redact(obj, forms, media)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"[{len(obj)} bytes removed]"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return _redact(str(obj), forms, media)


def strip_secret(obj: Any, secret: str | None) -> Any:
    """A copy of ``obj`` with every occurrence of ``secret`` in its strings (keys too) replaced.

    For a model's parsed answer (Jev's answers dict): only the key is cut, in each of
    its :func:`secret_forms`; the answer is otherwise kept as the model wrote it.
    Lists, tuples and dicts are walked without recursion, so a deeply nested answer
    cannot raise here.
    """
    if not isinstance(secret, str) or len(secret) < _MIN_SECRET_CHARS:
        return obj
    forms = secret_forms(secret)

    def fix(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        for form in forms:
            value = value.replace(form, REDACTED)
        return value

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
    """True when ``secret`` appears anywhere in ``obj``, in any of its :func:`secret_forms` (used by tests)."""
    if not secret:
        return False
    if isinstance(obj, dict):
        return any(has_secret(k, secret) or has_secret(v, secret) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(has_secret(v, secret) for v in obj)
    text = str(obj)
    return any(form in text for form in secret_forms(secret))
