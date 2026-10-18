"""Plain JSON values for trace payloads and run metadata.

``to_jsonable`` turns numpy scalars and arrays, paths, tuples, sets, pydantic
models, dataclasses, enums, dates, bytes and exceptions into dicts, lists,
strings, numbers, booleans and None. NaN and infinity become None, and a lone
surrogate (U+D800 to U+DFFF) in a string or a dict key becomes U+FFFD, so every
line loopkit writes is strict JSON that any parser reads back.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import json
import math
import re
from collections.abc import Mapping, Sequence, Set
from enum import Enum
from pathlib import PurePath
from typing import Any

import numpy as np
from pydantic import BaseModel

__all__ = ["dumps", "to_jsonable"]

# A high surrogate followed by a low one is a UTF-16 pair split into two code points;
# any other surrogate is lone. Python strings can hold either (a reply cut inside an
# emoji, bytes decoded with surrogateescape), UTF-8 cannot, and pydantic's JSON parser
# rejects the escape json.dumps writes for a lone one.
_SURROGATES = re.compile("[\ud800-\udbff][\udc00-\udfff]|[\ud800-\udfff]")


def to_jsonable(obj: Any) -> Any:
    """``obj`` as plain JSON types. Unknown objects fall back to ``str(obj)``."""
    t = type(obj)
    if t is str:
        return obj if obj.isascii() else _text(obj)
    if obj is None or t is bool or t is int:
        return obj
    if t is float:
        return obj if math.isfinite(obj) else None
    if t is dict:
        return {_key(k): to_jsonable(v) for k, v in obj.items()}
    if t is list or t is tuple:
        return [to_jsonable(v) for v in obj]
    return _convert(obj)


def dumps(obj: Any, ascii: bool = True) -> str:
    """Compact JSON for values already made safe by :func:`to_jsonable`.

    ``ascii=True`` escapes every non-ASCII character, so a line never contains a
    character (U+2028, NEL) that some line splitters treat as a line break.
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=ascii, allow_nan=False)


def _text(text: str) -> str:
    """``text`` with each lone surrogate replaced by U+FFFD and each split pair joined."""
    if text.isascii():
        return text
    return _SURROGATES.sub(_fix_surrogate, text)


def _fix_surrogate(match: re.Match[str]) -> str:
    pair = match.group()
    if len(pair) == 2:
        return chr(0x10000 + ((ord(pair[0]) - 0xD800) << 10) + (ord(pair[1]) - 0xDC00))
    return "\N{REPLACEMENT CHARACTER}"


def _convert(obj: Any) -> Any:
    if isinstance(obj, Enum):  # before str and int: str(E.A) is "E.A", not its value
        return to_jsonable(obj.value)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, str):  # np.str_ and other str subclasses
        return _text(str.__str__(obj))
    if isinstance(obj, np.ndarray):
        return _array(obj)
    if isinstance(obj, (np.datetime64, np.timedelta64)):
        return str(obj)
    if isinstance(obj, np.generic):  # complex and other numpy scalars
        return to_jsonable(obj.item())
    if isinstance(obj, BaseModel):
        # Python mode, then the rules above: numpy fields (arbitrary_types_allowed) work,
        # where mode="json" raises PydanticSerializationError. Excluded fields stay excluded.
        return to_jsonable(obj.model_dump(mode="python"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, PurePath):
        return _text(str(obj))
    if isinstance(obj, (bytes, bytearray, memoryview)):
        raw = bytes(obj)
        try:
            return raw.decode("utf-8")  # strict UTF-8 never yields a surrogate
        except UnicodeDecodeError:
            return "base64:" + base64.b64encode(raw).decode("ascii")
    if isinstance(obj, (set, frozenset, Set)):
        items = [to_jsonable(v) for v in obj]
        try:
            return sorted(items)
        except TypeError:  # mixed types: order by their JSON text so the output is stable
            return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(obj, Sequence):  # tuple and list subclasses, ranges, deques
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if isinstance(obj, dt.timedelta):
        return obj.total_seconds()
    if isinstance(obj, BaseException):
        text = str(obj)
        return _text(f"{type(obj).__name__}: {text}" if text else type(obj).__name__)
    if isinstance(obj, complex):
        return str(obj)
    tolist = getattr(obj, "tolist", None)  # array-likes from other libraries
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except Exception:  # fall through to the string form
            pass
    return _text(str(obj))


def _array(arr: np.ndarray) -> Any:
    kind = arr.dtype.kind
    if kind in "biu":
        return arr.tolist()
    if kind == "f":
        if arr.size and not np.isfinite(arr).all():
            return to_jsonable(arr.tolist())  # NaN and inf become None
        return arr.tolist()
    if kind in "mM":
        return arr.astype(str).tolist()
    return to_jsonable(arr.tolist())  # text (may hold surrogates), bytes, object, complex, structured


def _key(key: Any) -> str:
    """A dict key as JSON writes it (JSON object keys are strings)."""
    if type(key) is str:
        return _text(key)
    if isinstance(key, Enum):
        key = key.value
    if isinstance(key, str):
        return _text(str.__str__(key))
    if key is None:
        return "null"
    if isinstance(key, (bool, np.bool_)):
        return "true" if key else "false"
    if isinstance(key, (int, np.integer)):
        return str(int(key))
    if isinstance(key, (float, np.floating)):
        return repr(float(key))
    return _text(str(key))
