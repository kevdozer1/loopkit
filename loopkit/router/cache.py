"""Disk cache of model responses, keyed by the request hash.

One JSON file per response at ``<root>/<hash[:2]>/<hash>.json``. The default root
is ``LOOPKIT_DATA/cache/router`` (through :mod:`loopkit.paths`). Writes go to a
temp file in the same folder and are moved into place with ``os.replace``, so a
reader never sees half a record. A file that does not parse as a JSON object is a
miss. Records are scrubbed by the router before they are written (no key, no
base64); :meth:`DiskCache.put` cuts credential patterns once more on the way in.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

from .. import paths
from .redact import scrub

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_VERBATIM = ("text", "json_value")


class DiskCache:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else paths.cache_dir(create=False) / "router"

    def path_for(self, request_hash: str) -> Path:
        if not isinstance(request_hash, str) or not _HASH_RE.match(request_hash):
            raise ValueError(f"request hash must be 64 lowercase hex characters, got {request_hash!r}")
        return self.root / request_hash[:2] / f"{request_hash}.json"

    def get(self, request_hash: str) -> dict[str, Any] | None:
        path = self.path_for(request_hash)
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            return None
        except (OSError, ValueError):  # unreadable, or not UTF-8 (a corrupt file is a miss)
            return None
        try:
            record = json.loads(text)
        except (ValueError, RecursionError):
            return None
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            return None
        if record.get("request_hash") not in (None, request_hash):
            return None
        return record

    def put(self, request_hash: str, record: dict[str, Any]) -> Path | None:
        """Write ``record`` atomically. Returns the path, or None if the write failed (a warning is issued)."""
        path = self.path_for(request_hash)
        try:
            # credential patterns are cut from everything except the answer itself, which is kept as written
            kept = {k: record[k] for k in _VERBATIM if k in record}
            record = dict(scrub({k: v for k, v in record.items() if k not in _VERBATIM}, media=False))
            record.update(kept)
            record.setdefault("request_hash", request_hash)
            record.setdefault("created", time.time())
            text = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            # a NaN, an odd type or absurd nesting in a provider response: skip caching it, never fail the call
            warnings.warn(f"router cache skipped {request_hash[:12]}: record is not plain JSON "
                          f"({type(exc).__name__}: {str(exc)[:120]})", RuntimeWarning, stacklevel=2)
            return None
        tmp = path.with_name(f".{request_hash}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(6):
                try:
                    os.replace(tmp, path)
                    return path
                except PermissionError:  # Windows: a reader has the target open for a moment
                    if attempt == 5:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            warnings.warn(f"router cache write failed for {path.name}: {type(exc).__name__}: {exc}",
                          RuntimeWarning, stacklevel=2)
            try:
                tmp.unlink()
            except OSError:
                pass
        return None

    def __contains__(self, request_hash: str) -> bool:
        return self.get(request_hash) is not None
