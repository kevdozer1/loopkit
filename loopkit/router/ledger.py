"""Spend ledger and caps (pattern: statebench ``spend_guard.py`` and robolabel's reserve-then-reconcile).

The ledger is append-only JSONL at ``LOOPKIT_DATA/spend_ledger.jsonl``::

    {"type": "reserve", "id", "ts", "session", "model_key", "usd": projected, "request_hash", "tags"}
    {"type": "settle",  "id", "ts", "usd": actual, "usd_source"}

Every HTTP attempt of a paid call reserves its projected worst case before it is
sent and is settled at the measured cost afterwards (the router does this per
attempt, so a retry or a JSON-mode fallback is reserved too). Spent is the sum over
reserves of the settled amount when there is one, else the reserved amount. So an
attempt whose cost is never learned (a timeout, a dropped connection, a 5xx, a 200
without usage) stays counted at its worst case. The ledger is loopkit's own record;
it never seeds from a provider's usage page.

Records are written as ASCII JSON (``ensure_ascii``), one per line, and read back by
splitting the raw bytes on ``\\n`` only, so no character in a label can split a
record. Session names, model keys and phase labels must be single-line printable
text. A settle with no matching reserve is counted and reported (``summary()``
and a warning), never silently ignored.

Caps live in ``spend_caps.json`` next to the ledger::

    {"active_session": "S1", "session_caps": {"S1": 1.0},
     "model_caps": {"S4": {"opus": 2.0}}, "total_cap_usd": 21.0}

Caps must be finite non-negative numbers (NaN and Infinity are refused, so a bad
caps file fails closed). A paid reservation is refused when it would cross the
active session's cap, that session's cap for the model, or the total cap, and also
when one of those is already reached, even for a $0 projection. With no active
session every paid call is refused. Only the FakeModel at a $0 projection is free:
the router marks every other call paid.

The check and the append happen under one cross-process lock: an OS lock on a
persistent lock file (``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere).
The OS releases it when the holding process dies, so there is no stale-lock
breaking and the lock file is never deleted.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

from .. import paths
from .errors import RouterError, SpendCapExceeded

CAPS_FILENAME = "spend_caps.json"
LOCK_TIMEOUT_S = 120.0
_EPS = 1e-12
_ANY = object()
_ORPHANS_WARNED: dict[str, int] = {}  # ledger path -> orphan count already warned about by reserve()

if os.name == "nt":  # pragma: no cover - platform branch
    import msvcrt

    def _try_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # one byte at offset 0; raises OSError while another holds it

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:  # pragma: no cover - platform branch
    import fcntl

    def _try_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- labels
def is_label(value: Any) -> bool:
    """True for single-line printable text (no line or paragraph separators, no control characters)."""
    return isinstance(value, str) and value.isprintable()


def check_label(value: Any, what: str, *, allow_empty: bool = False) -> str:
    """``value`` when it is single-line printable text, else ValueError."""
    if not is_label(value) or (not allow_empty and not value.strip()):
        raise ValueError(f"{what} must be {'' if allow_empty else 'non-empty '}single-line printable text "
                         f"(no line breaks, separators or control characters), got {value!r}")
    return value


# --------------------------------------------------------------------------- lock
class FileLock:
    """A cross-process lock: an OS lock on a persistent lock file (never deleted, no stale breaking).

    The OS drops the lock when the holder closes the file or dies. Threads of one
    process exclude each other too, because each acquire opens its own handle.
    """

    def __init__(self, path: str | Path, *, timeout_s: float = LOCK_TIMEOUT_S, poll_s: float = 0.02):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            raise RouterError(f"lock {self.path} is already held by this FileLock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            while True:
                try:
                    _try_lock(fd)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RouterError(
                            f"could not take the spend-ledger lock {self.path} within {self.timeout_s:.0f} s: "
                            "another loopkit process holds it. The OS releases it when that process exits.") from None
                    time.sleep(self.poll_s)
        except BaseException:
            os.close(fd)  # closing the handle also drops a lock taken just before an interrupt
            raise
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            pass  # closing the handle below releases it anyway
        finally:
            os.close(fd)

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# --------------------------------------------------------------------------- caps file
def default_caps_path(ledger_path: str | Path | None = None) -> Path:
    base = Path(ledger_path) if ledger_path is not None else paths.ledger_path()
    return base.parent / CAPS_FILENAME


def _empty_caps() -> dict[str, Any]:
    return {"active_session": None, "session_caps": {}, "model_caps": {}, "total_cap_usd": None}


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a number of US dollars")


def _cap_value(value: Any, where: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RouterError(f"{path}: {where} must be a finite non-negative number of US dollars, got {value!r}. "
                          "Fix it or delete the file, then call set_active_session again.")
    return float(value)


def read_caps(path: str | Path) -> dict[str, Any]:
    """The caps file as a dict (the empty structure when it does not exist).

    NaN, Infinity, negative or non-numeric caps and a multi-line session name raise
    RouterError, so a damaged caps file refuses calls instead of admitting them.
    """
    p = Path(path)
    text = ""
    for attempt in range(10):
        try:
            text = p.read_text(encoding="utf-8-sig")
            break
        except FileNotFoundError:
            return _empty_caps()
        except PermissionError:  # Windows: a writer is replacing the file this instant
            if attempt == 9:
                raise
            time.sleep(0.02 * (attempt + 1))
    try:
        data = json.loads(text, parse_constant=_refuse_constant) if text.strip() else {}
    except (ValueError, RecursionError) as exc:
        raise RouterError(f"{p} is not valid JSON ({exc}). Fix it or delete it, then call "
                          "set_active_session again.") from None
    if not isinstance(data, dict):
        raise RouterError(f"{p} must hold a JSON object. Fix it or delete it, then call set_active_session again.")
    caps = _empty_caps()
    caps.update(data)
    for name in ("session_caps", "model_caps"):
        if not isinstance(caps.get(name), dict):
            caps[name] = {}
    active = caps.get("active_session")
    if active is not None and (not is_label(active) or not active.strip()):
        raise RouterError(f"{p}: active_session must be single-line printable text, got {active!r}. "
                          "Call set_active_session with a plain name.")
    caps["session_caps"] = {str(k): _cap_value(v, f"session_caps[{k!r}]", p) for k, v in caps["session_caps"].items()}
    model_caps: dict[str, dict[str, float]] = {}
    for session, per_model in caps["model_caps"].items():
        if not isinstance(per_model, dict):
            raise RouterError(f"{p}: model_caps[{session!r}] must map model keys to US dollars, got {per_model!r}")
        model_caps[str(session)] = {str(m): _cap_value(v, f"model_caps[{session!r}][{m!r}]", p)
                                    for m, v in per_model.items()}
    caps["model_caps"] = model_caps
    if caps.get("total_cap_usd") is not None:
        caps["total_cap_usd"] = _cap_value(caps["total_cap_usd"], "total_cap_usd", p)
    return caps


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _usd(value: Any, what: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number of US dollars, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number of US dollars, got {value!r}") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{what} must be a finite non-negative number of US dollars, got {value!r}")
    return v


def set_active_session(name: str, cap_usd: float, *, total_cap_usd: float | None = None,
                       model_caps: dict[str, float] | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Make ``name`` the active spend session with cap ``cap_usd`` and return the caps.

    Other sessions' caps are kept. ``total_cap_usd`` None keeps the existing total
    cap. ``model_caps`` (model key to USD) replaces this session's per-model caps;
    None keeps them. ``path`` is the caps file (default: ``spend_caps.json`` next to
    the ledger). The name must be single-line printable text.
    """
    check_label(name, "session name")
    caps_path = Path(path) if path is not None else default_caps_path()
    cap = _usd(cap_usd, "cap_usd")
    total = _usd(total_cap_usd, "total_cap_usd") if total_cap_usd is not None else None
    per_model = ({check_label(str(k), "model key"): _usd(v, f"model cap for {k}") for k, v in model_caps.items()}
                 if model_caps is not None else None)
    with FileLock(caps_path.with_name(caps_path.name + ".lock")):
        caps = read_caps(caps_path)
        caps["active_session"] = name
        caps["session_caps"][name] = cap
        if per_model is not None:
            caps["model_caps"][name] = per_model
        if total is not None:
            caps["total_cap_usd"] = total
        _write_json_atomic(caps_path, caps)
    return caps


# --------------------------------------------------------------------------- ledger
def _fmt(usd: float) -> str:
    return f"${usd:.4f}"


def _reached(spent: float, projected: float, cap: float) -> bool:
    """A paid reservation is refused when it would cross ``cap`` or the cap is already reached."""
    return spent + projected > cap + _EPS or spent >= cap - _EPS


class Ledger:
    """The append-only spend ledger plus the caps file beside it."""

    def __init__(self, path: str | Path | None = None, caps_path: str | Path | None = None, *,
                 lock_timeout_s: float = LOCK_TIMEOUT_S):
        self.path = Path(path) if path is not None else paths.ledger_path()
        self.caps_path = Path(caps_path) if caps_path is not None else default_caps_path(self.path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.lock_timeout_s = float(lock_timeout_s)

    def _lock(self) -> FileLock:
        return FileLock(self.lock_path, timeout_s=self.lock_timeout_s)

    # ------------------------------------------------------------------ caps
    def caps(self) -> dict[str, Any]:
        return read_caps(self.caps_path)

    def set_active_session(self, name: str, cap_usd: float, *, total_cap_usd: float | None = None,
                           model_caps: dict[str, float] | None = None) -> dict[str, Any]:
        return set_active_session(name, cap_usd, total_cap_usd=total_cap_usd, model_caps=model_caps,
                                  path=self.caps_path)

    # ------------------------------------------------------------------ file
    def _scan(self) -> tuple[list[dict[str, Any]], int]:
        """(every parseable record, the number of lines that did not parse)."""
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return [], 0
        out: list[dict[str, Any]] = []
        bad = 0
        for line in raw.split(b"\n"):  # bytes split on \n only: no other character ends a record
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line.decode("utf-8", errors="replace"))
            except (ValueError, RecursionError):
                bad += 1
                continue
            if isinstance(rec, dict):
                out.append(rec)
            else:
                bad += 1
        return out, bad

    def records(self) -> list[dict[str, Any]]:
        """Every parseable line. A truncated or corrupt line (a crash mid-write) is skipped."""
        return self._scan()[0]

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+b") as f:
            f.seek(0, os.SEEK_END)
            prefix = b""
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    prefix = b"\n"  # a crash left a truncated last line: start on a fresh one
                f.seek(0, os.SEEK_END)
            f.write(prefix + line.encode("ascii") + b"\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _index(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        reserves: dict[str, dict[str, Any]] = {}
        settles: dict[str, dict[str, Any]] = {}
        for rec in records:
            rid = rec.get("id")
            if not isinstance(rid, str):
                continue
            if rec.get("type") == "reserve":
                reserves[rid] = rec
            elif rec.get("type") == "settle":
                settles[rid] = rec  # the last settle wins
        return reserves, settles

    @staticmethod
    def _orphans(reserves: dict[str, dict[str, Any]], settles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return [rec for rid, rec in settles.items() if rid not in reserves]

    def _warn_orphans(self, orphans: list[dict[str, Any]], *, always: bool = False) -> None:
        """Warn about settles with no reserve: always from summary(), once per new count from reserve()."""
        key = str(self.path)
        if not orphans or (not always and _ORPHANS_WARNED.get(key) == len(orphans)):
            return
        _ORPHANS_WARNED[key] = len(orphans)
        usd = sum(self._amount(rec, None) for rec in orphans)
        warnings.warn(f"spend ledger {self.path}: {len(orphans)} settle record(s) ({_fmt(usd)}) have no matching "
                      "reserve, so their calls are missing from every cap. The ledger file was damaged or edited.",
                      RuntimeWarning, stacklevel=3)

    @staticmethod
    def _amount(reserve: dict[str, Any], settle: dict[str, Any] | None) -> float:
        value = settle.get("usd") if settle is not None else reserve.get("usd")
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return v if math.isfinite(v) and v > 0 else 0.0

    @classmethod
    def _spent(cls, reserves, settles, session: Any = _ANY, model_key: str | None = None) -> float:
        total = 0.0
        for rid, rec in reserves.items():
            if session is not _ANY and rec.get("session") != session:
                continue
            if model_key is not None and rec.get("model_key") != model_key:
                continue
            total += cls._amount(rec, settles.get(rid))
        return total

    # ------------------------------------------------------------------ API
    def spent(self, session: str | None = None, model_key: str | None = None) -> float:
        """USD spent (settled amounts, unsettled reservations at their projection). None means any."""
        reserves, settles = self._index(self.records())
        return self._spent(reserves, settles, _ANY if session is None else session, model_key)

    def reserve(self, projected: float, model_key: str, request_hash: str,
                tags: dict[str, Any] | None = None, *, paid: bool = True) -> str:
        """Reserve ``projected`` USD for one attempt, or raise SpendCapExceeded. Returns the reservation id.

        ``paid`` (the default) runs every check even at a $0 projection: an active
        session is required and a reached cap refuses the call. The router passes
        ``paid=False`` only for the FakeModel at its zero prices. A positive
        projection is always paid.
        """
        projected = _usd(projected, "projected cost")
        check_label(model_key, "model key")
        tags = dict(tags or {})
        if tags.get("phase") is not None:
            check_label(tags["phase"], "phase", allow_empty=True)
        paid = bool(paid) or projected > 0
        with self._lock():
            caps = self.caps()
            active = caps.get("active_session") or None
            reserves, settles = self._index(self.records())
            self._warn_orphans(self._orphans(reserves, settles))
            if paid:
                spent_all = self._spent(reserves, settles)
                if active is None:
                    raise SpendCapExceeded(
                        f"No active spend session, so the router refuses this paid call to {model_key} "
                        f"(projected {_fmt(projected)}). Call loopkit.router.set_active_session(name, cap_usd) "
                        "at the start of the session.",
                        scope="no_session", cap_usd=0.0, spent_usd=spent_all, projected_usd=projected)
                session_cap = caps["session_caps"].get(active)
                spent_session = self._spent(reserves, settles, active)
                if session_cap is None:
                    raise SpendCapExceeded(
                        f"Session {active} is active but has no cap in {self.caps_path.name}. "
                        "Call set_active_session with a cap.",
                        scope="session", cap_usd=0.0, spent_usd=spent_session, projected_usd=projected)
                session_cap = float(session_cap)
                if _reached(spent_session, projected, session_cap):
                    raise SpendCapExceeded(
                        f"Session {active} cap {_fmt(session_cap)} would be crossed or is reached: spent "
                        f"{_fmt(spent_session)}, this call to {model_key} projects {_fmt(projected)}.",
                        scope="session", cap_usd=session_cap, spent_usd=spent_session, projected_usd=projected)
                model_cap = (caps["model_caps"].get(active) or {}).get(model_key)
                if model_cap is not None:
                    model_cap = float(model_cap)
                    spent_model = self._spent(reserves, settles, active, model_key)
                    if _reached(spent_model, projected, model_cap):
                        raise SpendCapExceeded(
                            f"Model {model_key} cap {_fmt(model_cap)} in session {active} would be crossed or is "
                            f"reached: spent {_fmt(spent_model)}, this call projects {_fmt(projected)}.",
                            scope="model", cap_usd=model_cap, spent_usd=spent_model, projected_usd=projected)
                total_cap = caps.get("total_cap_usd")
                if total_cap is not None:
                    total_cap = float(total_cap)
                    if _reached(spent_all, projected, total_cap):
                        raise SpendCapExceeded(
                            f"Total cap {_fmt(total_cap)} would be crossed or is reached: spent {_fmt(spent_all)} "
                            f"over all sessions, this call to {model_key} projects {_fmt(projected)}.",
                            scope="total", cap_usd=total_cap, spent_usd=spent_all, projected_usd=projected)
            rid = uuid.uuid4().hex
            self._append({"type": "reserve", "id": rid, "ts": time.time(), "session": active,
                          "model_key": model_key, "usd": projected, "request_hash": request_hash,
                          "tags": tags})
        return rid

    def settle(self, rid: str, usd: float, usd_source: str, note: str | None = None) -> None:
        """Record a reservation's actual cost (``usd_source`` is ``provider`` or ``computed``)."""
        usd = _usd(usd, "settled cost")
        if usd_source not in ("provider", "computed"):
            raise ValueError("usd_source must be 'provider' or 'computed'")
        rec: dict[str, Any] = {"type": "settle", "id": rid, "ts": time.time(), "usd": usd, "usd_source": usd_source}
        if note:
            rec["note"] = note
        with self._lock():
            self._append(rec)

    def summary(self) -> dict[str, Any]:
        """Spend per session, per model and per session and model, plus unsettled reservations.

        ``orphan_settles`` counts settle records with no matching reserve (a damaged
        or edited ledger); when there are any, a RuntimeWarning says so.
        ``corrupt_lines`` counts lines that did not parse.
        """
        records, corrupt = self._scan()
        reserves, settles = self._index(records)
        orphans = self._orphans(reserves, settles)
        self._warn_orphans(orphans, always=True)
        by_session: dict[str, float] = {}
        by_model: dict[str, float] = {}
        by_session_model: dict[str, dict[str, float]] = {}
        unsettled_n = 0
        unsettled_usd = 0.0
        total = 0.0
        calls: set[str] = set()
        for rid, rec in reserves.items():
            tags = rec.get("tags") if isinstance(rec.get("tags"), dict) else {}
            calls.add(str(tags.get("call") or rid))  # one call may hold several attempt reservations
            amount = self._amount(rec, settles.get(rid))
            session = str(rec.get("session")) if rec.get("session") is not None else "(none)"
            model = str(rec.get("model_key"))
            total += amount
            by_session[session] = by_session.get(session, 0.0) + amount
            by_model[model] = by_model.get(model, 0.0) + amount
            by_session_model.setdefault(session, {})
            by_session_model[session][model] = by_session_model[session].get(model, 0.0) + amount
            if rid not in settles:
                unsettled_n += 1
                unsettled_usd += amount

        def r(d: dict[str, float]) -> dict[str, float]:
            return {k: round(v, 8) for k, v in sorted(d.items())}

        caps = self.caps()
        return {
            "ledger": str(self.path),
            "total_usd": round(total, 8),
            "calls": len(calls),
            "reservations": len(reserves),
            "by_session": r(by_session),
            "by_model": r(by_model),
            "by_session_model": {s: r(m) for s, m in sorted(by_session_model.items())},
            "unsettled": {"count": unsettled_n, "usd": round(unsettled_usd, 8)},
            "orphan_settles": {"count": len(orphans),
                               "usd": round(sum(self._amount(rec, None) for rec in orphans), 8)},
            "corrupt_lines": corrupt,
            "active_session": caps.get("active_session"),
            "caps": caps,
        }
