"""Reading traces back: the events of one run, the list of runs, and one table.

``load_runs`` is what reports build from: one row per event, the run's
``meta.json`` as ``meta_<key>`` columns, and the payload flattened into plain
columns. Columns declared by the payload schemas get their dtype from the schema
(``usd`` float64, token counts Int64, ``ok`` and ``passed`` boolean, text str),
so a column keeps its type even when no run has an event of that kind, and a
declared numeric column stays numeric even when another kind's extra field puts
text under the same name.
"""

from __future__ import annotations

import json
import math
import numbers
import types
import typing
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import paths
from ..schemas.trace import EVENT_KINDS, PAYLOAD_MODELS, TraceEvent
from ..schemas.verdict import Verdict
from .writer import EVENTS_FILE, VERDICT_EXTRA_FIELDS, check_run_id, iter_lines, parse_event_line, read_meta

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["BASE_COLUMNS", "PAYLOAD_JSON", "EventList", "events_path", "list_runs", "load_runs", "read_events"]

BASE_COLUMNS = ("run_id", "seq", "ts", "kind")
PAYLOAD_JSON = "payload_json"
_BASE_DTYPES = {"run_id": "str", "seq": "int64", "ts": "float64", "kind": "str"}
_ACCEPTS = {"boolean": {bool}, "Int64": {int}, "float64": {int, float}, "str": {str}}
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


class EventList(list):
    """The events of one run, plus what reading skipped.

    ``malformed`` counts lines that were not valid TraceEvents (a crash's partial
    last line, for example), ``malformed_lines`` holds their 1-based line numbers
    and ``path`` is the events file that was read.
    """

    def __init__(self, events: Iterable[TraceEvent] = (), malformed_lines: Iterable[int] = (),
                 path: Path | None = None):
        super().__init__(events)
        self.malformed_lines = list(malformed_lines)
        self.path = path

    @property
    def malformed(self) -> int:
        return len(self.malformed_lines)


def _is_run_id(value: str) -> bool:
    try:
        check_run_id(value)  # rejects a .jsonl suffix, so such a string is always a path
    except ValueError:
        return False
    return True


def _runs_root(root: str | Path | None) -> Path:
    return Path(root) if root is not None else paths.runs_dir(create=False)


def events_path(run: str | Path, root: str | Path | None = None) -> Path:
    """The events file for a run id (under ``root``, else ``paths.runs_dir()``), a run directory or an events file.

    A ``str`` that passes ``check_run_id`` is a run id. Anything else, and any
    ``Path``, is a path.
    """
    if isinstance(run, str) and _is_run_id(run):
        target = Path(root) / run if root is not None else paths.run_dir(run, create=False)
    else:
        target = Path(run)
    if target.is_dir():
        return target / EVENTS_FILE
    if target.is_file() or target.suffix.lower() == ".jsonl":
        return target
    return target / EVENTS_FILE


def read_events(run: str | Path, root: str | Path | None = None) -> EventList:
    """Every valid TraceEvent of one run, in file order.

    Lines that are not valid events are skipped and counted in ``.malformed``: bad
    JSON or UTF-8, a payload that fails its kind's model, a seq beyond int64. A
    line written with a lone surrogate escape (older writers did) is recovered,
    the surrogate read as U+FFFD. A run directory without an events file gives an
    empty list; a run that does not exist raises FileNotFoundError.
    """
    path = events_path(run, root)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        if path.name == EVENTS_FILE and path.parent.is_dir():
            return EventList(path=path)
        raise FileNotFoundError(f"no trace events for {str(run)!r}: {path} does not exist") from None
    events: list[TraceEvent] = []
    bad: list[int] = []
    for number, line in iter_lines(data):
        try:
            events.append(parse_event_line(line))
        except Exception:  # any failure on one line, RecursionError included, costs that line only
            bad.append(number)
    return EventList(events, bad, path)


def list_runs(root: str | Path | None = None) -> list[str]:
    """Sorted ids of the runs under ``root`` (default ``paths.runs_dir()``) that have an events file."""
    base = _runs_root(root)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / EVENTS_FILE).is_file())


def load_runs(root: str | Path | None = None, run_ids: Iterable[str] | None = None,
              kinds: Iterable[str] | None = None) -> "pd.DataFrame":
    """One row per event across runs, sorted by run_id then seq.

    Columns: ``run_id, seq, ts, kind``, then ``meta_<key>`` for each key in the
    runs' meta.json, then the payload's top-level keys as plain columns (dict and
    list values as JSON text), then ``payload_json`` with the whole payload. Every
    field the payload schemas declare for the selected kinds is present, typed,
    even with no rows. Without ``run_ids``, each run ``list_runs`` finds is read by
    its path under the root, never by resolving its name again.

    A payload key that would shadow a base or meta column, or ``payload_json``,
    gets a ``payload_`` prefix (``seq`` becomes ``payload_seq``), and a key that
    equals one of those prefixed names gets a numeric suffix (``payload_seq_2``).
    A declared numeric column stays numeric: a value that does not fit its dtype
    (text such as "2003" or "n/a", a bool, a fraction in an integer column) goes
    through ``pd.to_numeric(errors="coerce")`` and becomes NA when it does not
    parse. The raw value stays in ``payload_json``.

    ``df.attrs``:

    - ``malformed_lines``: each run read -> the number of lines it skipped.
    - ``renamed_payload_keys``: payload key -> its column, for every key stored
      under another name.
    - ``coerced_rows``: declared numeric column -> index labels of the rows whose
      value was coerced.
    """
    import pandas as pd

    scope = _scope(kinds)
    allowed = set(scope)
    if run_ids is None:
        base = _runs_root(root)
        sources: list[tuple[str, str | Path]] = [(name, base / name) for name in list_runs(root)]
    else:
        wanted = [run_ids] if isinstance(run_ids, (str, Path)) else list(run_ids)
        sources = [(str(run), run) for run in dict.fromkeys(wanted)]
    declared = dict(_declared_columns(scope))

    runs: list[tuple[dict[str, Any], list[TraceEvent]]] = []
    meta_cols: dict[str, None] = {}
    payload_keys: dict[str, None] = {}
    malformed: dict[str, int] = {}
    for name, source in sources:
        events = read_events(source, root=root)
        malformed[name] = events.malformed
        meta_row: dict[str, Any] = {}
        for key, value in read_meta(events.path.parent).items():
            col = f"meta_{key}"
            meta_cols.setdefault(col)
            meta_row[col] = _cell(value)
        kept = [event for event in events if event.kind in allowed]
        for event in kept:
            for key in event.payload:
                payload_keys.setdefault(key)
        runs.append((meta_row, kept))

    columns, renamed = _payload_columns(payload_keys)
    rows: list[dict[str, Any]] = []
    extra_cols: dict[str, None] = {}
    for meta_row, kept in runs:
        for event in kept:
            row = {"run_id": event.run_id, "seq": event.seq, "ts": event.ts, "kind": event.kind, **meta_row}
            for key, value in event.payload.items():
                col = columns[key]
                if col not in declared:
                    extra_cols.setdefault(col)
                row[col] = _cell(value)
            row[PAYLOAD_JSON] = _json_text(event.payload)
            rows.append(row)
    rows.sort(key=lambda r: (r["run_id"], r["seq"]))

    data: dict[str, pd.Series] = {}
    for col in BASE_COLUMNS:
        data[col] = pd.Series([r[col] for r in rows], dtype=_BASE_DTYPES[col])
    for col in meta_cols:
        data[col] = _column([r.get(col) for r in rows], None)
    coerced: dict[str, list[int]] = {}
    for col, dtype in declared.items():
        values = [r.get(col) for r in rows]
        if dtype in ("Int64", "float64"):
            data[col], changed = _numeric_column(values, dtype)
            if changed:
                coerced[col] = changed
        else:
            data[col] = _column(values, dtype)
    for col in extra_cols:
        data[col] = _column([r.get(col) for r in rows], None)
    data[PAYLOAD_JSON] = pd.Series([r[PAYLOAD_JSON] for r in rows], dtype="str")
    frame = pd.DataFrame(data)
    frame.attrs["malformed_lines"] = malformed
    frame.attrs["renamed_payload_keys"] = renamed
    frame.attrs["coerced_rows"] = coerced
    return frame


# ------------------------------------------------------------------- table helpers
def _scope(kinds: Iterable[str] | str | None) -> tuple[str, ...]:
    if kinds is None:
        return tuple(EVENT_KINDS)
    wanted = [kinds] if isinstance(kinds, str) else list(kinds)
    unknown = [k for k in wanted if not isinstance(k, str) or k not in PAYLOAD_MODELS]
    if unknown:
        raise ValueError(
            f"unknown trace event kind(s) {', '.join(map(repr, unknown))}; expected any of {', '.join(EVENT_KINDS)}"
        )
    return tuple(k for k in EVENT_KINDS if k in wanted)


def _dtype_for(annotation: Any) -> str | None:
    """The column dtype for a payload field's annotation (None: infer from the values)."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        return _dtype_for(rest[0]) if len(rest) == 1 else None
    if origin is typing.Literal:
        if all(isinstance(a, bool) for a in args):
            return "boolean"
        if all(isinstance(a, str) for a in args):
            return "str"
        if all(isinstance(a, int) and not isinstance(a, bool) for a in args):
            return "Int64"
        return None
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "Int64"
    if annotation is float:
        return "float64"
    if annotation is str:
        return "str"
    containers = (dict, list, tuple, set, frozenset)
    if origin in containers or annotation in containers:
        return "str"  # stored as JSON text
    return None


@lru_cache(maxsize=None)
def _declared_columns(kinds: tuple[str, ...]) -> dict[str, str | None]:
    cols: dict[str, str | None] = {}
    for kind in kinds:
        for name, field in PAYLOAD_MODELS[kind].model_fields.items():
            cols.setdefault(name, _dtype_for(field.annotation))
        if kind == "verdict":
            for name in VERDICT_EXTRA_FIELDS:
                cols.setdefault(name, _dtype_for(Verdict.model_fields[name].annotation))
    return cols


def _shadows(key: str) -> bool:
    """True for a payload key that would shadow a base column, a meta column or payload_json."""
    return key in BASE_COLUMNS or key == PAYLOAD_JSON or key.startswith("meta_")


def _payload_columns(keys: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    """The column for each payload key, and the keys stored under another name.

    A shadowing key gets the ``payload_`` prefix first, since that name is the
    documented one. Every other key keeps its own name when it is free. A key whose
    name is already taken (``payload_seq`` next to ``seq``) gets the first free
    ``_2``, ``_3``, ... suffix, so no two keys ever share a column.
    """
    keys = list(keys)
    columns = {key: "payload_" + key for key in keys if _shadows(key)}
    taken = set(BASE_COLUMNS) | {PAYLOAD_JSON} | set(columns.values())
    for key in keys:
        if key not in columns and key not in taken:
            columns[key] = key
            taken.add(key)
    for key in keys:
        if key not in columns:
            n = 2
            while f"{key}_{n}" in taken:
                n += 1
            columns[key] = f"{key}_{n}"
            taken.add(columns[key])
    renamed = {key: col for key, col in columns.items() if col != key}
    return columns, renamed


def _json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _cell(value: Any) -> Any:
    return _json_text(value) if isinstance(value, (dict, list)) else value


def _value_types(values: list[Any]) -> set[type]:
    seen: set[type] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            seen.add(bool)
        elif isinstance(value, int):
            seen.add(int)
        elif isinstance(value, float):
            seen.add(float)
        elif isinstance(value, str):
            seen.add(str)
        else:
            seen.add(object)
    return seen


def _inferred_dtype(seen: set[type]) -> Any:
    if not seen or object in seen:
        return object
    if seen == {bool}:
        return "boolean"
    if seen == {int}:
        return "Int64"
    if seen <= {int, float}:
        return "float64"
    if seen == {str}:
        return "str"
    return object


def _column(values: list[Any], declared: str | None) -> "pd.Series":
    """A typed column: the declared dtype when every value fits it, else one inferred from the values."""
    import pandas as pd

    seen = _value_types(values)
    dtype = declared if declared is not None and seen <= _ACCEPTS[declared] else _inferred_dtype(seen)
    try:
        return pd.Series(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError):
        return pd.Series(values, dtype=object)


def _fits(value: Any, dtype: str) -> bool:
    if type(value) is int:
        return _INT64_MIN <= value <= _INT64_MAX if dtype == "Int64" else value.bit_length() <= 1023
    return dtype == "float64" and type(value) is float


def _numeric_column(values: list[Any], dtype: str) -> tuple["pd.Series", list[int]]:
    """A declared numeric column (``Int64`` or ``float64``) that stays that dtype.

    Each value that does not fit goes through ``pd.to_numeric(errors="coerce")``;
    what still does not fit becomes NA. Returns the column and the positions of
    the values that were coerced.
    """
    import pandas as pd

    out = list(values)
    changed = [i for i, value in enumerate(values) if value is not None and not _fits(value, dtype)]
    for i in changed:
        out[i] = _coerce_number(values[i], dtype)
    return pd.Series(out, dtype=dtype), changed


def _coerce_number(value: Any, dtype: str) -> int | float | None:
    """``value`` through pd.to_numeric(errors="coerce"), as an int64 or float64 value, or None."""
    import pandas as pd

    if isinstance(value, bool):
        return None  # a flag is not a count or an amount
    try:
        num = pd.to_numeric(value, errors="coerce")
    except Exception:
        return None
    if isinstance(num, bool) or not isinstance(num, numbers.Real):
        return None
    if dtype == "Int64":
        if isinstance(num, numbers.Integral):
            whole = int(num)
        else:
            number = float(num)
            if not math.isfinite(number) or not number.is_integer():
                return None
            whole = int(number)
        return whole if _INT64_MIN <= whole <= _INT64_MAX else None
    try:
        number = float(num)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None
