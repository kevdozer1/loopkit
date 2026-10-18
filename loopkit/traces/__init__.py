"""Traces: append-only JSONL events per run, read back as events or as one pandas table.

Each run lives in ``LOOPKIT_DATA/runs/<run_id>/`` (or ``root/<run_id>/``) as
``events.jsonl``, ``meta.json`` and ``images/``. Every number in a report comes
from these events::

    from loopkit.traces import TraceWriter, load_runs

    with TraceWriter("t2_m1_r0", meta={"task_id": "T2", "model_key": "m1", "repeat": 0}) as trace:
        trace.node_start("locate", "find_rack", params={"camera": "overhead"})
        trace.node_end("locate", "find_rack", ok=True, outputs={"yaw_deg": 2.6}, sim_seconds=0.8)
        trace.verdict(verdict)  # a loopkit.schemas.Verdict, flattened into the payload

    calls = load_runs(kinds=["model_call"])  # provider, model, usd, input_tokens, ... as columns

A run has one writer at a time: a second TraceWriter on an open run, in any
process, raises RunLockedError. Do not log from a signal handler; set a flag
there and log from the main loop.
"""

from .jsonsafe import dumps, to_jsonable
from .reader import BASE_COLUMNS, PAYLOAD_JSON, EventList, events_path, list_runs, load_runs, read_events
from .writer import (
    EVENTS_FILE, IMAGES_DIR, LOCK_FILE, META_FILE, RunLockedError, TraceWriter, check_run_id, read_meta,
    safe_image_name,
)

__all__ = [
    "BASE_COLUMNS", "EVENTS_FILE", "EventList", "IMAGES_DIR", "LOCK_FILE", "META_FILE", "PAYLOAD_JSON",
    "RunLockedError", "TraceWriter", "check_run_id", "dumps", "events_path", "list_runs", "load_runs",
    "read_events", "read_meta", "safe_image_name", "to_jsonable",
]
