from __future__ import annotations

import dataclasses
import gc
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from loopkit.paths import REPO_ROOT
from loopkit.schemas import EVENT_KINDS, Evidence, Verdict
from loopkit.traces import (
    LOCK_FILE, RunLockedError, TraceWriter, check_run_id, list_runs, load_runs, read_events, read_meta, to_jsonable,
)
from loopkit.traces import writer as writer_module


def _call(**over):
    """A valid model_call payload; keyword arguments replace or add fields."""
    payload = dict(
        provider="fake", model="fake-writer-1", role="writer", input_tokens=1200, output_tokens=300,
        reasoning_tokens=40, usd=0.0125, usd_source="computed", latency_ms=812.5, cache_hit=False,
        request_hash="req-0001",
    )
    payload.update(over)
    return payload


def _pixels(h=8, w=10):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[2:5, 3:7] = (255, 40, 0)
    img[:, -1] = 200
    return img


# --------------------------------------------------------------------------- writing
def test_every_kind_round_trips(tmp_path):
    with TraceWriter("all_kinds", root=tmp_path) as w:
        written = [
            w.model_call(**_call(model_key="m1", phase="setup")),
            w.node_start("locate", "find_rack", params={"camera": "overhead"}, attempt=1),
            w.node_end("locate", "find_rack", False, error=RuntimeError("rack not found"), sim_seconds=1.5),
            w.verdict(Verdict(passed=False, score=0.0, failure_class="perception", failing_node="locate")),
            w.patch(1, "aaaa", "bbbb", "locate failed on 3 of 10 seeds"),
            w.intervention("paused for a look", actor="operator"),
            w.budget_stop("model", cap_usd=2.0, spent_usd=2.01, reason="projected cost over cap"),
            w.emit("intervention", {"reason": "explicit ts"}, ts=1_700_000_000.5),
        ]
    events = read_events("all_kinds", root=tmp_path)
    assert events.malformed == 0
    assert [e.kind for e in events] == [*EVENT_KINDS, "intervention"]
    assert [e.model_dump() for e in events] == [e.model_dump() for e in written]
    assert {e.run_id for e in events} == {"all_kinds"}
    assert events[0].payload["model_key"] == "m1"  # extra payload fields are kept
    assert events[2].payload["error"] == "RuntimeError: rack not found"
    assert events[2].payload["outputs"] == {}
    assert events[-1].ts == 1_700_000_000.5


def test_seq_increments_and_lines_are_compact_json(tmp_path):
    with TraceWriter("seq", root=tmp_path) as w:
        assert w.next_seq == 0 and not w.resumed
        seqs = [w.intervention(f"step {i}").seq for i in range(5)]
        assert w.next_seq == 5
    assert seqs == [0, 1, 2, 3, 4]
    raw = w.path.read_bytes()
    lines = raw.split(b"\n")
    assert lines[-1] == b"" and len(lines) == 6  # one line per event, newline terminated
    for i, line in enumerate(lines[:-1]):
        obj = json.loads(line)
        assert obj["seq"] == i and obj["run_id"] == "seq" and obj["kind"] == "intervention"
        assert line.decode("ascii") == json.dumps(obj, separators=(",", ":"))
    stamps = [e.ts for e in read_events(w.path)]
    assert stamps == sorted(stamps)


def test_resume_after_reopen_continues_seq(tmp_path):
    with TraceWriter("resume", root=tmp_path) as w:
        assert [w.intervention("a").seq, w.intervention("b").seq] == [0, 1]
    with TraceWriter("resume", root=tmp_path) as w:
        assert w.resumed and w.next_seq == 2 and w.malformed_on_open == 0
        assert w.intervention("c").seq == 2
    events = read_events("resume", root=tmp_path)
    assert [(e.seq, e.payload["reason"]) for e in events] == [(0, "a"), (1, "b"), (2, "c")]


def test_truncated_last_line_is_skipped_and_next_append_is_clean(tmp_path):
    with TraceWriter("crash", root=tmp_path) as w:
        for i in range(3):
            w.intervention(f"before crash {i}")
    good = w.path.read_bytes()
    partial = b'{"run_id":"crash","ts":1.0,"seq":3,"kind":"interv'
    with open(w.path, "ab") as fh:  # a crash mid-write leaves a partial line with no newline
        fh.write(partial)

    events = read_events("crash", root=tmp_path)
    assert [e.seq for e in events] == [0, 1, 2]
    assert events.malformed == 1 and events.malformed_lines == [4]

    with TraceWriter("crash", root=tmp_path) as w:
        assert w.next_seq == 3 and w.malformed_on_open == 1
        assert w.path.read_bytes() == good + partial + b"\n"  # opening only ends the partial line
        assert w.intervention("after crash").seq == 3
        w.intervention("and one more")
    data = w.path.read_bytes()
    assert data.startswith(good)  # nothing written before the crash was rewritten
    assert data.endswith(b"\n")
    events = read_events("crash", root=tmp_path)
    assert [e.seq for e in events] == [0, 1, 2, 3, 4]
    assert events[3].payload["reason"] == "after crash"
    assert events.malformed == 1  # the partial line is now a line of its own, still skipped
    assert load_runs(tmp_path).attrs["malformed_lines"] == {"crash": 1}


def test_complete_last_line_without_newline_is_kept(tmp_path):
    with TraceWriter("no_newline", root=tmp_path) as w:
        w.intervention("a")
        w.intervention("b")
    w.path.write_bytes(w.path.read_bytes().rstrip(b"\n"))
    with TraceWriter("no_newline", root=tmp_path) as w:
        assert w.next_seq == 2
        w.intervention("c")
    events = read_events("no_newline", root=tmp_path)
    assert [e.payload["reason"] for e in events] == ["a", "b", "c"] and events.malformed == 0


def test_failed_write_leaves_a_seq_gap_and_next_event_starts_a_fresh_line(tmp_path, monkeypatch):
    real_write = writer_module._write_all
    calls = {"n": 0}

    def disk_full_on_second_write(fh, data):
        calls["n"] += 1
        if calls["n"] == 2:
            fh.write(data[: len(data) // 2])  # half a line reaches the file, then the disk fills
            raise OSError("disk full")
        real_write(fh, data)

    monkeypatch.setattr(writer_module, "_write_all", disk_full_on_second_write)
    with TraceWriter("disk_full", root=tmp_path) as w:
        w.intervention("one")
        with pytest.raises(OSError, match="disk full"):
            w.intervention("two")
        assert w.next_seq == 2  # seq 1 was reserved before the write: a gap, never a duplicate
        w.intervention("three")
    events = read_events("disk_full", root=tmp_path)
    assert [(e.seq, e.payload["reason"]) for e in events] == [(0, "one"), (2, "three")]
    assert events.malformed == 1  # the half line


def test_non_ascii_text_round_trips_on_ascii_lines(tmp_path):
    odd = [chr(c) for c in (0xEF, 0x2013, 0xB0, 0x2028, 0x85, 0x4E2D)]  # diaeresis, en dash, degree, U+2028, NEL, CJK
    text = "yaw +2.6 " + " | ".join(odd) + " end"
    with TraceWriter("text", root=tmp_path) as w:
        w.intervention(text)
    raw = w.path.read_bytes()
    assert raw.isascii() and raw.count(b"\n") == 1
    assert read_events("text", root=tmp_path)[0].payload["reason"] == text
    assert load_runs(tmp_path)["reason"].iloc[0] == text


def test_meta_json_merges_and_new_keys_win(tmp_path):
    with TraceWriter("meta", meta={"task_id": "T2", "seed": 1000, "graph_hash": None}, root=tmp_path) as w:
        assert w.meta == {"task_id": "T2", "seed": 1000, "graph_hash": None}
    with TraceWriter("meta", meta={"seed": np.int64(1001), "unit": ("m1", "T2", 0)}, root=tmp_path) as w:
        assert w.meta == {"task_id": "T2", "seed": 1001, "graph_hash": None, "unit": ["m1", "T2", 0]}
        merged = w.update_meta({"graph_hash": "abc123"})
        assert merged["graph_hash"] == "abc123"
    on_disk = json.loads((tmp_path / "meta" / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == {"task_id": "T2", "seed": 1001, "graph_hash": "abc123", "unit": ["m1", "T2", 0]}
    # no temp files; the lock file stays (deleting it would let two writers lock two different files)
    assert sorted(p.name for p in (tmp_path / "meta").iterdir()) == [LOCK_FILE, "events.jsonl", "meta.json"]
    with TraceWriter("meta", root=tmp_path) as w:  # no meta argument: the file is left as it is
        assert w.meta == on_disk
    assert json.loads((tmp_path / "meta" / "meta.json").read_text(encoding="utf-8")) == on_disk


def test_save_image_writes_sanitized_never_clobbering_pngs(tmp_path):
    img = _pixels()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    png = buf.getvalue()
    with TraceWriter("img", root=tmp_path) as w:
        (w.images_dir).mkdir()
        (w.images_dir / "pre.png").write_bytes(b"someone else's file")
        first = w.save_image("front cam/seed:3", img)
        first_bytes = Path(first).read_bytes()
        second = w.save_image("front cam/seed:3", img[::-1])
        third = w.save_image("front cam/seed:3", img)
        from_pil = w.save_image("wrist.PNG", Image.fromarray(img))
        from_bytes = w.save_image("raw bytes", png)
        escape = w.save_image("../../outside", img)
        reserved = w.save_image("aux", img)
        unnamed = w.save_image("", img)
        pre = w.save_image("pre", img)
        with pytest.raises(ValueError, match="uint8"):
            w.save_image("float", img.astype(np.float32))
        with pytest.raises(ValueError):
            w.save_image("text", "not an image")

    names = [Path(p).name for p in (first, second, third, from_pil, from_bytes, escape, reserved, unnamed, pre)]
    assert names == [
        "front_cam_seed_3.png", "front_cam_seed_3_1.png", "front_cam_seed_3_2.png", "wrist.png", "raw_bytes.png",
        "_.._outside.png", "_aux.png", "image.png", "pre_1.png",
    ]
    for p in (first, second, third, from_pil, from_bytes, escape, reserved, unnamed, pre):
        assert Path(p).is_absolute() and Path(p).parent == w.images_dir
    assert Path(first).read_bytes() == first_bytes  # not overwritten by the later saves
    assert (w.images_dir / "pre.png").read_bytes() == b"someone else's file"
    assert Path(from_bytes).read_bytes() == png
    np.testing.assert_array_equal(np.asarray(Image.open(first)), img)
    np.testing.assert_array_equal(np.asarray(Image.open(second)), img[::-1])
    np.testing.assert_array_equal(np.asarray(Image.open(from_pil)), img)
    with Image.open(first) as check:
        assert check.format == "PNG" and check.size == (10, 8) and check.mode == "RGB"


def test_numpy_and_other_python_values_serialize(tmp_path):
    @dataclasses.dataclass
    class Grip:
        width_m: float
        closed: bool

    outputs = {
        "count": np.int64(4),
        "score": np.float32(0.5),
        "flag": np.bool_(True),
        "pos": np.array([0.1, 0.2, 0.3]),
        "grid": np.arange(6, dtype=np.uint8).reshape(2, 3),
        "gaps": np.array([1.0, np.nan]),
        "missing": float("nan"),
        "far": np.float64("inf"),
        "image": Path("images") / "a.png",
        "pair": (1, 2),
        "tags": {"b", "a"},
        "grip": Grip(np.float64(0.04), np.bool_(False)),
        "verdict": Verdict(passed=True, score=1.0),
        "day": np.datetime64("2026-10-17"),
        7: "int key",
    }
    expected = {
        "count": 4, "score": 0.5, "flag": True, "pos": [0.1, 0.2, 0.3], "grid": [[0, 1, 2], [3, 4, 5]],
        "gaps": [1.0, None], "missing": None, "far": None, "image": str(Path("images") / "a.png"),
        "pair": [1, 2], "tags": ["a", "b"], "grip": {"width_m": 0.04, "closed": False},
        "verdict": Verdict(passed=True, score=1.0).model_dump(mode="json"), "day": "2026-10-17", "7": "int key",
    }
    with TraceWriter("np", root=tmp_path) as w:
        event = w.node_end("locate", np.str_("find_rack"), np.bool_(True), outputs=outputs,
                           sim_seconds=np.float32(1.25), attempt_index=np.int32(2))
    assert event.payload["outputs"] == expected
    assert event.payload["ok"] is True and event.payload["sim_seconds"] == 1.25
    assert event.payload["attempt_index"] == 2 and type(event.payload["attempt_index"]) is int
    line = w.path.read_text(encoding="utf-8").splitlines()[0]
    assert "NaN" not in line and "Infinity" not in line  # strict JSON
    assert json.loads(line)["payload"]["outputs"] == expected
    assert read_events("np", root=tmp_path)[0].payload == event.payload


def test_invalid_payloads_raise_value_error_and_write_nothing(tmp_path):
    with TraceWriter("bad", root=tmp_path) as w:
        no_usd = _call()
        del no_usd["usd"]
        with pytest.raises(ValueError, match="usd") as err:
            w.model_call(**no_usd)
        message = str(err.value)
        assert "model_call" in message and "Field required" in message
        assert "errors.pydantic.dev" not in message
        with pytest.raises(ValueError, match="usd"):
            w.model_call(**_call(usd=-0.5))
        with pytest.raises(ValueError, match="role"):
            w.model_call(**_call(role="boss"))
        with pytest.raises(ValueError, match="unknown trace event kind 'model_calls'"):
            w.emit("model_calls", _call())
        with pytest.raises(ValueError, match="score"):
            w.verdict({"passed": True, "score": 1.5})
        with pytest.raises(ValueError, match="sim_seconds"):
            w.node_end("n", "s", True, sim_seconds=float("nan"))
        with pytest.raises(ValueError, match="dict"):
            w.emit("intervention", ["not", "a", "dict"])
        with pytest.raises(ValueError, match="ts"):
            w.emit("intervention", {"reason": "x"}, ts=float("nan"))
        loop: dict = {}
        loop["self"] = loop
        with pytest.raises(ValueError, match="nested"):
            w.emit("intervention", {"reason": "x", "loop": loop})
        with pytest.raises(ValueError):
            w.verdict("passed")
        assert w.next_seq == 0
        assert w.path.read_bytes() == b""
        assert w.model_call(**_call()).seq == 0
    assert len(read_events("bad", root=tmp_path)) == 1


def test_verdict_helper_flattens_verdicts_and_dicts(tmp_path):
    v = Verdict(
        passed=False, score=0.25, evidence=Evidence(metrics={"depth_mm": 3.5}, image_paths=["a.png"]),
        failure_class="alignment", failing_node="insert", task_id="T2", seed=1003, checker_id="t2_seated",
        summary="plug 2 mm short",
    )
    with TraceWriter("verdicts", root=tmp_path) as w:
        from_model = w.verdict(v, repeat=1)
        nested = w.verdict({"passed": True, "score": 1.0, "evidence": {"metrics": {"depth_mm": 9.0}}}, seed=1004)
        flat = w.verdict({"passed": True, "score": 1.0, "metrics": {"x": 1}}, summary="flat dict")
        override = w.verdict(v, seed=1005, metrics={"override": True})
    assert from_model.payload == {
        "passed": False, "score": 0.25, "failure_class": "alignment", "failing_node": "insert",
        "metrics": {"depth_mm": 3.5}, "image_paths": ["a.png"], "task_id": "T2", "seed": 1003,
        "checker_id": "t2_seated", "summary": "plug 2 mm short", "repeat": 1,
    }
    assert "evidence" not in nested.payload
    assert nested.payload["metrics"] == {"depth_mm": 9.0} and nested.payload["image_paths"] == []
    assert nested.payload["seed"] == 1004 and nested.payload["task_id"] is None and nested.payload["summary"] == ""
    assert flat.payload["metrics"] == {"x": 1} and flat.payload["summary"] == "flat dict"
    assert override.payload["seed"] == 1005 and override.payload["metrics"] == {"override": True}


def test_writer_is_thread_safe(tmp_path):
    n_threads, per_thread = 8, 50
    with TraceWriter("threads", root=tmp_path) as w:
        def work(i):
            for j in range(per_thread):
                w.intervention(f"thread {i} event {j}", actor="bot", thread=i)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert w.next_seq == n_threads * per_thread
    events = read_events("threads", root=tmp_path)
    assert events.malformed == 0
    assert [e.seq for e in events] == list(range(n_threads * per_thread))  # file order is seq order
    assert len({e.payload["reason"] for e in events}) == n_threads * per_thread


def test_closed_writer_refuses_events(tmp_path):
    w = TraceWriter("closed", root=tmp_path)
    w.intervention("one")
    w.close()
    w.close()
    assert w.closed
    with pytest.raises(ValueError, match="closed"):
        w.intervention("two")
    with pytest.raises(ValueError, match="closed"):  # the run lock went with close(): no meta writes either
        w.update_meta({"late": True})
    assert len(read_events("closed", root=tmp_path)) == 1


DEVICE_NAMES = ["nul", "NUL", "Nul.txt", "con", "aux.log", "prn", "COM1", "com9.tar.gz", "LPT1", "lpt9", "conin$",
                "CONOUT$", "nul .txt", "com\N{SUPERSCRIPT ONE}"]


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "x:y", ".", "..", " padded", "dot.", "t2_dump.jsonl", "X.JSONL",
                                 "cut\ud83d", *DEVICE_NAMES])
def test_bad_run_ids_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        TraceWriter(bad, root=tmp_path)
    assert list(tmp_path.iterdir()) == []  # rejected before anything is created


@pytest.mark.parametrize("good", ["console", "nul_run", "com10", "lpt", "auxiliary", "con-1", "run.json", "t2_m1_r0"])
def test_names_close_to_device_names_are_fine(tmp_path, good):
    assert check_run_id(good) == good
    with TraceWriter(good, root=tmp_path) as w:
        w.intervention("ok")
    assert list_runs(tmp_path) == [good] and len(read_events(good, root=tmp_path)) == 1


def test_default_root_is_the_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOPKIT_DATA", str(tmp_path / "data"))
    with TraceWriter("unit_1", meta={"repeat": 2}) as w:
        w.intervention("default root")
    assert w.dir == tmp_path / "data" / "runs" / "unit_1"
    assert (w.dir / "events.jsonl").is_file() and (w.dir / "meta.json").is_file()
    assert list_runs() == ["unit_1"]
    assert [e.payload["reason"] for e in read_events("unit_1")] == ["default root"]
    assert load_runs()["meta_repeat"].tolist() == [2]


# --------------------------------------------------------------------------- reading
def test_read_events_accepts_ids_dirs_and_files(tmp_path):
    with TraceWriter("r1", root=tmp_path) as w:
        w.intervention("x")
    for source in (read_events("r1", root=tmp_path), read_events(w.dir), read_events(w.path),
                   read_events(str(w.path)), read_events(str(w.dir))):
        assert [e.payload["reason"] for e in source] == ["x"]
        assert source.path == w.path
    (tmp_path / "no_events").mkdir()
    assert read_events("no_events", root=tmp_path) == []
    with pytest.raises(FileNotFoundError):
        read_events("missing", root=tmp_path)
    (tmp_path / "notes.txt").write_text("not a run", encoding="utf-8")
    assert list_runs(tmp_path) == ["r1"]  # only directories with an events file are runs
    assert list_runs(tmp_path / "nowhere") == []


def _two_runs(root):
    with TraceWriter("run_a", meta={"task_id": "T2", "model_key": "m_strong", "seed": 1000, "repeat": 0},
                     root=root) as w:
        w.model_call(**_call(usd=0.0125, input_tokens=1200, output_tokens=300, model_key="m_strong", phase="setup"))
        w.node_start("locate", "find_rack", params={"camera": "overhead"})
        w.node_end("locate", "find_rack", True, outputs={"yaw_deg": 2.6}, sim_seconds=0.8)
        w.model_call(**_call(usd=0.0, cache_hit=True, reasoning_tokens=None, model_key="m_strong", phase="patch"))
        w.verdict(Verdict(passed=False, score=0.4, failure_class="alignment", failing_node="insert", task_id="T2",
                          seed=1000))
    with TraceWriter("run_b", meta={"task_id": "T2", "model_key": "m_cheap", "seed": 1001, "repeat": 1,
                                    "session": "S1"}, root=root) as w:
        w.model_call(**_call(model="fake-cheap-1", usd=0.002, input_tokens=900, output_tokens=120,
                             reasoning_tokens=0, usd_source="provider", model_key="m_cheap", phase="setup"))
        w.verdict(Verdict(passed=True, score=1.0, task_id="T2", seed=1001))
        w.budget_stop("model", 0.002, 0.002, "cap reached")


def test_load_runs_flattens_model_calls_with_numeric_dtypes(tmp_path):
    _two_runs(tmp_path)
    df = load_runs(tmp_path)
    assert len(df) == 8
    assert list(df.columns[:4]) == ["run_id", "seq", "ts", "kind"]
    assert df.columns[-1] == "payload_json"
    assert list(zip(df["run_id"], df["seq"])) == [("run_a", i) for i in range(5)] + [("run_b", i) for i in range(3)]

    calls = df[df["kind"] == "model_call"]
    assert len(calls) == 3
    assert calls["usd"].sum() == pytest.approx(0.0145)
    assert calls.groupby("run_id")["usd"].sum().to_dict() == pytest.approx({"run_a": 0.0125, "run_b": 0.002})
    assert df["usd"].dtype == np.float64 and df["latency_ms"].dtype == np.float64
    for col in ("input_tokens", "output_tokens", "reasoning_tokens"):
        assert pd.api.types.is_integer_dtype(df[col]), col
    assert calls["input_tokens"].sum() == 1200 + 1200 + 900
    assert calls["reasoning_tokens"].isna().tolist() == [False, True, False]
    assert str(df["cache_hit"].dtype) == "boolean" and calls["cache_hit"].sum() == 1
    for col in ("provider", "model", "role", "usd_source", "request_hash", "model_key", "phase"):
        assert pd.api.types.is_string_dtype(df[col]), col
    assert calls["model"].tolist() == ["fake-writer-1", "fake-writer-1", "fake-cheap-1"]
    assert calls["phase"].tolist() == ["setup", "patch", "setup"]

    verdicts = df[df["kind"] == "verdict"]
    assert str(df["passed"].dtype) == "boolean" and verdicts["passed"].tolist() == [False, True]
    assert verdicts["score"].tolist() == [0.4, 1.0] and verdicts["seed"].tolist() == [1000, 1001]
    assert verdicts["failure_class"].iloc[0] == "alignment"
    ends = df[df["kind"] == "node_end"]
    assert str(df["ok"].dtype) == "boolean" and ends["ok"].tolist() == [True]
    assert json.loads(ends["outputs"].iloc[0]) == {"yaw_deg": 2.6}  # dict values are JSON text
    assert json.loads(df.loc[df["kind"] == "node_start", "params"].iloc[0]) == {"camera": "overhead"}
    assert df.loc[df["kind"] == "budget_stop", "cap_usd"].tolist() == [0.002]

    events = read_events("run_a", root=tmp_path) + read_events("run_b", root=tmp_path)
    assert [json.loads(text) for text in df["payload_json"]] == [e.payload for e in events]


def test_load_runs_meta_columns(tmp_path):
    _two_runs(tmp_path)
    df = load_runs(tmp_path)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    assert meta_cols == ["meta_task_id", "meta_model_key", "meta_seed", "meta_repeat", "meta_session"]
    assert list(df.columns[4:9]) == meta_cols  # right after the base columns
    by_run = df.groupby("run_id")
    assert by_run["meta_model_key"].first().to_dict() == {"run_a": "m_strong", "run_b": "m_cheap"}
    assert pd.api.types.is_integer_dtype(df["meta_seed"]) and pd.api.types.is_integer_dtype(df["meta_repeat"])
    assert by_run["meta_seed"].first().to_dict() == {"run_a": 1000, "run_b": 1001}
    assert df.loc[df["run_id"] == "run_a", "meta_session"].isna().all()
    assert (df.loc[df["run_id"] == "run_b", "meta_session"] == "S1").all()


def test_load_runs_filters_by_kind_and_run(tmp_path):
    _two_runs(tmp_path)
    calls = load_runs(tmp_path, kinds=["model_call"])
    assert set(calls["kind"]) == {"model_call"} and len(calls) == 3
    assert "usd" in calls.columns and "passed" not in calls.columns
    assert calls["usd"].sum() == pytest.approx(0.0145)
    stops = load_runs(tmp_path, kinds=["verdict", "budget_stop"])
    assert stops["kind"].tolist() == ["verdict", "verdict", "budget_stop"]
    only_b = load_runs(tmp_path, run_ids=["run_b"])
    assert set(only_b["run_id"]) == {"run_b"} and len(only_b) == 3
    with pytest.raises(ValueError, match="model_calls"):
        load_runs(tmp_path, kinds=["model_calls"])
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path, run_ids=["no_such_run"])


def test_load_runs_payload_keys_never_shadow_base_columns(tmp_path):
    with TraceWriter("odd", root=tmp_path) as w:
        w.intervention("odd keys", seq="not the seq", kind="not the kind", meta_seed=5, notes={"a": [1, 2]})
    row = load_runs(tmp_path).iloc[0]
    assert row["kind"] == "intervention" and row["seq"] == 0
    assert row["payload_seq"] == "not the seq" and row["payload_kind"] == "not the kind"
    assert row["payload_meta_seed"] == 5
    assert json.loads(row["notes"]) == {"a": [1, 2]}
    assert json.loads(row["payload_json"])["kind"] == "not the kind"


def test_load_runs_empty_inputs_give_typed_empty_frames(tmp_path):
    for df in (load_runs(tmp_path / "no_such_root"), load_runs(tmp_path)):
        assert df.empty and len(df) == 0
        assert {"run_id", "seq", "ts", "kind", "payload_json"} <= set(df.columns)
        assert df["usd"].dtype == np.float64 and df["usd"].sum() == 0
    with TraceWriter("idle", root=tmp_path):
        pass  # a run with no events yet
    assert list_runs(tmp_path) == ["idle"]
    idle = load_runs(tmp_path, kinds=["model_call"])
    assert idle.empty and "input_tokens" in idle.columns and "passed" not in idle.columns
    assert load_runs(tmp_path, run_ids=[]).empty


# --------------------------------------------------------------------------- helpers for raw files and children
def _raw_events(root, run_id, *lines):
    """A run folder holding ``lines`` as its events file, as another writer or a repair script left it."""
    run = Path(root) / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "events.jsonl").write_bytes(b"".join(line.encode("utf-8") + b"\n" for line in lines))
    return run


def _line(seq, reason, run_id="f", kind="intervention", payload=None):
    body = {"reason": reason} if payload is None else payload
    return json.dumps({"run_id": run_id, "ts": 1.0, "seq": seq, "kind": kind, "payload": body})


def _child(script, *args, **kwargs):
    """``script`` in a fresh interpreter from the repo root (offline)."""
    return subprocess.Popen([sys.executable, "-c", script, *map(str, args)], cwd=REPO_ROOT, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def _finish(child, timeout, input=None):
    """(stdout, stderr) of a child; a child that hangs (a deadlock) is killed and fails the test."""
    try:
        return child.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        out, err = child.communicate()
        pytest.fail(f"child still running after {timeout} s (deadlocked?)\n{out}\n{err}")


def _with_deadline(fn, seconds=30):
    """``fn()`` on a daemon thread, so a deadlock fails the test instead of hanging the suite."""
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # handed back to the test thread
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        pytest.fail(f"still running after {seconds} s: deadlocked")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


# --------------------------------------------------------------------------- one writer per run
def test_second_writer_in_this_process_is_refused_and_changes_nothing(tmp_path, monkeypatch):
    first = TraceWriter("shared", meta={"owner": "first"}, root=tmp_path)
    try:
        first.intervention("a")
        with pytest.raises(RunLockedError, match=r"run 'shared' already has an open TraceWriter in this process"):
            TraceWriter("shared", meta={"owner": "second"}, root=tmp_path)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RunLockedError, match="'shared'"):  # the same folder reached through a relative root
            TraceWriter("shared", root=".")
        assert first.intervention("b").seq == 1
        assert first.meta == {"owner": "first"}
    finally:
        first.close()
    assert json.loads((tmp_path / "shared" / "meta.json").read_text(encoding="utf-8")) == {"owner": "first"}
    assert (tmp_path / "shared" / LOCK_FILE).is_file()
    with TraceWriter("shared", root=tmp_path) as again:  # close() released the run
        assert again.next_seq == 2


def test_writer_dropped_without_close_frees_its_run(tmp_path):
    w = TraceWriter("dropped", root=tmp_path)
    w.intervention("a")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)  # the unclosed events file says so, as it should
        del w
        gc.collect()
    with TraceWriter("dropped", root=tmp_path) as again:
        assert again.next_seq == 1


_HOLD_RUN = r"""
import os, sys
from loopkit.traces import TraceWriter
w = TraceWriter(sys.argv[2], meta={"holder": "child"}, root=sys.argv[1])
w.intervention("from the child")
print("ready", os.getpid(), flush=True)
if sys.stdin.readline().strip() == "close":
    w.close()
    print("closed", flush=True)
"""


def _hold_run(root, run_id):
    child = _child(_HOLD_RUN, root, run_id)
    words = child.stdout.readline().split()
    if words[:1] != ["ready"]:
        child.kill()
        raise AssertionError(f"child did not open the run: {child.communicate(timeout=60)[1]}")
    return child, int(words[1])


def _open_once_free(run_id, root, timeout=20.0):
    """A TraceWriter as soon as a dead holder's lock is gone (the OS drops it as the process ends)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return TraceWriter(run_id, root=root)
        except RunLockedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_writer_in_another_process_holds_the_run_until_it_closes_or_dies(tmp_path):
    child, pid = _hold_run(tmp_path, "unit_7")
    try:
        with pytest.raises(RunLockedError, match=rf"run 'unit_7' is already open in another TraceWriter \(process {pid}\)"):
            TraceWriter("unit_7", meta={"holder": "parent"}, root=tmp_path)
        out, err = _finish(child, 60, input="close\n")
    finally:
        child.kill()
    assert child.returncode == 0 and "closed" in out, err
    with TraceWriter("unit_7", root=tmp_path) as w:  # closed by its owner: free, and the refused meta never landed
        assert w.next_seq == 1 and w.meta == {"holder": "child"}
        w.intervention("from the parent")

    child, pid = _hold_run(tmp_path, "unit_7")
    os.kill(pid, signal.SIGTERM)  # the interpreter dies holding the lock, no close()
    _finish(child, 60)
    with _open_once_free("unit_7", tmp_path) as w:
        assert w.next_seq == 3
    reasons = [e.payload["reason"] for e in read_events("unit_7", root=tmp_path)]
    assert reasons == ["from the child", "from the parent", "from the child"]


# --------------------------------------------------------------------------- lone surrogates
FFFD = "\N{REPLACEMENT CHARACTER}"  # what a lone surrogate becomes


def test_lone_surrogates_are_written_as_replacement_characters(tmp_path):
    cut_emoji = "model said: done \ud83d"  # a reply cut inside a surrogate pair
    undecodable = b"stderr: \x80".decode("utf-8", "surrogateescape")
    with TraceWriter("surrogates", meta={"note": "yaw \udc80", "k\ud800": 1}, root=tmp_path) as w:
        w.model_call(**_call(usd=0.25))
        w.model_call(**_call(usd=0.40, error="ValueError: " + cut_emoji))
        w.node_end("insert", "insert_plug", False, error=undecodable, sim_seconds=1.0,
                   outputs={"key \udfff": "v", "split pair": chr(0xD83D) + chr(0xDE00)})
    events = read_events("surrogates", root=tmp_path)
    assert events.malformed == 0 and [e.seq for e in events] == [0, 1, 2]
    assert events[1].payload["error"] == "ValueError: model said: done " + FFFD
    assert events[2].payload["error"] == "stderr: " + FFFD
    assert events[2].payload["outputs"] == {"key " + FFFD: "v", "split pair": "\U0001f600"}
    assert load_runs(tmp_path)["usd"].sum() == pytest.approx(0.65)
    meta = json.loads((tmp_path / "surrogates" / "meta.json").read_text(encoding="utf-8"))
    assert meta == {"note": "yaw " + FFFD, "k" + FFFD: 1}
    assert to_jsonable({"a\ud83d": ["b\udc80", np.array(["c\udfff"])]}) == {"a" + FFFD: ["b" + FFFD, ["c" + FFFD]]}


def test_lines_with_lone_surrogate_escapes_from_older_writers_are_recovered(tmp_path):
    texts = ["fine", "cut \ud83d", "low \udc80"]
    _raw_events(tmp_path, "old", *(_line(i, "", run_id="old", kind="model_call", payload=_call(usd=0.1, error=t))
                                   for i, t in enumerate(texts)))
    assert b"\\ud83d" in (tmp_path / "old" / "events.jsonl").read_bytes()  # what json.dumps wrote
    events = read_events("old", root=tmp_path)
    assert events.malformed == 0
    assert [e.payload["error"] for e in events] == ["fine", "cut " + FFFD, "low " + FFFD]
    df = load_runs(tmp_path)
    assert len(df) == 3 and df["usd"].sum() == pytest.approx(0.3)
    with TraceWriter("old", root=tmp_path) as w:
        assert w.next_seq == 3 and w.malformed_on_open == 0


# --------------------------------------------------------------------------- unreadable or odd meta.json
def test_unreadable_meta_costs_only_that_runs_meta_columns(tmp_path):
    for rid in ("unit_a", "unit_b", "unit_c"):
        with TraceWriter(rid, meta={"task_id": "T2", "model_key": rid}, root=tmp_path) as w:
            w.model_call(**_call())
    ansi = tmp_path / "unit_b" / "meta.json"  # PowerShell 5.1 Set-Content: the ANSI code page
    ansi_bytes = '{"task_id": "T2", "model_key": "unit_b", "note": "yaw +2.6\N{DEGREE SIGN}"}'.encode("cp1252")
    ansi.write_bytes(ansi_bytes)
    utf16 = tmp_path / "unit_c" / "meta.json"  # PowerShell 5.1 Out-File: UTF-16 with a BOM
    utf16.write_bytes('{"task_id": "T2", "model_key": "unit_c"}'.encode("utf-16"))

    with pytest.warns(UserWarning) as caught:
        df = load_runs(tmp_path)
    messages = [str(w.message) for w in caught]
    assert any(str(ansi) in m and "not UTF-8" in m for m in messages)
    assert any(str(utf16) in m and "not UTF-8" in m for m in messages)
    assert len(df) == 3 and df["usd"].sum() == pytest.approx(3 * 0.0125)
    assert df.loc[df["run_id"] == "unit_a", "meta_model_key"].tolist() == ["unit_a"]
    assert df.loc[df["run_id"] != "unit_a", "meta_model_key"].isna().all()

    with pytest.warns(UserWarning, match=re.escape(str(utf16))):
        assert read_meta(tmp_path / "unit_c") == {}
    with pytest.warns(UserWarning, match="not UTF-8"):
        with TraceWriter("unit_b", root=tmp_path) as w:  # reopening still works
            assert w.meta == {} and w.next_seq == 1
    assert ansi.read_bytes() == ansi_bytes  # opened without meta: the file is left alone


def test_busy_meta_is_retried_and_never_written_over(tmp_path, monkeypatch):
    with TraceWriter("busy", meta={"task_id": "T2"}, root=tmp_path):
        pass
    real_read = writer_module._read_file
    busy = {"left": 2}

    def busy_then_free(path):  # Windows: a reader caught the file in the instant it was replaced
        if busy["left"]:
            busy["left"] -= 1
            raise PermissionError(13, "The process cannot access the file", str(path))
        return real_read(path)

    monkeypatch.setattr(writer_module, "_read_file", busy_then_free)
    assert read_meta(tmp_path / "busy") == {"task_id": "T2"}  # two busy reads, the third works
    busy["left"] = 10**9  # busy for good
    with pytest.warns(UserWarning, match=r"meta\.json could not be read \(PermissionError"):
        assert read_meta(tmp_path / "busy") == {}
    with pytest.raises(PermissionError) as err:  # a merge needs the old keys: refuse rather than lose them
        TraceWriter("busy", meta={"repeat": 1}, root=tmp_path)
    assert "nothing was written" in " ".join(getattr(err.value, "__notes__", []))
    busy["left"] = 0
    assert json.loads((tmp_path / "busy" / "meta.json").read_text(encoding="utf-8")) == {"task_id": "T2"}
    with TraceWriter("busy", meta={"repeat": 1}, root=tmp_path) as w:  # the refused open released the run
        assert w.meta == {"task_id": "T2", "repeat": 1}


def test_malformed_meta_is_moved_aside_before_new_meta_is_written(tmp_path):
    meta = {"task_id": "T2", "model_key": "m_strong", "seed": 1000, "repeat": 2}
    with TraceWriter("unit", meta=meta, root=tmp_path) as w:
        w.model_call(**_call())
    path = tmp_path / "unit" / "meta.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"repeat": 2', '"repeat": 2, "note": "by hand",'),
                    encoding="utf-8")  # a hand edit that leaves a trailing comma
    hand_edit = path.read_bytes()

    with pytest.warns(UserWarning, match="not valid JSON"):
        with TraceWriter("unit", root=tmp_path) as w:  # no new meta: warn and leave the file where it is
            assert w.meta == {}
    assert path.read_bytes() == hand_edit

    with pytest.warns(UserWarning) as caught:
        with TraceWriter("unit", meta={"status": "resumed"}, root=tmp_path):
            pass
    moved = tmp_path / "unit" / "meta.json.corrupt-1"
    assert len(caught) == 1
    assert str(path) in str(caught[0].message) and str(moved) in str(caught[0].message)
    assert moved.read_bytes() == hand_edit  # every old key survives, beside the new file
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "resumed"}

    path.write_text("[1, 2]", encoding="utf-8")  # the next bad file gets the next free name
    with pytest.warns(UserWarning) as caught:
        with TraceWriter("unit", root=tmp_path) as w:  # opening warns, update_meta moves the file aside
            w.update_meta({"repeat": 3})
    messages = [str(m.message) for m in caught]
    assert len(messages) == 2 and "holds a JSON array, not an object" in messages[0]
    assert str(path.with_name("meta.json.corrupt-2")) in messages[1]
    assert path.with_name("meta.json.corrupt-2").read_text(encoding="utf-8") == "[1, 2]"
    assert json.loads(path.read_text(encoding="utf-8")) == {"repeat": 3}
    assert moved.read_bytes() == hand_edit


def test_meta_with_nan_or_infinity_can_still_take_updates(tmp_path):
    with TraceWriter("nan", meta={"task_id": "T2"}, root=tmp_path):
        pass
    path = tmp_path / "nan" / "meta.json"
    path.write_text(json.dumps({"task_id": "T2", "best_pass_rate": float("nan"), "spread": float("-inf")}),
                    encoding="utf-8")  # plain json.dumps writes NaN and -Infinity
    assert read_meta(tmp_path / "nan") == {"task_id": "T2", "best_pass_rate": None, "spread": None}
    with TraceWriter("nan", meta={"repeat": 1}, root=tmp_path) as w:
        merged = w.update_meta({"graph_hash": "g2"})
    assert merged == {"task_id": "T2", "best_pass_rate": None, "spread": None, "repeat": 1, "graph_hash": "g2"}
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text and json.loads(text) == merged


# --------------------------------------------------------------------------- seq reservation and re-entry
def test_interrupt_after_the_line_is_down_never_reuses_its_seq(tmp_path, monkeypatch):
    real_write = writer_module._write_all
    interrupt = {"armed": False}

    def ctrl_c_right_after_the_write(fh, data):
        real_write(fh, data)
        if interrupt["armed"]:
            interrupt["armed"] = False
            raise KeyboardInterrupt  # the line reached the file, emit has not returned yet

    monkeypatch.setattr(writer_module, "_write_all", ctrl_c_right_after_the_write)
    with TraceWriter("ctrl_c", root=tmp_path) as w:
        w.model_call(**_call())
        interrupt["armed"] = True
        with pytest.raises(KeyboardInterrupt):
            w.model_call(**_call(request_hash="interrupted"))
        assert w.next_seq == 2
        w.intervention("stopped by operator")
    events = read_events("ctrl_c", root=tmp_path)
    assert [(e.seq, e.kind) for e in events] == [(0, "model_call"), (1, "model_call"), (2, "intervention")]
    assert events[1].payload["request_hash"] == "interrupted" and events.malformed == 0


_CTRL_C_LOOP = r"""
import _thread, random, sys, threading
from loopkit.traces import TraceWriter, read_events
root, trials, runs_with_dupes = sys.argv[1], int(sys.argv[2]), 0
random.seed(7)
for t in range(trials):
    w = TraceWriter(f"r{t}", root=root)
    timer = threading.Timer(random.uniform(0.001, 0.004), _thread.interrupt_main)  # Ctrl+C, as the console sends it
    try:
        try:
            timer.start()
            while True:
                w.intervention("tick")
        except KeyboardInterrupt:
            w.intervention("stopped by operator")
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()
        w.close()
    seqs = [e.seq for e in read_events(f"r{t}", root=root)]
    runs_with_dupes += len(seqs) != len(set(seqs))
print("runs with a duplicate seq:", runs_with_dupes, flush=True)
"""


def test_real_ctrl_c_during_emit_never_duplicates_a_seq(tmp_path):
    out, err = _finish(_child(_CTRL_C_LOOP, tmp_path, 20), 120)
    assert "runs with a duplicate seq: 0" in out, out + err


def test_reentrant_emit_close_and_update_meta_raise_instead_of_deadlocking(tmp_path, monkeypatch):
    real_write = writer_module._write_all
    handler = {"call": None}

    def a_signal_handler_runs_inside_the_write(fh, data):
        if handler["call"] is not None:
            call, handler["call"] = handler["call"], None
            call()
        real_write(fh, data)

    monkeypatch.setattr(writer_module, "_write_all", a_signal_handler_runs_inside_the_write)

    def scenario():  # one thread throughout: the handler's call re-enters the writer it interrupted
        with TraceWriter("reenter", root=tmp_path) as w:
            for action, call, instead in (
                ("emit", lambda: w.intervention("from the handler"), "log"),
                ("update_meta", lambda: w.update_meta({"stopped": True}), "update the metadata"),
                ("close", w.close, "close the writer"),
            ):
                handler["call"] = call
                pattern = rf"{action} called re-entrantly on run 'reenter'.*set a flag in the handler and {instead} from the main loop"
                with pytest.raises(RuntimeError, match=pattern):
                    w.intervention("main loop")
            assert not w.closed
            assert w.intervention("after").seq == 3  # each refused handler cost the event it interrupted (seqs 0-2)

    _with_deadline(scenario)
    events = read_events("reenter", root=tmp_path)
    assert [(e.seq, e.payload["reason"]) for e in events] == [(3, "after")]
    assert not (tmp_path / "reenter" / "meta.json").exists()


_SIGINT_INSIDE_EMIT = r"""
import _thread, signal, sys
from loopkit.traces import TraceWriter, writer

w = TraceWriter("s", root=sys.argv[1])
real_write = writer._write_all
armed = {"on": False}

def write_with_ctrl_c(fh, data):
    if armed["on"]:
        armed["on"] = False
        _thread.interrupt_main()  # SIGINT for this (main) thread
        for _ in range(3):        # the handler runs at the next check, here, inside emit
            pass
    real_write(fh, data)

def on_sigint(signum, frame):     # the pattern the docs warn against: logging from the handler
    w.intervention("logged from the handler")

writer._write_all = write_with_ctrl_c
signal.signal(signal.SIGINT, on_sigint)
call = dict(provider="fake", model="m", role="writer", input_tokens=1, output_tokens=1, reasoning_tokens=None,
            usd=0.001, usd_source="computed", latency_ms=1.0, cache_hit=False, request_hash="h")
w.model_call(**call)
armed["on"] = True
try:
    w.model_call(**call)
except RuntimeError as exc:
    print("refused:", exc, flush=True)
w.intervention("logged from the main loop")
w.close()
print("finished", flush=True)
"""


def test_sigint_handler_that_logs_inside_emit_gets_an_error_not_a_hang(tmp_path):
    out, err = _finish(_child(_SIGINT_INSIDE_EMIT, tmp_path), 60)  # a deadlock is killed and fails here
    assert "refused: emit called re-entrantly on run 's'" in out, out + err
    assert "set a flag in the handler and log from the main loop" in out and "finished" in out, out + err
    events = read_events("s", root=tmp_path)
    assert [(e.seq, e.kind) for e in events] == [(0, "model_call"), (2, "intervention")]
    assert events[1].payload["reason"] == "logged from the main loop"


# --------------------------------------------------------------------------- pydantic models
def test_pydantic_models_with_numpy_fields_serialize(tmp_path):
    from pydantic import BaseModel, ConfigDict, Field, field_serializer

    class Detection(BaseModel):  # what a perception skill might return
        model_config = ConfigDict(arbitrary_types_allowed=True)
        label: str
        corners_px: np.ndarray
        score: np.float32
        debug: str = Field(default="internal", exclude=True)

    class Broken(BaseModel):
        value: int = 1

        @field_serializer("value")
        def _fail(self, value):
            raise RuntimeError("serializer failed")

    det = Detection(label="port", corners_px=np.array([[10, 20], [30, 40]], dtype=np.int32), score=np.float32(0.5))
    v = Verdict(passed=True, score=1.0, evidence=Evidence(metrics={"depth_mm": np.float32(8.5), "gap": np.array([0.1, 0.2])}))
    with TraceWriter("models", root=tmp_path) as w:
        end = w.node_end("find", "find_port", True, outputs={"det": det})
        verdict = w.verdict(v)
        with pytest.raises(ValueError, match=r"node_end payload for run 'models' is not JSON serializable.*serializer failed"):
            w.node_end("find", "find_port", True, outputs={"bad": Broken()})
        with pytest.raises(ValueError, match=r"verdict payload for run 'models' is not JSON serializable"):
            w.verdict(Broken())
        assert w.next_seq == 2
    assert end.payload["outputs"] == {"det": {"label": "port", "corners_px": [[10, 20], [30, 40]], "score": 0.5}}
    assert verdict.payload["metrics"] == {"depth_mm": 8.5, "gap": [0.1, 0.2]}
    assert [e.payload for e in read_events("models", root=tmp_path)] == [end.payload, verdict.payload]


# --------------------------------------------------------------------------- flattening
def test_payload_key_equal_to_a_prefixed_name_gets_a_numeric_suffix(tmp_path):
    with TraceWriter("a", root=tmp_path / "simple") as w:
        w.intervention("x", seq="from key seq", payload_seq="from key payload_seq")
    row = load_runs(tmp_path / "simple").iloc[0]
    assert row["seq"] == 0 and row["payload_seq"] == "from key seq" and row["payload_seq_2"] == "from key payload_seq"
    assert load_runs(tmp_path / "simple").attrs["renamed_payload_keys"] == {"seq": "payload_seq",
                                                                            "payload_seq": "payload_seq_2"}

    with TraceWriter("b", root=tmp_path / "crowded") as w:
        w.intervention("x", seq="key seq", payload_seq="key payload_seq", payload_seq_2="key payload_seq_2")
        w.intervention("y", payload_seq="literal only")  # one column means one key in every row
    df = load_runs(tmp_path / "crowded")
    assert df.attrs["renamed_payload_keys"] == {"seq": "payload_seq", "payload_seq": "payload_seq_3"}
    assert df["payload_seq"].tolist()[0] == "key seq" and pd.isna(df["payload_seq"].iloc[1])
    assert df["payload_seq_2"].tolist()[0] == "key payload_seq_2"
    assert df["payload_seq_3"].tolist() == ["key payload_seq", "literal only"]


def test_declared_numeric_columns_stay_numeric_when_another_kind_puts_text_there(tmp_path):
    with TraceWriter("b", root=tmp_path) as w:
        w.model_call(**_call(input_tokens=1200, usd=0.01))
        w.model_call(**_call(input_tokens=900, usd=0.02))
        w.verdict(Verdict(passed=True, score=1.0, seed=2001))
        w.verdict(Verdict(passed=False, score=0.4, failure_class="alignment", seed=2002))
        # extras that collide with declared names: a seed read from argv, a stray label, a flag, a fraction
        w.node_start("insert", "insert_plug", seed="2003", input_tokens="n/a", usd=True, score=0.5)
    df = load_runs(tmp_path)
    assert pd.api.types.is_integer_dtype(df["input_tokens"]) and pd.api.types.is_integer_dtype(df["seed"])
    assert df["usd"].dtype == np.float64 and df["score"].dtype == np.float64
    assert df["input_tokens"].sum() == 2100
    assert int(df["seed"].between(2000, 2019).sum()) == 3  # the held-out filter works
    start = int(df.index[df["kind"] == "node_start"][0])
    assert df.loc[start, "seed"] == 2003 and pd.isna(df.loc[start, "input_tokens"]) and pd.isna(df.loc[start, "usd"])
    assert df.loc[start, "score"] == 0.5  # a float in a float column fits as it is
    assert df.attrs["coerced_rows"] == {"input_tokens": [start], "usd": [start], "seed": [start]}
    raw = json.loads(df.loc[start, "payload_json"])
    assert raw["seed"] == "2003" and raw["input_tokens"] == "n/a" and raw["usd"] is True
    clean = load_runs(tmp_path, kinds=["model_call"])
    assert clean.attrs["coerced_rows"] == {} and clean.attrs["renamed_payload_keys"] == {}


# --------------------------------------------------------------------------- resume agrees with the reader
def test_resume_takes_every_seq_form_the_reader_accepts(tmp_path):
    forms = [0, 1.0, "2", " 3 ", "4.0", True]  # a repair script round-tripped seq through floats and text
    _raw_events(tmp_path, "f", *(_line(seq, f"line {i}") for i, seq in enumerate(forms)))
    assert [e.seq for e in read_events("f", root=tmp_path)] == [0, 1, 2, 3, 4, 1]
    with TraceWriter("f", root=tmp_path) as w:
        assert w.next_seq == 5 and w.malformed_on_open == 0
        assert w.intervention("appended").seq == 5
    seqs = [e.seq for e in read_events("f", root=tmp_path)]
    assert seqs.count(5) == 1 and max(seqs) == 5


def test_deeply_nested_line_does_not_stop_a_resume(tmp_path):
    _raw_events(tmp_path, "f", _line(0, "a"), "[" * 100_000 + "]" * 100_000)
    with TraceWriter("f", root=tmp_path) as w:
        assert w.next_seq == 1 and w.malformed_on_open == 1
        w.intervention("b")
    events = read_events("f", root=tmp_path)
    assert [e.seq for e in events] == [0, 1] and events.malformed_lines == [2]


def test_seq_beyond_int64_is_a_malformed_line(tmp_path):
    _raw_events(tmp_path, "f", _line(0, "a"), _line(10**20, "b"))
    events = read_events("f", root=tmp_path)
    assert [e.seq for e in events] == [0] and events.malformed_lines == [2]
    df = load_runs(tmp_path)  # used to raise OverflowError for the whole root
    assert df["seq"].tolist() == [0] and df.attrs["malformed_lines"] == {"f": 1}
    with TraceWriter("f", root=tmp_path) as w:
        assert w.next_seq == 1 and w.malformed_on_open == 1

    _raw_events(tmp_path, "last", _line(writer_module.MAX_SEQ, "the largest seq int64 holds", run_id="last"))
    assert load_runs(tmp_path, run_ids=["last"])["seq"].tolist() == [writer_module.MAX_SEQ]
    with TraceWriter("last", root=tmp_path) as w:  # the writer never hands out a seq the readers would drop
        with pytest.raises(ValueError, match="no seq left"):
            w.intervention("one too many")


# --------------------------------------------------------------------------- run ids and run folders
def test_load_runs_reads_listed_runs_by_path_never_from_the_cwd(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    with TraceWriter("unit_a", root=root) as w:
        w.model_call(**_call(usd=0.1))
    # a run folder named like an events file (an older writer allowed it), next to a CWD file of that name
    _raw_events(root, "t2_dump.jsonl", _line(0, "", run_id="t2_dump.jsonl", kind="model_call", payload=_call(usd=0.1)))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "t2_dump.jsonl").write_bytes(b"".join(
        _line(i, "", run_id="other", kind="model_call", payload=_call(usd=9.99)).encode() + b"\n" for i in range(5)))
    monkeypatch.chdir(cwd)
    assert list_runs(root) == ["t2_dump.jsonl", "unit_a"]
    df = load_runs(root)
    assert df.groupby("run_id").size().to_dict() == {"t2_dump.jsonl": 1, "unit_a": 1}
    assert df["usd"].sum() == pytest.approx(0.2)
    assert df.attrs["malformed_lines"] == {"t2_dump.jsonl": 0, "unit_a": 0}


@pytest.mark.skipif(os.name != "nt", reason="only Windows maps <folder>/nul to a device")
def test_run_folder_must_resolve_inside_its_root(tmp_path, monkeypatch):
    from loopkit import paths

    monkeypatch.setattr(writer_module, "check_run_id", lambda run_id: run_id)  # let NUL reach the backstop
    with pytest.raises(ValueError, match="does not resolve to a folder inside"):
        TraceWriter("nul", root=tmp_path)
    monkeypatch.setenv("LOOPKIT_DATA", str(tmp_path / "data"))
    with pytest.raises(ValueError, match="does not resolve to a folder inside"):
        paths.run_dir("nul")
    with pytest.raises(ValueError, match="does not resolve to a folder inside"):
        TraceWriter("nul")
    assert [p.name for p in tmp_path.rglob("*")] == ["data", "runs"]
