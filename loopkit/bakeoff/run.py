"""The bake-off runner. Resumable, capped, traced. No agent involved.

    uv run python -m loopkit.bakeoff.run --config config/bakeoff.yaml [--models ...] [--tasks ...]
        [--repeats N] [--dry-run] [--date YYYY-MM-DD] [--workers N] [--log PATH]

For each model, task and repeat (a "unit"): T2 and T3 run rehearse-and-patch on the training
seeds and score the best graph on the held-out seeds; T1 calls the model as the panel reader once
per held-out seed. Every unit is checkpointed to ``units/<model>__<task>__r<k>.json``, so a rerun
skips finished units (an error or a skip for a missing key runs again). A model without a key is
skipped up front and listed. Per-model and total caps go through the router ledger; a model that
hits its cap stops, a ``budget_stop`` event says where, and its remaining units are recorded as
skipped.

Repeats never share writer or reader cache entries: the repeat index is part of every request hash
(``cache_salt``). Held-out episode results are cached by (task, seed, graph hash), which is safe
because the simulator is deterministic; T1 episodes add the model and repeat to the key.

``--dry-run`` replaces every model with the FakeModel (scripted reference writer, schema-default
reader). It exercises the whole pipeline at $0 and its numbers describe the skill library, not a model.
It scores on dev seeds 600 to 619 unless ``--heldout-seeds`` says otherwise, because the bake-off
held-out seeds are never used before the bake-off.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from ..config import BAKEOFF_YAML, BakeoffConfig, load_bakeoff, load_models
from ..paths import REPO_ROOT, bakeoff_dir, check_disk_floor

T1_GRAPH = "T1"
# Unit statuses a rerun keeps. Anything else (an error, a skip for a missing key) runs again.
FINISHED = ("done", "no_graph", "budget_stop", "skipped_after_budget_stop")
# A dry run scores on these dev seeds unless told otherwise, never on the bake-off held-out seeds.
DRY_RUN_HELDOUT = "600-619"


class Tee:
    """Print to stdout and append to a UTF-8 log file, flushing every line."""

    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *parts: Any) -> None:
        line = " ".join(str(p) for p in parts)
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {line}", flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"[{stamp}] {line}\n")


def _read_unit(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def unit_name(model: str, task: str, repeat: int) -> str:
    return f"{model}__{task}__r{repeat}"


def git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def write_snapshot(out: Path, cfg: BakeoffConfig, models: list[str], tasks: list[str], repeats: int, dry: bool,
                   date: str, train: list[int] | None = None, heldout: list[int] | None = None) -> None:
    mcfg = load_models()
    used = {"train": train or cfg.train_seeds.seeds(), "heldout": heldout or cfg.heldout_seeds.seeds()}
    snap = {
        "run_date": date,
        "dry_run": dry,
        "git_commit": git_commit(),
        "cache_policy": "router disk cache on; the repeat index is part of every writer and reader request hash, so "
                        "repeats never share cache entries; held-out episode results are cached by (task, seed, graph "
                        "hash) because the simulator is deterministic (T1 adds the model and repeat to the key)",
        "models": {m: mcfg.get(m).model_dump(mode="json", exclude={"key"}) for m in models},
        "tasks": tasks,
        "repeats": repeats,
        "bakeoff": json.loads(cfg.model_dump_json()),
        "seeds_used": {k: f"{min(v)} to {max(v)} ({len(v)} seeds)" for k, v in used.items()},
        "seed_override": bool(train or heldout),
    }
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(snap, sort_keys=False, allow_unicode=True),
                                              encoding="utf-8", newline="\n")


def _episode_cache_path(out: Path, task: str, seed: int, key: str) -> Path:
    return out / "episodes" / f"{task}_s{seed}_{key}.json"


def score_heldout(out: Path, task: str, graph, seeds: list[int], *, workers: int, unit: str,
                  reader_model: str | None = None, reader_tag: str | None = None, chat_kwargs: dict | None = None,
                  trace=None) -> list[dict]:
    """Held-out episodes for one graph, with the deterministic episode cache.

    An episode with a reader in it is cached per unit (``reader_tag``, the model and repeat), since
    its result depends on that model's answers; an episode without one only on the graph.
    """
    from ..program.rehearse import run_episodes

    key = graph.graph_hash() + (f"_{reader_tag}" if reader_model else "")
    todo, results = [], {}
    for s in seeds:
        p = _episode_cache_path(out, task, s, key)
        if p.is_file():
            try:
                results[s] = {**json.loads(p.read_text(encoding="utf-8")), "cached": True}
                continue
            except (OSError, ValueError):
                pass
        todo.append(s)
    if todo:
        fresh = run_episodes(task, todo, graph, workers=workers, trace_root=str(out / "traces"),
                             run_prefix=f"{unit}_heldout", reader_model=reader_model, chat_kwargs=chat_kwargs)
        for r in fresh:
            results[r["seed"]] = {**r, "cached": False}
            if r.get("error") is None:
                p = _episode_cache_path(out, task, r["seed"], key)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(r, default=str), encoding="utf-8")
    ordered = [results[s] for s in seeds]
    if trace is not None:
        for r in ordered:
            payload = {"passed": bool(r["passed"]), "score": float(r["score"]),
                       "failure_class": None if r["passed"] else (r.get("failure_class") or "other"),
                       "failing_node": None if r["passed"] else r.get("failing_node"),
                       "metrics": r.get("metrics") or {}, "task_id": task, "seed": r["seed"], "split": "heldout",
                       "sim_seconds": r.get("sim_seconds"), "model_calls": r.get("model_calls", 0),
                       "model_usd": r.get("model_usd", 0.0), "cached": r.get("cached", False),
                       "graph_hash": graph.graph_hash(), "status": r.get("status")}
            if payload["score"] >= 1.0 and not payload["passed"]:
                payload["score"] = 0.999
            trace.emit("verdict", payload)
    return ordered


def run_unit(out: Path, model: str, task: str, repeat: int, cfg: BakeoffConfig, *, dry: bool, workers: int,
             log: Tee, train_override: list[int] | None = None, heldout_override: list[int] | None = None) -> dict:
    from ..program.rehearse import rehearse_and_patch
    from ..program.reference import reference_graph
    from ..router import SpendCapExceeded
    from ..schemas import ProgramGraph
    from ..sim.tasks import task_spec
    from ..skills import catalog
    from ..traces import TraceWriter

    unit = unit_name(model, task, repeat)
    route = "fake" if dry else model
    train = train_override or cfg.train_seeds.seeds()
    heldout = heldout_override or cfg.heldout_seeds.seeds()
    trace = TraceWriter(unit, root=out / "traces", meta={"model_key": model, "route": route, "task_id": task,
                                                          "repeat": repeat, "unit": unit, "kind": "unit", "dry_run": dry})
    salt = f"bakeoff:{out.name}:{model}:{task}:r{repeat}"
    record: dict[str, Any] = {"unit": unit, "model": model, "route": route, "task": task, "repeat": repeat,
                              "status": "error", "dry_run": dry}
    t0 = time.perf_counter()
    try:
        if task == T1_GRAPH:
            graph = ProgramGraph.model_validate(reference_graph("T1"))
            results = score_heldout(out, task, graph, heldout, workers=workers, unit=unit, reader_model=route,
                                    reader_tag=f"{model}_r{repeat}", trace=trace,
                                    chat_kwargs={"cache_salt": salt, "max_tokens": cfg.reader.max_tokens,
                                                 "reasoning": cfg.reader.reasoning})
            record.update(status="done", heldout=results, rehearsal=None, graph_hash=graph.graph_hash())
        else:
            spec = task_spec(task)
            reh = rehearse_and_patch(route, spec, catalog(), train, max_rounds=cfg.patch.max_rounds,
                                     stop_after_no_improvement=cfg.patch.stop_after_no_improvement, workers=workers,
                                     trace=trace, trace_root=str(out / "traces"), run_prefix=f"{unit}_train",
                                     cache_salt=salt, writer_max_tokens=cfg.writer.max_tokens,
                                     writer_reasoning=cfg.writer.reasoning)
            record["rehearsal"] = reh.summary()
            if reh.best_graph is None:
                # no program to run: the unit fails every held-out seed, and the trace says why
                results = [{"seed": sd, "passed": False, "score": 0.0, "failure_class": "other", "failing_node": None,
                            "metrics": {"no_graph": True}, "sim_seconds": 0.0, "model_calls": 0, "model_usd": 0.0,
                            "status": "no_graph", "cached": False} for sd in heldout]
                for r in results:
                    trace.emit("verdict", {**{k: r[k] for k in ("passed", "score", "failure_class", "failing_node",
                                                                  "metrics", "sim_seconds", "status")},
                                           "task_id": task, "seed": r["seed"], "split": "heldout",
                                           "summary": "no valid graph: the writer never produced a program that validated"})
                record.update(status="no_graph", heldout=results, graph_hash=None)
            else:
                (out / "graphs").mkdir(exist_ok=True)
                (out / "graphs" / f"{unit}.json").write_text(
                    json.dumps(reh.best_graph.model_dump(mode="json", exclude_none=True), indent=1), encoding="utf-8")
                results = score_heldout(out, task, reh.best_graph, heldout, workers=workers, unit=unit, trace=trace)
                record.update(status="done", heldout=results, graph_hash=reh.best_graph.graph_hash())
    except SpendCapExceeded as exc:
        record.update(status="budget_stop", error=str(exc))
        trace.emit("budget_stop", {"scope": getattr(exc, "scope", "cap"), "cap_usd": float(getattr(exc, "cap_usd", 0) or 0),
                                   "spent_usd": float(getattr(exc, "spent_usd", 0) or 0), "reason": str(exc)[:300],
                                   "unit": unit})
        log(f"  budget stop in {unit}: {exc}")
    except Exception as exc:  # noqa: BLE001
        record.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=6))
        log(f"  error in {unit}: {type(exc).__name__}: {exc}")
    finally:
        trace.close()
    record["wall_seconds"] = round(time.perf_counter() - t0, 1)
    hp = record.get("heldout") or []
    if hp:
        record["heldout_pass_rate"] = round(sum(1 for r in hp if r["passed"]) / len(hp), 4)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="loopkit bake-off runner")
    ap.add_argument("--config", default=str(BAKEOFF_YAML))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", default=None, help="output folder name (default: today); reuse it to resume")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--log", default=None, help="log file (default: <out>/run.log; '-' for stdout only)")
    ap.add_argument("--train-seeds", default=None, help="override, e.g. 1000-1002 (smoke runs only)")
    ap.add_argument("--heldout-seeds", default=None, help="override, e.g. 2000-2004 (smoke runs only)")
    ap.add_argument("--session-cap", type=float, default=None, help="override the session cap (default: total_cap_usd)")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_bakeoff(Path(args.config))
    models = args.models or list(cfg.models)
    optional_note = None
    if not args.models and cfg.optional_models:
        from .estimate import estimate

        added = estimate(cfg)["optional_included"]
        models += added
        optional_note = (f"optional models {cfg.optional_models}: added {added or 'none'} (added only if every "
                         f"model spending up to its cap stays under total_cap_usd ${cfg.total_cap_usd:.0f})")
    tasks = args.tasks or list(cfg.tasks)
    repeats = args.repeats or cfg.repeats
    date = args.date or dt.date.today().isoformat()
    name = f"{date}_dry" if args.dry_run and not args.date else date
    out = bakeoff_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "units").mkdir(exist_ok=True)
    log = Tee(None if args.log == "-" else Path(args.log) if args.log else out / "run.log")
    from ..program.rehearse import parse_seeds

    train = parse_seeds(args.train_seeds) if args.train_seeds else None
    heldout = parse_seeds(args.heldout_seeds) if args.heldout_seeds else None
    if args.dry_run and heldout is None:
        # the bake-off held-out seeds are never used before the bake-off (CLAUDE.md), not even by a dry run
        heldout = parse_seeds(DRY_RUN_HELDOUT)
    free = check_disk_floor()
    log(f"bake-off {cfg.name} -> {out} | models {models} | tasks {tasks} | repeats {repeats} | dry run {args.dry_run}")
    log(f"free disk GB {free}")
    if optional_note:
        log(optional_note)
    write_snapshot(out, cfg, models, tasks, repeats, args.dry_run, date, train, heldout)

    if not args.dry_run:
        from ..router.ledger import Ledger, set_active_session

        ledger = Ledger()
        spent_before = ledger.spent()
        session = f"bakeoff-{name}"
        cap = args.session_cap if args.session_cap is not None else cfg.total_cap_usd
        set_active_session(session, cap, total_cap_usd=round(spent_before + cap, 6),
                           model_caps={m: float(cfg.caps_usd[m]) for m in models if m in cfg.caps_usd})
        log(f"spend session {session}: cap ${cap:.2f}, per-model caps {[(m, cfg.caps_usd.get(m)) for m in models]}, "
            f"ledger total before ${spent_before:.4f}")
        from .estimate import estimate, format_estimate

        sub = cfg.model_copy(update={"models": models, "optional_models": [], "tasks": tasks, "repeats": repeats})
        log("cost estimate before launch:\n" + format_estimate(estimate(sub), sub))
    stopped_models: dict[str, str] = {}
    no_key: dict[str, str] = {}
    if not args.dry_run:
        from ..router.smoke import skip_reason

        mcfg = load_models()
        for model in models:
            reason = skip_reason(mcfg.get(model))
            if reason:
                no_key[model] = reason
                log(f"skip model {model}: {reason}")
    ctx_fake = None
    if args.dry_run:
        from ..program.fake_writer import reference_writer
        from ..router.adapters.fake import fake_script

        ctx_fake = fake_script(reference_writer)
        ctx_fake.__enter__()
    try:
        for model in models:
            for task in tasks:
                for k in range(1, repeats + 1):
                    unit = unit_name(model, task, k)
                    path = out / "units" / f"{unit}.json"
                    prior = _read_unit(path)
                    if prior.get("status") in FINISHED:
                        log(f"skip {unit} ({prior['status']})")
                        if prior["status"] == "budget_stop":
                            stopped_models.setdefault(model, unit)
                        continue
                    if model in no_key:
                        rec = {"unit": unit, "model": model, "task": task, "repeat": k, "status": "skipped_no_key",
                               "reason": no_key[model], "dry_run": args.dry_run}
                        path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                        continue
                    if model in stopped_models:
                        rec = {"unit": unit, "model": model, "task": task, "repeat": k, "status": "skipped_after_budget_stop",
                               "stopped_at": stopped_models[model], "dry_run": args.dry_run}
                        path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                        log(f"skip {unit}: {model} stopped at its cap in {stopped_models[model]}")
                        continue
                    check_disk_floor()
                    log(f"run {unit}")
                    rec = run_unit(out, model, task, k, cfg, dry=args.dry_run, workers=args.workers, log=log,
                                   train_override=train, heldout_override=heldout)
                    path.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
                    log(f"done {unit}: status {rec['status']}, held-out pass rate {rec.get('heldout_pass_rate')}, "
                        f"{rec['wall_seconds']} s")
                    if rec["status"] == "budget_stop":
                        stopped_models[model] = unit
    finally:
        if ctx_fake is not None:
            ctx_fake.__exit__(None, None, None)
    if not args.no_report:
        from .report import build_report

        paths = build_report(out)
        log(f"report: {paths}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
