"""Write, rehearse and patch one task with one model, then score the best graph on held-out seeds.

    uv run python -m loopkit.program.cli --model opus --task T2 --train-seeds 100-109 --heldout-seeds 500-519
    uv run python -m loopkit.program.cli --model fake --task T2 ...     # FakeModel with the reference graph

A paid model runs inside a spend session (``--session``, ``--cap``, ``--total-cap``) through the
router ledger. Writer calls, patch rounds and every episode are traced under
``LOOPKIT_DATA/runs/<run id>``; the per-round table, the held-out result and the best graph are
written to ``LOOPKIT_DATA/runs/<run id>/summary.json`` and ``best_graph.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from contextlib import nullcontext

from ..paths import check_disk_floor, run_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="rehearse-and-patch one task, then score on held-out seeds")
    ap.add_argument("--model", required=True, help="models.yaml key, or 'fake' for the scripted reference writer")
    ap.add_argument("--task", default="T2", choices=["T2", "T3"])
    ap.add_argument("--train-seeds", default="100-109")
    ap.add_argument("--heldout-seeds", default="500-519")
    ap.add_argument("--rounds", type=int, default=5, help="patch rounds after the initial program")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--session", default="S3")
    ap.add_argument("--cap", type=float, default=15.0, help="session cap in USD")
    ap.add_argument("--total-cap", type=float, default=21.0, help="cap over all sessions in USD")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    from ..router.adapters.fake import fake_script
    from ..sim.tasks import task_spec
    from ..skills import catalog
    from ..traces import TraceWriter
    from .fake_writer import reference_writer
    from .rehearse import parse_seeds, pass_rate, rehearse_and_patch, run_episodes

    check_disk_floor()
    fake = args.model == "fake"
    if not fake:
        from ..router.ledger import set_active_session

        set_active_session(args.session, args.cap, total_cap_usd=args.total_cap)
    train, heldout = parse_seeds(args.train_seeds), parse_seeds(args.heldout_seeds)
    run_id = args.run_id or f"{args.session.lower()}_{args.model}_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"
    root = run_dir(run_id)
    trace = TraceWriter(run_id, meta={"kind": "unit", "model_key": args.model, "task_id": args.task, "repeat": 1,
                                      "train_seeds": args.train_seeds, "heldout_seeds": args.heldout_seeds,
                                      "fake": fake})
    episodes_root = str(root / "episodes")
    t0 = time.perf_counter()
    with fake_script(reference_writer) if fake else nullcontext():
        reh = rehearse_and_patch(args.model, task_spec(args.task), catalog(), train, max_rounds=args.rounds,
                                 workers=args.workers, trace=trace, trace_root=episodes_root,
                                 run_prefix=f"{run_id}_train", cache_salt=run_id)
    held = []
    if reh.best_graph is not None:
        held = run_episodes(args.task, heldout, reh.best_graph, workers=args.workers, trace_root=episodes_root,
                            run_prefix=f"{run_id}_heldout")
        for r in held:
            trace.emit("verdict", {"passed": r["passed"], "score": r["score"] if not r["passed"] else 1.0,
                                   "failure_class": None if r["passed"] else (r["failure_class"] or "other"),
                                   "failing_node": None if r["passed"] else r["failing_node"],
                                   "metrics": r["metrics"], "task_id": args.task, "seed": r["seed"],
                                   "split": "heldout", "sim_seconds": r["sim_seconds"]})
        (root / "best_graph.json").write_text(json.dumps(reh.best_graph.model_dump(mode="json", exclude_none=True),
                                                         indent=1), encoding="utf-8")
    trace.close()
    rate = pass_rate(held) if held else 0.0
    classes = dict(Counter(r["failure_class"] for r in held if not r["passed"]))
    summary = {**reh.summary(), "run_id": run_id, "fake": fake, "train_seeds": args.train_seeds,
               "heldout_seeds": args.heldout_seeds, "heldout_pass_rate": rate,
               "heldout_passed": sum(1 for r in held if r["passed"]), "heldout_n": len(held),
               "heldout_failure_classes": classes,
               "heldout_failures": [{"seed": r["seed"], "class": r["failure_class"], "node": r["failing_node"],
                                     "summary": r["summary"], "error": r["last_error"]} for r in held if not r["passed"]],
               "wall_seconds": round(time.perf_counter() - t0, 1)}
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    print(f"{args.model} {args.task}: stopped because {reh.stopped_because}; best round {reh.best_round}")
    print(f"{'round':>5} {'train pass':>11} {'writer calls':>13} {'writer USD':>11}  failure classes")
    for r in reh.rounds:
        print(f"{r.round:>5} {r.passed:>5}/{r.n:<5} {r.writer_calls:>13} {r.writer_usd:>11.4f}  {r.failure_classes or ''}"
              + (f"  ({r.note})" if r.note else ""))
    print(f"held-out {args.heldout_seeds}: {summary['heldout_passed']}/{len(held)} = {100 * rate:.0f}%, "
          f"failure classes {classes or 'none'}; writer calls {reh.writer_calls}, ${reh.usd:.4f}")
    print(f"summary: {root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
