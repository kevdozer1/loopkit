"""Rehearse-and-patch, episode running and held-out scoring.

1. Run the graph on N training seeds (in parallel processes, one SimBackend each). Episodes are
   run and graded by ``loopkit.sim.episode``; this module only sees the plain-data records.
2. Summarize the failures as text: failing node, its error and outputs, checker metrics, and
   what differed in the scene from the passing seeds. No raw images in the patch prompt.
3. Send the summary with the current graph to the writer; validate the new graph.
4. Stop after ``max_rounds`` patch rounds, at a perfect pass rate, or when the pass rate fails to
   improve for ``stop_after_no_improvement`` rounds. Keep the best graph.

Every round is logged as a ``patch`` event; every writer call as a ``model_call`` event (by the router).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..schemas import ProgramGraph, SkillSpec, TaskSpec

DEFAULT_WORKERS = 10


# ------------------------------------------------------------------------------ episodes
def _episode_star(args: tuple) -> dict:
    from ..sim.episode import run_episode  # harness side: grades with the checker

    return run_episode(*args)


def run_episodes(task_id: str, seeds: list[int], graph: ProgramGraph, *, workers: int = DEFAULT_WORKERS,
                 trace_root: str | None = None, run_prefix: str | None = None, reader_model: str | None = None,
                 chat_kwargs: dict | None = None, episode_fn: Callable | None = None) -> list[dict]:
    gj = graph.model_dump(mode="json", exclude_none=True)
    jobs = [(task_id, s, gj, trace_root, f"{run_prefix}_s{s}" if run_prefix else None, None, reader_model, chat_kwargs)
            for s in seeds]
    if episode_fn is not None:
        return [episode_fn(*j) for j in jobs]
    if workers <= 1 or len(jobs) == 1:
        return [_episode_star(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(_episode_star, jobs))


# ------------------------------------------------------------------------------ failure summary
def _fmt_outputs(outputs: dict) -> str:
    text = json.dumps(outputs, separators=(",", ":"), default=str)
    return text if len(text) <= 240 else text[:237] + "..."


def summarize_failures(results: list[dict], graph: ProgramGraph | None = None, max_examples: int = 6) -> str:
    """Text for the patch prompt: what failed, where, and what differed in the scene from passing seeds."""
    n = len(results)
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    lines = [f"Rehearsal on {n} training runs (each a different scene): {len(passed)} passed, {len(failed)} failed."]
    if not failed:
        return "\n".join(lines)
    classes = Counter(r["failure_class"] for r in failed)
    lines.append("Failure classes: " + ", ".join(f"{k} {v}" for k, v in classes.most_common()) + ".")
    nodes = Counter(r["failing_node"] or "(none)" for r in failed)
    lines.append("Failing node: " + ", ".join(f"{k} in {v} of {len(failed)}" for k, v in nodes.most_common()) + ".")
    lines.append("Examples:")
    for r in failed[:max_examples]:
        node_hist = [h for h in r["history"] if h["node"] == r["failing_node"] and not h["ok"]]
        err = r.get("last_error") or (node_hist[-1]["error"] if node_hist else "") or "none"
        err = err.splitlines()[0][:300] if err else "none"
        lines.append(f"- run ended '{r['status']}'; failing node '{r['failing_node']}', error: {err}. Checker: "
                     f"{r['summary']} ({r['failure_class']}).")
        if node_hist and node_hist[-1].get("outputs"):
            lines.append(f"  outputs of the failing node: {_fmt_outputs(node_hist[-1]['outputs'])}")
    diffs = scene_differences(passed, failed)
    if diffs:
        lines.append("What differed in the scene between the failing and the passing runs:")
        lines += [f"- {d}" for d in diffs]
    return "\n".join(lines)


def scene_differences(passed: list[dict], failed: list[dict], top: int = 3) -> list[str]:
    keys = sorted({k for r in passed + failed for k in r.get("scene", {})})
    out = []
    scored = []
    for k in keys:
        fv = [r["scene"][k] for r in failed if k in r.get("scene", {})]
        pv = [r["scene"][k] for r in passed if k in r.get("scene", {})]
        if not fv:
            continue
        if k.startswith("target_"):
            c = Counter(fv)
            common, cnt = c.most_common(1)[0]
            share_f = cnt / len(fv)
            share_p = (sum(1 for v in pv if v == common) / len(pv)) if pv else 0.0
            if share_f - share_p >= 0.4 and cnt >= 2:
                scored.append((share_f - share_p, f"{k.replace('_', ' ')} was {common} in {cnt} of {len(fv)} failures"
                                                  f" but in {round(share_p * len(pv))} of {len(pv)} passes"))
            continue
        fm = float(np.mean(fv))
        if pv:
            pm = float(np.mean(pv))
            spread = float(np.std(fv + pv)) or 1.0
            effect = abs(fm - pm) / spread
            if effect >= 0.8:
                scored.append((effect, f"{k.replace('_', ' ')}: {fm:+.1f} on average in failures vs {pm:+.1f} in passes"))
        elif abs(fm) > 0:
            scored.append((0.1, f"{k.replace('_', ' ')}: {fm:+.1f} on average in failures"))
    scored.sort(key=lambda t: -t[0])
    out = [s for _, s in scored[:top]]
    return out


# ------------------------------------------------------------------------------ the loop
@dataclass
class RoundRecord:
    round: int
    graph_hash: str | None
    pass_rate: float
    passed: int
    n: int
    writer_ok: bool
    writer_attempts: int
    writer_usd: float
    writer_calls: int
    failure_classes: dict[str, int] = field(default_factory=dict)
    note: str = ""


@dataclass
class RehearsalResult:
    task_id: str
    model_key: str
    best_graph: ProgramGraph | None
    best_round: int
    best_pass_rate: float
    rounds: list[RoundRecord]
    stopped_because: str
    usd: float
    writer_calls: int
    train_results: list[list[dict]] = field(default_factory=list)

    def summary(self) -> dict:
        return {"task": self.task_id, "model": self.model_key, "best_round": self.best_round,
                "best_pass_rate": self.best_pass_rate, "stopped_because": self.stopped_because,
                "usd": round(self.usd, 6), "writer_calls": self.writer_calls,
                "rounds": [asdict(r) for r in self.rounds],
                "best_graph_hash": self.best_graph.graph_hash() if self.best_graph else None}


def first_observation_images(task_id: str, seed: int) -> tuple[list[np.ndarray], tuple[str, ...]]:
    from ..sim.episode import first_observation_images as first

    return first(task_id, seed)


def rehearse_and_patch(model_key: str, task: TaskSpec, catalog: dict[str, SkillSpec], train_seeds: list[int], *,
                       max_rounds: int = 5, stop_after_no_improvement: int = 2, workers: int = DEFAULT_WORKERS,
                       trace=None, trace_root: str | None = None, run_prefix: str = "rehearse",
                       chat: Callable | None = None, cache_salt: str = "", chat_kwargs: dict | None = None,
                       episode_fn: Callable | None = None, images: list | None = None,
                       image_names: tuple[str, ...] = (), writer_max_tokens: int = 16000,
                       writer_reasoning: str | dict = "highest") -> RehearsalResult:
    from .writer import write_graph

    if images is None:
        images, image_names = first_observation_images(task.id, train_seeds[0])
    rounds: list[RoundRecord] = []
    all_results: list[list[dict]] = []
    usd, calls = 0.0, 0
    w = write_graph(model_key, task, catalog, images, image_names=image_names, chat=chat, trace=trace,
                    cache_salt=f"{cache_salt}|r0", chat_kwargs=chat_kwargs, max_tokens=writer_max_tokens,
                    reasoning=writer_reasoning)
    usd += w.usd
    calls += len(w.calls)
    if trace is not None:
        trace.emit("patch", {"round": 0, "graph_before_hash": None,
                             "graph_after_hash": w.graph.graph_hash() if w.graph else None,
                             "failure_summary": "", "writer_ok": w.ok, "writer_attempts": w.attempts,
                             "writer_errors": w.errors[:5]})
    if w.graph is None:
        rounds.append(RoundRecord(0, None, 0.0, 0, len(train_seeds), False, w.attempts, w.usd, len(w.calls),
                                  note="no valid graph: " + "; ".join(w.errors[:2])))
        return RehearsalResult(task.id, model_key, None, 0, 0.0, rounds, "no valid initial graph", usd, calls, all_results)
    graph = w.graph
    best_graph, best_rate, best_round = graph, -1.0, 0
    no_improve = 0
    stopped = f"reached {max_rounds} patch rounds"
    for r in range(0, max_rounds + 1):
        results = run_episodes(task.id, train_seeds, graph, workers=workers, trace_root=trace_root,
                               run_prefix=f"{run_prefix}_r{r}", episode_fn=episode_fn)
        all_results.append(results)
        n_pass = sum(1 for x in results if x["passed"])
        rate = n_pass / len(results)
        classes = dict(Counter(x["failure_class"] for x in results if not x["passed"]))
        rec = RoundRecord(r, graph.graph_hash(), rate, n_pass, len(results), True, w.attempts, w.usd, len(w.calls), classes)
        rounds.append(rec)
        if rate > best_rate:
            best_graph, best_rate, best_round = graph, rate, r
            no_improve = 0
        elif r > 0:
            no_improve += 1
        if rate >= 1.0:
            stopped = "all training runs passed"
            break
        if no_improve >= stop_after_no_improvement:
            stopped = f"no improvement for {stop_after_no_improvement} rounds"
            break
        if r == max_rounds:
            break
        summary = summarize_failures(results, graph)
        w = write_graph(model_key, task, catalog, None, previous=graph, failure_summary=summary, chat=chat,
                        trace=trace, cache_salt=f"{cache_salt}|r{r + 1}", chat_kwargs=chat_kwargs,
                        max_tokens=writer_max_tokens, reasoning=writer_reasoning)
        usd += w.usd
        calls += len(w.calls)
        new_graph = w.graph if w.graph is not None else graph
        if trace is not None:
            trace.emit("patch", {"round": r + 1, "graph_before_hash": graph.graph_hash(),
                                 "graph_after_hash": new_graph.graph_hash(), "failure_summary": summary,
                                 "writer_ok": w.ok, "writer_attempts": w.attempts, "writer_errors": w.errors[:5],
                                 "pass_rate_before": rate})
        if w.graph is None:
            rec.note = "patch did not validate; kept the previous graph"
        graph = new_graph
    return RehearsalResult(task.id, model_key, best_graph, best_round, max(best_rate, 0.0), rounds, stopped, usd, calls,
                           all_results)


def pass_rate(results: list[dict]) -> float:
    return sum(1 for r in results if r["passed"]) / len(results) if results else 0.0


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out
