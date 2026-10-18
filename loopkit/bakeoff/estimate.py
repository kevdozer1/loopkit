"""Cost estimate for a bake-off config, printed before anything launches.

    uv run python -m loopkit.bakeoff.estimate [--config config/bakeoff.yaml] [--from <bakeoff folder>]

Input tokens come from the real prompts: the writer prompt for T2 and T3 (TaskSpec, skill catalog,
example, 3 camera images, the graph JSON Schema), the patch prompt (previous graph plus a failure
summary) and the T1 reader prompt (1 front image). They are priced with the router's own
projection (``loopkit.router.core.project``), so the numbers match what the ledger reserves.

Three numbers per model:

* worst case: every unit uses every patch round and every repair, and every call runs to its full
  output allowance. The runner stops a model at its cap, so spend never passes ``min(worst, cap)``.
* typical: per-call output tokens and calls per unit from a measured run when ``--from`` names one
  (a bake-off folder with traces); otherwise the assumptions printed with the table.
* the cap.

The total is the sum of ``min(typical, cap)`` over the configured models. An optional model is
added only if the sum of ``min(worst, cap)``, with it included, stays under ``total_cap_usd``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import BAKEOFF_YAML, BakeoffConfig, load_bakeoff, load_models

# Assumptions used when no measured run is given. They are printed with every estimate.
TYPICAL = {
    "writer_output_tokens": 6000,     # graph JSON (~1.5k tokens) plus reasoning at the highest setting
    "patch_rounds": 3,                # of max_rounds 5; stops early at 100% or after 2 flat rounds
    "attempts_per_call": 1.3,         # validate-and-repair: 1 to 4 attempts per writer turn
    "reader_output_tokens": 500,      # a 12-port JSON report, reasoning disabled
    "summary_chars": 2500,            # a failure summary in the patch prompt
}


@dataclass
class Prompts:
    """Prompt sizes in characters, plus image counts, for each kind of call."""

    writer_chars: dict[str, int]
    patch_chars: dict[str, int]
    repair_extra_chars: dict[str, int]
    reader_chars: int
    writer_images: int = 3
    reader_images: int = 1


@dataclass
class ModelEstimate:
    model: str
    worst_usd: float
    typical_usd: float
    cap_usd: float | None
    per_task: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str = "assumptions"
    price_note: str | None = None

    @property
    def bounded_typical(self) -> float:
        return min(self.typical_usd, self.cap_usd) if self.cap_usd is not None else self.typical_usd

    @property
    def bounded_worst(self) -> float:
        return min(self.worst_usd, self.cap_usd) if self.cap_usd is not None else self.worst_usd


def _chars(messages: list[dict], schema: dict | None) -> int:
    n = sum(len(m["content"]) for m in messages)
    if schema is not None:
        n += len(json.dumps(schema, sort_keys=True, separators=(",", ":")))
    return n


def measure_prompts(tasks: list[str]) -> Prompts:
    from ..program.prompt import build_messages, repair_message
    from ..program.reference import reference_graph
    from ..program.writer import GRAPH_SCHEMA
    from ..schemas import ProgramGraph
    from ..sim.tasks import task_spec
    from ..skills import catalog
    from ..skills.perceive import READER_SYSTEM

    cat = catalog()
    names = ("overhead", "front", "wrist")
    writer, patch, repair = {}, {}, {}
    for t in tasks:
        if t == "T1":
            continue
        spec = task_spec(t)
        graph = ProgramGraph.model_validate(reference_graph(t))
        writer[t] = _chars(build_messages(spec, cat, image_names=names), GRAPH_SCHEMA)
        summary = "x" * TYPICAL["summary_chars"]
        patch[t] = _chars(build_messages(spec, cat, previous=graph, failure_summary=summary), GRAPH_SCHEMA)
        answer = json.dumps(graph.model_dump(mode="json", exclude_none=True))
        repair[t] = len(answer) + len(repair_message("x" * 600)["content"])
    t1 = task_spec("T1")
    question = ("Report every port 1 to 12 as JSON: {\"1\": {\"occupied\": true|false, \"led\": \"green|amber|red|off\"}, "
                "..., \"12\": {...}}. Read the port numbers from the label strip.")
    reader = _chars([{"role": "system", "content": READER_SYSTEM}, {"role": "user", "content": question}],
                    t1.output_schema)
    return Prompts(writer, patch, repair, reader)


def measured_rates(folder: Path) -> dict[str, dict[str, float]]:
    """Per model: mean output tokens per writer and reader call, writer calls per unit, from traces."""
    from ..traces import load_runs

    df = load_runs(folder / "traces", kinds=["model_call"])
    if df.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    units = df[df.get("meta_kind") == "unit"] if "meta_kind" in df.columns else df
    for model, g in df.groupby("model_key"):
        w = g[g["role"] == "writer"]
        r = g[g["role"] == "reader"]
        rec: dict[str, float] = {}
        if len(w):
            rec["writer_output_tokens"] = float(w["output_tokens"].dropna().mean()) if w["output_tokens"].notna().any() else math.nan
            n_units = units[(units["model_key"] == model) & (units["role"] == "writer")]["run_id"].nunique()
            rec["writer_calls_per_unit"] = len(w) / max(1, n_units)
        if len(r):
            rec["reader_output_tokens"] = float(r["output_tokens"].dropna().mean()) if r["output_tokens"].notna().any() else math.nan
        out[str(model)] = rec
    return out


def estimate_model(model: str, cfg: BakeoffConfig, prompts: Prompts, tasks: list[str], *,
                   measured: dict[str, float] | None = None) -> ModelEstimate:
    from ..router.core import prices, project

    models = load_models()
    entry = models.get(model)
    pin, pout, _cached, fallback = prices(entry, models)
    rounds_max = cfg.patch.max_rounds
    attempts_max = 4  # 1 + MAX_REPAIRS
    writer_out_max = cfg.writer.max_tokens
    reader_out_max = cfg.reader.max_tokens
    reps = cfg.repeats
    heldout = cfg.heldout_seeds.count
    m = measured or {}
    w_out = m.get("writer_output_tokens") if not math.isnan(m.get("writer_output_tokens", math.nan)) else None
    w_out = w_out or TYPICAL["writer_output_tokens"]
    r_out = m.get("reader_output_tokens") if not math.isnan(m.get("reader_output_tokens", math.nan)) else None
    r_out = r_out or TYPICAL["reader_output_tokens"]
    per_task: dict[str, dict[str, float]] = {}
    worst = typical = 0.0
    for t in tasks:
        if t == "T1":
            est_in, worst_call, _ = project(entry, models, prompts.reader_chars, prompts.reader_images, reader_out_max)
            typ_call = est_in * pin / 1e6 + r_out * pout / 1e6
            w_t, t_t = worst_call * heldout * reps, typ_call * heldout * reps
        else:
            in0, worst0, _ = project(entry, models, prompts.writer_chars[t], prompts.writer_images, writer_out_max)
            inp, worstp, _ = project(entry, models, prompts.patch_chars[t], 0, writer_out_max)
            extra_in = math.ceil(prompts.repair_extra_chars[t] / 3.5)
            # worst: every turn uses all 4 attempts and each repair attempt carries the growing conversation
            def turn_worst(base_in: int, base_worst: float) -> float:
                total = 0.0
                for a in range(attempts_max):
                    total += base_worst + (a * extra_in) * pin / 1e6
                return total
            w_unit = turn_worst(in0, worst0) + rounds_max * turn_worst(inp, worstp)
            att = TYPICAL["attempts_per_call"]
            if "writer_calls_per_unit" in m:
                calls = m["writer_calls_per_unit"]
                turns = 1 + TYPICAL["patch_rounds"]
                att = max(1.0, calls / turns)
            typ_turn0 = att * (in0 * pin / 1e6 + w_out * pout / 1e6) + (att - 1) * extra_in * pin / 1e6
            typ_turnp = att * (inp * pin / 1e6 + w_out * pout / 1e6) + (att - 1) * extra_in * pin / 1e6
            t_unit = typ_turn0 + TYPICAL["patch_rounds"] * typ_turnp
            w_t, t_t = w_unit * reps, t_unit * reps
        per_task[t] = {"worst_usd": w_t, "typical_usd": t_t}
        worst += w_t
        typical += t_t
    cap = cfg.caps_usd.get(model)
    return ModelEstimate(model, worst, typical, float(cap) if cap is not None else None, per_task,
                         "measured run" if measured else "assumptions",
                         f"prices from {fallback} (this entry has none)" if fallback else None)


def estimate(cfg: BakeoffConfig, *, measured_from: Path | None = None,
             tasks: list[str] | None = None) -> dict[str, Any]:
    tasks = tasks or list(cfg.tasks)
    prompts = measure_prompts(tasks)
    measured = measured_rates(measured_from) if measured_from else {}
    rows = [estimate_model(m, cfg, prompts, tasks, measured=measured.get(m)) for m in cfg.models]
    total = sum(r.bounded_typical for r in rows)
    bounded_worst = sum(r.bounded_worst for r in rows)
    optional_rows, included = [], []
    for m in cfg.optional_models:
        r = estimate_model(m, cfg, prompts, tasks, measured=measured.get(m))
        optional_rows.append(r)
        # the stricter test: even if every model spent up to its cap, the run would stay under the total cap
        if bounded_worst + r.bounded_worst <= cfg.total_cap_usd:
            total += r.bounded_typical
            bounded_worst += r.bounded_worst
            included.append(m)
    return {"rows": rows, "optional": optional_rows, "optional_included": included, "total_typical": total,
            "total_bounded_worst": min(bounded_worst, cfg.total_cap_usd), "total_cap": cfg.total_cap_usd,
            "prompts": prompts, "measured_from": str(measured_from) if measured_from else None, "tasks": tasks}


def format_estimate(res: dict[str, Any], cfg: BakeoffConfig) -> str:
    lines = [f"Bake-off estimate for {cfg.name}: tasks {', '.join(res['tasks'])}, {cfg.repeats} repeats, "
             f"{cfg.train_seeds.count} training and {cfg.heldout_seeds.count} held-out seeds.",
             "Source: " + (f"measured run {res['measured_from']}" if res["measured_from"] else
                           "prompt sizes from the real prompts and the typical-case assumptions below (no measured run)"),
             "", f"{'model':<11}{'typical':>10}{'worst':>10}{'cap':>8}{'counted':>10}  per task (typical)"]
    for r in res["rows"] + res["optional"]:
        tag = "" if r in res["rows"] else (" optional, included" if r.model in res["optional_included"]
                                            else " optional, left out")
        per = ", ".join(f"{t} ${v['typical_usd']:.2f}" for t, v in r.per_task.items())
        cap = f"${r.cap_usd:.0f}" if r.cap_usd is not None else "none"
        counted = f"${r.bounded_typical:.2f}" if (r in res["rows"] or r.model in res["optional_included"]) else "-"
        lines.append(f"{r.model:<11}{'$%.2f' % r.typical_usd:>10}{'$%.2f' % r.worst_usd:>10}{cap:>8}{counted:>10}  "
                     f"{per}{tag}" + (f" ({r.price_note})" if r.price_note else ""))
    lines += ["", f"Estimated total: ${res['total_typical']:.2f} typical (each model counted up to its cap). "
                  f"Hard ceiling: ${res['total_bounded_worst']:.2f}, because every model stops at its cap and the "
                  f"run stops at total_cap_usd ${res['total_cap']:.0f}."]
    if not res["measured_from"]:
        a = TYPICAL
        lines.append(f"Typical-case assumptions: {a['writer_output_tokens']} output tokens per writer call, "
                     f"{a['patch_rounds']} patch rounds per unit, {a['attempts_per_call']} attempts per writer turn, "
                     f"{a['reader_output_tokens']} output tokens per reader call. Worst case: all "
                     f"{cfg.patch.max_rounds} patch rounds, 4 attempts per turn, every call at max_tokens "
                     f"({cfg.writer.max_tokens} writer, {cfg.reader.max_tokens} reader).")
    p = res["prompts"]
    lines.append("Prompt sizes (characters, before images): writer " +
                 ", ".join(f"{t} {n:,}" for t, n in p.writer_chars.items()) + "; patch " +
                 ", ".join(f"{t} {n:,}" for t, n in p.patch_chars.items()) + f"; reader {p.reader_chars:,}. "
                 f"Images: {p.writer_images} per initial writer call, {p.reader_images} per reader call.")
    if res["total_typical"] > res["total_cap"]:
        lines.append("The estimate is above total_cap_usd. Options: cut repeats to 2, or lower the caps of the "
                     "most expensive writers. The config is not edited here.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Estimate the cost of a bake-off config")
    ap.add_argument("--config", default=str(BAKEOFF_YAML))
    ap.add_argument("--from", dest="measured_from", default=None, help="a bake-off folder with traces to measure")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_bakeoff(Path(args.config))
    res = estimate(cfg, measured_from=Path(args.measured_from) if args.measured_from else None)
    if args.json:
        print(json.dumps({"total_typical": round(res["total_typical"], 2),
                          "total_bounded_worst": round(res["total_bounded_worst"], 2),
                          "rows": [{"model": r.model, "typical": round(r.typical_usd, 3),
                                    "worst": round(r.worst_usd, 3), "cap": r.cap_usd} for r in res["rows"]]},
                         indent=1))
    else:
        print(format_estimate(res, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
