"""The bake-off report, built from traces.

    uv run python -m loopkit.bakeoff.report --date YYYY-MM-DD

Every number comes from the trace events under ``LOOPKIT_DATA/bakeoff/<date>/traces``:

* unit runs (meta kind "unit"): the writer's ``model_call`` events, the held-out ``verdict``
  events (split "heldout"), the ``patch`` events and any ``budget_stop`` events;
* episode runs (meta kind "episode"): training-round verdicts (run ids ``<unit>_train_r<k>_s<seed>``)
  give the pass rate per patch round, and the reader's ``model_call`` events for T1.

The unit checkpoints (``units/*.json``) are read only to list the units that never ran (a model
without a key, or one stopped at its cap) and why.

Writes ``results.parquet`` (one row per held-out run), ``report.md`` (Level 1 line, Level 1.5
paragraph, Level 2 tables), ``pass_vs_cost.png`` (one dot per model, one panel per task) and copies
report.md to ``reports/bakeoff_<date>.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import REPO_ROOT, bakeoff_dir
from ..traces import load_runs

Z95 = 1.959963984540054
IMAGE_TOKENS_DEFAULT = 1600  # the router's projection default when models.yaml has no image_tokens
NOISE_NOTE = ("Power: pooled over 3 repeats a model and task has 60 held-out runs, so a gap smaller than about 15 to "
              "18 points is within noise. On one repeat of 20 runs it is about 30 points.")
TRAIN_RE = re.compile(r"^(?P<unit>.+)_train_r(?P<round>\d+)_s(?P<seed>\d+)$")
HELDOUT_RE = re.compile(r"^(?P<unit>.+)_heldout_s(?P<seed>\d+)$")


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # exact at the edges: c + m is 1 for k = n and c - m is 0 for k = 0, up to rounding
    lo = 0.0 if k == 0 else max(0.0, c - m)
    hi = 1.0 if k == n else min(1.0, c + m)
    return lo, hi


def _num(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)


def _spend(calls: pd.DataFrame) -> pd.Series:
    """What each call cost the ledger: ``ledger_usd`` (includes attempts lost to timeouts or dropped
    connections), else the answer's own ``usd``."""
    return _num(_col(calls, "ledger_usd")).fillna(_num(calls["usd"])).fillna(0.0)


def unit_of(run_id: str) -> str:
    """The unit an episode run belongs to, or the run id itself for a unit run."""
    for rx in (TRAIN_RE, HELDOUT_RE):
        m = rx.match(run_id)
        if m:
            return m.group("unit")
    return run_id


# ------------------------------------------------------------------------------ tables from traces
def heldout_rows(units: pd.DataFrame) -> pd.DataFrame:
    v = units[units["kind"] == "verdict"]
    v = v[_col(v, "split") == "heldout"]
    return pd.DataFrame({
        "unit": v["run_id"].astype(str), "model": _col(v, "meta_model_key"), "task": _col(v, "meta_task_id"),
        "repeat": _num(_col(v, "meta_repeat")), "seed": _num(_col(v, "seed")), "passed": v["passed"].astype(bool),
        "score": _num(v["score"]), "failure_class": _col(v, "failure_class"),
        "sim_seconds": _num(_col(v, "sim_seconds")), "runtime_model_calls": _num(_col(v, "model_calls")).fillna(0),
        "runtime_usd": _num(_col(v, "model_usd")).fillna(0.0), "graph_hash": _col(v, "graph_hash"),
        "cached_episode": _col(v, "cached"), "dry_run": _col(v, "meta_dry_run"), "status": _col(v, "status"),
    }).reset_index(drop=True)


def call_rows(df: pd.DataFrame) -> pd.DataFrame:
    mc = df[df["kind"] == "model_call"].copy()
    mc["unit"] = [unit_of(str(r)) for r in mc["run_id"]]
    return mc


def training_rounds(episodes: pd.DataFrame) -> pd.DataFrame:
    v = episodes[episodes["kind"] == "verdict"]
    rows = []
    for run_id, passed in zip(v["run_id"], v["passed"]):
        m = TRAIN_RE.match(str(run_id))
        if m:
            rows.append({"unit": m.group("unit"), "round": int(m.group("round")), "seed": int(m.group("seed")),
                         "passed": bool(passed)})
    return pd.DataFrame(rows, columns=["unit", "round", "seed", "passed"])


def unit_checkpoints(out: Path) -> list[dict]:
    rows = []
    for p in sorted((out / "units").glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            rows.append({"unit": p.stem, "status": "unreadable checkpoint"})
    return rows


def _image_tokens(snapshot: dict, model: str) -> int:
    entry = (snapshot.get("models") or {}).get(model) or {}
    return int(entry.get("image_tokens") or IMAGE_TOKENS_DEFAULT)


def summarize(out: Path, snapshot: dict | None = None) -> dict[str, Any]:
    snapshot = snapshot or {}
    df = load_runs(out / "traces")
    empty = {"heldout": pd.DataFrame(), "per_model_task": pd.DataFrame(), "calls": pd.DataFrame(), "stops": [],
             "rounds": pd.DataFrame(), "reasoning": {}, "events": len(df), "spend_by_model": {},
             "no_graph_units": []}
    if df.empty:
        return empty
    kind = _col(df, "meta_kind")
    units, episodes = df[kind == "unit"], df[kind == "episode"]
    held = heldout_rows(units)
    calls = call_rows(df)
    rounds = training_rounds(episodes)
    stops = [{"unit": r["run_id"], "model": r.get("meta_model_key"), "reason": r.get("reason")}
             for _, r in units[units["kind"] == "budget_stop"].iterrows()]
    best_round: dict[str, int] = {}
    for unit, g in rounds.groupby("unit"):
        rates = g.groupby("round")["passed"].mean()
        best_round[unit] = int(rates[rates == rates.max()].index.min())
    # the reasoning setting each model's calls actually sent, per role (docs/03 rule 1)
    reasoning: dict[str, set[str]] = {}
    unit_model = dict(zip(units["run_id"].astype(str), _col(units, "meta_model_key")))
    for _, c in calls.iterrows():
        model = unit_model.get(c["unit"]) or c.get("model_key")
        setting = c.get("reasoning_setting")
        text = setting if isinstance(setting, str) else json.dumps(setting)
        reasoning.setdefault(f"{model} {c.get('role')}", set()).add(text)
    rows = []
    for (model, task), h in held.groupby(["model", "task"], sort=True):
        n, k = len(h), int(h["passed"].sum())
        lo, hi = wilson(k, n)
        per_rep = h.groupby("repeat")["passed"].mean()
        unit_ids = sorted(h["unit"].unique())
        c = calls[calls["unit"].isin(unit_ids)]
        writer, runtime = c[c["role"] == "writer"], c[c["role"] != "writer"]
        usd = float(_spend(c).sum()) if len(c) else 0.0
        in_tok = float(_num(c["input_tokens"]).sum()) if len(c) else 0.0
        out_tok = float(_num(c["output_tokens"]).sum()) if len(c) else 0.0
        img_tok = float(_num(_col(c, "image_count")).fillna(0).sum()) * _image_tokens(snapshot, str(model))
        succ = h[h["passed"]]
        fc = Counter(h.loc[~h["passed"], "failure_class"].fillna("other"))
        rounds_u = [best_round[u] for u in unit_ids if u in best_round]
        rows.append({
            "model": model, "task": task, "runs": n, "passed": k, "pass_rate": k / n if n else math.nan,
            "ci_low": lo, "ci_high": hi, "repeats": int(len(per_rep)),
            "repeat_mean": float(per_rep.mean()) if len(per_rep) else math.nan,
            "repeat_sd": float(per_rep.std(ddof=1)) if len(per_rep) > 1 else math.nan,
            "usd": usd, "usd_per_success": usd / k if k else math.nan,
            "setup_calls_per_unit": len(writer) / max(1, len(unit_ids)),
            "runtime_calls_per_run": len(runtime) / n if n else math.nan,
            "failed_calls": int((~_col(c, "ok").fillna(True).astype(bool)).sum()) if len(c) else 0,
            "sim_s_per_success": float(succ["sim_seconds"].mean()) if len(succ) else math.nan,
            "patch_rounds_to_best": float(np.mean(rounds_u)) if rounds_u else math.nan,
            "input_tokens": in_tok, "output_tokens": out_tok,
            # n/a when the reported input tokens cannot include the images (the FakeModel counts text only)
            "image_share_est": img_tok / in_tok if in_tok and img_tok <= in_tok else math.nan,
            "failure_classes": dict(fc),
        })
    spend_by_model: dict[str, float] = {}
    if len(calls):
        owner = [unit_model.get(u) or mk for u, mk in zip(calls["unit"], _col(calls, "model_key"))]
        for model, usd_call in zip(owner, _spend(calls)):
            spend_by_model[str(model)] = spend_by_model.get(str(model), 0.0) + float(usd_call)
    no_graph = sorted({u for u, st in zip(held["unit"], _col(held, "status")) if st == "no_graph"})         if "status" in held.columns else []
    return {"heldout": held, "per_model_task": pd.DataFrame(rows), "calls": calls, "stops": stops,
            "rounds": rounds, "reasoning": {m: sorted(v) for m, v in reasoning.items()}, "events": len(df),
            "spend_by_model": spend_by_model, "no_graph_units": no_graph}


# ------------------------------------------------------------------------------ markdown
def _isnan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _pct(x: float) -> str:
    return "n/a" if _isnan(x) else f"{100 * x:.0f}%"


def _usd(x: float) -> str:
    if _isnan(x):
        return "n/a"
    return f"${x:.4f}" if x < 1 else f"${x:.2f}"


def _fmt(x: float, spec: str = "{:.1f}") -> str:
    return "n/a" if _isnan(x) else spec.format(x)


def render_markdown(out: Path, s: dict[str, Any], snapshot: dict[str, Any], units: list[dict]) -> str:
    pm = s["per_model_task"]
    dry = bool(snapshot.get("dry_run"))
    date = snapshot.get("run_date", out.name)
    name = (snapshot.get("bakeoff") or {}).get("name", "")
    lines = [f"# Bake-off {name} {date}{' (dry run)' if dry else ''}", ""]
    if dry:
        lines += ["**Dry run.** Every model slot was played by the FakeModel: the writer returns the hand-written "
                  "reference graph and the T1 reader returns the smallest answer its schema allows. The numbers "
                  "describe the skill library and the pipeline at $0, not any model.", ""]
    not_run = [u for u in units if u.get("status") not in ("done", "no_graph")]
    if pm.empty:
        lines += ["Level 1: no held-out results in the traces.", ""]
    else:
        parts = []
        for task, g in pm.groupby("task", sort=True):
            top = g[g["pass_rate"] == g["pass_rate"].max()].sort_values("usd_per_success")
            names = list(top["model"])
            who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] + " tied"
            parts.append(f"{task} {_pct(top['pass_rate'].iloc[0])} ({who})")
        lines += [f"Level 1: best held-out pass rate per task: {', '.join(parts)}.", ""]
        n_models = pm["model"].nunique()
        total_usd = sum(s["spend_by_model"].values())
        para = (f"{n_models} model{'s' if n_models != 1 else ''} ran {', '.join(sorted(pm['task'].unique()))} on "
                f"held-out seeds, {int(pm['runs'].sum())} scored runs in all, for {_usd(total_usd)} of model calls. ")
        if (pm["task"] == "T1").any():
            para += "On T1 every model call is a runtime call: the model reads the panel once per run. "
        if pm["task"].isin(["T2", "T3"]).any():
            para += ("On T2 and T3 the model only writes the program. The graph then runs with no model in the loop, "
                     "so runtime calls are 0 and all cost is setup. ")
        lines += ["Level 1.5: " + para + NOISE_NOTE, ""]
    if s["stops"]:
        lines += ["Stopped at a cap: " + "; ".join(f"{x['model']} in {x['unit']} ({x['reason']})" for x in s["stops"])
                  + ".", ""]
    if s["no_graph_units"]:
        lines += ["No valid graph (scored as a failure on every held-out seed): " + ", ".join(s["no_graph_units"]) + ".",
                  ""]
    if not_run:
        lines += ["Units without results:", ""]
        for u in not_run:
            why = u.get("reason") or u.get("error") or u.get("stopped_at") or ""
            lines.append(f"- {u.get('unit')}: {u.get('status')}{(' (' + str(why)[:200] + ')') if why else ''}")
        lines.append("")
    if not pm.empty:
        lines += ["## Level 2", "", "### Held-out pass rate", "",
                  "| Task | Model | Passed | Pass rate | Wilson 95% | Repeats | Repeat mean | Repeat sd |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in pm.sort_values(["task", "pass_rate"], ascending=[True, False]).itertuples():
            sd = "n/a" if _isnan(r.repeat_sd) else f"{100 * r.repeat_sd:.0f} pts"
            lines.append(f"| {r.task} | {r.model} | {r.passed}/{r.runs} | {_pct(r.pass_rate)} | {_pct(r.ci_low)} to "
                         f"{_pct(r.ci_high)} | {r.repeats} | {_pct(r.repeat_mean)} | {sd} |")
        lines += ["", NOISE_NOTE, "", "### Cost, calls and tokens", "",
                  "| Task | Model | Model spend | Cost per success | Setup calls per unit | Runtime calls per run | "
                  "Failed calls | Input tokens | Output tokens | Image share of input (est.) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in pm.sort_values(["task", "model"]).itertuples():
            lines.append(f"| {r.task} | {r.model} | {_usd(r.usd)} | {_usd(r.usd_per_success)} | "
                         f"{_fmt(r.setup_calls_per_unit)} | {_fmt(r.runtime_calls_per_run, '{:.2f}')} | {r.failed_calls} | "
                         f"{int(r.input_tokens):,} | {int(r.output_tokens):,} | {_pct(r.image_share_est)} |")
        caps = (snapshot.get("bakeoff") or {}).get("caps_usd") or {}
        if s["spend_by_model"]:
            lines += ["", "### Spend per model, every unit", "", "| Model | Spend | Cap |", "|---|---|---|"]
            for m, v in sorted(s["spend_by_model"].items()):
                cap = caps.get(m)
                lines.append(f"| {m} | {_usd(v)} | {'$%.2f' % cap if cap is not None else 'n/a'} |")
        lines += ["", "Setup calls are writer calls (the initial program, patches and repairs). Runtime calls are model calls "
                  "made while a held-out run executes. Image share is an estimate: images sent times the model's "
                  "image token estimate, over the input tokens the provider reported.", "",
                  "### Cycle time, patching and failures", "",
                  "| Task | Model | Sim seconds per success | Patch rounds to best graph | Failure classes |",
                  "|---|---|---|---|---|"]
        for r in pm.sort_values(["task", "model"]).itertuples():
            fc = ", ".join(f"{k} {v}" for k, v in sorted(r.failure_classes.items(), key=lambda kv: -kv[1])) or "none"
            lines.append(f"| {r.task} | {r.model} | {_fmt(r.sim_s_per_success)} | {_fmt(r.patch_rounds_to_best)} | {fc} |")
        lines += ["", "Patch rounds to best graph: the earliest round whose training pass rate was the unit's best "
                  "(0 is the initial program), averaged over repeats. T1 has no patch loop.", ""]
    bo = snapshot.get("bakeoff") or {}
    models = snapshot.get("models") or {}
    lines += ["## Run facts", "",
              "- Model ids: " + (", ".join(f"{k} = {v.get('id')} via {v.get('adapter')}" for k, v in models.items())
                                 or "none recorded"),
              "- Reasoning setting sent, per model and role: " + ("; ".join(f"{m}: {', '.join(v)}"
                                                                         for m, v in sorted(s["reasoning"].items()))
                                                               or "no model calls"),
              f"- Run date {date}, git commit {snapshot.get('git_commit')}",
              f"- Cache policy: {snapshot.get('cache_policy')}",
              _seeds_line(snapshot, bo),
              f"- Built from {s['events']} trace events under traces/ (model_call, verdict, patch, budget_stop). Unit "
              "checkpoints are used only for the list of units without results.", ""]
    return "\n".join(lines)


def _seeds_line(snapshot: dict, bo: dict) -> str:
    used = snapshot.get("seeds_used") or {}
    train = used.get("train") or _seed_text(bo.get("train_seeds"))
    held = used.get("heldout") or _seed_text(bo.get("heldout_seeds"))
    line = f"- Seeds: training {train}, held-out {held}, repeats {snapshot.get('repeats')}"
    if snapshot.get("seed_override"):
        line += (". The seeds were overridden for this run, so it is a smoke run, not the bake-off: the config's "
                 f"seeds are training {_seed_text(bo.get('train_seeds'))} and held-out {_seed_text(bo.get('heldout_seeds'))}")
    return line


def _seed_text(r: Any) -> str:
    if isinstance(r, dict) and "start" in r and "count" in r:
        return f"{r['start']} to {r['start'] + r['count'] - 1}"
    return str(r)


# ------------------------------------------------------------------------------ figure
def plot_pass_vs_cost(pm: pd.DataFrame, path: Path, dry: bool) -> Path:
    """One panel per task, one dot per model: x cost per held-out success, y pass rate with its Wilson interval.

    One mark color: the dots are entities, named by direct labels in ink (no color legend needed).
    A model with no success has no cost per success and is listed under its panel instead of plotted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    ink, muted, grid, mark, surface = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"
    tasks = sorted(pm["task"].unique()) if not pm.empty else ["T1", "T2", "T3"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.3 * len(tasks), 4.1), sharey=True, dpi=150, squeeze=False)
    fig.patch.set_facecolor(surface)
    for ax, task in zip(axes[0], tasks):
        ax.set_facecolor(surface)
        sub = pm[pm["task"] == task] if not pm.empty else pm
        shown = sub[sub["usd_per_success"].notna()] if not sub.empty else sub
        hidden = sub[sub["usd_per_success"].isna()] if not sub.empty else sub
        xs = shown["usd_per_success"].to_numpy(dtype=float) if len(shown) else np.array([])
        xmax = float(xs.max()) if len(xs) and xs.max() > 0 else 1.0
        log_x = len(xs) > 1 and xs.min() > 0 and xs.max() / xs.min() > 20  # costs a decade or more apart
        if log_x:
            ax.set_xscale("log")
            ax.set_xlim(xs.min() / 2, xs.max() * 6)
        else:
            ax.set_xlim(-0.05 * xmax, 1.45 * xmax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"${v:g}"))
        ax.set_ylim(-0.02, 1.06)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        placed: list[tuple[float, float]] = []
        for r, x in sorted(zip(shown.itertuples(), xs), key=lambda t: -t[0].pass_rate):
            y = r.pass_rate
            ax.errorbar([x], [y], yerr=[[max(0.0, y - r.ci_low)], [max(0.0, r.ci_high - y)]], fmt="none", ecolor=muted,
                        elinewidth=1.0, capsize=0, zorder=2)
            ax.plot([x], [y], "o", color=mark, markersize=8, markeredgecolor=surface, markeredgewidth=2, zorder=3)
            ly = y
            for px, py in placed:  # stack labels that would collide
                near = abs(math.log10(px / x)) < 0.25 if log_x else abs(px - x) < 0.3 * xmax
                if near and abs(py - ly) < 0.06:
                    ly = py - 0.065
            placed.append((x, ly))
            ax.annotate(r.model, (x, ly), xytext=(8, 0), textcoords="offset points", fontsize=8, color=ink,
                        va="center")
        if len(hidden):
            ax.text(0.0, -0.2, "no successes, not plotted: " + ", ".join(hidden["model"]), transform=ax.transAxes,
                    fontsize=7, color=muted)
        ax.set_title(task, color=ink, fontsize=10, loc="left")
        ax.set_xlabel("model USD per held-out success", color=muted, fontsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0][0].set_ylabel("held-out pass rate, Wilson 95%", color=muted, fontsize=8)
    fig.suptitle("Pass rate vs cost per success" + ("  (dry run: FakeModel, $0)" if dry else ""), color=ink,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------ entry points
def build_report(out: Path, *, copy_to_reports: bool = True) -> dict[str, str]:
    import yaml

    out = Path(out)
    snap_path = out / "config_snapshot.yaml"
    snapshot = yaml.safe_load(snap_path.read_text(encoding="utf-8")) if snap_path.is_file() else {}
    s = summarize(out, snapshot)
    held = s["heldout"]
    paths = {"report": str(out / "report.md"), "figure": str(out / "pass_vs_cost.png")}
    if not held.empty:
        held.to_parquet(out / "results.parquet", index=False)
        paths["results"] = str(out / "results.parquet")
    md = render_markdown(out, s, snapshot, unit_checkpoints(out))
    (out / "report.md").write_text(md, encoding="utf-8", newline="\n")
    plot_pass_vs_cost(s["per_model_task"], out / "pass_vs_cost.png", bool(snapshot.get("dry_run")))
    if copy_to_reports:
        dest = REPO_ROOT / "reports" / f"bakeoff_{out.name}.md"
        shutil.copyfile(out / "report.md", dest)
        paths["copied_to"] = str(dest)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the bake-off report from traces")
    ap.add_argument("--date", required=True, help="the bake-off folder name under LOOPKIT_DATA/bakeoff")
    ap.add_argument("--no-copy", action="store_true", help="do not copy report.md into reports/")
    args = ap.parse_args(argv)
    print(json.dumps(build_report(bakeoff_dir() / args.date, copy_to_reports=not args.no_copy), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
