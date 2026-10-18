"""S4: bake-off runner, report and cost estimate. No paid calls: the FakeModel or synthetic traces only."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from loopkit import paths
from loopkit.bakeoff import report as R
from loopkit.bakeoff import run as B
from loopkit.config import load_bakeoff
from loopkit.router import SpendCapExceeded
from loopkit.traces import TraceWriter


@pytest.fixture(autouse=True)
def _isolated_data(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOPKIT_DATA", str(tmp_path / "data"))


# ------------------------------------------------------------------------------ Wilson interval
def test_wilson_matches_known_values():
    lo, hi = R.wilson(0, 20)
    assert lo == 0.0 and abs(hi - 0.1611) < 1e-3
    lo, hi = R.wilson(20, 20)
    assert abs(lo - 0.8389) < 1e-3 and hi == 1.0
    lo, hi = R.wilson(30, 60)
    assert abs(lo - 0.3773) < 1e-3 and abs(hi - 0.6227) < 1e-3
    assert all(math.isnan(x) for x in R.wilson(0, 0))


# ------------------------------------------------------------------------------ report from traces
def _call(role: str, usd: float, inp: int, out: int, images: int = 0, ledger: float | None = None) -> dict:
    p = {"provider": "openrouter", "model": "m", "role": role, "input_tokens": inp, "output_tokens": out,
         "reasoning_tokens": None, "usd": usd, "usd_source": "provider", "latency_ms": 10.0, "cache_hit": False,
         "request_hash": "h" * 8, "image_count": images, "ok": True, "reasoning_setting": {"effort": "high"}}
    if ledger is not None:
        p["ledger_usd"] = ledger
    return p


def _verdict(passed: bool, seed: int, cls: str = "alignment", **extra) -> dict:
    v = {"passed": passed, "score": 1.0 if passed else 0.4, "failure_class": None if passed else cls,
         "failing_node": None if passed else "slide", "metrics": {}, "task_id": "T2", "seed": seed}
    return {**v, **extra}


def _write_unit(root, model: str, task: str, repeat: int, outcomes: list[bool], *, writer_usd: float,
                rounds: list[list[bool]] | None = None, reader_calls: int = 0):
    unit = B.unit_name(model, task, repeat)
    tw = TraceWriter(unit, root=root, meta={"model_key": model, "task_id": task, "repeat": repeat, "unit": unit,
                                            "kind": "unit", "dry_run": False})
    if task != "T1":
        tw.emit("model_call", _call("writer", writer_usd, 9000, 4000, images=3, ledger=writer_usd + 0.01))
        tw.emit("model_call", _call("writer", writer_usd, 6000, 3000))
        tw.emit("patch", {"round": 0, "graph_before_hash": None, "graph_after_hash": "g0", "failure_summary": ""})
    for i, ok in enumerate(outcomes):
        tw.emit("verdict", _verdict(ok, 2000 + i, split="heldout", sim_seconds=40.0 + i, model_calls=0,
                                    model_usd=0.0, graph_hash="g0", cached=False))
    tw.close()
    for r, round_out in enumerate(rounds or []):
        for j, ok in enumerate(round_out):
            ep = TraceWriter(f"{unit}_train_r{r}_s{1000 + j}", root=root, meta={"kind": "episode", "task_id": task})
            ep.emit("verdict", _verdict(ok, 1000 + j))
            ep.close()
    for j in range(reader_calls):
        ep = TraceWriter(f"{unit}_heldout_s{2000 + j}", root=root, meta={"kind": "episode", "task_id": task})
        ep.emit("model_call", _call("reader", 0.002, 2400, 300, images=1))
        ep.close()
    return unit


def _snapshot(out, models):
    import yaml

    snap = {"run_date": out.name, "dry_run": False, "git_commit": "abc1234", "cache_policy": "repeat in hash",
            "models": {m: {"id": f"vendor/{m}", "adapter": "openrouter"} for m in models}, "repeats": 2,
            "bakeoff": {"name": "rack_cell_v1", "train_seeds": {"start": 1000, "count": 10},
                        "heldout_seeds": {"start": 2000, "count": 20}}}
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(snap), encoding="utf-8")


def test_report_numbers_come_from_the_traces(tmp_path):
    out = tmp_path / "bakeoff" / "2026-10-18"
    traces = out / "traces"
    _write_unit(traces, "alpha", "T2", 1, [True] * 15 + [False] * 5, writer_usd=0.10,
                rounds=[[True, False], [True, True]])
    _write_unit(traces, "alpha", "T2", 2, [True] * 17 + [False] * 3, writer_usd=0.10, rounds=[[True, True]])
    _write_unit(traces, "beta", "T2", 1, [False] * 20, writer_usd=0.01, rounds=[[False, False], [False, False]])
    _write_unit(traces, "alpha", "T1", 1, [True] * 3 + [False], writer_usd=0.0, reader_calls=4)
    tw = TraceWriter(B.unit_name("beta", "T3", 1), root=traces,
                     meta={"model_key": "beta", "task_id": "T3", "repeat": 1, "kind": "unit"})
    tw.emit("budget_stop", {"scope": "model", "cap_usd": 2.0, "spent_usd": 2.01, "reason": "beta hit its cap"})
    tw.close()
    (out / "units").mkdir(parents=True)
    (out / "units" / "gamma__T2__r1.json").write_text(json.dumps(
        {"unit": "gamma__T2__r1", "status": "skipped_no_key", "reason": "no key: set OPENROUTER_API_KEY"}))
    _snapshot(out, ["alpha", "beta"])

    s = R.summarize(out)
    pm = s["per_model_task"].set_index(["model", "task"])
    a = pm.loc[("alpha", "T2")]
    assert a["runs"] == 40 and a["passed"] == 32 and a["pass_rate"] == 0.8
    lo, hi = R.wilson(32, 40)
    assert a["ci_low"] == pytest.approx(lo) and a["ci_high"] == pytest.approx(hi)
    assert a["repeat_mean"] == pytest.approx((0.75 + 0.85) / 2)
    assert a["repeat_sd"] == pytest.approx(pd.Series([0.75, 0.85]).std(ddof=1))
    # spend is the ledger figure when present: 2 units x (0.11 + 0.10)
    assert a["usd"] == pytest.approx(0.42) and a["usd_per_success"] == pytest.approx(0.42 / 32)
    assert a["setup_calls_per_unit"] == 2 and a["runtime_calls_per_run"] == 0
    assert a["patch_rounds_to_best"] == pytest.approx(0.5)  # round 1 in repeat 1, round 0 in repeat 2
    assert a["failure_classes"] == {"alignment": 8}
    assert a["sim_s_per_success"] == pytest.approx((sum(40 + i for i in range(15)) + sum(40 + i for i in range(17))) / 32)
    assert a["image_share_est"] == pytest.approx(2 * 3 * 1600 / (2 * 15000))
    b = pm.loc[("beta", "T2")]
    assert b["passed"] == 0 and math.isnan(b["usd_per_success"])
    t1 = pm.loc[("alpha", "T1")]
    assert t1["runtime_calls_per_run"] == 1 and t1["setup_calls_per_unit"] == 0
    assert t1["usd"] == pytest.approx(0.008)
    assert s["stops"] and s["stops"][0]["model"] == "beta"

    paths_out = R.build_report(out, copy_to_reports=False)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.splitlines()[2].startswith("Level 1: best held-out pass rate per task: T1 75% (alpha), T2 80% (alpha)")
    assert "Level 1.5:" in md and "15 to 18 points" in md and "## Level 2" in md
    assert "Stopped at a cap: beta" in md
    assert "gamma__T2__r1: skipped_no_key (no key: set OPENROUTER_API_KEY)" in md
    assert "| T2 | alpha | 32/40 | 80% |" in md
    assert 'alpha writer: {"effort":"high"}' in md
    res = pd.read_parquet(paths_out["results"])
    assert len(res) == 64 and set(res["model"]) == {"alpha", "beta"}
    assert (out / "pass_vs_cost.png").stat().st_size > 10_000


def test_report_with_no_results_still_renders(tmp_path):
    out = tmp_path / "bakeoff" / "empty"
    (out / "traces").mkdir(parents=True)
    R.build_report(out, copy_to_reports=False)
    assert "no held-out results" in (out / "report.md").read_text(encoding="utf-8")
    assert (out / "pass_vs_cost.png").is_file()


# ------------------------------------------------------------------------------ runner
def test_a_spend_cap_ends_the_write_and_the_run():
    from loopkit.program.writer import write_graph
    from loopkit.runtime.interpreter import run_graph
    from loopkit.schemas import ProgramGraph
    from loopkit.sim.tasks import task_spec
    from loopkit.skills import catalog
    from loopkit.skills.base import skill

    def capped(*args, **kwargs):
        raise SpendCapExceeded("cap", scope="model", cap_usd=1.0, spent_usd=1.0)

    with pytest.raises(SpendCapExceeded):
        write_graph("fake", task_spec("T2"), catalog(), None, chat=capped)

    @skill("other")
    def t_capped(ctx) -> None:
        """Hits the cap."""
        capped()

    class _Backend:
        sim_time = 0.0
        stopped = False

        def stop(self):
            self.stopped = True

    g = ProgramGraph.model_validate({
        "nodes": [{"id": "a", "skill": "t_capped", "params": {}}, {"id": "ok", "skill": "done", "params": {}},
                  {"id": "no", "skill": "abort", "params": {}}],
        "edges": [{"src": "a", "dst": "ok", "on": "success"}, {"src": "a", "dst": "no", "on": "failure"}],
        "entry": "a", "terminal_nodes": ["ok", "no"]})
    be = _Backend()
    with pytest.raises(SpendCapExceeded):
        run_graph(g, be, task_spec("T2", 1))
    assert be.stopped


def test_run_unit_records_a_budget_stop(monkeypatch, tmp_path):
    import loopkit.program.rehearse as reh

    def capped(*args, **kwargs):
        raise SpendCapExceeded("model alpha at its $2 cap", scope="model", cap_usd=2.0, spent_usd=2.0)

    monkeypatch.setattr(reh, "rehearse_and_patch", capped)
    out = tmp_path / "bk"
    (out / "traces").mkdir(parents=True)
    cfg = load_bakeoff()
    rec = B.run_unit(out, "alpha", "T2", 1, cfg, dry=False, workers=1, log=B.Tee(None))
    assert rec["status"] == "budget_stop"
    s = R.summarize(out)
    assert s["stops"] == [{"unit": "alpha__T2__r1", "model": "alpha", "reason": "model alpha at its $2 cap"}]


def test_models_without_a_key_are_skipped_and_retried_later(monkeypatch, tmp_path):
    import loopkit.router.smoke as smoke

    monkeypatch.setattr(smoke, "skip_reason", lambda entry: "no key: set OPENROUTER_API_KEY in .env")
    calls = []
    monkeypatch.setattr(B, "run_unit", lambda *a, **k: calls.append(a) or {})
    assert B.main(["--models", "opus", "--tasks", "T2", "--repeats", "1", "--date", "nokey", "--no-report"]) == 0
    unit = paths.bakeoff_dir() / "nokey" / "units" / "opus__T2__r1.json"
    assert json.loads(unit.read_text(encoding="utf-8"))["status"] == "skipped_no_key" and not calls
    assert B._read_unit(unit)["status"] not in B.FINISHED  # a rerun with the key runs it


@pytest.mark.skipif(not (paths.panda_dir() / "panda.xml").is_file(), reason="Menagerie Panda not cloned")
def test_dry_run_end_to_end_and_resume():
    args = ["--dry-run", "--models", "opus", "--tasks", "T2", "--repeats", "1", "--train-seeds", "1000",
            "--heldout-seeds", "600", "--date", "dry_test", "--workers", "1", "--no-report"]
    assert B.main(args) == 0
    out = paths.bakeoff_dir() / "dry_test"
    rec = json.loads((out / "units" / "opus__T2__r1.json").read_text(encoding="utf-8"))
    assert rec["status"] == "done" and len(rec["heldout"]) == 1 and rec["route"] == "fake"
    assert (out / "graphs" / "opus__T2__r1.json").is_file()
    snap = (out / "config_snapshot.yaml").read_text(encoding="utf-8")
    assert "seed_override: true" in snap
    R.build_report(out, copy_to_reports=False)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "(dry run)" in md and "| T2 | opus |" in md and "smoke run, not the bake-off" in md
    # a rerun skips the finished unit
    assert B.main(args) == 0
    assert "skip opus__T2__r1 (done)" in (out / "run.log").read_text(encoding="utf-8")


def test_dry_run_never_scores_on_the_bakeoff_heldout_seeds(monkeypatch):
    seen = {}

    def fake_unit(out, model, task, repeat, cfg, **kw):
        seen["heldout"] = kw.get("heldout_override")
        return {"status": "done", "wall_seconds": 0.0}

    monkeypatch.setattr(B, "run_unit", fake_unit)
    B.main(["--dry-run", "--models", "opus", "--tasks", "T2", "--repeats", "1", "--date", "dry_seeds", "--no-report"])
    assert seen["heldout"] and min(seen["heldout"]) >= 600 and max(seen["heldout"]) < 1000


# ------------------------------------------------------------------------------ estimate
def test_estimate_is_bounded_by_the_caps():
    from loopkit.bakeoff.estimate import estimate, format_estimate

    cfg = load_bakeoff()
    res = estimate(cfg)
    assert 0 < res["total_typical"] <= res["total_bounded_worst"] <= cfg.total_cap_usd
    for r in res["rows"]:
        assert r.typical_usd < r.worst_usd and r.bounded_worst <= cfg.caps_usd[r.model]
        assert set(r.per_task) == set(cfg.tasks)
    text = format_estimate(res, cfg)
    assert "Estimated total" in text and "no measured run" in text


def test_a_unit_without_a_valid_graph_fails_every_heldout_seed(monkeypatch, tmp_path):
    import loopkit.program.rehearse as reh
    from loopkit.program.rehearse import RehearsalResult, RoundRecord

    def no_graph(model_key, task, *a, **k):
        rounds = [RoundRecord(0, None, 0.0, 0, 3, False, 4, 0.0, 4, note="no valid graph: bad_json")]
        return RehearsalResult(task.id, model_key, None, 0, 0.0, rounds, "no valid initial graph", 0.0, 4)

    monkeypatch.setattr(reh, "rehearse_and_patch", no_graph)
    out = tmp_path / "bk"
    (out / "traces").mkdir(parents=True)
    rec = B.run_unit(out, "alpha", "T2", 1, load_bakeoff(), dry=False, workers=1, log=B.Tee(None),
                     heldout_override=[600, 601, 602])
    assert rec["status"] == "no_graph" and [r["passed"] for r in rec["heldout"]] == [False] * 3
    s = R.summarize(out)
    row = s["per_model_task"].iloc[0]
    assert (row["model"], row["runs"], row["passed"]) == ("alpha", 3, 0)
    assert s["no_graph_units"] == ["alpha__T2__r1"]
    md = R.render_markdown(out, s, {}, [])
    assert "No valid graph (scored as a failure on every held-out seed): alpha__T2__r1." in md


def test_wilson_is_exact_at_the_edges_and_the_figure_takes_a_perfect_score(tmp_path):
    for n in (1, 20, 60, 61, 997):
        assert R.wilson(n, n)[1] == 1.0 and R.wilson(0, n)[0] == 0.0
    lo, hi = R.wilson(60, 60)
    pm = pd.DataFrame([{"task": "T2", "model": "alpha", "pass_rate": 1.0, "ci_low": lo, "ci_high": hi,
                        "usd_per_success": 0.0}])
    R.plot_pass_vs_cost(pm, tmp_path / "f.png", True)
    assert (tmp_path / "f.png").stat().st_size > 5_000
