# loopkit build summary (S0 to S4)

Built 2026-10-17 from KICKOFF.md in one session. This page pulls together the per-phase reports (S0.md to S4.md), PROGRESS.md and the dry-run bake-off report. Where a section draws on one report, it names it.

## Where it stands

The loop is built and runs end to end. A model writes a program graph from a typed skill library, the graph is rehearsed in MuJoCo, a privileged checker scores it, failures go back to the writer as text, and every call and verdict is traced. A bake-off runner repeats that across models and builds a report from the traces.

No real model has been called yet. Every key in `.env` is empty, so every model call so far went to the FakeModel. As the writer it returns a hand-written reference graph, and as the T1 panel reader it returns the smallest answer its schema allows. Those numbers measure the skill library and the pipeline, not any model. The simulator checks in S2 use privileged oracles and no model at all. Spend is $0 of the $21 build cap.

The overnight bake-off is ready to start once a key is in place. It is estimated at $27.16 typical and at most $40.27, because the ledger stops each model at its cap.

## Phase status

| Phase | Status | Gate in KICKOFF.md | Result |
|---|---|---|---|
| S0 bootstrap | done | pytest passes, smoke.png shows the Panda | 15 tests passed, the Panda renders |
| S1 schemas, router, traces | partial | tests pass, live table prints for every model with a key, models.yaml verified where possible | all tests pass after three review rounds, and models.yaml was checked against the OpenRouter catalogue and provider docs. The live smoke called 0 of 9 models: no key is set |
| S2 sim task suite | done | oracle at least 95% on 50 dev seeds for T2 and T3, do-nothing scores 0, determinism, 3 mp4s | oracle T2 50/50 and T3 50/50, do-nothing 0 on all 3 tasks, determinism holds, 3 mp4s written |
| S3 skills, writer, rehearsal | blocked | T2 at least 80% on seeds 500 to 519 within 5 rounds, or the honest number after the allowed fixes (needs a writer key) | no writer key. With the FakeModel's reference graph: T2 20/20 after one skill-fix round (19/20 before), T3 20/20 |
| S4 bake-off runner | partial | dry run gives a full report, real 2-model T2 run under $5, full-run estimate printed | dry run: full report and figure. Real run blocked (no key). Estimate printed from real prompt sizes |

Spend in every phase: $0. The ledger holds 422 FakeModel calls (51 writer, 371 reader) at $0 and no paid call.

## What was built

| Part | Where | What it does |
|---|---|---|
| Schemas | `loopkit/schemas` | TaskSpec, SkillSpec, ProgramGraph, Observation, Verdict, TraceEvent (pydantic v2), JSON Schema export, and a graph validator that returns every error with a code and a fix hint |
| Router | `loopkit/router` | `chat()` over OpenRouter, Elorian, the System One decisions protocol (Jev, Liquid d1) and the FakeModel. Disk cache, a spend ledger that reserves every HTTP attempt before it is sent, session, per-model and total caps, one trace event per call |
| Traces | `loopkit/traces` | Append-only JSONL per run with one locked writer, and `load_runs()` into a typed pandas table |
| Simulator | `loopkit/sim` | The rack_cell MuJoCo scene (Menagerie Panda, 4U rack, 2 slots, 12-port patch panel with LEDs and labels, parts tray), SimBackend with 3 cameras, T1 to T3 TaskSpecs, checkers, oracles and the episode grader |
| Skills | `loopkit/skills` | 14 typed skills: classical-CV perception, motion, manipulation with force guards and search, a wrist-camera calibration dance, `read_panel` through the router, `done` and `abort` |
| Interpreter | `loopkit/runtime` | Runs a ProgramGraph on a Backend: refs, retries, visit bounds, timeouts, safety stops, trace events. SO-101 backend stub |
| Writer and rehearsal | `loopkit/program` | Writer prompt with validate-and-repair (up to 3 repairs), rehearse-and-patch with text failure summaries, and a CLI for the S3 acceptance run |
| Bake-off | `loopkit/bakeoff`, `setup/start_bakeoff.ps1` | Resumable, capped runner, a report built from traces, a cost estimate, and the overnight start script |

## Results

### Simulator (S2.md)

| Check | Result |
|---|---|
| T2 oracle, dev seeds 0 to 49 | 50/50 |
| T3 oracle, dev seeds 0 to 49 | 50/50 |
| Wider sweep, seeds 50 to 249 | T2 200/200, T3 199/200 (seed 152: a 66.9 N safety stop in the pre-grasp move) |
| Do-nothing, seeds 0 to 49 | 0/50 on T1, T2 and T3, max score 0 |
| Determinism | same seed gives identical MJCF, model arrays and initial Observation (joint state and all 3 camera images, byte for byte) |
| Sim seconds per oracle episode | T1 2.0, T2 38.5 mean (39.9 max), T3 35.7 mean (36.5 max). Budgets: 20 s for T1, 90 s for T2 and T3 |

The plug fits its port with 0.5 mm total clearance, as specified, so the 1.0 mm fallback was not needed. The 1U box is scaled to 150 x 200 x 44.45 mm and 1.0 kg so the Panda can carry it.

### Skill library with the scripted writer (S3.md)

| Run | Training seeds 100 to 109 | Held-out seeds 500 to 519 |
|---|---|---|
| T2, original library | 9/10 | 19/20 |
| T2, after one skill-fix round | 10/10 | 20/20 |
| T3, both versions | 10/10 | 20/20 |

After the fix, the T2 reference graph also passed 100/100 on dev seeds 100 to 199. Those seeds include the training seeds and seeds 103 and 146, which showed the problem, so they are not independent of the fix.

The fix: the box's position along its 200 mm width came from a thin bezel strip (0.37 mm std, 2.0 mm worst against simulator truth), while the rack rails leave 1 mm per side. It now comes from the whole top face (0.07 mm std, 0.33 mm worst). `slide_into` also gained a guarded rail entry with a sideways search, and it now fails when the box is not seated instead of reporting success.

### Bake-off dry run (S4.md, bakeoff_2026-10-17_dry.md)

All 6 main models, T1 to T3, 3 repeats: 54 units and 1,080 scored runs in 11.5 minutes at $0, with the FakeModel in every model slot. Dev seeds 600 to 619 stood in for the held-out seeds. The run predates the code that adds the optional luna model, so a dry run now would also cover luna.

| Task | Every model slot | Wilson 95% |
|---|---|---|
| T1 | 0/60 | 0% to 6% (by construction: the FakeModel reader does not read the image) |
| T2 | 60/60 | 94% to 100% |
| T3 | 60/60 | 94% to 100% |

### Cost estimate for the overnight run (S4.md)

| Model | Typical | Cap |
|---|---|---|
| opus | $5.80 | $8 |
| astra | $14.51 | $15 |
| sol | $2.90 | $5 |
| argon | $2.90 | $8 |
| glimpse | $0.87 | $3 |
| cheap_open | $0.03 | $2 |
| luna (optional, included) | $0.15 | $1 |
| Total | $27.16 | ceiling $40.27, under the $50 total cap |

This is a projection from the real prompt sizes with assumed output tokens, not a measured run.

## Review and hardening

Three review rounds ran on the router, schemas and traces (S1.md). Round 1 was part of S1, before its commit. Two more followed: round 2 covered all three areas, and round 3 covered the router and schemas. Each round had one fix agent and one independent verifier per area. The verifiers re-ran every earlier repro, tried to break the fixes and ran the full suite. Every fix in rounds 2 and 3 has a regression test that fails on the code before it. The fixes that mattered most for an unattended paid run:

- A refused connection held its full projected cost per attempt, so a network blip could fill a model's cap with spend that never happened. It now settles at $0.
- The answer parser's code-fence regex was cubic: one degenerate answer (a fence and 2,000 spaces) took 11.6 s. A linear scan now parses that answer in under a millisecond, and it returns the same results as the old regex on 400,000 random strings.
- Several paths could carry the API key into a trace, a cache entry or an error message (escaped spellings, response metadata, clipped errors). All are cut now, except one low-severity case listed in S1.md: a key written with escapes in prose outside any JSON value is kept as written.
- A Ctrl+C during a trace writer's open or close could leave the run locked. Under stress: 0 of 3,000 interrupted open/close cycles and 0 of 6,000 interrupted closes leak, down from 111 and 40.

Defects caught while building and testing S3 and S4, all fixed:

- `slide_into` reported success with the box 38.7 mm short of the back stop.
- The writer swallowed a spend-cap stop, so a capped model would never have recorded where it stopped (caught before the S3 commit).
- The T1 episode cache was keyed on the route, which is "fake" for every model in a dry run, so models would have shared answers (caught in a small dry run before the S4 commit).
- The report crashed at a perfect score, because the Wilson upper bound rounded to just under 1.
- The reader ignored the `reader:` settings in bakeoff.yaml (fixed in 4994f5f).
- A unit whose writer never produced a valid graph would have been left out of the pass rates (caught before the S4 commit). It now counts as a failure on every held-out seed, and the report names it.

The perfect-score crash, the spend-cap stop and the no-graph unit have regression tests. The reader settings, the T1 cache key and the not-seated failure in `slide_into` do not yet (`tests/test_skills.py` covers the box-localization half of that fix).

## Safety and honesty notes

- The bake-off held-out seeds 2000 to 2019 have never been used. Dry runs default to dev seeds 600 to 619.
- Neither S3 number is a clean held-out result. Seed 512 is in the S3 held-out set and prompted the skill fix, so the post-fix T2 20/20 on seeds 500 to 519 is not clean. The dev sweep is not clean either: seeds 100 to 199 include seeds 103 and 146, which also showed the problem, and seeds 100 to 159 were used to measure the localization error. The bake-off held-out seeds will give a clean number.
- `setup/start_bakeoff.ps1` was checked with PowerShell's parser but never run, since running it would start the bake-off.
- No key was printed, logged or committed, `context/` was never committed, and nothing was pushed.
- One tool call was denied. During the review, a verifier subagent's PowerShell `Stop-Process` on its own hung scratch process was refused, and the subagent then used `taskkill` on the same processes, which breaks KICKOFF's rule against retrying a denied action in another form. Nothing outside the scratchpad was affected. PROGRESS.md records it.

## Open issues

- No live model call has been made, so each provider adapter is tested only against mocked HTTP, and no real model has seen the writer prompt yet.
- glimpse's id `elorian-1` comes from the platform docs. Its live model list needs a key and was never read.
- The typical cost estimate rests on assumed output tokens. Reasoning models at their highest setting may write more, which the caps bound.
- astra's typical estimate sits at its $15 cap, so it may stop before its last T3 repeat. Raising the cap to $20 keeps the ceiling at $45.27. The config was not edited.
- The writer's 900 s deadline also covers connecting. A shorter connect timeout would fail faster on a dead route.
- Occupied-port plugs are the same yellow as amber LEDs. A reader model could confuse them on a poorly lit seed.
- Three fixes have no regression test yet (listed under Review and hardening).
- CLAUDE.md rule 1 names the employer in the sentence that forbids naming it. Suggested wording: "Nothing here names any employer."
- `claude --version` reports 2.1.81, and README step 3 asks for 2.1.259 or later for `setup\run_session.ps1`.
- Low-severity residuals from the last review round are listed in S1.md.

## What to do next

1. Make a dedicated OpenRouter key for loopkit with a credit limit of about $70, and put it in `C:\Users\Kevin\projects\loopkit\.env` as `OPENROUTER_API_KEY`.
2. Live smoke: `uv run python -m loopkit.router.smoke`
3. S3 gate: `uv run python -m loopkit.program.cli --model opus --task T2 --train-seeds 100-109 --heldout-seeds 500-519`
4. S4 smoke on dev seeds: `uv run python -m loopkit.bakeoff.run --models opus cheap_open --tasks T2 --repeats 1 --heldout-seeds 600-619 --date s4_smoke --session-cap 5`, then `uv run python -m loopkit.bakeoff.estimate --from <LOOPKIT_DATA>\bakeoff\s4_smoke` for a measured estimate.
5. Decide on the astra cap.
6. Overnight bake-off: `powershell -ExecutionPolicy Bypass -File setup\start_bakeoff.ps1`

Run the commands from the repo root.

## By the numbers

- 13 commits in the build session, from `cf18348` (spec) to `1636ab3` (final status).
- 69 Python modules, 14,495 lines in `loopkit/`: router 4,328, schemas 2,596, sim 2,234, traces 1,500, skills 1,347, bakeoff 1,069, program 672, runtime 328, and 421 in the top-level `config.py`, `paths.py` and `__init__.py`.
- 593 tests in 7,391 lines, all passing in the final run: schemas 343, router 89, traces 82, paths and config 25, sim 18, program 12, skills 12, bakeoff 11, smoke 1.

## Where things are

| What | Where |
|---|---|
| Per-phase reports and status | `reports/S0.md` to `reports/S4.md`, `reports/PROGRESS.md` |
| Dry-run bake-off report | `reports/bakeoff_2026-10-17_dry.md`, with the figure, traces and `results.parquet` in `LOOPKIT_DATA/bakeoff/2026-10-17_dry/` |
| Oracle runs and videos | `LOOPKIT_DATA/runs/oracles/` (seeds 0 to 49), `LOOPKIT_DATA/videos/` (T1, T2 and T3 oracle mp4s, seed 7) |
| S3 FakeModel runs | `LOOPKIT_DATA/runs/s3_fake_T2/` and `s3_fake_T3/` (original library), `s3_fake_T2_v2/` and `s3_fake_T3_v2/` (after the skill fix) |
| Spend ledger and caps | `LOOPKIT_DATA/spend_ledger.jsonl`, `LOOPKIT_DATA/spend_caps.json` |

`LOOPKIT_DATA` defaults to `D:\loopkit_data`.

## Commit history (build session)

| Commit | What |
|---|---|
| cf18348 | S0: initial spec, docs, configs and session prompts |
| 64f20ba | S0: uv package, paths and config, Panda smoke render |
| 2360d48 | S1: schemas, router and traces, reviewed and fixed |
| 935bbcb | S2: rack_cell scene, SimBackend, checkers, oracles |
| 7391022 | S3: skill library, interpreter, writer and rehearse-and-patch |
| bb187d6 | S4: bake-off runner, report and cost estimate, dry run |
| 4994f5f | S4: T1 reader settings from bakeoff.yaml |
| 7478a71 | S4: keep Windows awake during a bake-off run |
| 93bf941 | S1 follow-up: review round 2, router and schemas |
| 65eefe7 | S1 follow-up: trace writer survives Ctrl+C |
| c059d3c | S1 follow-up: review round 3, router and schemas |
| 3f75611 | S1 follow-up: drop the provider body when an answer held the key, writer finds the graph after a JSON example in prose |
| 1636ab3 | Final: PROGRESS with the last test run |
