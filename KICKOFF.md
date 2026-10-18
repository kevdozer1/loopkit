# loopkit kickoff: build S0 to S4 in one run

You are the coding agent for loopkit, Kevin Hopkins' personal research repo. Build the full sim loop and the bake-off runner this weekend, phase by phase, without waiting on Kevin between phases. Stop before the overnight bake-off and hand Kevin one command to start it.

## 0. Before you write any code

Read these in order. They are the spec. Where they disagree, the earlier file wins.

1. `CLAUDE.md`, the rules
2. `docs/01_architecture.md`, the interfaces
3. `docs/02_task_suite.md`, the scene and tasks
4. `docs/03_bakeoff_protocol.md`
5. `docs/05_machine_gotchas.md`
6. `docs/06_reuse_map.md`, then skim the statebench and robolabel files it names
7. `docs/04_prior_art.md`
8. `sessions/S0.md` to `sessions/S4.md`
9. Optional, for intent only: `context/README.md`. Never copy from `context/` into committed files.

If `reports/PROGRESS.md` exists, you are resuming. Read it and every `reports/S*.md`, run `git log --oneline -20` and `uv run pytest -q`, then continue from the first phase that is not marked done.

## 1. The phases

Run each phase by following its session file. A phase is done only when its acceptance check passes and is shown with real numbers.

| Phase | Session file | Cap | Gate to move on |
|---|---|---|---|
| S0 | sessions/S0.md | $0 | pytest passes, smoke.png shows the Panda |
| S1 | sessions/S1.md | $1 | tests pass, live table prints for every model with a key, models.yaml verified where possible |
| S2 | sessions/S2.md | $0 | oracle at least 95% on 50 dev seeds for T2 and T3 (T3 may use the 1.0 mm fallback), do-nothing scores 0, determinism holds, 3 mp4s |
| S3 | sessions/S3.md | $15 | T2 at least 80% on seeds 500 to 519 within 5 rounds, or the honest number after the allowed fixes |
| S4 | sessions/S4.md | $5 | FakeModel dry run gives a full report, real 2-model T2 run under $5, full-run estimate printed |

Total cap for this whole run: $21. The router ledger enforces it. Set the active session cap at the start of each phase.

After each phase:
1. Run the full test suite.
2. Commit with a message like `S2: rack_cell scene, T1-T3 checkers, oracles 98% T2 / 96% T3`.
3. Write `reports/S<n>.md`: what ran, results with numbers, open issues, Kevin's exact next action.
4. Update `reports/PROGRESS.md`: a table of phases with status (done, partial, blocked), the headline number, spend so far, and the time.

Then start the next phase immediately.

## 2. When something goes wrong

- **Missing API key.** Skip that model, list it in the report and in PROGRESS.md, and keep going. S3 needs at least one writer key. If there is none, finish S3 with FakeModel only, mark the gate blocked, and still build S4 with its dry run.
- **A gate fails.** Try the stop rule in the session file. If the gate still fails after the allowed attempts, mark the phase partial with the honest number and move on, as long as the next phase can run on what exists. Never fake, lower or reword a gate to pass it.
- **A tool or command is denied.** Do not retry the same command in a different form. Note what was denied and why you needed it in PROGRESS.md, choose a path that does not need it, and continue.
- **Network.** Allowed: pypi, github (Menagerie clone), openrouter.ai, platform.elorian.ai, api.typesafe.ai and provider doc pages. Nothing else is needed.
- **Disk floor.** Below 10 GB free on C: or D:, stop and report.
- **Context getting long.** Keep PROGRESS.md current so a resumed session loses nothing.

## 3. What you must not do

- Push, add a remote, or touch any other repo. Read statebench and robolabel only as docs/06_reuse_map.md allows.
- Write outside `C:\Users\Kevin\projects\loopkit` and `D:\loopkit_data`.
- Print, log or commit a key. Commit anything under `context/`.
- Start the full bake-off, or any paid run beyond the session caps.
- Put company names, employer details or anything from `context/` into code, configs, commits or reports.

## 4. How to finish

When S4 is done (or the furthest phase you can reach):

1. `uv run pytest -q` once more and paste the summary line into PROGRESS.md.
2. Final commit.
3. End with a message to Kevin in this shape, plain language, no em dashes, no semicolons:

```
Level 1: <one sentence on where the loop stands>
Phases: S0 <status, number>, S1 ..., S2 ..., S3 ..., S4 ...
Spend: $<total> of $21
Blocked or skipped: <list, or none>
Overnight bake-off: estimated $<x>. Start it with:
  powershell -ExecutionPolicy Bypass -File setup\start_bakeoff.ps1
Your next action: <one exact action>
```
