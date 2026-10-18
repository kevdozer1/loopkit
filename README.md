# loopkit

A small model-agnostic loop for robot task programs. Any frontier model writes a program graph from a typed skill library, the graph is rehearsed in MuJoCo, a checker scores it, failures go back to the writer as text, and every call and verdict is traced. A bake-off runs the loop across models and reports pass rate, cost per success and big-model calls.

Personal project and prior IP of Kevin Hopkins.

## Start here (Kevin)

1. Copy `.env.example` to `.env`. Fill in `OPENROUTER_API_KEY` (a new key just for loopkit with a credit limit is best), `ELORIAN_API_KEY` and `TYPESAFE_JEV_API_KEY` (same as statebench's).
2. Merge `setup\claude_settings_snippet.json` into `C:\Users\Kevin\.claude\settings.json`. autoMode is only read from user settings, so it has to go there.
3. Check `claude --version` is 2.1.259 or later.
4. Start the build. Either:
   - Watch it: `cd C:\Users\Kevin\projects\loopkit`, run `claude --permission-mode auto`, and paste `Read KICKOFF.md and follow it exactly.`
   - Leave it: `powershell -ExecutionPolicy Bypass -File setup\run_session.ps1`
5. When it ends, read `reports\PROGRESS.md`, then start the overnight bake-off with the command it prints.

## What is in the folder

| Path | What it is |
|---|---|
| `KICKOFF.md` | The coding prompt. Builds S0 to S4 with gates, stops before the overnight run |
| `CLAUDE.md` | Rules every agent session follows |
| `docs/` | The spec: architecture, task suite, bake-off protocol, prior art, machine gotchas, reuse map |
| `config/` | `models.yaml` (ids and prices, some to verify in S1) and `bakeoff.yaml` |
| `sessions/` | S0 to S7 plus the optional GaP study, one prompt per session |
| `setup/` | Settings snippet and the headless runner |
| `reports/` | One report per session, PROGRESS.md, bake-off reports |
| `context/` | Private reference (gitignored): this week's research docs and podcast blueprints |

## Session order

S0 bootstrap, S1 core interfaces, S2 sim tasks, S3 skills and writer, S4 bake-off runner (KICKOFF runs these 5), then the overnight bake-off as plain Python, then S5 SO-101 demo, S6 exporters, S7 memo. Run S5 to S7 one at a time with `setup\run_session.ps1 -Session S6`, or interactively for S5 and S7.
