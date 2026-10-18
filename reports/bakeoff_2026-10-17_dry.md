# Bake-off rack_cell_v1 2026-10-17 (dry run)

**Dry run.** Every model slot was played by the FakeModel: the writer returns the hand-written reference graph and the T1 reader returns the smallest answer its schema allows. The numbers describe the skill library and the pipeline at $0, not any model.

Level 1: best held-out pass rate per task: T1 0% (argon, astra, cheap_open, glimpse, opus and sol tied), T2 100% (argon, astra, cheap_open, glimpse, opus and sol tied), T3 100% (argon, astra, cheap_open, glimpse, opus and sol tied).

Level 1.5: 6 models ran T1, T2, T3 on held-out seeds, 1080 scored runs in all, for $0.0000 of model calls. On T1 every model call is a runtime call: the model reads the panel once per run. On T2 and T3 the model only writes the program. The graph then runs with no model in the loop, so runtime calls are 0 and all cost is setup. Power: pooled over 3 repeats a model and task has 60 held-out runs, so a gap smaller than about 15 to 18 points is within noise. On one repeat of 20 runs it is about 30 points.

## Level 2

### Held-out pass rate

| Task | Model | Passed | Pass rate | Wilson 95% | Repeats | Repeat mean | Repeat sd |
|---|---|---|---|---|---|---|---|
| T1 | argon | 0/60 | 0% | 0% to 6% | 3 | 0% | 0 pts |
| T1 | astra | 0/60 | 0% | 0% to 6% | 3 | 0% | 0 pts |
| T1 | cheap_open | 0/60 | 0% | 0% to 6% | 3 | 0% | 0 pts |
| T1 | glimpse | 0/60 | 0% | 0% to 6% | 3 | 0% | 0 pts |
| T1 | opus | 0/60 | 0% | 0% to 6% | 3 | 0% | 0 pts |
| T1 | sol | 0/60 | 0% | 0% to 6% | 3 | 0% | 0 pts |
| T2 | argon | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T2 | astra | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T2 | cheap_open | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T2 | glimpse | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T2 | opus | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T2 | sol | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T3 | argon | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T3 | astra | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T3 | cheap_open | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T3 | glimpse | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T3 | opus | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |
| T3 | sol | 60/60 | 100% | 94% to 100% | 3 | 100% | 0 pts |

Power: pooled over 3 repeats a model and task has 60 held-out runs, so a gap smaller than about 15 to 18 points is within noise. On one repeat of 20 runs it is about 30 points.

### Cost, calls and tokens

| Task | Model | Model spend | Cost per success | Setup calls per unit | Runtime calls per run | Failed calls | Input tokens | Output tokens | Image share of input (est.) |
|---|---|---|---|---|---|---|---|---|---|
| T1 | argon | $0.0000 | n/a | 0.0 | 1.00 | 0 | 52,800 | 7,440 | n/a |
| T1 | astra | $0.0000 | n/a | 0.0 | 1.00 | 0 | 52,800 | 7,440 | n/a |
| T1 | cheap_open | $0.0000 | n/a | 0.0 | 1.00 | 0 | 52,800 | 7,440 | n/a |
| T1 | glimpse | $0.0000 | n/a | 0.0 | 1.00 | 0 | 52,800 | 7,440 | n/a |
| T1 | opus | $0.0000 | n/a | 0.0 | 1.00 | 0 | 52,800 | 7,440 | n/a |
| T1 | sol | $0.0000 | n/a | 0.0 | 1.00 | 0 | 52,800 | 7,440 | n/a |
| T2 | argon | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 16,023 | 1,071 | 90% |
| T2 | astra | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 16,023 | 1,071 | 90% |
| T2 | cheap_open | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 16,023 | 1,071 | 90% |
| T2 | glimpse | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 16,023 | 1,071 | 90% |
| T2 | opus | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 16,023 | 1,071 | 90% |
| T2 | sol | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 16,023 | 1,071 | 90% |
| T3 | argon | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 18,165 | 1,062 | 79% |
| T3 | astra | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 18,165 | 1,062 | 79% |
| T3 | cheap_open | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 18,165 | 1,062 | 79% |
| T3 | glimpse | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 18,165 | 1,062 | 79% |
| T3 | opus | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 18,165 | 1,062 | 79% |
| T3 | sol | $0.0000 | $0.0000 | 1.0 | 0.00 | 0 | 18,165 | 1,062 | 79% |

### Spend per model, every unit

| Model | Spend | Cap |
|---|---|---|
| argon | $0.0000 | $8.00 |
| astra | $0.0000 | $15.00 |
| cheap_open | $0.0000 | $2.00 |
| glimpse | $0.0000 | $3.00 |
| opus | $0.0000 | $8.00 |
| sol | $0.0000 | $5.00 |

Setup calls are writer calls (the initial program, patches and repairs). Runtime calls are model calls made while a held-out run executes. Image share is an estimate: images sent times the model's image token estimate, over the input tokens the provider reported.

### Cycle time, patching and failures

| Task | Model | Sim seconds per success | Patch rounds to best graph | Failure classes |
|---|---|---|---|---|
| T1 | argon | n/a | n/a | perception 60 |
| T1 | astra | n/a | n/a | perception 60 |
| T1 | cheap_open | n/a | n/a | perception 60 |
| T1 | glimpse | n/a | n/a | perception 60 |
| T1 | opus | n/a | n/a | perception 60 |
| T1 | sol | n/a | n/a | perception 60 |
| T2 | argon | 49.3 | 0.0 | none |
| T2 | astra | 49.3 | 0.0 | none |
| T2 | cheap_open | 49.3 | 0.0 | none |
| T2 | glimpse | 49.3 | 0.0 | none |
| T2 | opus | 49.3 | 0.0 | none |
| T2 | sol | 49.3 | 0.0 | none |
| T3 | argon | 45.1 | 0.0 | none |
| T3 | astra | 45.1 | 0.0 | none |
| T3 | cheap_open | 45.1 | 0.0 | none |
| T3 | glimpse | 45.1 | 0.0 | none |
| T3 | opus | 45.1 | 0.0 | none |
| T3 | sol | 45.1 | 0.0 | none |

Patch rounds to best graph: the earliest round whose training pass rate was the unit's best (0 is the initial program), averaged over repeats. T1 has no patch loop.

## Run facts

- Model ids: opus = anthropic/claude-opus-5.5 via openrouter, astra = openai/gpt-6-astra via openrouter, sol = openai/gpt-6.1-sol via openrouter, argon = google/gemini-4-argon via openrouter, glimpse = elorian-1 via elorian, cheap_open = nex-agi/nex-n2.5-mini via openrouter
- Reasoning setting sent, per model and role: argon reader: {"enabled":false}; argon writer: {"effort":"high"}; astra reader: {"enabled":false}; astra writer: {"effort":"high"}; cheap_open reader: {"enabled":false}; cheap_open writer: {"effort":"high"}; glimpse reader: {"enabled":false}; glimpse writer: {"effort":"high"}; opus reader: {"enabled":false}; opus writer: {"effort":"high"}; sol reader: {"enabled":false}; sol writer: {"effort":"high"}
- Run date 2026-10-17, git commit 64f20ba
- Cache policy: router disk cache on; the repeat index is part of every writer and reader request hash, so repeats never share cache entries; held-out episode results are cached by (task, seed, graph hash) because the simulator is deterministic (T1 adds reader and repeat to the key)
- Seeds: training 1000 to 1009 (10 seeds), held-out 600 to 619 (20 seeds), repeats 3. The seeds were overridden for this run, so it is a smoke run, not the bake-off: the config's seeds are training 1000 to 1009 and held-out 2000 to 2019
- Built from 9312 trace events under traces/ (model_call, verdict, patch, budget_stop). Unit checkpoints are used only for the list of units without results.
