# 03. Bake-off protocol

Level 1: every model gets the same tasks, skills, images and budget, with 3 repeats on held-out seeds. The report shows the numbers a deployment cares about and public leaderboards leave out: big-model calls per task, cost per success, and how quickly a model fixes its own failures.

## Models and roles

| Role | Models | What it answers |
|---|---|---|
| Program writer (T2, T3) | opus, astra, sol, argon, glimpse, cheap_open (keys in `config/models.yaml`) | Who writes the best program, at what cost |
| Panel reader (T1) | The same 6 | Whether cheap visual reasoning holds up on LEDs and labels |
| Runtime branches (S6 probe) | d1, jev, a small local classifier, with the frontier choice as reference | Whether a System 1 model can take over repeated calls |

## Rules

1. Same TaskSpec, skill catalog, camera images and spend cap for every model. Use each API's highest reasoning setting for the writer and record it.
2. 10 training seeds (1000 to 1009) for rehearse-and-patch, 20 held-out seeds (2000 to 2019) for the score, 3 repeats. Held-out seeds are never used before the bake-off.
3. Record exact model ids, the run date and the router cache policy. Repeats must not share cache entries: include the repeat index in the request hash.
4. Know the power. Pooled over 3 repeats that is 60 held-out runs per model and task, so a gap smaller than about 15 to 18 points is within noise. On one repeat of 20 runs it is about 30 points. Say this in the report next to any comparison.
5. A model that hits its cap stops, and the report says where it stopped.
6. Checkpoint after every unit (model, task, repeat) so a crash resumes where it stopped.

## Metrics

| Metric | Why it matters |
|---|---|
| Held-out pass rate with a Wilson 95% interval | The headline, with its uncertainty attached |
| Mean and spread over the 3 repeats | How stable a model's programs are |
| Cost per success | What a deployment pays, failures included |
| Big-model calls, setup vs runtime | The number Kulakov's design drives toward 0 at runtime |
| Sim seconds per success | A stand-in for cycle time |
| Patch rounds to the best graph | How well a model uses feedback |
| Failure classes | Where the next skill or checker should go |
| Tokens in and out, with image share of input | AA-Robotics found about 70% of spend went to input context |

## Outputs

`LOOPKIT_DATA/bakeoff/<date>/`: `traces/`, `results.parquet`, `report.md` (Level 1 line, Level 1.5 paragraph, Level 2 tables), `pass_vs_cost.png` (one dot per model, one panel per task), `config_snapshot.yaml`. Copy `report.md` into `reports/bakeoff_<date>.md` in the repo. It is small text.

## Cost

Rough estimate: about $10 per model at Astra prices and under $50 for the full run. S4 prints a real estimate from its smoke run before anything launches.
