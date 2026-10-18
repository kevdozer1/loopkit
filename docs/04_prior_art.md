# 04. Prior art: what loopkit borrows and why

Short briefs from Kevin's October 2026 research. Each ends with what to take into loopkit.

## Kulakov, MicroFactory (Oct 15, 2026)

An agent on Claude Opus and GPT-6 Astra designs fixtures, writes the computer vision code and guides a human through cell setup. The generated code runs the robot with no model in the loop. The agent returns only when something new shows up, then edits the pipeline. Claims: all operations replaced, a 1.5 hour error-free run, 5 minute task switch after setup. Founder claims, one video.

Tricks: a flip platform that turns a random pile into a known orientation, a $300 screw feeder, and a "little dance" that collects a few offset moves and fits a linear regression from image error to pose correction before insertion.

Take: the System 2 writes code, the System 1 is the code. Simplify the cell before writing code. Build `dance()` and `aligned_insert()` as skills. Count runtime big-model calls and drive them toward 0.

Source: https://x.com/ihorbeaver/status/2110866012333838675

## GaP, Graph-as-Policy (arXiv 2607.05369, Jul 2026, Apache-2.0)

An orchestrator splits a task into segments. Skill agents build a graph of perception, planning and control nodes from a 51-skill library. The graph is rehearsed in parallel sim, failures analyzed, graph patched until the score plateaus. An edge interpreter runs the finished graph with no agents. Results: 121 of 130 real cable insertions (93%), grocery orders 25 of 25, a popcorn task from 33% to 94% in sim. One LLM writing plain Python with no graph scored 0. Slow: about 30 s per insertion and 67 s per pick against an industry norm near 7 s. Needs Linux and a 24 GB GPU.

Take: graphs with typed skills, validate before execution, rehearse-and-patch, agent-free interpreter.

Sources: https://arxiv.org/abs/2607.05369, https://github.com/graph-robots/graph-as-policy

## CaP-X (arXiv 2603.22435, ICML 2026, MIT license)

A gym, benchmark (187 tasks) and RL recipe for code-as-policy agents. RL on the coding model with simulator rewards lifted a 7B model on Cube Lift from 25% to 80% in sim, and it reached 84% on a real Franka. Raw images in the agent's context hurt. Text descriptions of what changed in the scene helped every model. Human expert code reached 88.5% single-turn. Models trail on one try and close the gap with feedback.

Take: failure summaries as text, an RL env whose action is a program and whose reward is the checker (S6), and patch rounds as a reported metric.

Source: https://arxiv.org/pdf/2603.22435

## AA-Robotics v0.1 (Artificial Analysis, Oct 14, 2026)

5 frontier models drove a real 6-axis arm through coding harnesses on 8 tasks. Scores: GPT-6 Astra 69.4, GPT-6.1 Sol 64.6, Gemini 4 Argon 55.3, Claude Opus 5.5 48.8, Claude Fable 5.1 46.3. About 30 minutes per task against 20 s for a human. Cost per task $1.93 (Sol) to $16.36 (Astra). About 70% of spend went to input context, mostly images. Top failure: not knowing when to stop.

Take: rankings flip by task, so run a standing bake-off. Track image share of input tokens. Give every task an end condition a checker can score, and classify `stopping` failures.

Source: https://artificialanalysis.ai/articles/aa-robotics

## Fast System 1 decision models

- Jev (Typesafe, Sep 15): typed decisions with calibrated probabilities in 70 to 500 ms, $0.042 per million input tokens, output free. Kevin already has a key and a working backend in statebench.
- d1 (Liquid AI, Oct 5): the same idea with vision, 200 to 300 ms, $0.04 per million input tokens, text access on OpenRouter.
- Both are vendor numbers from sim demos.

Take: the S6 probe. Log runtime branch decisions so they can be distilled later.

Sources: https://www.datacamp.com/blog/system-one-models-jev, https://www.liquid.ai/blog/d1-decision-model

## Glimpse (Elorian, Oct 16, 2026)

A visual thinking model in research preview that zooms into regions and rewatches video while it reasons. $0.60 per million input and $3.00 per million output. Elorian's own numbers: WGO 42.70 vs Astra 39.42, ERQA tie at 83.75, Astra still leads on 6 of 11 evals. Robot control is listed as future work.

Take: a bake-off slot as panel reader (T1) and checker cross-check. Expect it to trail on code-heavy writing.

Source: https://elorian.ai/blog/introducing-glimpse

## SGS and AlphaNIST (arXiv 2610.12465, 2610.13354)

Sample training tasks the policy solves only sometimes. SGS scaled to about 1M parallel envs with plain PPO and a binary reward. AlphaNIST solved all 20 NIST assembly tasks at 98.2% in sim, and 3 tiny-part tasks only learned inside the multi-task policy.

Take: when the RL env exists (S6), sample seeds and task variants by mid-range pass rate. Checkers make this possible.
