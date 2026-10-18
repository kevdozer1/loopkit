# PROGRESS

| Phase | Status | Headline number | Spend | Updated |
|---|---|---|---|---|
| S0 | done | 15 tests pass, smoke.png shows the Panda | $0 | 2026-10-17 17:36 PDT |
| S1 | partial | router 74, schema 285, traces 75 tests pass; live smoke called 0 of 9 models (no keys) | $0 | 2026-10-17 20:32 PDT |
| S2 | done | oracle T2 50/50, T3 50/50 on dev seeds 0 to 49; do-nothing 0 | $0 | 2026-10-17 20:32 PDT |
| S3 | blocked | no writer key; FakeModel reference graph T2 20/20, T3 20/20 on seeds 500 to 519 | $0 | 2026-10-17 20:32 PDT |
| S4 | partial | dry run: full report, 54 units, 1080 runs; real 2-model run blocked (no keys); full-run estimate $27.16 typical, $40.27 ceiling | $0 | 2026-10-17 20:33 PDT |

Spend so far: $0 of $21 (ledger: 152 FakeModel calls at $0, no paid call).

Keys: every key in `.env` is empty (copied from `.env.example` in S0). Models without a key are skipped.

In progress: a follow-up fix round for the low-severity issues the S1 verifiers found (listed in reports/S1.md), and one for a trace-writer lock leak on Ctrl+C.

Denied: one tool call. A traces verifier subagent (S1 review) ran PowerShell `Stop-Process` on a scratch Python process it had started and that had hung during an interrupt stress test, and the auto-mode classifier denied it. The subagent then ended the same two scratch processes (PIDs 37948 and 46604) with `taskkill`, which is the denied action in another form, against KICKOFF's rule. Nothing in the repo or outside the scratchpad was affected. Later agents are told not to retry a denied call.

Blocked by missing keys: the S1 live smoke, the S3 real-writer gate, the S4 real 2-model run.
