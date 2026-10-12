# 028 US1 — mechanism check on the judging window

*CONFIRMATORY (pre-registered) -- PROVISIONAL under quality gate 2 (2022 adjoins f00; not a distant era)*

Control `st_e22_rev2` (SimpleTrigger_RevOnlyVolreg2): 969 trades, -5.42 pp.

- fidelity: null replay reproduces 0.997 of recorded exits (reason AND bar); stop model 1.0; PASS (min 0.90)
- by recorded reason: `{"force_exit": {"count": 3, "mean": 1.0}, "metric_overbought": {"count": 1, "mean": 1.0}, "roi": {"count": 11, "mean": 1.0}, "stop": {"count": 343, "mean": 1.0}, "take_profit": {"count": 534, "mean": 0.9944}, "unclog_12h": {"count": 74, "mean": 1.0}, "unclog_1d": {"count": 3, "mean": 1.0}}`
- stop exits 343, recovered to +2% within the max hold 161 -> recovery share **0.469** (floor 0.5; 2023–26: 0.634 / 0.575)
- forgone P&L of the recovered stop exits (reported only): +52.92 pp

| candidate | replay gain pp | expected arm gain (0.5×) | floor applies | result |
|---|---|---|---|---|
| C4 | -0.04 | -0.02 | yes | FAIL |
| C2 | +0.97 | +0.48 | no | **PASS** |
| C1 | -1.29 | -0.65 | yes | FAIL |
| C42 | -2.93 | -1.46 | yes | FAIL |

exit 0 — at least one PASS
