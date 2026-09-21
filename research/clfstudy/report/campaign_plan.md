# Unattended architecture-search campaign — forward_mae

**Written 2026-09-21, to run unattended for several weeks.**

## Goal

Find the architecture that performs best on the chosen label,
`forward_mae` H12, min_gain/min_loss 1.5%, max_drawdown 10%.

Judged only on classification metrics against the generated labels: balanced
accuracy, macro-F1 and MCC as headline, MCC the ranking key. No backtests, no
P&L, no trade-level metrics anywhere (spec FR-036).

## Why this label

Phase 29 mirrored the phase-25 architecture grid with only the label swapped.
`forward_mae` beat strict triple barrier at **11 of 11** matched
architecture x window cells (+0.0006 to +0.0069). Every individual gap is
small, but 11/11 one-signed is ~1-in-2000 under a coin flip, so the effect is
systematic rather than noise.

## What is already known (do not re-litigate)

| Finding | Evidence |
|---|---|
| Architecture ranking is label-independent: LSTM > Transformer > TCN > MLP | phases 25 and 29, both labels |
| LSTM improves with lookback; MLP and plain TCN *degrade* with it | phase 29 |
| The seq_len ladder saturates around 64 | phases 27, 30, 31 |
| Capacity is inert: 14x parameter range spans 0.0027, the two smallest cells best | phase 28 |
| Dilated TCN (rf31) beats plain TCN, gap widening with window | phase 29 |

So the live questions are **which architecture**, **which input set**, and
whether component *combinations* (conv front-end, residual, attention pooling)
add anything. Not capacity, and not window length.

## Funnel

Screening runs on **evaluation window A only** with 2 seeds, which is the cheap
probe established in phases 30-32 (~5 min/run at seq_len 64). Only finalists pay
for the full three-window, three-seed protocol.

| Stage | What | Windows | Seeds | Selection rule |
|---|---|---|---|---|
| A | Census: every registered arch, seq_len 64, raw8 | A | 0,1 | top 8 by mean MCC |
| B | Input set: stage-A top 8 x {raw8, ind24, both32} | A | 0,1 | top 6 (arch, input_set) cells |
| C | Refinement: stage-B top 6 x small hp grid x seq_len {32,64} | A | 0,1 | top 4 cells |
| D | Promotion: stage-C top 4 + RIDGE + MAJORITY/PERSISTENCE tripwires | A,B,C | 0,1,2 | final ranking |
| E | Report: leaderboard, per-stage findings, open questions | — | — | — |

Stage A deliberately includes architectures the study has never scored on this
label, including the production `LSTM` (conv front-end + residual) and the
literature types added in phase 6.

## Operating rules

* **One trainer at a time.** Parallel trainers caused an OS out-of-memory on
  2026-09-16. The driver never launches a second.
* **Every run isolated** in its own subprocess (phase.py default), so a crashed
  or OOM-killed run costs one row, not the campaign.
* **Watchdog**: a trainer that adds no result row for 90 minutes is killed and
  the queue moves on. One hung MLX process must not block weeks of work.
* **Resumable**: rows key on `config_id`, so re-running any stage skips
  completed work. The campaign can be killed and restarted at any point.
* **Exclusions**: Mamba is excluded — ~48 GB Metal peak and ~68 min/run, and an
  unattended OOM would put the machine at risk. Any other arch flagged as
  pathologically expensive at seq_len 64 is excluded on the same grounds and
  named in the report.

## Known caveats to carry into the report

* `PERSISTENCE` scores ~0.77 on these labels against a best model of ~0.23. Most
  of the predictable structure in the target is label autocorrelation, much of it
  from zero-volume gap-fill bars (22-55% of 15m bars for six of eleven pairs).
  Feature 020 covers this; the campaign does not attempt to fix it, and every
  conclusion here is conditional on it.
* Screening on window A alone is a real compromise: window A saturates earliest
  in the seq_len ladder, and extrapolating from it over-predicted phase 31. Stage
  D exists because window-A rank order is not trusted as final.
* `scaler_mode` stays at the `robust` default for comparability. Flipping it to
  `robust_std` is deliberate future work, bundled with feature 020 Strand A.

## Running it / checking on it

Start or restart (safe at any time — every stage is idempotent and rows key on
`config_id`, so a restart resumes rather than repeats):

```
cd ~/projects/freqtrade/user_data/strategies
nohup research/clfstudy/report/logs/campaign.sh &
```

Check progress:

```
source ../../.venv/bin/activate && export PYTHONPATH=.
python -m research.clfstudy.campaign --status          # stage state + completion counts
python -m research.clfstudy.leaderboard                # current full leaderboard
tail -40 research/clfstudy/report/logs/campaign.log
cat research/clfstudy/report/logs/campaign_watchdog.log # empty is good
```

`research/clfstudy/report/logs/CAMPAIGN_DONE` appears when every stage has
finished. Stage decisions are written as JSON under `research/clfstudy/results/`
so the selection at each gate is auditable after the fact.

### Known limitations of the unattended setup

* **A reboot stops the campaign.** The driver is a plain background process with
  no launchd agent, so a macOS update or a power cut ends it. `autorestart` is on
  so the Mac comes back, but the campaign needs the one command above to resume.
  It loses no completed work.
* `caffeinate -i` holds off idle sleep for the campaign's lifetime. AC power is
  currently set to `sleep 0` anyway; this covers a settings change or a switch to
  battery mid-run.
* The watchdog only kills stalled *trainers*. If the driver process itself hangs,
  nothing recovers it automatically.
