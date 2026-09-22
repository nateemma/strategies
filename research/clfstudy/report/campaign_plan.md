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
* **Exclusions** (measured from `results.jsonl`, not assumed — this machine has
  **64 GB** of physical RAM):

  | arch | peak Metal | max s/run | verdict |
  |---|---|---|---|
  | ROCKET | **71,842 MB** | 1,883 | excluded — exceeds physical RAM; already has a `failed` and a `collapsed` row |
  | Mamba | 48,467 MB | 4,062 | excluded — 76% of RAM, and by far the slowest |
  | TSMamba | 17,826 MB | 1,586 | kept |
  | Mamba2 | 13,299 MB | 1,130 | kept |
  | LSTM_FAM (reference) | 17–20 GB | — | — |

  Peak memory is roughly flat in `seq_len` (LSTM_FAM: 19.9 GB at 16, 17.3 GB at
  64, 32.8 GB at 256), so the seq_len-16 measurements above are a fair guide at
  64. Mamba2 and TSMamba sit in the same band as the models already running
  nightly and are NOT excluded.

* **Never run before, on any label**: XLSTM, PATCHTST, ITRANSFORMER, TSMIXER,
  TRANSFORMER_MODERN. Expect first-contact failures; stage A is where they
  surface, and one arch failing must not abort a stage. PATCHTST additionally
  raises unless `seq_len >= 4 * patch_len`, so it runs with `patch_len` 8 rather
  than its default 16 to stay valid at seq_len 32.

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

## RESULT — campaign complete 2026-09-21 19:53 PDT (rc=0)

All four stages finished with no watchdog interventions. Stage D, all three windows,
3 seeds, ranked by pooled MCC:

| arch | seq_len | A | B | C | pooled | macro-F1 | bal-acc | params | s/run |
|---|---|---|---|---|---|---|---|---|---|
| PERSISTENCE | — | 0.7677 | 0.7517 | 0.7806 | **0.7667** | 0.8494 | 0.8494 | — | 0 |
| LSTM_W64 | 64 | 0.2842±0.0026 | 0.1803±0.0025 | 0.2430±0.0014 | **0.2358** | 0.4642 | 0.4784 | 46,467 | 624 |
| LSTM_RAW | 64 | 0.2857±0.0016 | 0.1798±0.0024 | 0.2404±0.0011 | **0.2353** | 0.4658 | 0.4777 | 6,907 | 457 |
| GRU_FAM | 64 | 0.2829±0.0029 | 0.1775±0.0033 | 0.2437±0.0016 | **0.2347** | 0.4626 | 0.4759 | 14,403 | 569 |
| LSTM_RAW | 32 | 0.2839±0.0025 | 0.1818±0.0046 | 0.2373±0.0006 | **0.2343** | 0.4660 | 0.4760 | 6,907 | 200 |
| RIDGE | 64 | 0.2249±0.0042 | 0.1366±0.0015 | 0.1645±0.0014 | 0.1754 | 0.4217 | 0.4212 | 1,539 | 35 |
| RIDGE | 32 | 0.2269±0.0004 | 0.1335±0.0007 | 0.1580±0.0027 | 0.1728 | 0.4202 | 0.4197 | 771 | 31 |

`MAJORITY` collapsed to all-Hold on every window, as designed (MCC 0).

### Findings

1. **Recommended: `LSTM_RAW` at seq_len 32, input set `raw8`.** The top four cells are a
   statistical tie — 0.2358 to 0.2343, a 0.0015 spread against per-window seed standard
   deviations of 0.0006-0.0046. Within a tie, choose the cheapest: `LSTM_RAW` at 32 uses
   **6,907 parameters and 200 s/run against `LSTM_W64`'s 46,467 and 624 s** — 6.7x fewer
   weights and 3x faster for the same score. It also has the best macro-F1 of the four.

2. **The architecture family is the only lever that moved anything.** All four finalists
   are recurrent; every one clears RIDGE by ~0.060, far above the 0.02 viability margin.
   Within the family nothing separates: width (6.9K vs 46K parameters), gating (LSTM vs
   GRU) and window length (32 vs 64) are all inside noise. This is the fourth independent
   confirmation that capacity is inert on this task.

3. **The campaign did not beat what we already had.** The pre-campaign best,
   `LSTM_FAM` hidden 32 / depth 2 at seq_len 64, scored 0.2343 pooled. The campaign's
   winner scores 0.2358. A 36-architecture census, the literature types included, bought
   +0.0015 — i.e. nothing. The value delivered was **negative information**: it closed off
   attention, convolutional, state-space, mixer and patching families, all of which lost by
   0.02-0.05, and it found a model 6.7x smaller at the same score.

4. **`PERSISTENCE` still scores 0.7667 against a best model of 0.2358.** Every number above
   is a contest for a small share of the signal; the bulk of what is predictable in this
   label is its own autocorrelation, much of it gap-fill artefact. See feature 020.

### Open

* `LSTM_FAM` never reached stages B-D: its default hp (hidden 64, depth 1) diverges to NaN
  at seq_len 64, so it failed the census. Its working setting (hidden 32, depth 2) matches
  the campaign winner. Carrying it forward needs a per-arch default override.
* Everything here is window-A-screened before stage D. Window B remains far harder than A
  and C for every model (0.18 against 0.28/0.24), and no architecture changed that.
