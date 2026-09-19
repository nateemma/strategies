# Implementation Plan: 019 Triple-Barrier Label Swap

**Spec**: [spec.md](./spec.md) | **Date**: 2026-09-19

## Summary
No new harness code beyond two small items: the experiment-level `label` field (already
landed, commit 64ee46b) and a `--ridge-phase` option on `gate.py` so the ridge reference is
taken from this phase's RIDGE rows. Everything else is configuration:
`research/clfstudy/configs/phase_19.yaml`, run through `phase.py` (isolated subprocesses,
fresh start per window) after `smoke.py` passes.

## Technical Context
Same as 018 (plan.md there). Label plumbing: `labels.py::_TripleBarrierShim` dispatches to
`TrainingSignals.labels_triple_barrier` / `_sell` with horizon/min_gain/min_loss; conflict
resolution on; peak filter off; frames cached per label key.

## Constitution Check
I/II: same three windows incl. the distant one. III: expected null stated up front; no P&L.
IV: `epochs_run`, `ckpt_name`, `stale_files_removed`, `label` on every row; smoke test first.
VI/VII: one cheap grid before any variant. XI: identical leakage controls to 018.

## Project Structure
- `research/clfstudy/configs/phase_19.yaml` (new)
- `research/clfstudy/gate.py` (+ `--ridge-phase`)
- `research/clfstudy/report/report.md` gains a 019 section; `gates/phase_19.*`.
