# Quickstart: Label-Type Screen (034)

From `user_data/strategies`; environment: `set -a; source ~/.freqtrade_env; set +a`.
Alias below: `PY="PYTHONPATH=. ../../.venv/bin/python -m"`.

1. **Build** classes and plans: `$PY research.label_screen.build_arms`
   → expect 7 `NNNC_LS_<TAG>` + 8 `_SZ` classes; tests show control keys unchanged.
2. **S0a audit** (no model): `$PY research.label_screen.audit`
   → `audit.json`: every candidate's BUY share per early fold, mfe_mae_ratio's overlap on record,
   each declared extra changes labels, feature lists equal the control's.
3. **Commit the pre-registration** (`research/label_screen/preregistration.json/.md`).
4. **f00 smoke**: `$PY research.label_screen.train --tag FMAE --folds f00`, its signal table, and
   `pick --tag FMAE --fold f00 --role sz --verify` → weights at a new key, threshold at 4 dp.
5. **Chain** (≈ 11 h): train all trainable candidates on f00–f07, then tables and sizer picks
   (one fold per process).
6. **S0 + S1**: `instrument`, then `separation`; expect the control's gap to reproduce 0.072.
7. **Survivors**: `$PY research.label_screen.survivors` → commit `survivors.json`.
8. **S2**: entry-rate picks for survivors, then `evaluation.run_folds --plan plan_ls_<tag>_<role>
   --folds f00,...,f07` per arm; `judge`.
9. **S3** (winner only): train f08–f14, pick, backtest, `judge`; GBR repeat on a pass.

Byte-identity check (SC-003): re-run `wf_aro_g24` on f00 and f07 into scratch tags and compare
trades hashes with the recorded ones.
