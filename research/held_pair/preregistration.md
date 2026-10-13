# 031 held-pair add (DCA) — pre-registration

Machine-readable version: [preregistration.json](preregistration.json). Spec:
`specs/031-held-pair-dca/`.

**Setting (frozen, not re-tuned).**
- `HELD_PAIR_ADD = {}`: one add, last fill ≥ 1 h old, on any fresh entry signal for a held pair.
- NNNC at its 24 h grace default.
- Selected from the held-pair screen's EARLY half only (DCA:all:1h at 24 h, commit 07ff0dc).

**Books.**

| book | arm | control | confirmatory window | exploratory |
|---|---|---|---|---|
| ARO | `hp_aro_dca` | `wf_aro_g24` | f08–f14 | f00–f07 (fidelity run, exempt record) |
| GBR | `hp_gbr_dca` | `fs_gbr_gr` | f08–f14 | f00–f07 |
| GFM | `hp_gfm_dca` | `fs_gfm_gr` | f08–f14 | f00–f07 |
| MLX | `hp_mlx_dca` | `fs_mlx_gr` | f08–f14 | f00–f07 |
| ST reversion | `st_e22_hp_rev` / `st_hp_rev` | `st_e22_g24_rev` / `st_g24_rev` | 2022 e00–e02 | 2023–26 |
| ST momentum | `st_e22_hp_mom` / `st_hp_mom` | `st_e22_g24_mom` / `st_g24_mom` | 2022 e00–e02 | 2023–26 |

**Screen predictions (early half, pp).**

| model | prediction | adds |
|---|---|---|
| ARO | +26.17 | 234 |
| GBR | +42.64 | 341 |
| GFM | +6.98 | 284 |
| MLX | +9.59 | 48 |

SimpleTrigger has no screen.

**Rule.**
- **NNNC.** `judge_one` PASS over f00–f14: early > 0 AND late > 0 AND paired AND both risk
  verdicts map to pass. The maps are copied from the NNNC framework round 2. Minimum control
  trades: 200.
- **SimpleTrigger.** `judge_2022` PASS on e00–e02: gain > 0 AND paired AND both risk verdicts map
  to pass. There is no minimum control trade count (st_grace convention).
- **INCONCLUSIVE** on either rule: zero adds, a thin control, or an inconclusive map.
- **Gate 5.** An ADOPTED gain under 1 pp needs a fresh-tag repeat of one fold.
- **Per-book verdicts only.** The switch stays off by default; enabling it is the user's decision.

**Reported.**
- exposure (average and peak committed stake);
- max drawdown;
- adds and their counters;
- the crowd-out split;
- the leverage control. A gain no larger than it is "explained by extra capital".

**Exempt records** (older than this commit, labelled EXPLORATORY): `hp_smoke` f00, and `hp_aro_dca`
f00–f07, the fidelity run.

**Fidelity** (before this commit): precision 97.5%, recall 99.6%, 0 misaligned rows; arm early
+26.46 vs screen +26.17. PASS.

**Evidence labels.**
- NNNC late half: CONFIRMATORY, PROVISIONAL. It is one era; no NNNC model covers 2022.
- SimpleTrigger 2022: CONFIRMATORY.
- Everything else: EXPLORATORY.
