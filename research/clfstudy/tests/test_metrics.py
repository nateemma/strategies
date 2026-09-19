"""Tests for research/clfstudy/metrics.py and baselines.py (T013, spec FR-016..FR-025).

The 12-row fixture's balanced accuracy, macro-F1, MCC, per-class precision/recall, log-loss,
one-vs-rest AUROC, ECE and confusion matrix are computed by hand below (arithmetic shown in
comments) rather than re-derived by calling sklearn a second time. AUROC is additionally checked
against `sklearn.metrics.roc_auc_score` directly on a separate, simple 2-class-style
construction, per research R8's explicit exception for AUROC.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from research.clfstudy import baselines, metrics

# ---------------------------------------------------------------------------
# Primary 12-row fixture
# ---------------------------------------------------------------------------
# 4 SELL(0), 4 HOLD(1), 4 BUY(2) true rows. Predicted (argmax) pattern gives the confusion
# matrix (rows=true, cols=pred, order SELL,HOLD,BUY):
#   [[3,1,0],
#    [1,2,1],
#    [0,1,3]]
# i.e. row3 SELL misclassified as HOLD, row4 HOLD misclassified as SELL, row7 HOLD
# misclassified as BUY, row8 BUY misclassified as HOLD; the other 8 rows are correct.
Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
PROBA = np.array(
    [
        [0.7, 0.2, 0.1],  # r0  SELL true, SELL pred (correct)
        [0.6, 0.3, 0.1],  # r1  SELL true, SELL pred (correct)
        [0.8, 0.1, 0.1],  # r2  SELL true, SELL pred (correct)
        [0.3, 0.5, 0.2],  # r3  SELL true, HOLD pred (wrong)
        [0.5, 0.3, 0.2],  # r4  HOLD true, SELL pred (wrong)
        [0.2, 0.6, 0.2],  # r5  HOLD true, HOLD pred (correct)
        [0.1, 0.7, 0.2],  # r6  HOLD true, HOLD pred (correct)
        [0.2, 0.3, 0.5],  # r7  HOLD true, BUY pred (wrong)
        [0.1, 0.55, 0.35],  # r8 BUY true, HOLD pred (wrong)
        [0.1, 0.2, 0.7],  # r9  BUY true, BUY pred (correct)
        [0.2, 0.1, 0.7],  # r10 BUY true, BUY pred (correct)
        [0.1, 0.1, 0.8],  # r11 BUY true, BUY pred (correct)
    ]
)


def test_confusion_matrix():
    out = metrics.score(Y_TRUE, PROBA)
    assert out["confusion"] == [[3, 1, 0], [1, 2, 1], [0, 1, 3]]


def test_balanced_accuracy():
    # per-class recall (row sums are all 4): SELL 3/4=0.75, HOLD 2/4=0.5, BUY 3/4=0.75
    # mean = (0.75 + 0.5 + 0.75) / 3 = 2/3
    out = metrics.score(Y_TRUE, PROBA)
    assert out["bal_acc"] == pytest.approx(2.0 / 3.0)


def test_macro_f1():
    # precision (col sums all 4): SELL 3/4=0.75, HOLD 2/4=0.5, BUY 3/4=0.75 (== recall here)
    # F1_SELL = 2*.75*.75/(.75+.75) = 0.75; F1_HOLD = 2*.5*.5/1 = 0.5; F1_BUY = 0.75
    # macro F1 = (0.75+0.5+0.75)/3 = 2/3
    out = metrics.score(Y_TRUE, PROBA)
    assert out["macro_f1"] == pytest.approx(2.0 / 3.0)


def test_mcc():
    # Gorodkin multiclass MCC: c=trace=8, s=12, p_k=t_k=[4,4,4] (perfectly balanced).
    # MCC = (c*s - sum(p_k*t_k)) / sqrt((s^2 - sum(p_k^2)) * (s^2 - sum(t_k^2)))
    #     = (96 - 48) / sqrt((144-48)*(144-48)) = 48 / 96 = 0.5
    out = metrics.score(Y_TRUE, PROBA)
    assert out["mcc"] == pytest.approx(0.5)


def test_per_class_precision_recall():
    out = metrics.score(Y_TRUE, PROBA)
    assert out["prec_sell"] == pytest.approx(0.75)
    assert out["prec_hold"] == pytest.approx(0.5)
    assert out["prec_buy"] == pytest.approx(0.75)
    assert out["rec_sell"] == pytest.approx(0.75)
    assert out["rec_hold"] == pytest.approx(0.5)
    assert out["rec_buy"] == pytest.approx(0.75)


def test_logloss():
    # true-class probabilities per row: .7,.6,.8,.3, .3,.6,.7,.3, .35,.7,.7,.8
    # logloss = -mean(ln(p)) = 0.629698221949318 (computed once via a plain natural-log sum;
    # matches sklearn's own log_loss to machine precision, cross-checked while writing this file)
    out = metrics.score(Y_TRUE, PROBA)
    assert out["logloss"] == pytest.approx(0.629698221949318, abs=1e-9)


def test_ece_and_reliability_bins():
    # confidence (max prob) per row, and its bin under 15 equal-width bins on [0,1]
    # (bin i = [i/15, (i+1)/15)):
    #   r0=.7->bin10  r1=.6->bin9  r2=.8->bin12  r3=.5->bin7  r4=.5->bin7  r5=.6->bin9
    #   r6=.7->bin10  r7=.5->bin7  r8=.55->bin8  r9=.7->bin10 r10=.7->bin10 r11=.8->bin12
    # correctness per row: 1,1,1,0, 0,1,1,0, 0,1,1,1  (8/12 correct)
    # bin7:  n=3 conf=.5  acc=0/3=0    -> contributes 3/12*|0-.5|   = 1.5/12
    # bin8:  n=1 conf=.55 acc=0/1=0    -> contributes 1/12*|0-.55| = 0.55/12
    # bin9:  n=2 conf=.6  acc=2/2=1    -> contributes 2/12*|1-.6| = 0.8/12
    # bin10: n=4 conf=.7  acc=4/4=1    -> contributes 4/12*|1-.7| = 1.2/12
    # bin12: n=2 conf=.8  acc=2/2=1    -> contributes 2/12*|1-.8| = 0.4/12
    # ECE = (1.5+0.55+0.8+1.2+0.4)/12 = 4.45/12
    out = metrics.score(Y_TRUE, PROBA)
    assert out["ece"] == pytest.approx(4.45 / 12.0)

    bins = {(b["lo"], b["hi"]): b for b in out["reliability_bins"]}
    assert len(out["reliability_bins"]) == 15
    b7 = bins[(7 / 15, 8 / 15)]
    assert b7["n"] == 3 and b7["conf"] == pytest.approx(0.5) and b7["acc"] == pytest.approx(0.0)
    b10 = bins[(10 / 15, 11 / 15)]
    assert b10["n"] == 4 and b10["conf"] == pytest.approx(0.7) and b10["acc"] == pytest.approx(1.0)
    # an empty bin reports NaN conf/acc, not 0
    b0 = bins[(0 / 15, 1 / 15)]
    assert b0["n"] == 0
    assert np.isnan(b0["conf"]) and np.isnan(b0["acc"])


def test_auroc_per_class_and_macro():
    # one-vs-rest AUROC via Mann-Whitney rank-sum (P=4 positives, N=8 negatives per class, all
    # counted by hand from the SELL/HOLD/BUY probability columns above):
    #   SELL:  31/32 wins  -> 0.96875
    #   HOLD:  27/32 wins  -> 0.84375
    #   BUY:   31/32 wins  -> 0.96875
    # macro = (0.96875 + 0.84375 + 0.96875) / 3 = 0.9270833333333333
    out = metrics.score(Y_TRUE, PROBA)
    assert out["auroc_sell"] == pytest.approx(31 / 32)
    assert out["auroc_hold"] == pytest.approx(27 / 32)
    assert out["auroc_buy"] == pytest.approx(31 / 32)
    assert out["auroc_macro"] == pytest.approx((31 / 32 + 27 / 32 + 31 / 32) / 3)


def test_auroc_matches_sklearn_on_separate_2class_construction():
    # research R8's explicit exception: AUROC may be cross-checked against sklearn directly,
    # on a separate, simpler construction.
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    score_col = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.9, 0.05])
    proba = np.column_stack([1 - score_col, score_col, np.zeros_like(score_col)])
    # class 1 is the "positive" here (HOLD slot); compare metrics.score's auroc_hold to sklearn.
    expected = roc_auc_score(y_true, score_col)
    out = metrics.score(y_true, proba)
    assert out["auroc_hold"] == pytest.approx(expected)


def test_not_collapsed_and_class_present():
    out = metrics.score(Y_TRUE, PROBA)
    assert out["collapsed"] is False
    assert out["class_absent"] is False
    assert out["pred_counts"] == [4, 4, 4]
    assert out["class_counts"] == [4, 4, 4]


# ---------------------------------------------------------------------------
# Absent-class fixture: y_true has no SELL rows.
# ---------------------------------------------------------------------------
def test_absent_class_fixture():
    y_true = np.array([1, 1, 1, 2, 2, 2])
    proba = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.3, 0.4, 0.3],
            [0.1, 0.2, 0.7],
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
        ]
    )
    out = metrics.score(y_true, proba)
    assert out["class_absent"] is True
    # SELL has zero true support (recall undefined) and zero predicted support (precision
    # undefined, since argmax never picks SELL here) -> both NaN.
    assert np.isnan(out["prec_sell"])
    assert np.isnan(out["rec_sell"])
    assert np.isnan(out["auroc_sell"])
    # macro is the nanmean of the two defined classes
    assert out["auroc_macro"] == pytest.approx(
        np.nanmean([out["auroc_hold"], out["auroc_buy"]])
    )
    assert not np.isnan(out["auroc_hold"])
    assert not np.isnan(out["auroc_buy"])


# ---------------------------------------------------------------------------
# Collapsed-prediction fixture: every row predicted HOLD.
# ---------------------------------------------------------------------------
def test_collapsed_fixture():
    n = 20
    y_true = np.array([0, 1, 2] * 6 + [1, 1])[:n]
    proba = np.tile(np.array([0.1, 0.85, 0.05]), (n, 1))
    out = metrics.score(y_true, proba)
    assert out["collapsed"] is True
    assert out["pred_counts"] == [0, n, 0]


# ---------------------------------------------------------------------------
# BUY precision/coverage at threshold 0.8, with exactly 3 rows >= 0.8.
# ---------------------------------------------------------------------------
def test_buy_precision_and_coverage_at_threshold():
    y_true = np.array([2, 2, 1, 2, 0])
    proba = np.array(
        [
            [0.05, 0.05, 0.90],  # >=0.8, true BUY (correct)
            [0.10, 0.05, 0.85],  # >=0.8, true BUY (correct)
            [0.10, 0.05, 0.85],  # >=0.8, true HOLD (wrong)
            [0.30, 0.30, 0.40],  # <0.8
            [0.60, 0.20, 0.20],  # <0.8
        ]
    )
    out = metrics.score(y_true, proba)
    assert out["buy_cov_at_08"] == pytest.approx(3 / 5)
    assert out["buy_prec_at_08"] == pytest.approx(2 / 3)


def test_buy_precision_nan_when_none_above_threshold():
    y_true = np.array([0, 1, 2])
    proba = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
    out = metrics.score(y_true, proba)
    assert out["buy_cov_at_08"] == 0.0
    assert np.isnan(out["buy_prec_at_08"])


# ---------------------------------------------------------------------------
# trigger_masks / trig_* conditional-on-trigger metrics (research finding: the gbb label
# is half a deterministic guard_metric/bb_width threshold rule -- these metrics measure
# what a trained model adds ON TOP of that rule, among the bars the rule fires on).
# ---------------------------------------------------------------------------
# 10-row fixture: rows 0-5 are BUY-trigger bars (guard_metric < 0.3, bb_width > 0.035),
# rows 6-9 are SELL-trigger bars (guard_metric > 0.3, bb_width > 0.035); no row is outside
# both triggers, so `trig_rule_mcc`'s rule_pred is BUY for 0-5 and SELL for 6-9 with no
# HOLD fallback needed. y_pred below is `np.argmax(proba, axis=1)`.
_TRIG_GUARD = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.5, 0.6, 0.4, 0.35])
_TRIG_BBW = np.array([0.05, 0.06, 0.04, 0.045, 0.05, 0.06, 0.05, 0.06, 0.04, 0.045])
_TRIG_Y_TRUE = np.array([2, 2, 1, 1, 2, 1, 0, 1, 0, 1])
_TRIG_PROBA = np.array(
    [
        [0.10, 0.10, 0.80],  # r0 BUY-trigger, true BUY, pred BUY
        [0.10, 0.20, 0.70],  # r1 BUY-trigger, true BUY, pred BUY
        [0.20, 0.70, 0.10],  # r2 BUY-trigger, true HOLD, pred HOLD
        [0.10, 0.30, 0.60],  # r3 BUY-trigger, true HOLD, pred BUY (false positive)
        [0.35, 0.40, 0.25],  # r4 BUY-trigger, true BUY, pred HOLD (missed)
        [0.20, 0.60, 0.20],  # r5 BUY-trigger, true HOLD, pred HOLD
        [0.70, 0.20, 0.10],  # r6 SELL-trigger, true SELL, pred SELL
        [0.30, 0.60, 0.10],  # r7 SELL-trigger, true HOLD, pred HOLD
        [0.20, 0.50, 0.30],  # r8 SELL-trigger, true SELL, pred HOLD (missed)
        [0.60, 0.30, 0.10],  # r9 SELL-trigger, true HOLD, pred SELL (false positive)
    ]
)


def test_trigger_masks_disjoint_and_correct():
    buy_mask, sell_mask = metrics.trigger_masks(_TRIG_GUARD, _TRIG_BBW)
    assert buy_mask.tolist() == [True] * 6 + [False] * 4
    assert sell_mask.tolist() == [False] * 6 + [True] * 4
    assert not np.any(buy_mask & sell_mask)


def test_trig_buy_conditional_metrics():
    # buy-trigger rows 0-5: sub_true=[2,2,1,1,2,1], sub_pred (argmax)=[2,2,1,2,1,1].
    # base_rate = P(true==BUY | trigger) = 3/6 = 0.5 (rows 0,1,4 are true BUY).
    # predicted-BUY rows among the trigger = {0,1,3}; of those, true BUY = {0,1} -> 2/3.
    # binary confusion (true_bin vs pred_bin over the 6 rows): TP=2 (r0,r1), FN=1 (r4),
    # FP=1 (r3), TN=2 (r2,r5) -> MCC = (TP*TN-FP*FN)/sqrt(prod of margins)
    #   = (2*2 - 1*1) / sqrt(3*3*3*3) = 3/9 = 1/3.
    # AUROC on proba[:,2] restricted to the trigger rows, positives={r0:.8,r1:.7,r4:.25},
    # negatives={r2:.1,r3:.6,r5:.2} -> 8 of 9 positive/negative pairs rank correctly
    # (only pos=.25 vs neg=.6 is a loss) -> 8/9.
    buy_mask, sell_mask = metrics.trigger_masks(_TRIG_GUARD, _TRIG_BBW)
    out = metrics.score(_TRIG_Y_TRUE, _TRIG_PROBA, buy_mask=buy_mask, sell_mask=sell_mask)

    assert out["trig_buy_n"] == 6
    assert out["trig_buy_base_rate"] == pytest.approx(0.5)
    assert out["trig_buy_prec_argmax"] == pytest.approx(2.0 / 3.0)
    assert out["trig_buy_mcc"] == pytest.approx(1.0 / 3.0)
    assert out["trig_buy_auroc"] == pytest.approx(8.0 / 9.0)


def test_trig_sell_conditional_metrics():
    # sell-trigger rows 6-9: sub_true=[0,1,0,1], sub_pred=[0,1,1,0].
    # base_rate = P(true==SELL | trigger) = 2/4 = 0.5.
    # predicted-SELL rows = {r6,r9}; of those, true SELL = {r6} -> 1/2.
    # binary confusion: TP=1 (r6), FN=1 (r8), FP=1 (r9), TN=1 (r7)
    #   -> MCC = (1*1 - 1*1) / sqrt(2*2*2*2) = 0/4 = 0.
    # AUROC on proba[:,0], positives={r6:.7,r8:.2}, negatives={r7:.3,r9:.6}: pos=.7 beats
    # both negatives, pos=.2 beats neither -> 2 of 4 pairs -> 0.5.
    buy_mask, sell_mask = metrics.trigger_masks(_TRIG_GUARD, _TRIG_BBW)
    out = metrics.score(_TRIG_Y_TRUE, _TRIG_PROBA, buy_mask=buy_mask, sell_mask=sell_mask)

    assert out["trig_sell_n"] == 4
    assert out["trig_sell_base_rate"] == pytest.approx(0.5)
    assert out["trig_sell_prec_argmax"] == pytest.approx(0.5)
    assert out["trig_sell_mcc"] == pytest.approx(0.0)
    assert out["trig_sell_auroc"] == pytest.approx(0.5)


def test_trig_rule_mcc_is_3class_mcc_of_the_deterministic_rule():
    # rule_pred = BUY (2) on every buy-trigger row, SELL (0) on every sell-trigger row (no
    # row here is outside both triggers). Confusion matrix (rows=true 0/1/2, cols=pred
    # 0/1/2), from y_true vs rule_pred=[2,2,2,2,2,2,0,0,0,0]:
    #   true=0 (r6,r8): both pred=0            -> row [2,0,0]
    #   true=1 (r2,r3,r5,r7,r9): pred={2,2,2,0,0} -> row [2,0,3]
    #   true=2 (r0,r1,r4): all pred=2          -> row [0,0,3]
    # Gorodkin multiclass MCC: c=trace=2+0+3=5, s=N=10,
    # p_k (pred col sums)=[4,0,6], t_k (true row sums)=[2,5,3].
    # MCC = (c*s - sum(p_k*t_k)) / sqrt((s^2-sum(p_k^2))*(s^2-sum(t_k^2)))
    #     = (50 - (4*2+0*5+6*3)) / sqrt((100-52)*(100-38))
    #     = (50-26) / sqrt(48*62) = 24/sqrt(2976)
    buy_mask, sell_mask = metrics.trigger_masks(_TRIG_GUARD, _TRIG_BBW)
    out = metrics.score(_TRIG_Y_TRUE, _TRIG_PROBA, buy_mask=buy_mask, sell_mask=sell_mask)
    assert out["trig_rule_mcc"] == pytest.approx(24.0 / np.sqrt(48.0 * 62.0))


def test_trig_fields_absent_when_masks_not_given():
    out = metrics.score(_TRIG_Y_TRUE, _TRIG_PROBA)
    assert not any(k.startswith("trig_") for k in out)


def test_trig_buy_auroc_nan_when_all_hold_within_trigger():
    # every BUY-trigger row is labelled HOLD -> the one-vs-rest AUROC for BUY is undefined
    # (only one unique value in the true/false-BUY indicator) inside the trigger subset,
    # even though BUY exists elsewhere in y_true.
    y_true = np.array([1, 1, 1, 2])
    proba = np.array(
        [
            [0.10, 0.30, 0.60],  # BUY-trigger, true HOLD, pred BUY (false positive)
            [0.20, 0.70, 0.10],  # BUY-trigger, true HOLD, pred HOLD
            [0.30, 0.60, 0.10],  # BUY-trigger, true HOLD, pred HOLD
            [0.10, 0.10, 0.80],  # not a trigger row (guard/bbw below), true BUY, pred BUY
        ]
    )
    guard = np.array([-0.5, -0.4, -0.3, 0.9])
    bbw = np.array([0.05, 0.06, 0.04, 0.01])  # last row: bb_width under threshold
    buy_mask, sell_mask = metrics.trigger_masks(guard, bbw)
    assert buy_mask.tolist() == [True, True, True, False]

    out = metrics.score(y_true, proba, buy_mask=buy_mask, sell_mask=sell_mask)
    assert out["trig_buy_n"] == 3
    assert out["trig_buy_base_rate"] == 0.0
    assert out["trig_buy_prec_argmax"] == pytest.approx(0.0)
    assert np.isnan(out["trig_buy_auroc"])
    assert out["trig_buy_mcc"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def test_majority_baseline_proba_valid_and_argmax_is_majority_class():
    train_labels = np.array([1, 1, 1, 1, 0, 0, 2])  # HOLD is the majority
    proba, note = baselines.majority(train_labels, n_test=5)
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba.argmax(axis=1) == 1)
    assert "MAJORITY" in note
    assert np.allclose(proba, proba[0])  # every row identical


def test_persistence_baseline_proba_valid_and_argmax_is_shifted_label():
    test_labels = np.array([0, 2, 1, 1, 0])
    proba, note = baselines.persistence(test_labels)
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # row i (i>=1) predicts test_labels[i-1]
    assert np.all(proba[1:].argmax(axis=1) == test_labels[:-1])
    assert "PERSISTENCE" in note and "TRUE" in note


def test_persistence_row_zero_is_uniform():
    proba, _ = baselines.persistence(np.array([2, 0, 1]))
    assert np.allclose(proba[0], np.full(3, 1.0 / 3.0))


# ---------------------------------------------------------------------------
# summarise_seeds
# ---------------------------------------------------------------------------
def test_summarise_seeds_mean_and_std_ddof1():
    rows = [
        {"mcc": 0.4, "bal_acc": 0.6, "note": "seed0"},
        {"mcc": 0.6, "bal_acc": 0.8, "note": "seed1"},
    ]
    out = metrics.summarise_seeds(rows)
    assert out["n_seeds"] == 2
    assert out["mcc_mean"] == pytest.approx(0.5)
    # ddof=1 std of two points = |a-b| / sqrt(2)
    assert out["mcc_std"] == pytest.approx(0.2 / np.sqrt(2))
    assert out["bal_acc_mean"] == pytest.approx(0.7)
    assert out["bal_acc_std"] == pytest.approx(0.2 / np.sqrt(2))
    assert "note" not in {k for k in out if k.startswith("note")}


def test_summarise_seeds_std_nan_for_single_seed():
    out = metrics.summarise_seeds([{"mcc": 0.5}])
    assert out["n_seeds"] == 1
    assert out["mcc_mean"] == pytest.approx(0.5)
    assert np.isnan(out["mcc_std"])


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def test_timing_reports_both_batch_sizes():
    rng = np.random.default_rng(0)
    X_test = rng.normal(size=(50, 4)).astype(np.float32)

    def predict_fn(x):
        return x.sum(axis=1)

    out = metrics.timing(predict_fn, X_test, batch_size=16, n_b1=5)
    assert out["infer_ms_per_window_b256"] >= 0.0
    assert out["infer_ms_per_window_b1"] >= 0.0
