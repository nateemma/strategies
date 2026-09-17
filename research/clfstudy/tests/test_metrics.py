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
