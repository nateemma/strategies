"""Per-fold classifier training and encoder inheritance (T011, T012).

Contract: specs/010-labelfree-warmstart-finetune/contracts/warm-start.md

The production model is NEVER reused. It was trained on data that later folds treat as
held out, so inheriting it would leak the test range into the initialisation — the same
class of error as ranking on the training range, which cost feature 009 a full run.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.labelfree.policy import ClassifierEncoderPolicy      # noqa: E402

# Only the encoder transfers. The classifier's decode() path — bottleneck, out, softmax —
# is a LABEL decision and must not be inherited by a policy that emits exposure.
ENCODER_PREFIXES = ("resize", "conv1", "ln_conv", "lstm", "ln_lstm")
EXCLUDED_PREFIXES = ("bottleneck", "out")


@dataclass(frozen=True)
class WarmStartSource:
    fold_id: str
    train_start: str
    train_end: str
    weights_ref: str
    label_threshold: float
    inherited_tensors: tuple[str, ...]

    def describe(self) -> dict:
        return {"fold_id": self.fold_id,
                "train_range": [self.train_start, self.train_end],
                "weights_ref": self.weights_ref,
                "label_threshold": self.label_threshold,
                "n_inherited": len(self.inherited_tensors)}


def encoder_tensors(weights: dict) -> dict:
    """The encoder subset of a classifier's saved weights."""
    return {k: v for k, v in weights.items()
            if k.split(".")[0] in ENCODER_PREFIXES}


def load_encoder(policy: ClassifierEncoderPolicy, weights: dict) -> list[str]:
    """Copy encoder weights into the policy. Returns the tensor names transferred.

    Raises rather than warns on any mismatch: a silent partial load leaves most of the
    encoder random while the arm still reports itself as warm-started, which would be
    indistinguishable from Arm B and reported as one.
    """
    enc = encoder_tensors(weights)
    if not enc:
        raise ValueError("no encoder tensors found in the source weights")
    leaked = [k for k in weights if k.split(".")[0] in EXCLUDED_PREFIXES]
    have = {k: v.shape for k, v in tree_flatten(policy.parameters())}

    transferred, problems = [], []
    for k, v in enc.items():
        if k not in have:
            problems.append(f"{k}: not present in the policy")
        elif tuple(have[k]) != tuple(v.shape):
            problems.append(f"{k}: shape {tuple(v.shape)} != policy {tuple(have[k])}")
        else:
            transferred.append(k)
    if problems:
        raise ValueError("encoder transfer mismatch:\n  " + "\n  ".join(problems))

    policy.update(_nest({k: weights[k] for k in transferred}))
    mx.eval(policy.parameters())
    return sorted(transferred) + []  # excluded tensors are simply never copied


def _nest(flat: dict) -> dict:
    """{'lstm.Wx': arr} -> {'lstm': {'Wx': arr}} for Module.update()."""
    out: dict = {}
    for k, v in flat.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def differs_from_random(policy: ClassifierEncoderPolicy, n_features: int,
                        seed: int = 0) -> bool:
    """Proof that something was actually inherited.

    A warm-start that loads nothing is behaviourally identical to Arm B and would be
    reported as a warm-start result. This is the check that makes that impossible.
    """
    mx.random.seed(seed)
    ref = ClassifierEncoderPolicy(n_features=n_features,
                                  turnover_control=policy.turnover_control)
    a = dict(tree_flatten(policy.parameters()))
    b = dict(tree_flatten(ref.parameters()))
    for k in a:
        if k.split(".")[0] not in ENCODER_PREFIXES:
            continue
        if not np.allclose(np.asarray(a[k]), np.asarray(b[k])):
            return True
    return False


def check_learning_rates(encoder_lr: float, head_lr: float) -> None:
    """FR-005: the inherited encoder trains an order of magnitude below the new head."""
    if encoder_lr * 10.0 > head_lr + 1e-12:
        raise ValueError(
            f"encoder_lr {encoder_lr} must be <= head_lr/10 ({head_lr / 10.0}); "
            "fine-tuning an inherited encoder at the head's rate destroys the prior")


# ---------------------------------------------------------------------------
# Per-fold classifier training (T011)
# ---------------------------------------------------------------------------

def train_fold_classifier(pairs: list[str], train_start: str, train_end: str,
                          fold_id: str, min_gain: float, seq_len: int = 16,
                          horizon: int | None = None, seed: int = 1,
                          out_dir: Path | None = None) -> WarmStartSource:
    """Train a classifier on ONE fold's training range and save its weights.

    The production model is never reused: it saw data later folds treat as held out, so
    inheriting it would leak the test range into the initialisation.

    `MLXClassifierNary.train()` accepts numpy tensors directly, so no freqtrade backtest is
    involved — the per-fold cost is a plain function call.
    """
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "NNNC"))
    sys.path.insert(0, str(ROOT / "Predictors"))
    sys.path.insert(0, str(ROOT / "utils"))

    from NNNClassifierMLX import NNNClassifierMLX_LSTM
    from DataframePopulator import DataframePopulator
    from regime import regimedata as rd
    from research.labelfree import features as FT
    from research.labelfree import labels as LB

    h = horizon if horizon is not None else LB.DEFAULT_HORIZON
    X_all, Y_all = [], []
    for p in pairs:
        feats, _px, _vq, _cols = FT.build_features(p, train_start, train_end)
        df = rd.load_ohlcv(p.split("/")[0])
        df = df[(df.index >= pd.Timestamp(train_start, tz="UTC")) &
                (df.index <= pd.Timestamp(train_end, tz="UTC"))].reset_index()
        pop = DataframePopulator().add_minimal_indicators(df.copy())
        tgt = LB.directional_label(pop, horizon=h, min_gain=min_gain)

        n = min(len(feats), len(tgt.values))
        win = FT.sequence_tensor(pd.DataFrame(feats[:n]), seq_len)   # (n-seq+1, seq, F)
        # A window ending at bar i carries the label of bar i.
        y = tgt.values[seq_len - 1: n]
        ok = tgt.valid_mask[seq_len - 1: n]
        m = min(len(win), len(y), len(ok))
        X_all.append(win[:m][ok[:m]])
        Y_all.append(y[:m][ok[:m]])

    X = np.concatenate(X_all).astype(np.float32)
    Y = np.concatenate(Y_all).astype(int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]
    cut = int(0.8 * len(X))

    # MLXBasePredictor requires (pair, seq_len, num_features). `pair` is only an identity
    # tag here: this classifier is trained on the POOLED fold panel, matching the
    # production strategy's aggregate_pairs semantics, not on a single pair.
    clf = NNNClassifierMLX_LSTM(f"pooled@{fold_id}", seq_len, X.shape[-1])
    clf.train_seed = seed

    # Pin the model path BEFORE training. train() calls save() at the end, and an unset
    # path makes os.makedirs('') raise after the whole run has completed — AGENT_GUIDE's
    # silent-model-path note, in its loudest form.
    out_dir = out_dir or (ROOT / "saved_data" / "labelfree_warmstart")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = out_dir / f"clf_{fold_id}_g{min_gain}_s{seed}.safetensors"
    clf.set_model_path(str(ref))

    clf.train(X[:cut], X[cut:], LB.one_hot(Y[:cut]), LB.one_hot(Y[cut:]), force_train=True)
    flat = dict(tree_flatten(clf.model.parameters()))

    enc = sorted(encoder_tensors(flat))
    return WarmStartSource(fold_id, train_start, train_end, str(ref), min_gain, tuple(enc))
