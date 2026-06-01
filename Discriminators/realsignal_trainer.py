"""Training loop for the per-class RealsignalClassifier.

For each class c in [0, num_classes):
  * y = (real_class_idx == c).astype(int)
  * BCE training with AdamW, early stopping on held-out validation
  * Saves to ``<model_root>/class_<c>/``

The positive class (one specific NNNC label) is the minority in 2 of 3
cases (class-0 Sell ~10%, class-2 Buy ~10%, class-1 Hold ~80% on this
dataset). We handle imbalance by oversampling positives via
``pos_oversample`` so the training loss isn't dominated by Hold rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from Discriminators.RealsignalClassifier import RealsignalClassifier, class_subdir


@dataclass
class RealsignalTrainConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.2
    early_stop_patience: int = 8
    verbose: bool = True
    seed: int = 0
    # Oversample positives so each minibatch sees roughly balanced
    # pos/neg. Computed at training time from the actual class balance
    # in the real data — passing a ratio override here forces a
    # specific oversample factor instead.
    pos_oversample: Optional[float] = None


def _to_mx(arr: np.ndarray) -> mx.array:
    return mx.array(np.ascontiguousarray(arr, dtype=np.float32))


def _bce_loss(model: RealsignalClassifier, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    z = logits
    return mx.mean(mx.maximum(z, 0) - z * y + mx.log1p(mx.exp(-mx.abs(z))))


def train_realsignal_classifiers(
    real_features: np.ndarray,
    real_class_idx: np.ndarray,
    num_classes: int,
    save_root: str,
    config: Optional[RealsignalTrainConfig] = None,
) -> Dict[int, Dict[str, float]]:
    """Train one binary classifier per class and persist each. Returns a
    metrics dict keyed by class index."""
    cfg = config or RealsignalTrainConfig()
    if real_features.ndim != 2:
        raise ValueError(f"real_features must be 2D, got shape {real_features.shape}")
    if len(real_features) != len(real_class_idx):
        raise ValueError("real_features and real_class_idx length mismatch")

    real_features = np.asarray(real_features, dtype=np.float32)
    real_class_idx = np.asarray(real_class_idx, dtype=np.int64).flatten()
    num_features = int(real_features.shape[1])

    all_metrics: Dict[int, Dict[str, float]] = {}

    classes_present = np.unique(real_class_idx)
    if cfg.verbose:
        counts = {int(c): int((real_class_idx == c).sum()) for c in classes_present}
        print(
            f"    RealsignalClassifier training: F={num_features} "
            f"num_classes={num_classes} per-class counts={counts}"
        )

    for c in range(num_classes):
        c_int = int(c)
        if not (real_class_idx == c_int).any():
            if cfg.verbose:
                print(f"      class {c_int}: no positives — skipping")
            continue
        metrics = _train_one(
            class_idx=c_int,
            real_features=real_features,
            real_class_idx=real_class_idx,
            num_features=num_features,
            save_root=save_root,
            cfg=cfg,
        )
        all_metrics[c_int] = metrics

    return all_metrics


def _train_one(
    class_idx: int,
    real_features: np.ndarray,
    real_class_idx: np.ndarray,
    num_features: int,
    save_root: str,
    cfg: RealsignalTrainConfig,
) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.seed + class_idx)

    y = (real_class_idx == class_idx).astype(np.float32)
    pos_idx = np.where(y == 1.0)[0]
    neg_idx = np.where(y == 0.0)[0]

    if cfg.pos_oversample is not None:
        oversample = float(cfg.pos_oversample)
    else:
        # Default: oversample positives so they are no fewer than 25%
        # of the *training* mix. 1:3 pos:neg is a reasonable balance
        # for BCE without flipping the class prior too aggressively.
        neg_count = len(neg_idx)
        pos_count = max(1, len(pos_idx))
        target_pos_share = 0.25
        target_pos_count = int(neg_count * target_pos_share / (1.0 - target_pos_share))
        oversample = max(1.0, target_pos_count / pos_count)

    if oversample > 1.0:
        pos_repeat = int(np.ceil(oversample))
        pos_idx_oversampled = np.tile(pos_idx, pos_repeat)[:int(len(pos_idx) * oversample)]
    else:
        pos_idx_oversampled = pos_idx

    all_idx = np.concatenate([pos_idx_oversampled, neg_idx])
    rng.shuffle(all_idx)

    X = real_features[all_idx]
    y = y[all_idx]

    n_val = max(1, int(len(X) * cfg.val_fraction))
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    if cfg.verbose:
        print(
            f"      class {class_idx}: train={len(X_tr)} val={len(X_val)} "
            f"(real pos={len(pos_idx)} neg={len(neg_idx)} "
            f"oversample={oversample:.2f}×)"
        )

    model = RealsignalClassifier(num_features=num_features)
    opt = optim.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_and_grad = nn.value_and_grad(model, _bce_loss)

    best_val = float("inf")
    best_state = None
    patience = 0
    n_train = len(X_tr)
    indices = np.arange(n_train)

    X_val_mx = _to_mx(X_val)
    y_val_mx = _to_mx(y_val)
    metrics: Dict[str, float] = {}

    for epoch in range(cfg.epochs):
        rng.shuffle(indices)
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            batch_idx = indices[start:start + cfg.batch_size]
            x_b = _to_mx(X_tr[batch_idx])
            y_b = _to_mx(y_tr[batch_idx])
            loss, grads = loss_and_grad(model, x_b, y_b)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            train_loss_sum += float(loss.item())
            n_batches += 1
        train_loss = train_loss_sum / max(1, n_batches)

        model.eval()
        val_loss = float(_bce_loss(model, X_val_mx, y_val_mx).item())
        val_logits = model(X_val_mx)
        val_pred = (mx.sigmoid(val_logits) >= 0.5).astype(mx.float32)
        val_acc = float(mx.mean((val_pred == y_val_mx).astype(mx.float32)).item())

        metrics["train_bce"] = float(train_loss)
        metrics["val_bce"] = float(val_loss)
        metrics["val_acc"] = float(val_acc)
        metrics["epoch"] = int(epoch + 1)

        if cfg.verbose and (epoch < 3 or (epoch + 1) % 5 == 0):
            print(
                f"        Epoch {epoch+1}/{cfg.epochs} | "
                f"train_bce={train_loss:.4f} val_bce={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience = 0
            best_state = _snapshot_params(model)
            metrics["best_val_loss"] = float(best_val)
            metrics["best_val_acc"] = float(val_acc)
            metrics["best_epoch"] = int(epoch + 1)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                if cfg.verbose:
                    print(
                        f"        Early stop at epoch {epoch+1} "
                        f"(best val_bce={best_val:.4f})"
                    )
                break

    if best_state is not None:
        _restore_params(model, best_state)
        mx.eval(model.parameters())

    save_path = class_subdir(save_root, class_idx)
    model.save(save_path)
    if cfg.verbose:
        print(f"        Saved to {save_path}")

    return metrics


def _snapshot_params(model: nn.Module) -> Dict:
    def _copy(node):
        if isinstance(node, dict):
            return {k: _copy(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_copy(v) for v in node]
        if isinstance(node, mx.array):
            return mx.array(np.array(np.asarray(node), copy=True))
        return node
    return _copy(model.parameters())


def _restore_params(model: nn.Module, snapshot: Dict) -> None:
    model.update(snapshot)
