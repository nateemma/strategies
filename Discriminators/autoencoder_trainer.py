"""Training loop for the per-class RealAutoencoder.

For each class c in [0, num_classes):
  * X_train = real_features[real_class_idx == c]
  * y_train = X_train (identity reconstruction)
  * MSE training with AdamW, early stopping on val MSE
  * Saves to ``<model_root>/class_<c>/``

No class one-hot input — each AE handles a single class. No
oversampling — for AE training we just want as much real same-class
data as we have; class imbalance doesn't apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from Discriminators.RealAutoencoder import RealAutoencoder, class_subdir


@dataclass
class AutoencoderTrainConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.2
    early_stop_patience: int = 8
    verbose: bool = True
    seed: int = 0


def _to_mx(arr: np.ndarray) -> mx.array:
    return mx.array(np.ascontiguousarray(arr, dtype=np.float32))


def _mse_loss(model: RealAutoencoder, x: mx.array) -> mx.array:
    recon = model(x)
    return mx.mean((x - recon) ** 2)


def train_autoencoders(
    real_features: np.ndarray,
    real_class_idx: np.ndarray,
    num_classes: int,
    save_root: str,
    config: Optional[AutoencoderTrainConfig] = None,
) -> Dict[int, Dict[str, float]]:
    """Train one autoencoder per class. Returns metrics dict keyed by
    class index."""
    cfg = config or AutoencoderTrainConfig()
    if real_features.ndim != 2:
        raise ValueError(f"real_features must be 2D, got shape {real_features.shape}")
    if len(real_features) != len(real_class_idx):
        raise ValueError("real_features and real_class_idx length mismatch")

    real_features = np.asarray(real_features, dtype=np.float32)
    real_class_idx = np.asarray(real_class_idx, dtype=np.int64).flatten()
    num_features = int(real_features.shape[1])

    all_metrics: Dict[int, Dict[str, float]] = {}

    if cfg.verbose:
        counts = {int(c): int((real_class_idx == c).sum())
                  for c in np.unique(real_class_idx)}
        print(
            f"    RealAutoencoder training: F={num_features} "
            f"num_classes={num_classes} per-class counts={counts}"
        )

    for c in range(num_classes):
        c_int = int(c)
        cls_mask = real_class_idx == c_int
        if not cls_mask.any():
            if cfg.verbose:
                print(f"      class {c_int}: no real samples — skipping")
            continue
        metrics = _train_one(
            class_idx=c_int,
            X=real_features[cls_mask],
            num_features=num_features,
            save_root=save_root,
            cfg=cfg,
        )
        all_metrics[c_int] = metrics

    return all_metrics


def _train_one(
    class_idx: int,
    X: np.ndarray,
    num_features: int,
    save_root: str,
    cfg: AutoencoderTrainConfig,
) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.seed + class_idx)

    # Shuffle then split
    perm = rng.permutation(len(X))
    X = X[perm]
    n_val = max(1, int(len(X) * cfg.val_fraction))
    X_val = X[:n_val]
    X_tr = X[n_val:]

    if cfg.verbose:
        print(
            f"      class {class_idx}: train={len(X_tr)} val={len(X_val)}"
        )

    model = RealAutoencoder(num_features=num_features)
    opt = optim.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_and_grad = nn.value_and_grad(model, _mse_loss)

    best_val = float("inf")
    best_state = None
    patience = 0
    n_train = len(X_tr)
    indices = np.arange(n_train)
    X_val_mx = _to_mx(X_val)
    metrics: Dict[str, float] = {}

    for epoch in range(cfg.epochs):
        rng.shuffle(indices)
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            batch_idx = indices[start:start + cfg.batch_size]
            x_b = _to_mx(X_tr[batch_idx])
            loss, grads = loss_and_grad(model, x_b)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            train_loss_sum += float(loss.item())
            n_batches += 1
        train_loss = train_loss_sum / max(1, n_batches)

        model.eval()
        val_loss = float(_mse_loss(model, X_val_mx).item())

        metrics["train_mse"] = float(train_loss)
        metrics["val_mse"] = float(val_loss)
        metrics["epoch"] = int(epoch + 1)

        if cfg.verbose and (epoch < 3 or (epoch + 1) % 5 == 0):
            print(
                f"        Epoch {epoch+1}/{cfg.epochs} | "
                f"train_mse={train_loss:.6f} val_mse={val_loss:.6f}"
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience = 0
            best_state = _snapshot_params(model)
            metrics["best_val_mse"] = float(best_val)
            metrics["best_epoch"] = int(epoch + 1)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                if cfg.verbose:
                    print(
                        f"        Early stop at epoch {epoch+1} "
                        f"(best val_mse={best_val:.6f})"
                    )
                break

    if best_state is not None:
        _restore_params(model, best_state)
        mx.eval(model.parameters())

    # Per-sample MSE percentiles on the held-out validation set — these
    # are the calibration anchor for ``gan_synth_autoencoder_threshold``.
    # A threshold equal to p95 means "synth must reconstruct as well as
    # 95% of real samples in this class". Use p90 to tighten (if drift
    # flags appear in the augmentation diagnostic), p99 to loosen (if
    # pass rate falls below 50%).
    model.eval()
    recon_val = model(X_val_mx)
    per_sample_mse = mx.mean((X_val_mx - recon_val) ** 2, axis=1)
    mx.eval(per_sample_mse)
    mse_np = np.asarray(per_sample_mse.tolist(), dtype=np.float64)
    p50 = float(np.percentile(mse_np, 50))
    p90 = float(np.percentile(mse_np, 90))
    p95 = float(np.percentile(mse_np, 95))
    p99 = float(np.percentile(mse_np, 99))
    metrics["real_mse_p50"] = p50
    metrics["real_mse_p90"] = p90
    metrics["real_mse_p95"] = p95
    metrics["real_mse_p99"] = p99
    if cfg.verbose:
        print(
            f"        Real-sample MSE percentiles (class {class_idx}): "
            f"p50={p50:.6f}  p90={p90:.6f}  p95={p95:.6f}  p99={p99:.6f}"
        )
        print(
            f"        Suggested gan_synth_autoencoder_threshold = {p95:.4f} "
            f"(p95). Tighten to {p90:.4f} (p90) if drift flags appear; "
            f"loosen to {p99:.4f} (p99) if pass rate drops below 50%."
        )

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
