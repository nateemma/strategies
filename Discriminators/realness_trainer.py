"""Training loop for the unified RealnessDiscriminator.

Takes a real pool and a dict of (gan_name → synth, gan_name → synth_labels)
contributions, builds a balanced binary training set, trains the
discriminator with BCE + Adam + early stopping on a held-out validation
split, and returns the trained model.

Training-set construction:
  * Real samples — labeled y=0, paired with their true class indices.
  * Synth samples — labeled y=1, paired with the class the GAN was
    conditioned on. Pooled across GANs.
  * Downsampling — synth is downsampled to the real-count so the loss
    isn't dominated by whichever GAN generated the most rows.

No special class-imbalance handling: the class label is a feature input
(via one-hot concat), so the discriminator sees both the claimed class
AND the features. That's the whole point of cross-label awareness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from Discriminators.RealnessDiscriminator import RealnessDiscriminator, one_hot_class


@dataclass
class RealnessTrainConfig:
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


def _bce_loss(model: RealnessDiscriminator, x: mx.array, c: mx.array, y: mx.array) -> mx.array:
    logits = model(x, c)
    # mlx-style numerically-stable BCE with logits:
    #   loss = max(z, 0) - z * y + log(1 + exp(-|z|))
    z = logits
    return mx.mean(mx.maximum(z, 0) - z * y + mx.log1p(mx.exp(-mx.abs(z))))


def train_realness_discriminator(
    real_features: np.ndarray,
    real_class_idx: np.ndarray,
    synth_by_gan: Dict[str, Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    config: Optional[RealnessTrainConfig] = None,
) -> Tuple[RealnessDiscriminator, Dict[str, float]]:
    """Train and return a unified RealnessDiscriminator.

    Args:
        real_features:  (N_real, F) ndarray, normalised the same way the
                        classifier sees it (i.e. post-scaler).
        real_class_idx: (N_real,) integer class indices for the real rows.
        synth_by_gan:   dict of ``gan_name → (synth_features, synth_class_idx)``.
                        Each entry contributes positives; pooled and
                        downsampled to balance against real.
        num_classes:    width of the class one-hot.
        config:         training hyperparams.

    Returns:
        ``(trained_model, metrics_dict)``.
        ``metrics_dict`` includes final train/val loss and val accuracy.
    """
    cfg = config or RealnessTrainConfig()
    rng = np.random.default_rng(cfg.seed)

    if real_features.ndim != 2:
        raise ValueError(f"real_features must be 2D, got shape {real_features.shape}")
    if len(real_features) != len(real_class_idx):
        raise ValueError(
            f"real_features and real_class_idx length mismatch: "
            f"{len(real_features)} vs {len(real_class_idx)}"
        )
    num_features = int(real_features.shape[1])

    # --- Assemble positives (synth) pooled across GAN families --------- #
    synth_feats_list = []
    synth_cls_list = []
    for gan_name, (sf, sc) in synth_by_gan.items():
        if sf is None or len(sf) == 0:
            continue
        if sf.shape[1] != num_features:
            raise ValueError(
                f"synth from {gan_name} has F={sf.shape[1]}, expected {num_features}"
            )
        synth_feats_list.append(np.asarray(sf, dtype=np.float32))
        synth_cls_list.append(np.asarray(sc, dtype=np.int64).flatten())

    if not synth_feats_list:
        raise ValueError("synth_by_gan empty — nothing to train against")

    synth_features = np.concatenate(synth_feats_list, axis=0)
    synth_class_idx = np.concatenate(synth_cls_list, axis=0)

    # Downsample synth to match real count so neither side dominates.
    n_target = min(len(real_features), len(synth_features))
    if len(real_features) > n_target:
        keep = rng.choice(len(real_features), size=n_target, replace=False)
        real_features = real_features[keep]
        real_class_idx = np.asarray(real_class_idx)[keep]
    if len(synth_features) > n_target:
        keep = rng.choice(len(synth_features), size=n_target, replace=False)
        synth_features = synth_features[keep]
        synth_class_idx = synth_class_idx[keep]

    # --- Build (X, c, y) and split train/val --------------------------- #
    X = np.concatenate([real_features, synth_features], axis=0).astype(np.float32)
    c_idx = np.concatenate([real_class_idx, synth_class_idx], axis=0).astype(np.int64)
    y = np.concatenate([
        np.zeros(len(real_features), dtype=np.float32),
        np.ones(len(synth_features), dtype=np.float32),
    ])

    # One-hot the class indices once.
    one_hot = np.zeros((len(c_idx), num_classes), dtype=np.float32)
    one_hot[np.arange(len(c_idx)), c_idx.clip(0, num_classes - 1)] = 1.0

    # Shuffle then split.
    perm = rng.permutation(len(X))
    X, one_hot, y = X[perm], one_hot[perm], y[perm]
    n_val = max(1, int(len(X) * cfg.val_fraction))
    X_val, c_val, y_val = X[:n_val], one_hot[:n_val], y[:n_val]
    X_tr, c_tr, y_tr = X[n_val:], one_hot[n_val:], y[n_val:]

    # --- Model + optimizer -------------------------------------------- #
    model = RealnessDiscriminator(num_features=num_features, num_classes=num_classes)
    opt = optim.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    loss_and_grad = nn.value_and_grad(model, _bce_loss)

    best_val = float("inf")
    best_state = None
    patience = 0
    metrics: Dict[str, float] = {}

    X_val_mx = _to_mx(X_val)
    c_val_mx = _to_mx(c_val)
    y_val_mx = _to_mx(y_val)

    n_train = len(X_tr)
    indices = np.arange(n_train)

    if cfg.verbose:
        print(
            f"    RealnessDiscriminator training: "
            f"{n_train} train, {len(X_val)} val, F={num_features}, C={num_classes}"
        )
        print(
            f"      real={len(real_features)} synth(pooled)={len(synth_features)} "
            f"(from {len(synth_feats_list)} GAN(s))"
        )

    for epoch in range(cfg.epochs):
        rng.shuffle(indices)
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for start in range(0, n_train, cfg.batch_size):
            batch_idx = indices[start:start + cfg.batch_size]
            x_batch = _to_mx(X_tr[batch_idx])
            c_batch = _to_mx(c_tr[batch_idx])
            y_batch = _to_mx(y_tr[batch_idx])

            loss, grads = loss_and_grad(model, x_batch, c_batch, y_batch)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)

            train_loss_sum += float(loss.item())
            n_batches += 1

        train_loss = train_loss_sum / max(1, n_batches)

        # --- Val ----- #
        model.eval()
        val_loss = float(_bce_loss(model, X_val_mx, c_val_mx, y_val_mx).item())
        val_logits = model(X_val_mx, c_val_mx)
        val_pred = (mx.sigmoid(val_logits) >= 0.5).astype(mx.float32)
        val_acc = float(mx.mean((val_pred == y_val_mx).astype(mx.float32)).item())

        if cfg.verbose and (epoch < 5 or (epoch + 1) % 5 == 0):
            print(
                f"      Epoch {epoch+1}/{cfg.epochs} | "
                f"train_bce={train_loss:.4f} val_bce={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience = 0
            best_state = {k: v for k, v in model.parameters().items()} if False else None
            # MLX parameters() returns a tree; we use the model itself as the
            # canonical store and restore by retraining-from-here equivalence.
            # Save current weights to a temp dict-of-arrays for restore.
            best_state = _snapshot_params(model)
            metrics["best_val_loss"] = best_val
            metrics["best_val_acc"] = val_acc
            metrics["best_epoch"] = epoch + 1
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                if cfg.verbose:
                    print(
                        f"      Early stop at epoch {epoch+1} "
                        f"(best val_bce={best_val:.4f} @ epoch {metrics['best_epoch']})"
                    )
                break

    if best_state is not None:
        _restore_params(model, best_state)
        mx.eval(model.parameters())

    metrics["final_val_loss"] = float(val_loss)
    metrics["final_val_acc"] = float(val_acc)
    metrics["epochs_run"] = epoch + 1

    if cfg.verbose:
        print(
            f"    Training complete. Best val_bce={metrics.get('best_val_loss', float('nan')):.4f} "
            f"(val_acc={metrics.get('best_val_acc', float('nan')):.4f})"
        )

    return model, metrics


def _snapshot_params(model: nn.Module) -> Dict:
    """Deep-copy current model parameters into a plain dict tree."""
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
    """Reload a snapshot into the model in place."""
    model.update(snapshot)
