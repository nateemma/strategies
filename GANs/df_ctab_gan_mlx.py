"""
CTAB-GAN+ — MLX (Apple Metal) backend, single-task variant.

VGM-encoded continuous features and a single one-hot class condition.
Multi-task variant lives in ``df_mt_ctab_gan_mlx.py``.  Both classes
share VGM encoding, evaluation, and training callbacks via
``mlx_ctab_helpers``.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd

from GANs.mlx_ctab_helpers import (
    BestCheckpointTracker,
    DivergenceMonitor,
    EarlyStopping,
    ReduceLROnPlateau,
    evaluate_with_dataframes,
    fit_vgm_columns,
    inverse_transform_vgm,
    transform_vgm,
)


class GumbelSoftmax(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def __call__(self, x):
        return mx.softmax(x / self.temperature, axis=-1)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        continuous_info: List[Tuple[int, int]],
        hidden_dim: int = 256,
    ):
        """cond_dim is the total conditioning vector size — class one-hot
        concatenated with (optional) pair one-hot. Caller assembles the
        cond vector outside this class."""
        super().__init__()
        self.continuous_info = continuous_info
        self.total_dim = sum(v + m for v, m in continuous_info)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Per-column heads — registered as named attributes so MLX's
        # parameters() walk picks them up.
        self._output_layer_names: List[str] = []
        for i, (val_dim, num_modes) in enumerate(continuous_info):
            scalar_name = f"output_scalar_{i}"
            setattr(self, scalar_name, nn.Linear(hidden_dim, 1))
            self._output_layer_names.append(scalar_name)

            if num_modes > 0:
                mode_name = f"output_modes_{i}"
                setattr(self, mode_name, nn.Linear(hidden_dim, num_modes))
                self._output_layer_names.append(mode_name)

    def __call__(self, z, c):
        h = self.model(mx.concatenate([z, c], axis=-1))
        outputs: List[Any] = []
        layer_idx = 0
        for i, (val_dim, num_modes) in enumerate(self.continuous_info):
            scalar_layer = getattr(self, self._output_layer_names[layer_idx])
            outputs.append(mx.tanh(scalar_layer(h)))
            layer_idx += 1
            if num_modes > 0:
                mode_layer = getattr(self, self._output_layer_names[layer_idx])
                outputs.append(mx.softmax(mode_layer(h) / 0.2, axis=-1))
                layer_idx += 1
        return mx.concatenate(outputs, axis=-1)


class Critic(nn.Module):
    def __init__(self, total_dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(total_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 1),
        )

    def __call__(self, x, c):
        return self.model(mx.concatenate([x, c], axis=-1))


class CTABGANMLX:
    """MLX transition of CTAB-GAN+ for purely continuous data using VGM encoding."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        epochs: int = 100,
        batch_size: int = 512,
        n_critic: int = 5,
        learning_rate: float = 2e-4,
        gp_weight: float = 10.0,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-4,
        reduce_lr_patience: int = 10,
        reduce_lr_factor: float = 0.5,
        reduce_lr_min: float = 1e-6,
        eval_frequency: int = 5,
        eval_num_samples: int = 1000,
        monitor_metric: str = "eval_quality",
        random_seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        if monitor_metric != "eval_quality":
            raise ValueError(
                f"monitor_metric={monitor_metric!r} is not supported by the MLX "
                f"backend — only 'eval_quality' is available.  Set "
                f"eval_frequency=0 to disable eval-based monitoring entirely "
                f"(falls back to combined-loss divergence guard only)."
            )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.learning_rate = learning_rate
        self.gp_weight = gp_weight
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_min = reduce_lr_min
        self.eval_frequency = eval_frequency
        self.eval_num_samples = eval_num_samples
        self.monitor_metric = monitor_metric
        self.random_seed = random_seed
        self.verbose = verbose

        self.is_fitted = False
        self.num_classes = 0
        # Pair conditioning — set when fit() is called with pair_labels.
        # num_pairs == 0 means no pair conditioning (backward compatible).
        self.num_pairs: int = 0
        self.pair_names: Optional[List[str]] = None  # optional pair_id → label
        self.gen: Optional[Generator] = None
        self.critic: Optional[Critic] = None
        self.gen_opt: Optional[optim.Adam] = None
        self.critic_opt: Optional[optim.Adam] = None

        self.vgm_models: Dict[str, Any] = {}
        self.column_info: Dict[str, dict] = {}
        self.column_order: List[str] = []
        self.continuous_columns: List[str] = []
        self.continuous_info: List[Tuple[int, int]] = []
        self.total_dim: int = 0

    @property
    def cond_dim(self) -> int:
        """Total conditioning vector size: class one-hot + (optional) pair one-hot."""
        return self.num_classes + self.num_pairs

    # ------------------------------------------------------------------ #
    # fit()                                                              #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        dataframe: pd.DataFrame,
        labels: np.ndarray,
        categorical_columns: List[str] = [],
        validation_split: float = 0.1,
        pair_labels: Optional[np.ndarray] = None,
        pair_names: Optional[List[str]] = None,
    ):
        """Fit the GAN. With pair_labels provided, the generator learns
        P(features | class, pair); without, it falls back to class-only
        conditioning (backward compatible).

        Args:
            pair_labels: shape (N,) integer indices OR (N, num_pairs) one-hot.
                If None, no pair conditioning is used.
            pair_names: optional list of pair names indexed by pair_id, stored
                in metadata so callers can resolve "BTC/USDT" → 0 at generate
                time.
        """
        if categorical_columns:
            warnings.warn(
                f"CTABGANMLX is continuous-only; ignoring categorical_columns="
                f"{list(categorical_columns)}.  Use the TF backend if you need "
                f"categorical support.",
                stacklevel=2,
            )

        df = dataframe.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

        self.column_order = list(df.columns)
        self.continuous_columns = list(df.columns)
        self.num_classes = labels.shape[1] if labels.ndim > 1 else int(labels.max()) + 1

        # Pair conditioning setup
        if pair_labels is not None:
            pair_arr = np.asarray(pair_labels)
            if pair_arr.ndim == 1:
                self.num_pairs = int(pair_arr.max()) + 1
                pair_oh = np.eye(self.num_pairs, dtype=np.float32)[pair_arr.astype(int)]
            else:
                self.num_pairs = pair_arr.shape[1]
                pair_oh = pair_arr.astype(np.float32)
            if pair_oh.shape[0] != len(df):
                raise ValueError(
                    f"pair_labels length {pair_oh.shape[0]} != dataframe length {len(df)}"
                )
            self.pair_names = list(pair_names) if pair_names is not None else None
        else:
            self.num_pairs = 0
            pair_oh = None
            self.pair_names = None

        if self.verbose:
            print(f"    Fitting VGM models for {len(df.columns)} columns...")
        self.vgm_models, self.column_info, self.continuous_info = fit_vgm_columns(
            df,
            integer_columns=(),
            random_state=self.random_seed if self.random_seed is not None else 42,
            verbose=False,
        )
        self.total_dim = sum(v + m for v, m in self.continuous_info)

        encoded = transform_vgm(df, self.vgm_models, self.column_info, self.column_order)
        if labels.ndim == 1:
            class_oh = np.eye(self.num_classes, dtype=np.float32)[labels.astype(int)]
        else:
            class_oh = labels.astype(np.float32)

        # Assemble cond = [class_oh, pair_oh] (pair part omitted if num_pairs=0)
        if pair_oh is not None:
            cond = np.concatenate([class_oh, pair_oh], axis=1)
        else:
            cond = class_oh

        self.gen = Generator(
            self.latent_dim, self.cond_dim, self.continuous_info, self.hidden_dim
        )
        self.critic = Critic(self.total_dim, self.cond_dim, self.hidden_dim)
        self.gen_opt = optim.Adam(learning_rate=self.learning_rate, betas=(0.5, 0.9))
        self.critic_opt = optim.Adam(learning_rate=self.learning_rate, betas=(0.5, 0.9))

        self._run_training(df, encoded, cond)
        self.is_fitted = True
        self._post_train_diagnostics(df, cond, class_oh=class_oh, pair_oh=pair_oh)

    def _post_train_diagnostics(
        self,
        real_df: pd.DataFrame,
        real_cond: np.ndarray,
        class_oh: Optional[np.ndarray] = None,
        pair_oh: Optional[np.ndarray] = None,
    ) -> None:
        """Print per-class (and optionally per-(class, pair)) generator
        output stats after training.

        When pair_oh is provided, the diagnostic iterates over each pair and
        each class within that pair so you can see whether the generator's
        conditional distribution actually tracks the per-pair real
        distribution (the failure mode that motivated pair conditioning).
        """
        try:
            print("\n  --- CTABGANMLX end-of-training diagnostics ---")
            real_vals = real_df.values.astype("float32")
            # Resolve class indices from class_oh if provided, else from full
            # cond (backward compat: cond was just class_oh before).
            if class_oh is None:
                class_oh = real_cond
            class_idx = np.argmax(class_oh, axis=1)

            if pair_oh is None or self.num_pairs == 0:
                # Class-only diagnostic
                for c in range(self.num_classes):
                    mask = class_idx == c
                    if not mask.any():
                        continue
                    real_mean = real_vals[mask].mean(axis=0)
                    gen_df = self.generate(min(200, mask.sum()), class_label=c)
                    gen_mean = gen_df.values.astype("float32").mean(axis=0)
                    rm_str = " ".join(f"{m:+.2f}" for m in real_mean)
                    gm_str = " ".join(f"{m:+.2f}" for m in gen_mean)
                    print(f"  Class {c}: real means = [{rm_str}]")
                    print(f"           gen  means = [{gm_str}]")
            else:
                # Per-(pair, class) diagnostic — the whole point of pair
                # conditioning. Limit to first 4 pairs to keep output sane.
                pair_idx = np.argmax(pair_oh, axis=1)
                pairs_to_show = sorted(np.unique(pair_idx).tolist())[:4]
                for p in pairs_to_show:
                    pname = (
                        self.pair_names[p]
                        if self.pair_names and p < len(self.pair_names)
                        else f"pair_{p}"
                    )
                    for c in range(self.num_classes):
                        mask = (class_idx == c) & (pair_idx == p)
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        real_mean = real_vals[mask].mean(axis=0)
                        gen_df = self.generate(
                            min(200, n), class_label=c, pair_label=p
                        )
                        gen_mean = gen_df.values.astype("float32").mean(axis=0)
                        rm_str = " ".join(f"{m:+.2f}" for m in real_mean)
                        gm_str = " ".join(f"{m:+.2f}" for m in gen_mean)
                        print(f"  {pname} class {c} (n={n}): real = [{rm_str}]")
                        print(f"  {' ' * (len(pname) + 9)}gen  = [{gm_str}]")
                if len(np.unique(pair_idx)) > 4:
                    print(
                        f"  ... {len(np.unique(pair_idx)) - 4} more pairs "
                        "omitted from diagnostic output"
                    )
            print("  --- end diagnostics ---\n")
        except Exception as exc:
            print(f"  MLX diagnostics failed: {exc}")

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #

    def _run_training(
        self,
        original_df: pd.DataFrame,
        encoded: np.ndarray,
        cond: np.ndarray,
    ) -> None:
        X_mx = mx.array(encoded, dtype=mx.float32)
        C_mx = mx.array(cond, dtype=mx.float32)
        n_samples = encoded.shape[0]

        gp_weight = float(self.gp_weight)

        def loss_critic(critic_model, real_x, real_c, z, fake_c):
            fake_x = self.gen(z, fake_c)
            real_score = critic_model(real_x, real_c)
            fake_score = critic_model(fake_x, fake_c)
            w_loss = mx.mean(fake_score) - mx.mean(real_score)
            alpha = mx.random.uniform(shape=(real_x.shape[0], 1))
            interpolated = alpha * real_x + (1.0 - alpha) * fake_x

            # WGAN-GP requires the per-sample gradient ||∇ critic(x_i)||.
            # mx.grad of mx.mean(critic(x)) divides each row by N — using
            # mx.sum recovers the unscaled per-sample gradient.  Without
            # this, the penalty pulls gradient norms toward N instead of
            # toward 1, causing catastrophic loss explosion at any batch
            # size larger than ~64.
            def score_fn(x):
                return mx.sum(critic_model(x, real_c))

            gp_grad = mx.grad(score_fn)(interpolated)
            gp = mx.mean((mx.sqrt(mx.sum(gp_grad ** 2, axis=1) + 1e-8) - 1.0) ** 2)
            return w_loss + gp_weight * gp

        def loss_gen(gen_model, z, c):
            return -mx.mean(self.critic(gen_model(z, c), c))

        critic_grad_fn = nn.value_and_grad(self.critic, loss_critic)
        gen_grad_fn = nn.value_and_grad(self.gen, loss_gen)

        eval_enabled = self.eval_frequency > 0
        ckpt_mode = "max" if eval_enabled else "min"
        ckpt = BestCheckpointTracker(mode=ckpt_mode, min_delta=self.early_stopping_min_delta)
        early = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            mode=ckpt_mode,
        )
        lr_reduce = ReduceLROnPlateau(
            patience=self.reduce_lr_patience,
            factor=self.reduce_lr_factor,
            min_lr=self.reduce_lr_min,
            mode=ckpt_mode,
        )
        diverge = DivergenceMonitor()

        if self.verbose:
            mode_str = "eval-quality" if eval_enabled else "combined-loss (legacy)"
            print(
                f"    Starting MLX CTAB-GAN+ training "
                f"({self.epochs} epochs, monitor={mode_str})..."
            )
            start_t = time.time()

        for epoch in range(self.epochs):
            if self.random_seed is not None:
                rng = np.random.RandomState(self.random_seed + epoch)
                perm = rng.permutation(n_samples)
            else:
                perm = np.random.permutation(n_samples)
            perm_mx = mx.array(perm)

            d_losses: List[float] = []
            g_losses: List[float] = []
            last_d: float = 0.0
            last_g: float = 0.0

            for i in range(0, n_samples, self.batch_size):
                idx = perm_mx[i : i + self.batch_size]
                real_x = X_mx[idx]
                real_c = C_mx[idx]
                bs = real_x.shape[0]

                for _ in range(self.n_critic):
                    z_c = mx.random.normal((bs, self.latent_dim))
                    l_c, g_c = critic_grad_fn(self.critic, real_x, real_c, z_c, real_c)
                    self.critic_opt.update(self.critic, g_c)
                    mx.eval(self.critic.parameters(), self.critic_opt.state)
                    last_d = float(l_c.item())
                    d_losses.append(last_d)

                z = mx.random.normal((bs, self.latent_dim))
                l_g, g_g = gen_grad_fn(self.gen, z, real_c)
                self.gen_opt.update(self.gen, g_g)
                mx.eval(self.gen.parameters(), self.gen_opt.state)
                last_g = float(l_g.item())
                g_losses.append(last_g)

            mx.eval(self.gen.parameters(), self.critic.parameters())
            avg_d = float(np.mean(d_losses)) if d_losses else 0.0
            avg_g = float(np.mean(g_losses)) if g_losses else 0.0

            # Divergence detection uses the LAST batch's losses, not the
            # epoch mean — early-epoch transient spikes inflate the mean
            # past the absolute ceilings (300/500) on large datasets even
            # when training is genuinely converging.  This matches the
            # pre-helper-refactor MLX behaviour.
            if diverge.update(last_d, last_g, ckpt.best):
                if self.verbose:
                    print(
                        f"    Stopping early at epoch {epoch + 1} due to divergence "
                        f"(last-batch d={last_d:.2f}, g={last_g:.2f}; "
                        f"epoch-mean d={avg_d:.2f}, g={avg_g:.2f}). "
                        f"Best epoch was {ckpt.best_epoch + 1}."
                    )
                break

            if eval_enabled:
                run_eval = (epoch + 1) % self.eval_frequency == 0
                if run_eval:
                    eval_metrics = self._evaluate_for_training(original_df)
                    if eval_metrics is not None:
                        quality = float(
                            eval_metrics.get("overall_score", {}).get("overall_quality", 0.0)
                        )
                        ckpt.update(quality, epoch, self.gen, self.critic)
                        stop_now = early.update(quality)
                        reduced, new_lr = lr_reduce.update(
                            quality, [self.gen_opt, self.critic_opt]
                        )
                        if self.verbose:
                            block = eval_metrics.get("overall_score", {})
                            msg = (
                                f"    Epoch {epoch + 1}/{self.epochs}  "
                                f"d={avg_d:.4f} g={avg_g:.4f}  "
                                f"quality={quality:.4f} (best={ckpt.best:.4f} @ ep {ckpt.best_epoch + 1})  "
                                f"div={block.get('diversity_score', 0.0):.3f} "
                                f"corr={block.get('correlation_score', 0.0):.3f} "
                                f"stat={block.get('statistical_score', 0.0):.3f} "
                                f"valid={block.get('validity_score', 0.0):.3f}  "
                                f"lr={float(self.gen_opt.learning_rate):.2e}"
                            )
                            if reduced:
                                msg += f"  ↓ LR -> {new_lr:.2e}"
                            print(msg)
                        if stop_now:
                            if self.verbose:
                                print(
                                    f"    Early stopping at epoch {epoch + 1}. "
                                    f"Best epoch {ckpt.best_epoch + 1} "
                                    f"(quality={ckpt.best:.4f})."
                                )
                            break
                elif self.verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"    Epoch {epoch + 1}/{self.epochs}  d={avg_d:.4f} g={avg_g:.4f}  "
                        f"lr={float(self.gen_opt.learning_rate):.2e}"
                    )
            else:
                # Legacy fallback — combined-loss best-checkpoint, no early stop.
                # Uses last-batch losses to match the pre-helper-refactor behaviour.
                combined = abs(last_d) + abs(last_g)
                ckpt.update(combined, epoch, self.gen, self.critic)
                if self.verbose and epoch % 10 == 0:
                    print(
                        f"    Epoch {epoch}/{self.epochs} | "
                        f"Critic Loss: {last_d:.4f} | Gen Loss: {last_g:.4f}"
                    )

            mx.clear_cache()

        if self.verbose:
            print(f"    CTAB-GAN MLX training complete in {time.time() - start_t:.2f}s")

        if ckpt.has_checkpoint:
            restored = ckpt.restore(self.gen, self.critic)
            if restored:
                mx.eval(self.gen.parameters(), self.critic.parameters())
                if self.verbose:
                    print(
                        f"    Restored best model from epoch {ckpt.best_epoch + 1} "
                        f"(score={ckpt.best:.4f})."
                    )

        ckpt.cleanup()

    def _evaluate_for_training(self, original_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        try:
            n = min(self.eval_num_samples, len(original_df))
            if n <= 0:
                return None
            if self.random_seed is not None:
                rng = np.random.RandomState(self.random_seed + 12345)
                idx = rng.choice(len(original_df), n, replace=False)
            else:
                rng = np.random
                idx = np.random.choice(len(original_df), n, replace=False)
            real_eval = original_df.iloc[idx]
            # For pair-conditioned models, generate with a uniform random
            # mix of (class, pair) — that approximates the training data's
            # marginal distribution, which is what the evaluator compares
            # against.  Build the cond vector explicitly so we don't trip
            # the "pair_label required" guard in generate().
            if self.num_pairs > 0:
                pair_ids = rng.randint(0, self.num_pairs, n)
                class_ids = rng.randint(0, self.num_classes, n)
                pair_oh = np.eye(self.num_pairs, dtype=np.float32)[pair_ids]
                class_oh = np.eye(self.num_classes, dtype=np.float32)[class_ids]
                cond = np.concatenate([class_oh, pair_oh], axis=1)
                gen_df = self.generate(num_samples=n, condition_vector=cond)
            else:
                gen_df = self.generate(num_samples=n)
            return evaluate_with_dataframes(
                real_eval,
                gen_df,
                continuous_columns=self.continuous_columns,
                column_info=self.column_info,
                column_order=self.column_order,
            )
        except Exception as exc:
            if self.verbose:
                print(f"    Evaluation failed: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # _transform — kept for backwards compat with any external callers   #
    # ------------------------------------------------------------------ #

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        return transform_vgm(df, self.vgm_models, self.column_info, self.column_order)

    # ------------------------------------------------------------------ #
    # generate()                                                         #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        num_samples: int,
        class_label: Optional[int] = None,
        condition_vector: Optional[np.ndarray] = None,
        pair_label: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic samples.

        Args:
            num_samples: number of rows to generate.
            class_label: integer class index. Required when condition_vector
                is None and the model has class conditioning.
            condition_vector: full (num_samples, cond_dim) one-hot conditioning
                matrix. Overrides class_label / pair_label when provided.
            pair_label: integer pair index. Required when the model was fit
                with pair_labels (i.e., self.num_pairs > 0). Ignored otherwise.
        """
        if not self.is_fitted and self.gen is None:
            raise ValueError("Model not fitted")

        if condition_vector is not None:
            c = condition_vector[:num_samples]
            if c.shape[0] < num_samples:
                c = np.tile(c, (int(np.ceil(num_samples / c.shape[0])), 1))[:num_samples]
        else:
            # Build class one-hot
            if class_label is not None:
                class_oh = np.zeros((num_samples, self.num_classes), dtype=np.float32)
                class_oh[:, class_label] = 1.0
            else:
                class_oh = np.eye(self.num_classes, dtype=np.float32)[
                    np.random.randint(0, self.num_classes, num_samples)
                ]

            # Build pair one-hot if the model has pair conditioning
            if self.num_pairs > 0:
                if pair_label is None:
                    raise ValueError(
                        f"This model was fit with pair conditioning "
                        f"(num_pairs={self.num_pairs}); pair_label must be "
                        f"provided to generate()."
                    )
                pair_oh = np.zeros((num_samples, self.num_pairs), dtype=np.float32)
                pair_oh[:, pair_label] = 1.0
                c = np.concatenate([class_oh, pair_oh], axis=1)
            else:
                c = class_oh

        z = mx.random.normal((num_samples, self.latent_dim))
        c_mx = mx.array(c, dtype=mx.float32)
        gen_data = self.gen(z, c_mx)
        mx.eval(gen_data)
        gen_np = np.array(gen_data)

        decoded = inverse_transform_vgm(
            gen_np, self.vgm_models, self.column_info, self.column_order
        )
        return pd.DataFrame(decoded, columns=self.column_order)

    # ------------------------------------------------------------------ #
    # evaluate()                                                         #
    # ------------------------------------------------------------------ #

    def evaluate_with_dataframes(
        self, real_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> Dict[str, Any]:
        return evaluate_with_dataframes(
            real_data,
            generated_data,
            continuous_columns=self.continuous_columns,
            column_info=self.column_info,
            column_order=self.column_order,
        )

    # ------------------------------------------------------------------ #
    # save() / load()                                                    #
    # ------------------------------------------------------------------ #

    def save(self, path: str, **extra_metadata: Any):
        if self.gen is None:
            raise RuntimeError("No fitted model to save.")
        os.makedirs(path, exist_ok=True)
        meta: Dict[str, Any] = {
            "column_info": self.column_info,
            "vgm_models": self.vgm_models,
            "column_order": self.column_order,
            "continuous_columns": self.continuous_columns,
            "continuous_info": self.continuous_info,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "num_pairs": self.num_pairs,
            "pair_names": self.pair_names,
            "total_dim": self.total_dim,
        }
        for key, value in extra_metadata.items():
            if value is None:
                continue
            if key in ("min_buy_gain_threshold", "min_sell_loss_threshold"):
                meta[key] = float(value)
            elif key == "training_type":
                meta[key] = int(value)
            else:
                meta[key] = value
        with open(os.path.join(path, "metadata_mlx.pkl"), "wb") as f:
            pickle.dump(meta, f)
        self.gen.save_weights(os.path.join(path, "gen_mlx.safetensors"))

    def load(self, path: str) -> Dict[str, Any]:
        with open(os.path.join(path, "metadata_mlx.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.column_info = meta["column_info"]
        self.vgm_models = meta["vgm_models"]
        self.column_order = meta["column_order"]
        self.continuous_columns = meta.get("continuous_columns", list(self.column_order))
        self.continuous_info = meta["continuous_info"]
        self.latent_dim = meta["latent_dim"]
        self.hidden_dim = meta.get("hidden_dim", 256)
        self.num_classes = meta["num_classes"]
        # Backward compatible — pre-pair-conditioning models lack these keys.
        self.num_pairs = meta.get("num_pairs", 0)
        self.pair_names = meta.get("pair_names")
        self.total_dim = meta["total_dim"]

        self.gen = Generator(
            self.latent_dim, self.cond_dim, self.continuous_info, self.hidden_dim
        )
        self.gen.load_weights(os.path.join(path, "gen_mlx.safetensors"))
        self.is_fitted = True

        # Return ALL persisted metadata keys, not just a hardcoded subset.
        # See sibling CTABGANMLXMT.load — save() accepts arbitrary
        # extra_metadata (including future-added keys like 'horizon'),
        # so load must mirror that or the validator sees the saved file
        # as "missing" keys it actually has.
        return dict(meta)
