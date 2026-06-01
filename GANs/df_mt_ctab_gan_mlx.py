"""
Multi-Task CTAB-GAN+ — MLX (Apple Metal) backend.

Mirrors the API of ``GANs.df_mt_ctab_gan.CTABGANPlusMT`` so callers can
switch between TF and MLX transparently via ``GANInterface`` /
``GANs.backends.ctab_gan``:

    model = CTABGANMLXMT(epochs=300, batch_size=512)
    model.fit(dataframe, labels={"trading": ..., "regime": ...})
    gen_df, labels_dict = model.generate(num_samples=1000)
    model.save("/path/to/dir", min_buy_gain_threshold=0.016)
    model.load("/path/to/dir")

Differences from the TF variant (intentional, scoped):

  * Continuous-only.  Categorical columns are warned about and
    dropped — the MLX variants don't carry the categorical encoding
    machinery.
  * No PacGAN — the existing single-task MLX class doesn't have it
    either, and adding it isn't on this iteration's scope.
  * Monitor metric is ``eval_quality`` only.  The TF version's
    multi-metric switching isn't ported.

Shared logic (VGM encoding, evaluation, callbacks) lives in
``mlx_ctab_helpers`` so both MLX classes stay in sync.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# ===========================================================================
# Networks — same shape as the single-task MLX generator/critic, with
# ``num_classes`` replaced by ``cond_dim`` (the concatenated multi-task
# condition vector dimension).
# ===========================================================================


class _Generator(nn.Module):
    """Conditional generator producing per-column (scalar + mode probs)."""

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        continuous_info: List[Tuple[int, int]],
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.continuous_info = continuous_info
        self.total_dim = sum(v + m for v, m in continuous_info)

        self.trunk = nn.Sequential(
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
        self._head_names: List[str] = []
        for i, (val_dim, num_modes) in enumerate(continuous_info):
            scalar_name = f"head_scalar_{i}"
            setattr(self, scalar_name, nn.Linear(hidden_dim, 1))
            self._head_names.append(scalar_name)
            if num_modes > 0:
                mode_name = f"head_mode_{i}"
                setattr(self, mode_name, nn.Linear(hidden_dim, num_modes))
                self._head_names.append(mode_name)

    def __call__(self, z, c):
        h = self.trunk(mx.concatenate([z, c], axis=-1))
        outputs: List[Any] = []
        idx = 0
        for val_dim, num_modes in self.continuous_info:
            scalar_layer = getattr(self, self._head_names[idx])
            outputs.append(mx.tanh(scalar_layer(h)))
            idx += 1
            if num_modes > 0:
                mode_layer = getattr(self, self._head_names[idx])
                outputs.append(mx.softmax(mode_layer(h) / 0.2, axis=-1))
                idx += 1
        return mx.concatenate(outputs, axis=-1)


class _Critic(nn.Module):
    """Conditional Wasserstein critic."""

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


# ===========================================================================
# Multi-task CTAB-GAN+ — MLX
# ===========================================================================


def _concat_task_labels(task_labels: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate task labels in sorted-key order for a stable condition vector."""
    return np.concatenate([task_labels[k] for k in sorted(task_labels.keys())], axis=1)


class CTABGANMLXMT:
    """MLX backend for multi-task CTAB-GAN+ (continuous-only)."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        batch_size: int = 512,
        epochs: int = 300,
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
    ) -> None:
        if monitor_metric != "eval_quality":
            raise ValueError(
                f"monitor_metric={monitor_metric!r} is not supported by the MLX "
                f"multi-task variant — only 'eval_quality' is available.  Use the "
                f"TF backend if you need loss-based monitoring."
            )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
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

        # Populated by fit() / load().
        self.is_fitted: bool = False
        self.gen: Optional[_Generator] = None
        self.critic: Optional[_Critic] = None
        self.gen_opt: Optional[optim.Adam] = None
        self.critic_opt: Optional[optim.Adam] = None

        self.column_order: List[str] = []
        self.continuous_columns: List[str] = []
        self.vgm_models: Dict[str, Any] = {}
        self.column_info: Dict[str, dict] = {}
        self.continuous_info: List[Tuple[int, int]] = []
        self.total_dim: int = 0

        self.task_label_dims: Dict[str, int] = {}
        self.sorted_tasks: List[str] = []
        self.total_cond_dim: int = 0

    # ------------------------------------------------------------------ #
    # fit()                                                              #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        dataframe: pd.DataFrame,
        labels: Dict[str, np.ndarray],
        categorical_columns: Sequence[str] = (),
        validation_split: float = 0.1,
    ) -> None:
        if dataframe.empty:
            raise ValueError("dataframe cannot be empty")
        if not isinstance(labels, dict) or not labels:
            raise ValueError("labels must be a non-empty dict of task arrays")
        sizes = {k: arr.shape[0] for k, arr in labels.items()}
        if len(set(sizes.values())) > 1:
            raise ValueError(f"all task labels must share batch size; got {sizes}")
        n_samples = next(iter(sizes.values()))
        if n_samples != len(dataframe):
            raise ValueError(
                f"dataframe rows ({len(dataframe)}) != label batch size ({n_samples})"
            )

        if categorical_columns:
            warnings.warn(
                f"CTABGANMLXMT is continuous-only; ignoring categorical_columns="
                f"{list(categorical_columns)}.  Use the TF backend if you need "
                f"categorical support.",
                stacklevel=2,
            )

        df = dataframe.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
        self.column_order = list(df.columns)
        self.continuous_columns = list(df.columns)

        labels_processed: Dict[str, np.ndarray] = {}
        for task, arr in labels.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                num_classes = int(arr.max()) + 1
                labels_processed[task] = np.eye(num_classes, dtype=np.float32)[arr.astype(int)]
            elif arr.ndim == 2:
                labels_processed[task] = arr.astype(np.float32)
            else:
                raise ValueError(f"Task '{task}' labels must be 1D or 2D")
        self.task_label_dims = {k: v.shape[1] for k, v in labels_processed.items()}
        self.sorted_tasks = sorted(self.task_label_dims.keys())
        self.total_cond_dim = sum(self.task_label_dims.values())

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
        cond = _concat_task_labels(labels_processed)

        self.gen = _Generator(
            self.latent_dim, self.total_cond_dim, self.continuous_info, self.hidden_dim
        )
        self.critic = _Critic(self.total_dim, self.total_cond_dim, self.hidden_dim)
        self.gen_opt = optim.Adam(learning_rate=self.learning_rate, betas=(0.5, 0.9))
        self.critic_opt = optim.Adam(learning_rate=self.learning_rate, betas=(0.5, 0.9))

        self._run_training(df, encoded, cond)
        self.is_fitted = True
        self._post_train_diagnostics(df, labels_processed)

    def _post_train_diagnostics(
        self,
        real_df: pd.DataFrame,
        real_labels: Dict[str, np.ndarray],
    ) -> None:
        """Per-class generator-output stats after training, MT version.

        Reports real vs generated per-feature mean for each class of the
        primary task ('trading' if present, else first sorted task).
        Other tasks fixed at class 0 so the comparison isolates the
        primary task's conditioning.
        """
        try:
            print("\n  --- MT CTABGANMLX end-of-training diagnostics ---")
            primary = "trading" if "trading" in self.sorted_tasks else self.sorted_tasks[0]
            primary_idx = np.argmax(real_labels[primary], axis=1)
            real_vals = real_df.values.astype("float32")
            for c in range(self.task_label_dims[primary]):
                mask = primary_idx == c
                if not mask.any():
                    continue
                real_mean = real_vals[mask].mean(axis=0)
                n = min(200, int(mask.sum()))
                task_labels_full = {
                    task: np.tile(
                        np.eye(self.task_label_dims[task], dtype=np.float32)[
                            c if task == primary else 0
                        ],
                        (n, 1),
                    )
                    for task in self.sorted_tasks
                }
                gen_df, _ = self.generate(n, task_labels=task_labels_full)
                gen_mean = gen_df.values.astype("float32").mean(axis=0)
                rm_str = " ".join(f"{m:+.2f}" for m in real_mean)
                gm_str = " ".join(f"{m:+.2f}" for m in gen_mean)
                print(f"  '{primary}' Class {c}: real means = [{rm_str}]")
                print(f"                  gen  means = [{gm_str}]")
            print("  --- end diagnostics ---\n")
        except Exception as exc:
            print(f"  MT MLX diagnostics failed: {exc}")

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
            gp = mx.mean(
                (mx.sqrt(mx.sum(gp_grad ** 2, axis=1) + 1e-8) - 1.0) ** 2
            )
            return w_loss + gp_weight * gp

        def loss_gen(gen_model, z, c):
            return -mx.mean(self.critic(gen_model(z, c), c))

        critic_grad_fn = nn.value_and_grad(self.critic, loss_critic)
        gen_grad_fn = nn.value_and_grad(self.gen, loss_gen)

        ckpt = BestCheckpointTracker(mode="max", min_delta=self.early_stopping_min_delta)
        early = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            mode="max",
        )
        lr_reduce = ReduceLROnPlateau(
            patience=self.reduce_lr_patience,
            factor=self.reduce_lr_factor,
            min_lr=self.reduce_lr_min,
            mode="max",
        )
        diverge = DivergenceMonitor()

        if self.verbose:
            print(
                f"    Starting MLX MT CTAB-GAN+ training "
                f"({self.epochs} epochs, eval every {self.eval_frequency}, "
                f"early-stop patience {self.early_stopping_patience})..."
            )
            start_t = time.time()

        stopped_early = False

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
            # when training is genuinely converging.
            if diverge.update(last_d, last_g, ckpt.best):
                if self.verbose:
                    print(
                        f"    Stopping early at epoch {epoch + 1} due to divergence "
                        f"(last-batch d={last_d:.2f}, g={last_g:.2f}; "
                        f"epoch-mean d={avg_d:.2f}, g={avg_g:.2f}). "
                        f"Best epoch was {ckpt.best_epoch + 1} "
                        f"(quality={ckpt.best:.4f})."
                    )
                stopped_early = True
                break

            run_eval = (
                self.eval_frequency > 0
                and (epoch + 1) % self.eval_frequency == 0
            )
            quality: Optional[float] = None
            eval_metrics: Optional[Dict[str, Any]] = None
            if run_eval:
                eval_metrics = self._evaluate_for_training(original_df)
                if eval_metrics is not None:
                    quality = float(
                        eval_metrics.get("overall_score", {}).get("overall_quality", 0.0)
                    )

            if quality is not None:
                ckpt.update(quality, epoch, self.gen, self.critic)
                stop_now = early.update(quality)
                reduced, new_lr = lr_reduce.update(quality, [self.gen_opt, self.critic_opt])

                if self.verbose:
                    overall_block = eval_metrics.get("overall_score", {}) if eval_metrics else {}
                    msg = (
                        f"    Epoch {epoch + 1}/{self.epochs}  "
                        f"d={avg_d:.4f} g={avg_g:.4f}  "
                        f"quality={quality:.4f} (best={ckpt.best:.4f} @ ep {ckpt.best_epoch + 1})  "
                        f"div={overall_block.get('diversity_score', 0.0):.3f} "
                        f"corr={overall_block.get('correlation_score', 0.0):.3f} "
                        f"stat={overall_block.get('statistical_score', 0.0):.3f} "
                        f"valid={overall_block.get('validity_score', 0.0):.3f}  "
                        f"lr={float(self.gen_opt.learning_rate):.2e}"
                    )
                    if reduced:
                        msg += f"  ↓ LR -> {new_lr:.2e}"
                    print(msg)

                if stop_now:
                    if self.verbose:
                        print(
                            f"    Early stopping at epoch {epoch + 1} "
                            f"(no improvement for {self.early_stopping_patience} eval cycles). "
                            f"Best epoch {ckpt.best_epoch + 1} (quality={ckpt.best:.4f})."
                        )
                    stopped_early = True
                    break
            elif self.verbose and (epoch + 1) % max(self.eval_frequency, 1) == 0:
                print(
                    f"    Epoch {epoch + 1}/{self.epochs}  d={avg_d:.4f} g={avg_g:.4f}  "
                    f"lr={float(self.gen_opt.learning_rate):.2e}"
                )

            mx.clear_cache()

        if self.verbose:
            print(f"    MLX MT CTAB-GAN+ training complete in {time.time() - start_t:.2f}s")

        if ckpt.has_checkpoint:
            restored = ckpt.restore(self.gen, self.critic)
            if restored:
                mx.eval(self.gen.parameters(), self.critic.parameters())
                if self.verbose:
                    print(
                        f"    Restored best model from epoch {ckpt.best_epoch + 1} "
                        f"(quality={ckpt.best:.4f})."
                    )

        ckpt.cleanup()
        _ = stopped_early  # marker — read by tests/debug

    def _evaluate_for_training(self, original_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate and score a sample for during-training quality monitoring."""
        try:
            n = min(self.eval_num_samples, len(original_df))
            if n <= 0:
                return None
            if self.random_seed is not None:
                rng = np.random.RandomState(self.random_seed + 12345)
                idx = rng.choice(len(original_df), n, replace=False)
            else:
                idx = np.random.choice(len(original_df), n, replace=False)
            real_eval = original_df.iloc[idx]
            gen_df, _ = self.generate(num_samples=n, task_labels=None)
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
    # generate()                                                         #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        num_samples: int,
        task_labels: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        if self.gen is None:
            raise RuntimeError("Model is not fitted; call fit() or load() first.")

        if task_labels is None:
            if self.random_seed is not None:
                rng = np.random.RandomState(self.random_seed + 2000)
            else:
                rng = np.random
            task_labels = {}
            for task in self.sorted_tasks:
                num_classes = self.task_label_dims[task]
                idx = rng.randint(0, num_classes, size=num_samples)
                task_labels[task] = np.eye(num_classes, dtype=np.float32)[idx]
        else:
            if set(task_labels.keys()) != set(self.task_label_dims.keys()):
                raise ValueError(
                    f"task_labels must contain all tasks: {list(self.task_label_dims.keys())}"
                )
            for task in self.sorted_tasks:
                expected = (num_samples, self.task_label_dims[task])
                if task_labels[task].shape != expected:
                    raise ValueError(
                        f"Task '{task}' label shape {task_labels[task].shape} "
                        f"!= expected {expected}"
                    )

        cond = _concat_task_labels(task_labels)
        z = mx.random.normal((num_samples, self.latent_dim))
        c_mx = mx.array(cond, dtype=mx.float32)
        gen_x = self.gen(z, c_mx)
        mx.eval(gen_x)
        gen_np = np.array(gen_x)

        decoded = inverse_transform_vgm(
            gen_np, self.vgm_models, self.column_info, self.column_order
        )
        return pd.DataFrame(decoded, columns=self.column_order), task_labels

    # ------------------------------------------------------------------ #
    # evaluate()                                                         #
    # ------------------------------------------------------------------ #

    def evaluate_with_dataframes(
        self, real_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Public wrapper around the shared helper."""
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

    _META_FILE: str = "metadata_mlx.pkl"
    _GEN_FILE: str = "gen_mlx.safetensors"
    _CRITIC_FILE: str = "critic_mlx.safetensors"

    def save(self, filepath: str, **extra_metadata: Any) -> None:
        """Persist generator/critic weights and metadata.

        Accepts arbitrary ``extra_metadata`` kwargs and stores them in the
        pickled metadata dict. The permissive signature mirrors the
        single-task ``CTABGANMLX.save`` so that ``_master_save_kwargs``
        on the Create-class trainer can evolve (e.g. add ``horizon``,
        ``feature_set_id``, …) without breaking persistence on every
        backend. Known-typed keys (``min_buy_gain_threshold``,
        ``min_sell_loss_threshold``, ``training_type``) are cast for
        downstream determinism; everything else is stored as-is.
        """
        if self.gen is None:
            raise RuntimeError("No fitted model to save.")
        os.makedirs(filepath, exist_ok=True)
        self.gen.save_weights(os.path.join(filepath, self._GEN_FILE))
        self.critic.save_weights(os.path.join(filepath, self._CRITIC_FILE))

        meta: Dict[str, Any] = {
            "column_order": self.column_order,
            "continuous_columns": self.continuous_columns,
            "column_info": self.column_info,
            "vgm_models": self.vgm_models,
            "continuous_info": self.continuous_info,
            "total_dim": self.total_dim,
            "task_label_dims": self.task_label_dims,
            "total_cond_dim": self.total_cond_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
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

        with open(os.path.join(filepath, self._META_FILE), "wb") as f:
            pickle.dump(meta, f)

    def load(self, filepath: str) -> Dict[str, Any]:
        meta_path = os.path.join(filepath, self._META_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata file not found at {meta_path} — model may have been "
                f"saved with the TF backend (use that backend to load)."
            )
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.column_order = meta["column_order"]
        self.continuous_columns = meta["continuous_columns"]
        self.column_info = meta["column_info"]
        self.vgm_models = meta["vgm_models"]
        self.continuous_info = meta["continuous_info"]
        self.total_dim = meta["total_dim"]
        self.task_label_dims = meta["task_label_dims"]
        self.sorted_tasks = sorted(self.task_label_dims.keys())
        self.total_cond_dim = meta["total_cond_dim"]
        self.latent_dim = meta["latent_dim"]
        self.hidden_dim = meta.get("hidden_dim", 256)

        self.gen = _Generator(
            self.latent_dim, self.total_cond_dim, self.continuous_info, self.hidden_dim
        )
        self.critic = _Critic(self.total_dim, self.total_cond_dim, self.hidden_dim)
        self.gen.load_weights(os.path.join(filepath, self._GEN_FILE))
        self.critic.load_weights(os.path.join(filepath, self._CRITIC_FILE))
        self.gen_opt = optim.Adam(learning_rate=self.learning_rate, betas=(0.5, 0.9))
        self.critic_opt = optim.Adam(learning_rate=self.learning_rate, betas=(0.5, 0.9))
        self.is_fitted = True

        # Return ALL persisted metadata keys, not just a hardcoded subset.
        # The save() method accepts arbitrary **extra_metadata and writes
        # whatever the caller passes — _master_save_kwargs in particular
        # has grown (horizon, etc.). Whitelisting on load drops those new
        # keys before the validator sees them, which makes pre-existing
        # GAN saves look like they're missing fields they actually have.
        # Internal structural keys (column_info, vgm_models, …) are already
        # bound to self above; returning them again costs nothing but
        # extra dict refs.
        return dict(meta)
