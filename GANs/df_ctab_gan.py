"""
CTAB-GAN+ for conditional tabular data generation.

CTAB-GAN+ is specifically designed for tabular data with mixed data types
(continuous and categorical). It uses conditional generation to create
samples of specific classes.

API:
    model = CTABGANPlus()
    model.fit(dataframe, labels, categorical_columns=['col1', 'col2'])
    generated_samples = model.generate(num_samples=1000, class_label=1)
"""

from __future__ import annotations

from GANs.ctab_gan_base import CTABGANPlusBase, CTABGANPlusEnhancedMixin

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning
import warnings
import os
import pickle
from typing import List, Optional, Dict, Any
from scipy.spatial.distance import pdist


@tf.keras.utils.register_keras_serializable()
class GumbelSoftmax(layers.Layer):
    """Gumbel-Softmax layer for categorical feature generation.
    Returns sharp distributions during training while remaining differentiable,
    and returns standard softmax probabilities during inference."""

    def __init__(self, temperature=0.2, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, logits, training=None):
        if training:
            u = tf.random.uniform(
                tf.shape(logits), minval=1e-5, maxval=1.0 - 1e-5, dtype=logits.dtype
            )
            gumbel_noise = -tf.math.log(-tf.math.log(u))

            # Use value property if it's a Variable, otherwise use it directly
            temp_val = (
                self.temperature.value()
                if hasattr(self.temperature, "value")
                else self.temperature
            )
            temp = tf.cast(temp_val, logits.dtype)

            return tf.nn.softmax((logits + gumbel_noise) / temp, axis=-1)
        else:
            return tf.nn.softmax(logits, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


class CTABGANPlus(CTABGANPlusBase):
    """
    CTAB-GAN+ model for conditional tabular data generation.

    Handles mixed data types (continuous and categorical) and generates
    samples conditioned on class labels.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        generator_layers: List[int] = [256, 256],
        discriminator_layers: List[int] = [256, 256],
        batch_size: int = 1024,  # Power of 2 for optimal GPU utilization
        epochs: int = 300,
        learning_rate: float = 2e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        gp_weight: float = 10.0,
        verbose: bool = True,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-4,
        reduce_lr_patience: int = 10,
        reduce_lr_factor: float = 0.5,
        reduce_lr_min: float = 1e-6,
        pac: int = 10,  # PacGAN packing factor to prevent mode collapse
        monitor_metric: str = "eval_quality",  # "g_loss", "d_loss", "combined", "eval_quality", "eval_diversity", etc.
        eval_frequency: int = 5,  # Evaluate every N epochs for display when using loss metrics.
        # When using eval metrics for best epoch selection, evaluation
        # happens EVERY epoch and is displayed every epoch.
        eval_num_samples: int = 1000,  # Number of samples for evaluation
        random_seed: Optional[
            int
        ] = 42,  # Random seed for reproducibility (None for non-deterministic)
        integer_columns: List[
            str
        ] = [],  # Columns to treat as simple continuous (linear) without VGM
        n_critic: int = 5,  # WGAN-GP discriminator updates per generator update
    ):
        """
        Initialize CTAB-GAN+ model.

        Args:
            latent_dim: Dimension of latent noise vector
            generator_layers: List of hidden layer sizes for generator
            discriminator_layers: List of hidden layer sizes for discriminator
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for Adam optimizer
            beta_1: Beta1 for Adam optimizer
            beta_2: Beta2 for Adam optimizer
            gp_weight: Weight for gradient penalty
            verbose: Whether to print training progress
            early_stopping_patience: Number of epochs to wait before stopping if no improvement
            early_stopping_min_delta: Minimum change to qualify as an improvement
            reduce_lr_patience: Number of epochs to wait before reducing learning rate
            reduce_lr_factor: Factor by which to reduce learning rate
            reduce_lr_min: Minimum learning rate
            monitor_metric: Metric to monitor for early stopping/LR reduction.
                          Options: "g_loss", "d_loss", "combined", "eval_quality",
                          "eval_diversity", "eval_correlation", "eval_statistical",
                          "eval_validity". Evaluation metrics (eval_*) are preferred
                          as they better reflect model quality.
            eval_frequency: Evaluate model every N epochs during training (0 to disable, default: 10)
            eval_num_samples: Number of samples to generate for evaluation (default: 1000)
            random_seed: Random seed for reproducibility. Set to None for non-deterministic
                        behavior. Default: 42
        """
        self.latent_dim = latent_dim
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gp_weight = gp_weight
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_min = reduce_lr_min
        self.pac = pac
        self.monitor_metric = monitor_metric
        self.eval_frequency = eval_frequency
        self.eval_num_samples = eval_num_samples
        self.random_seed = random_seed
        self.integer_columns = integer_columns
        self.n_critic = n_critic

        # Temperature variable for Gumbel-Softmax annealing
        self.temperature = tf.Variable(
            0.9, trainable=False, dtype=tf.float32, name="gumbel_temperature"
        )

        # Set random seeds for reproducibility
        if self.random_seed is not None:
            self._set_random_seeds(self.random_seed)

        # Will be set during fit()
        self.categorical_columns: List[str] = []
        self.continuous_columns: List[str] = []
        self.num_classes: int = 0
        self.num_features: int = 0
        self.vgm_models: Dict[str, BayesianGaussianMixture] = (
            {}
        )  # For Variational Gaussian Mixture
        self.column_info: Dict[str, Any] = {}
        self.generator: Optional[Model] = None
        self.discriminator: Optional[Model] = None
        self.gan: Optional[Model] = None
        self.auxiliary: Optional[Model] = (
            None  # Used by CTABGANPlusEnhanced for downstream loss
        )
        self.is_fitted = False

        # Enable mixed precision training for better GPU utilization (2x speedup on modern GPUs)
        self.use_mixed_precision = True
        if self.use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                if self.verbose:
                    print("    Mixed precision training enabled (float16/float32)")
            except (AttributeError, RuntimeError):
                # Mixed precision not available - that's okay
                self.use_mixed_precision = False
                if self.verbose:
                    print("    Mixed precision not available, using float32")

        # Configure GPU for better utilization (must be called after verbose is set)
        self._configure_gpu()


    def _configure_gpu(self):
        """Configure GPU for optimal utilization and suggest batch size."""
        # Check available GPUs
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 0:
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # # Collect GPU info for batch size recommendations
                # gpu_details = []
                # for gpu in gpus:
                #     memory_gb = None
                #     device_name = gpu.name

                #     try:
                #         # Try to get device details (may include memory info)
                #         device_details = tf.config.experimental.get_device_details(gpu)
                #         if isinstance(device_details, dict):
                #             if "device_name" in device_details:
                #                 device_name = device_details["device_name"]
                #             # Some GPU drivers report memory in device details
                #             if "memory" in device_details:
                #                 memory_gb = device_details["memory"] / (1024**3)
                #     except (AttributeError, RuntimeError, KeyError):
                #         # Device details not available - that's okay
                #         pass

                #     gpu_details.append({"name": device_name, "memory_gb": memory_gb})

                # if self.verbose:
                #     print(
                #         f"    Configured {len(gpus)} GPU(s) with memory growth enabled"
                #     )
                #     for detail in gpu_details:
                #         mem_str = (
                #             f" ({detail['memory_gb']:.1f} GB)"
                #             if detail["memory_gb"]
                #             else ""
                #         )
                #         print(f"      GPU: {detail['name']}{mem_str}")

                #     # Try to get GPU memory info using TensorFlow's memory info
                #     try:
                #         # For GPUs that support it, get actual memory stats
                #         gpu_memory_info = None
                #         if len(gpus) > 0:
                #             # Try to get memory info (works on CUDA GPUs, may not work on Metal/AMD)
                #             try:
                #                 from tensorflow.python.client import device_lib

                #                 local_devices = device_lib.list_local_devices()
                #                 for device in local_devices:
                #                     if "GPU" in device.name:
                #                         if hasattr(device, "memory_limit"):
                #                             gpu_memory_info = device.memory_limit / (
                #                                 1024**3
                #                             )
                #                             break
                #             except (ImportError, AttributeError):
                #                 pass
                #     except Exception:
                #         pass

                #     # Suggest batch size based on GPU memory or system RAM (for unified memory)
                #     suggested_batch = self.batch_size
                #     memory_info_str = ""

                #     if gpu_details and gpu_details[0]["memory_gb"]:
                #         memory_gb = gpu_details[0]["memory_gb"]
                #         memory_info_str = f"{memory_gb:.1f} GB GPU memory"
                #         if memory_gb >= 24:
                #             suggested_batch = 4096
                #         elif memory_gb >= 16:
                #             suggested_batch = 2048
                #         elif memory_gb >= 8:
                #             suggested_batch = 2048  # Can often handle this
                #         else:
                #             suggested_batch = 1024
                #     elif gpu_memory_info:
                #         memory_gb = gpu_memory_info
                #         memory_info_str = f"{memory_gb:.1f} GB GPU memory"
                #         if memory_gb >= 24:
                #             suggested_batch = 4096
                #         elif memory_gb >= 16:
                #             suggested_batch = 2048
                #         elif memory_gb >= 8:
                #             suggested_batch = 2048
                #         else:
                #             suggested_batch = 1024
                #     else:
                #         # On unified memory systems (Apple Silicon), suggest based on typical capabilities
                #         # With 64GB system RAM, unified memory GPUs can often handle larger batches
                #         memory_info_str = "unified memory (GPU shares system RAM)"
                #         suggested_batch = (
                #             2048  # Conservative estimate for unified memory
                #         )
                #         print(
                #             f"    💡 Note: GPU memory not directly reported (unified memory system?). "
                #             f"With 64GB system RAM, consider trying batch_size=2048 or 4096"
                #         )

                #     if self.batch_size < suggested_batch:
                #         print(
                #             f"    💡 Tip: Consider increasing batch_size to {suggested_batch} "
                #             f"for better GPU utilization (current: {self.batch_size})"
                #         )
                #         if memory_info_str:
                #             print(f"      Based on {memory_info_str}")
                #     else:
                #         if memory_info_str:
                #             print(
                #                 f"    Batch size {self.batch_size} is appropriate for {memory_info_str}"
                #             )
            except RuntimeError as e:
                if self.verbose:
                    print(f"    GPU configuration warning: {e}")
        else:
            if self.verbose:
                print("    No GPU detected, using CPU")
                if self.batch_size > 1024:
                    print(
                        f"    💡 Tip: Large batch_size ({self.batch_size}) may slow down "
                        f"CPU training. Consider reducing to 512-1024 for CPU."
                    )


    def fit(
        self,
        dataframe: pd.DataFrame,
        labels: np.ndarray,
        categorical_columns: List[str],
        validation_split: float = 0.1,
    ):
        """
        Fit the CTAB-GAN+ model to the provided dataframe.

        Args:
            dataframe: Input dataframe with mixed data types
            labels: Class labels (1D array of class indices or one-hot encoded)
            categorical_columns: List of column names that are categorical
            validation_split: Fraction of data to use for validation
        """
        if dataframe.empty:
            raise ValueError("Dataframe cannot be empty")

        # Store column information
        self.categorical_columns = categorical_columns
        self.continuous_columns = [
            col for col in dataframe.columns if col not in categorical_columns
        ]

        # Log column distribution if verbose
        if self.verbose:
            print(
                f"    Data columns: {len(dataframe.columns)} total, "
                f"{len(self.categorical_columns)} categorical, "
                f"{len(self.continuous_columns)} continuous"
            )
            if self.continuous_columns:
                print("    Fitting VGM models for continuous columns...")

        # Process labels
        if labels.ndim == 1:
            # Convert to one-hot if needed
            num_classes = int(labels.max()) + 1
            labels_one_hot = np.eye(num_classes, dtype=np.float32)[labels.astype(int)]
        else:
            labels_one_hot = labels.astype(np.float32)
            num_classes = labels_one_hot.shape[1]

        self.num_classes = num_classes
        self.num_features = len(dataframe.columns)

        # Analyze categorical columns to get their unique values
        self.column_info = {}
        categorical_info = []
        continuous_info = []
        self.column_order = list(dataframe.columns)  # Preserve original order

        for col in dataframe.columns:
            if col in categorical_columns:
                unique_vals = sorted(dataframe[col].unique())
                num_categories = len(unique_vals)
                # Create mapping from category value to index
                cat_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                idx_to_cat = unique_vals

                # For binary columns (0/1), we only need 2 categories
                # For multi-category, use all unique values
                self.column_info[col] = {
                    "type": "categorical",
                    "num_categories": num_categories,
                    "unique_values": unique_vals,
                    "cat_to_idx": cat_to_idx,
                    "idx_to_cat": idx_to_cat,
                }
                categorical_info.append(num_categories)
            else:
                col_min = float(dataframe[col].min())
                col_max = float(dataframe[col].max())
                col_mean = float(dataframe[col].mean())
                col_std = float(dataframe[col].std())

                if col in self.integer_columns:
                    # Skip VGM for linear/integer features like indices
                    if self.verbose:
                        progress = len(self.vgm_models) + 1
                        print(
                            f"        [{progress}/{len(self.continuous_columns)}] {col}... Linear (Skipped VGM)",
                            flush=True,
                        )

                    self.vgm_models[col] = None
                    self.column_info[col] = {
                        "type": "continuous",
                        "min": col_min,
                        "max": col_max,
                        "mean": col_mean,
                        "std": col_std,
                        "vgm_components": 0,  # 0 signifies no VGM modes
                    }
                    # A simple continuous column uses 1 scalar and 0 modes
                    continuous_info.append((1, 0))
                    continue

                # Ensure data is 2D and drop NaNs for fitting
                clean_data = dataframe[col].dropna().values.reshape(-1, 1)

                # Cap VGM components by sample count to avoid over-parameterization
                # on small datasets (10 components × ~3 params each needs O(100s)
                # of samples per column to be identifiable).
                n_components_capped = max(1, min(10, len(clean_data) // 20))

                bgm = BayesianGaussianMixture(
                    n_components=n_components_capped,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )

                if self.verbose:
                    progress = len(self.vgm_models) + 1
                    print(
                        f"        [{progress}/{len(self.continuous_columns)}] {col}...",
                        end=" ",
                        flush=True,
                    )

                # Handle edge case where column is constant
                if len(np.unique(clean_data)) > 1:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        bgm.fit(clean_data)
                    if self.verbose:
                        print("Done")
                else:
                    if self.verbose:
                        print("Constant (Skipped)")
                    # Constant column: fake a single component
                    bgm.means_ = np.array([[clean_data[0, 0]]])
                    bgm.covariances_ = np.array([[[1e-4]]])
                    bgm.weights_ = np.array([1.0])
                    bgm.n_components = 1
                    bgm.predict_proba = lambda x: np.ones((len(x), 1))

                self.vgm_models[col] = bgm

                # Save info
                self.column_info[col] = {
                    "type": "continuous",
                    "min": col_min,
                    "max": col_max,
                    "mean": col_mean,
                    "std": col_std,
                    "vgm_components": bgm.n_components,
                }
                # A continuous column uses 1 scalar + C components in one-hot mode
                continuous_info.append((1, bgm.n_components))

        # Store categorical info for model creation
        self.categorical_info = categorical_info

        # Calculate continuous dimension sizes
        self.continuous_info = continuous_info

        self.num_categorical_features = len(categorical_columns)
        self.num_continuous_features = len(continuous_info)

        # Verify total matches dataframe columns
        total_features = self.num_categorical_features + self.num_continuous_features
        if total_features != len(dataframe.columns):
            raise ValueError(
                f"Feature count mismatch: categorical ({self.num_categorical_features}) + "
                f"continuous ({self.num_continuous_features}) = {total_features}, "
                f"but dataframe has {len(dataframe.columns)} columns"
            )

        # Calculate expanded feature dimension for encoding
        self.categorical_dim = sum(self.categorical_info)
        # Continuous dimension = sum(value (1) + num_modes)
        self.continuous_dim = sum(val + modes for val, modes in self.continuous_info)

        self.total_feature_dim = self.continuous_dim + self.categorical_dim

        # Create models
        self._create_models()

        # Prepare training data with one-hot encoded categorical columns
        train_data = self._transform_data(dataframe)
        train_labels = labels_one_hot

        # Split validation if needed
        val_data = None
        val_labels = None
        if validation_split > 0:
            split_idx = int(len(train_data) * (1 - validation_split))
            train_data, val_data = train_data[:split_idx], train_data[split_idx:]
            train_labels, val_labels = (
                train_labels[:split_idx],
                train_labels[split_idx:],
            )

        # Store original dataframe for evaluation
        self._original_dataframe = dataframe.copy()

        if self.verbose:
            print("\n    Pre-processing complete. Starting GAN training...")

        # Train the model
        self._train(
            train_data, train_labels, val_data, val_labels, original_dataframe=dataframe
        )

        self.is_fitted = True

    def generate(
        self,
        num_samples: int,
        class_label: Optional[int] = None,
        class_probs: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Args:
            num_samples: Number of samples to generate
            class_label: Specific class label to generate (if None, uses class_probs)
            class_probs: Probability distribution over classes (if None, uses uniform)

        Returns:
            Generated dataframe with same columns as input
        """
        # Check if generator exists (can generate during training)
        if self.generator is None:
            raise ValueError("Generator model not created. Must call fit() first.")

        # Prepare class labels (use seeded random if seed is set)
        if self.random_seed is not None:
            rng = np.random.RandomState(self.random_seed + 2000)
        else:
            rng = np.random

        if class_label is not None:
            labels = np.eye(self.num_classes, dtype=np.float32)[class_label]
            labels = np.tile(labels, (num_samples, 1))
        elif class_probs is not None:
            # Sample from distribution
            classes = rng.choice(self.num_classes, size=num_samples, p=class_probs)
            labels = np.eye(self.num_classes, dtype=np.float32)[classes]
        else:
            # Uniform distribution
            classes = rng.randint(0, self.num_classes, size=num_samples)
            labels = np.eye(self.num_classes, dtype=np.float32)[classes]

        # Generate noise using TensorFlow (GPU-accelerated)
        # Note: TensorFlow random operations respect the global seed set in __init__
        noise = tf.random.normal((num_samples, self.latent_dim), dtype=tf.float32)
        labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Generate samples - use direct call instead of predict for better GPU utilization
        generated = self.generator([noise, labels_t], training=False)

        # Convert to numpy for post-processing
        generated = generated.numpy()

        # Split output into continuous and categorical parts
        continuous_output = generated[:, : self.continuous_dim]
        categorical_output = generated[:, self.continuous_dim :]

        # Process continuous columns using VGM denormalization
        continuous_values = {}
        cont_offset = 0

        for idx, col in enumerate(self.continuous_columns):
            info = self.column_info[col]
            vgm_components = info["vgm_components"]

            # Extract scalar and mode probabilities
            scalar = continuous_output[:, cont_offset]
            offset_mode = cont_offset + 1
            mode_probs = continuous_output[
                :, offset_mode : offset_mode + vgm_components
            ]

            cont_offset += 1 + vgm_components

            if vgm_components > 0:
                # Use VGM to reconstruct the value
                # Get chosen mode for each sample
                modes = np.argmax(mode_probs, axis=1)

                bgm = self.vgm_models[col]
                means = bgm.means_.reshape(1, -1)
                stds = np.sqrt(bgm.covariances_).reshape(1, -1)

                chosen_means = means[0, modes]
                chosen_stds = stds[0, modes]

                # Reverse the standardization
                denormalized = (scalar * 4 * chosen_stds) + chosen_means
            else:
                # Simple Min-Max reverse scaling
                denormalized = (
                    0.5 * (scalar + 1) * (info["max"] - info["min"]) + info["min"]
                )

            # Clip between observed min/max to prevent extreme outliers
            continuous_values[col] = np.clip(denormalized, info["min"], info["max"])

        # Process categorical columns (recover category from one-hot probabilities)
        categorical_values = {}
        cat_offset = 0
        for idx, col in enumerate(self.categorical_columns):
            info = self.column_info[col]
            num_cats = info["num_categories"]

            # Extract probability vector for this column
            probs = categorical_output[:, cat_offset : cat_offset + num_cats]
            cat_offset += num_cats

            # Argmax to find predicted category
            category_indices = np.argmax(probs, axis=1)

            idx_to_cat = info["idx_to_cat"]
            categorical_values[col] = np.array(
                [idx_to_cat[int(cat_idx)] for cat_idx in category_indices]
            )

        # Create dataframe in original column order
        data_dict = {}
        for col in self.column_order:
            if col in categorical_values:
                data_dict[col] = categorical_values[col]
            else:
                data_dict[col] = continuous_values[col]

        generated_df = pd.DataFrame(data_dict, columns=self.column_order)
        return generated_df

    def _create_models(self):
        """Create generator and discriminator models."""
        # Generator
        noise_input = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(self.num_classes,))

        # Concatenate noise and label
        x = layers.Concatenate()([noise_input, label_input])

        # Generator layers
        for layer_size in self.generator_layers:
            x = layers.Dense(layer_size)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        # Output layer splits into continuous branches (scalar + mode_probs) and categorical branches
        outputs = []
        for i, (val_dim, num_modes) in enumerate(self.continuous_info):
            # Scalar output (value within mode)
            scalar = layers.Dense(1, activation="tanh", name=f"cont_val_{i}")(x)
            outputs.append(scalar)

            if num_modes > 0:
                # Categorical output (which mode)
                mode_logits = layers.Dense(num_modes, name=f"cont_mode_logits_{i}")(x)
                mode_probs = GumbelSoftmax(
                    temperature=self.temperature, name=f"cont_mode_probs_{i}"
                )(mode_logits)
                outputs.append(mode_probs)

        cat_outputs = []
        for i, num_cats in enumerate(self.categorical_info):
            logits = layers.Dense(num_cats, name=f"cat_logits_{i}")(x)
            probs = GumbelSoftmax(temperature=self.temperature, name=f"cat_probs_{i}")(
                logits
            )
            cat_outputs.append(probs)

        if cat_outputs:
            all_outputs = layers.Concatenate()(outputs + cat_outputs)
        else:
            all_outputs = (
                layers.Concatenate()(outputs) if len(outputs) > 1 else outputs[0]
            )

        self.generator = Model(
            [noise_input, label_input], all_outputs, name="ctab_gan_generator"
        )

        # Discriminator (1-Lipschitz, no normalization, no dropout)
        # For PacGAN, we handle multiple samples at once.
        # The input dimensions are scaled by pac factor.
        data_input = layers.Input(shape=(self.total_feature_dim * self.pac,))
        label_input_d = layers.Input(shape=(self.num_classes * self.pac,))

        # Concatenate data and label (now both are packed)
        x_d = layers.Concatenate()([data_input, label_input_d])

        # Discriminator layers
        for layer_size in self.discriminator_layers:
            x_d = layers.Dense(layer_size)(x_d)
            x_d = layers.LeakyReLU(0.2)(x_d)

        # Output: single value (Wasserstein distance)
        output_d = layers.Dense(1)(x_d)

        self.discriminator = Model(
            [data_input, label_input_d], output_d, name="ctab_gan_discriminator"
        )

        # Compile models with mixed precision loss scaling if enabled
        if self.use_mixed_precision:
            optimizer_d = tf.keras.mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                )
            )
        else:
            optimizer_d = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
            )

        self.discriminator.compile(
            optimizer=optimizer_d,
            loss=self._wasserstein_loss,
        )

        # GAN (generator + discriminator)
        self.discriminator.trainable = False

        # When compiling the GAN wrapper, we need to pass packed samples to the discriminator
        # So instead of letting Keras handle the GAN wrapper directly, we define a wrapper
        # where the inputs are already shaped as pac-multiples
        gan_noise_input = layers.Input(shape=(self.pac * self.latent_dim,))
        gan_label_input = layers.Input(shape=(self.pac * self.num_classes,))

        # Reshape to run through generator one-by-one (batch size becomes real batch_size)
        reshaped_noise = layers.Reshape((self.pac, self.latent_dim))(gan_noise_input)
        reshaped_label = layers.Reshape((self.pac, self.num_classes))(gan_label_input)

        # We need a Custom Layer or TimeDistributed to run the Generator over the pac dimension
        # A simpler way is to just train generator manually in train step and skip Keras GAN compile
        # But for backward compatibility with Keras Model tracking, we create a dummy GAN model

        # Since we use custom training loops (_train_generator_step), the self.gan is ONLY used
        # to hold the optimizer and track trainable variables. We don't actually call self.gan([x])
        # anywhere during training!
        # So we can just create a dummy model or fix the shape. Let's fix the shape properly:
        gen_output = self.generator([noise_input, label_input])
        # We won't connect them in a Functional model directly because PacGAN requires batch reshaping
        # which is weird to represent in standard Keras stateless graphs.
        # Instead, we just compile the models individually with their optimizers!

        # Compile GAN with mixed precision loss scaling if enabled
        if self.use_mixed_precision:
            optimizer_g = tf.keras.mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                )
            )
        else:
            optimizer_g = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
            )

        # Store optimizer on generator instead of dummy GAN wrapper
        self.generator.optimizer = optimizer_g
        self.discriminator.trainable = True


    def _gradient_penalty(self, real, fake, labels):
        """Calculate gradient penalty for WGAN-GP."""
        # Convert inputs to tensors and ensure consistent float32 dtype
        if not tf.is_tensor(real):
            real = tf.convert_to_tensor(real, dtype=tf.float32)
        else:
            real = tf.cast(real, dtype=tf.float32)

        if not tf.is_tensor(fake):
            fake = tf.convert_to_tensor(fake, dtype=tf.float32)
        else:
            fake = tf.cast(fake, dtype=tf.float32)

        if not tf.is_tensor(labels):
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        else:
            labels = tf.cast(labels, dtype=tf.float32)

        batch_size = tf.shape(real)[0]
        # Use float32 explicitly for alpha to ensure dtype consistency
        alpha = tf.random.uniform((batch_size, 1), 0.0, 1.0, dtype=tf.float32)
        interpolated = alpha * real + (1.0 - alpha) * fake

        # PacGAN reshaping
        pac_size = tf.shape(interpolated)[0] // self.pac
        interpolated_pac = tf.reshape(
            interpolated, [pac_size, self.pac * self.total_feature_dim]
        )
        labels_pac = tf.reshape(labels, [pac_size, self.pac * self.num_classes])

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_pac)
            pred = self.discriminator([interpolated_pac, labels_pac], training=True)

        grads = gp_tape.gradient(pred, [interpolated_pac])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
        gp = tf.reduce_mean(tf.square(norm - 1.0))
        return gp

    @tf.function  # Graph compilation for better GPU utilization (XLA disabled for compatibility)
    def _train_discriminator_step(self, real_data, real_labels, noise):
        """Compiled training step for discriminator."""
        with tf.GradientTape() as d_tape:
            # Generate fake data
            fake_data = self.generator([noise, real_labels], training=True)

            # PacGAN reshaping
            pac_size = tf.shape(real_data)[0] // self.pac
            real_data_pac = tf.reshape(
                real_data, [pac_size, self.pac * self.total_feature_dim]
            )
            fake_data_pac = tf.reshape(
                fake_data, [pac_size, self.pac * self.total_feature_dim]
            )
            real_labels_pac = tf.reshape(
                real_labels, [pac_size, self.pac * self.num_classes]
            )

            # Discriminator scores
            real_scores = self.discriminator(
                [real_data_pac, real_labels_pac], training=True
            )
            fake_scores = self.discriminator(
                [fake_data_pac, real_labels_pac], training=True
            )

            # Wasserstein loss
            d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)

            # Gradient penalty
            gp = self._gradient_penalty(real_data, fake_data, real_labels)
            gp = tf.cast(gp, dtype=d_loss.dtype)
            gp_weight_t = tf.cast(self.gp_weight, dtype=d_loss.dtype)
            d_loss = d_loss + gp_weight_t * gp

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        d_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in d_grads]
        valid_grads = [
            (g, v)
            for g, v in zip(d_grads, self.discriminator.trainable_variables)
            if g is not None
        ]
        if valid_grads:
            self.discriminator.optimizer.apply_gradients(valid_grads)
        return d_loss

    @tf.function  # Graph compilation for better GPU utilization (XLA disabled for compatibility)
    def _train_generator_step(self, labels, noise, real_data=None):
        """Compiled training step for generator. real_data optional for subclasses (info/downstream loss)."""
        with tf.GradientTape() as g_tape:
            fake_data = self.generator([noise, labels], training=True)

            # PacGAN reshaping
            pac_size = tf.shape(fake_data)[0] // self.pac
            fake_data_pac = tf.reshape(
                fake_data, [pac_size, self.pac * self.total_feature_dim]
            )
            labels_pac = tf.reshape(labels, [pac_size, self.pac * self.num_classes])

            fake_scores = self.discriminator([fake_data_pac, labels_pac], training=True)
            g_loss = -tf.reduce_mean(fake_scores)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        g_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in g_grads]
        valid_grads = [
            (g, v)
            for g, v in zip(g_grads, self.generator.trainable_variables)
            if g is not None
        ]
        if valid_grads:
            self.generator.optimizer.apply_gradients(valid_grads)
        return g_loss

    def _train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        original_dataframe: Optional[pd.DataFrame] = None,
    ):
        """Train the CTAB-GAN+ model."""
        n_samples = len(train_data)
        steps_per_epoch = n_samples // self.batch_size

        # Pre-convert data to tensors for better GPU utilization
        # Convert entire dataset to tensors once (more efficient than per-batch conversion)
        train_data_t = tf.constant(train_data, dtype=tf.float32)
        train_labels_t = tf.constant(train_labels, dtype=tf.float32)

        # Early stopping and LR reduction tracking
        # Determine if we're maximizing (eval metrics) or minimizing (losses)
        is_maximizing = self.monitor_metric.startswith("eval_")
        best_metric = float("-inf") if is_maximizing else float("inf")
        patience_counter = 0
        lr_patience_counter = 0
        best_generator_weights = None
        best_discriminator_weights = None
        best_epoch = 0
        divergence_recovery_count = 0  # number of times we restored due to divergence
        DIVERGENCE_D_LOSS_THRESHOLD = -12.0  # d_loss below this = D too strong
        DIVERGENCE_G_LOSS_THRESHOLD = 12.0  # g_loss above this = G collapsing

        # Identify indices for each class for Training-By-Sampling (TBS)
        labels_argmax = np.argmax(train_labels, axis=1)
        class_indices = [
            np.where(labels_argmax == c)[0] for c in range(self.num_classes)
        ]
        # Filter out any classes with zero samples just in case
        class_indices = [idx for idx in class_indices if len(idx) > 0]
        num_present_classes = len(class_indices)

        # Calculate samples per class to maintain batch size
        samples_per_class = self.batch_size // num_present_classes

        # Inform user about evaluation frequency
        if is_maximizing and self.verbose:
            print(
                f"    Using '{self.monitor_metric}' for best epoch selection. "
                f"Evaluation will run every epoch to ensure no best epoch is missed."
            )

        for epoch in range(self.epochs):
            d_losses = []
            g_losses = []

            for step in range(steps_per_epoch):
                # Training-By-Sampling: Create balanced batch with equal representation of all classes
                batch_indices_list = []

                # Use seeded random state if needed
                if self.random_seed is not None:
                    rng = np.random.RandomState(
                        self.random_seed + epoch * steps_per_epoch + step
                    )
                else:
                    rng = np.random

                for idx_array in class_indices:
                    # Sample with replacement to ensure we can always fill the batch
                    sampled = rng.choice(
                        idx_array, size=samples_per_class, replace=True
                    )
                    batch_indices_list.append(sampled)

                batch_indices = np.concatenate(batch_indices_list)
                rng.shuffle(batch_indices)  # Interleave classes

                # Make sure batch_size is divisible by pac
                # Truncate if necessary (rarely an issue as batch_size is usually power of 2 and pac=10)
                if len(batch_indices) % self.pac != 0:
                    remainder = len(batch_indices) % self.pac
                    batch_indices = batch_indices[:-remainder]

                # Convert to tensor and gather
                batch_indices_t = tf.constant(batch_indices, dtype=tf.int32)
                batch_data_t = tf.gather(train_data_t, batch_indices_t)
                batch_labels_t = tf.gather(train_labels_t, batch_indices_t)
                batch_size_actual = tf.shape(batch_data_t)[0]

                # Train discriminator n_critic steps (WGAN-GP)
                for _ in range(self.n_critic):
                    noise = tf.random.normal(
                        (batch_size_actual, self.latent_dim), dtype=tf.float32
                    )
                    d_loss = self._train_discriminator_step(
                        batch_data_t, batch_labels_t, noise
                    )
                    d_losses.append(float(d_loss))

                # Optional: train auxiliary on real batch (used by CTABGANPlusEnhanced)
                if (
                    hasattr(self, "auxiliary")
                    and self.auxiliary is not None
                    and hasattr(self, "_train_auxiliary_step")
                ):
                    self._train_auxiliary_step(batch_data_t, batch_labels_t)

                # Train generator using compiled step (real_data passed for info/downstream loss in Enhanced)
                noise = tf.random.normal(
                    (batch_size_actual, self.latent_dim), dtype=tf.float32
                )
                g_loss = self._train_generator_step(batch_labels_t, noise, batch_data_t)
                g_losses.append(float(g_loss))

            # Anneal Gumbel-Softmax Temperature
            # Decay from 0.9 to 0.1 over the training process smoothly
            progress = epoch / max(1, self.epochs - 1)
            # Linear decay could work, but exponential usually looks nicer
            new_temp = max(0.1, 0.9 * tf.math.exp(-3.0 * progress))
            self.temperature.assign(new_temp)

            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)

            # Divergence detection: restore best weights and reduce LR to recover
            diverged = (
                avg_d_loss < DIVERGENCE_D_LOSS_THRESHOLD
                or avg_g_loss > DIVERGENCE_G_LOSS_THRESHOLD
                or np.isnan(avg_d_loss)
                or np.isnan(avg_g_loss)
                or not np.isfinite(avg_d_loss)
                or not np.isfinite(avg_g_loss)
            )
            if (
                diverged
                and best_generator_weights is not None
                and divergence_recovery_count < 2
            ):
                divergence_recovery_count += 1
                self.generator.set_weights(best_generator_weights)
                self.discriminator.set_weights(best_discriminator_weights)
                current_lr = float(self.discriminator.optimizer.learning_rate.numpy())
                new_lr = max(current_lr * 0.5, self.reduce_lr_min)
                self.discriminator.optimizer.learning_rate.assign(new_lr)
                self.generator.optimizer.learning_rate.assign(new_lr)
                lr_patience_counter = 0
                patience_counter = min(
                    patience_counter, self.early_stopping_patience // 2
                )
                if self.verbose:
                    print(
                        f"    ⚠ Divergence detected (d_loss={avg_d_loss:.2f}, g_loss={avg_g_loss:.2f}). "
                        f"Restoring best weights from epoch {best_epoch}, reducing LR to {new_lr:.2e} "
                        f"(recovery {divergence_recovery_count}/2)"
                    )
                if divergence_recovery_count >= 2:
                    if self.verbose:
                        print(
                            "    Stopping after 2 divergence recoveries; best model already restored."
                        )
                    break

            # Run evaluation if needed (before calculating metric for best epoch selection)
            # If using eval metrics for best epoch selection, evaluate EVERY epoch
            # Otherwise, evaluate only at eval_frequency intervals
            eval_metrics = None
            is_using_eval_metric = self.monitor_metric.startswith("eval_")
            should_evaluate_for_selection = (
                is_using_eval_metric and original_dataframe is not None
            )
            should_evaluate_for_display = (
                self.eval_frequency > 0
                and (epoch + 1) % self.eval_frequency == 0
                and original_dataframe is not None
            )

            if should_evaluate_for_selection or should_evaluate_for_display:
                try:
                    # Sample subset of real data for evaluation
                    eval_sample_size = min(
                        len(original_dataframe), self.eval_num_samples
                    )
                    if self.random_seed is not None:
                        rng = np.random.RandomState(self.random_seed + epoch + 1000)
                        eval_indices = rng.choice(
                            len(original_dataframe), eval_sample_size, replace=False
                        )
                        eval_real = original_dataframe.iloc[eval_indices]
                    else:
                        eval_real = original_dataframe.sample(
                            n=eval_sample_size, random_state=epoch
                        )

                    # Evaluate
                    eval_metrics = self.evaluate(
                        real_data=eval_real,
                        num_samples=self.eval_num_samples,
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"    Evaluation failed: {e}")

            # Calculate monitored metric based on evaluation or training losses
            current_metric = None
            use_eval_metric = is_using_eval_metric and eval_metrics is not None

            if use_eval_metric:
                # Use evaluation metrics when available
                overall_score = eval_metrics.get("overall_score", {})
                if self.monitor_metric == "eval_quality":
                    current_metric = overall_score.get("overall_quality", 0.0)
                elif self.monitor_metric == "eval_diversity":
                    current_metric = overall_score.get("diversity_score", 0.0)
                elif self.monitor_metric == "eval_correlation":
                    current_metric = overall_score.get("correlation_score", 0.0)
                elif self.monitor_metric == "eval_statistical":
                    current_metric = overall_score.get("statistical_score", 0.0)
                elif self.monitor_metric == "eval_validity":
                    current_metric = overall_score.get("validity_score", 0.0)
                else:
                    # Fall back to overall_quality if metric name not recognized
                    current_metric = overall_score.get("overall_quality", 0.0)
            else:
                # Use training losses (fallback or explicit choice)
                if self.monitor_metric == "g_loss":
                    current_metric = avg_g_loss
                elif self.monitor_metric == "d_loss":
                    current_metric = avg_d_loss
                else:  # "combined" or default
                    current_metric = avg_d_loss + abs(avg_g_loss)

            # Skip metric update if evaluation metric requested but not available this epoch
            # Only update best model when evaluation metrics are available if using eval metrics
            should_update = True
            if self.monitor_metric.startswith("eval_"):
                if eval_metrics is None:
                    # Skip update this epoch - wait for next evaluation
                    should_update = False
                elif current_metric is None:
                    should_update = False

            if current_metric is None:
                current_metric = best_metric  # No change

            # Check for improvement (maximize eval metrics, minimize losses)
            improved = False
            if should_update:
                if is_maximizing:
                    improved = current_metric > (
                        best_metric + self.early_stopping_min_delta
                    )
                else:
                    improved = current_metric < (
                        best_metric - self.early_stopping_min_delta
                    )

            if improved and should_update:
                best_metric = current_metric
                best_epoch = epoch + 1
                patience_counter = 0
                lr_patience_counter = 0
                # Save best weights
                best_generator_weights = [
                    np.copy(w) for w in self.generator.get_weights()
                ]
                best_discriminator_weights = [
                    np.copy(w) for w in self.discriminator.get_weights()
                ]

                if self.verbose and use_eval_metric:
                    metric_name = self.monitor_metric.replace("eval_", "")
                    print(
                        f"    ✓ New best {metric_name}: {best_metric:.4f} at epoch {best_epoch}"
                    )

            else:
                if should_update:
                    patience_counter += 1
                    lr_patience_counter += 1

            # Learning rate reduction
            if lr_patience_counter >= self.reduce_lr_patience:
                current_lr = float(self.discriminator.optimizer.learning_rate.numpy())
                new_lr = max(current_lr * self.reduce_lr_factor, self.reduce_lr_min)
                if new_lr < current_lr:
                    self.discriminator.optimizer.learning_rate.assign(new_lr)
                    self.generator.optimizer.learning_rate.assign(new_lr)
                    if self.verbose:
                        print(f"    Reducing learning rate to {new_lr:.2e}")
                    lr_patience_counter = 0

            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(
                        f"    Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {self.early_stopping_patience} epochs)"
                    )
                    if best_epoch > 0:
                        metric_str = f"{self.monitor_metric}={best_metric:.4f}"
                        print(
                            f"    Restoring best model from epoch {best_epoch} "
                            f"({metric_str})"
                        )
                # Restore best weights
                if best_generator_weights is not None:
                    self.generator.set_weights(best_generator_weights)
                if best_discriminator_weights is not None:
                    self.discriminator.set_weights(best_discriminator_weights)
                break

            # Evaluation metrics already computed above if needed for best epoch selection
            # Keep them for display purposes

            if self.verbose:
                current_lr = float(self.discriminator.optimizer.learning_rate.numpy())
                lr_str = f", lr: {current_lr:.2e}"
                eval_str = ""

                # Always display eval metrics if they were computed (helps assess training progress)
                if eval_metrics is not None:
                    overall = eval_metrics.get("overall_score", {}).get(
                        "overall_quality", 0.0
                    )
                    diversity = eval_metrics.get("overall_score", {}).get(
                        "diversity_score", 0.0
                    )
                    corr = eval_metrics.get("overall_score", {}).get(
                        "correlation_score", 0.0
                    )
                    div_ratio = eval_metrics.get("diversity", {}).get(
                        "diversity_ratio", 0.0
                    )
                    stat_score = eval_metrics.get("overall_score", {}).get(
                        "statistical_score", 0.0
                    )
                    validity = eval_metrics.get("overall_score", {}).get(
                        "validity_score", 0.0
                    )

                    # Show best metric value when using eval metrics for selection
                    # Include debug info for correlation if it's suspiciously low
                    corr_info = ""
                    if corr < 0.1:
                        # Check correlation metrics for diagnostics
                        corr_metrics = eval_metrics.get("correlation", {})
                        num_cont = corr_metrics.get("num_continuous_cols", "?")
                        total_cols = corr_metrics.get("total_columns", "?")
                        warning = corr_metrics.get("warning", "")
                        real_std = corr_metrics.get("real_corr_std", None)
                        gen_std = corr_metrics.get("gen_corr_std", None)

                        if num_cont != "?":
                            corr_info = f" [cont={num_cont}/total={total_cols}]"
                            if real_std is not None and gen_std is not None:
                                corr_info += (
                                    f" [std_r={real_std:.3e},std_g={gen_std:.3e}]"
                                )
                            if warning:
                                corr_info += f" [!{warning[:25]}]"

                    # Include debug info for statistical score if it's suspiciously low
                    stat_info = ""
                    if stat_score < 0.1:
                        stat_metrics = eval_metrics.get("statistics", {})
                        mean_err = stat_metrics.get("mean_error_avg", None)
                        std_err = stat_metrics.get("std_error_avg", None)
                        cat_err = stat_metrics.get("categorical_error_avg", None)
                        mean_err_max = stat_metrics.get("mean_error_max", None)
                        worst_mean_col = stat_metrics.get("worst_mean_error_col", "")

                        if mean_err is not None:
                            stat_info = f" [mean_err={mean_err:.3f}"
                            if std_err is not None:
                                stat_info += f",std_err={std_err:.3f}"
                            if cat_err is not None:
                                stat_info += f",cat_err={cat_err:.3f}"
                            stat_info += "]"
                            if mean_err_max is not None and mean_err_max > mean_err * 2:
                                stat_info += f" [worst_mean={mean_err_max:.3f}]"

                    # Append stat_info to stat_score display
                    stat_display = f"{stat_score:.3f}{stat_info}"

                    if is_using_eval_metric:
                        eval_str = (
                            f", eval: quality={overall:.3f} [best: {best_metric:.3f}], "
                            f"div={diversity:.3f} (ratio={div_ratio:.3f}), "
                            f"corr={corr:.3f}{corr_info}, stat={stat_display}, valid={validity:.3f}"
                        )
                    else:
                        eval_str = (
                            f", eval: quality={overall:.3f}, div={diversity:.3f} (ratio={div_ratio:.3f}), "
                            f"corr={corr:.3f}{corr_info}, stat={stat_display}, valid={validity:.3f}"
                        )

                print(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"d_loss: {avg_d_loss:.4f}, g_loss: {avg_g_loss:.4f}{lr_str}{eval_str}"
                )

        # Restore best weights if training completed without early stopping
        # (Early stopping already restores weights, so this handles normal completion)
        if (
            best_generator_weights is not None
            and best_discriminator_weights is not None
        ):
            if patience_counter < self.early_stopping_patience:
                # Training completed normally, restore best model
                if self.verbose:
                    print(
                        f"\n    Training completed. Restoring best model from epoch {best_epoch} "
                        f"({self.monitor_metric}={best_metric:.4f})"
                    )
                self.generator.set_weights(best_generator_weights)
                self.discriminator.set_weights(best_discriminator_weights)

        if hasattr(self, "_post_train_diagnostics"):
            self._post_train_diagnostics(train_data, train_labels)

    def save(
        self,
        filepath: str,
        min_buy_gain_threshold: Optional[float] = None,
        min_sell_loss_threshold: Optional[float] = None,
        training_type: Optional[int] = None,
    ):
        """Save the model to disk.

        Args:
            filepath: Directory to save the model
            min_buy_gain_threshold: Minimum buy gain threshold used for training labels (stored in metadata)
            min_sell_loss_threshold: Minimum sell loss threshold used for training labels (stored in metadata)
            training_type: Training type (label method) used for training labels (stored in metadata)
        """
        os.makedirs(filepath, exist_ok=True)
        self.generator.save(os.path.join(filepath, "generator.keras"))
        self.discriminator.save(os.path.join(filepath, "discriminator.keras"))

        # Save metadata
        metadata = {
            "categorical_columns": self.categorical_columns,
            "continuous_columns": self.continuous_columns,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "column_info": self.column_info,
            "column_order": self.column_order,
            "latent_dim": self.latent_dim,
            "generator_layers": self.generator_layers,
            "discriminator_layers": self.discriminator_layers,
            "vgm_models": self.vgm_models,
            "continuous_info": self.continuous_info,
            "continuous_dim": self.continuous_dim,
            "integer_columns": self.integer_columns,
        }

        # Store training thresholds and training_type if provided (for consistency with strategy)
        if min_buy_gain_threshold is not None:
            metadata["min_buy_gain_threshold"] = float(min_buy_gain_threshold)
        if min_sell_loss_threshold is not None:
            metadata["min_sell_loss_threshold"] = float(min_sell_loss_threshold)
        if training_type is not None:
            metadata["training_type"] = int(training_type)

        with open(os.path.join(filepath, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    def load(self, filepath: str) -> Dict[str, Optional[float]]:
        """Load the model from disk.

        Returns:
            Dictionary with 'min_buy_gain_threshold', 'min_sell_loss_threshold',
            and 'training_type' if present in metadata
        """
        # Load generator (needed for generation)
        self.generator = tf.keras.models.load_model(
            os.path.join(filepath, "generator.keras")
        )
        # Load discriminator without compiling (we don't need it for generation)
        # The custom loss function can't be deserialized, so we load without compile
        self.discriminator = tf.keras.models.load_model(
            os.path.join(filepath, "discriminator.keras"), compile=False
        )

        # Load metadata
        with open(os.path.join(filepath, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.categorical_columns = metadata["categorical_columns"]
        self.continuous_columns = metadata["continuous_columns"]
        self.num_classes = metadata["num_classes"]
        self.num_features = metadata["num_features"]
        self.column_info = metadata["column_info"]
        self.latent_dim = metadata["latent_dim"]
        self.generator_layers = metadata["generator_layers"]
        self.discriminator_layers = metadata["discriminator_layers"]
        self.vgm_models = metadata.get("vgm_models", {})
        self.continuous_info = metadata.get("continuous_info", [])
        self.continuous_dim = metadata.get("continuous_dim", 0)
        self.integer_columns = metadata.get("integer_columns", [])

        # Restore column_order (preserve original order from training)
        if "column_order" in metadata:
            self.column_order = metadata["column_order"]
        else:
            # Fallback: reconstruct from column_info keys (preserves order in Python 3.7+)
            self.column_order = list(self.column_info.keys())

        # Recalculate feature counts (needed for generate())
        self.num_categorical_features = len(self.categorical_columns)
        self.num_continuous_features = len(self.continuous_columns)

        # Reconstruct categorical_info from column_info
        self.categorical_info = [
            self.column_info[col]["num_categories"] for col in self.categorical_columns
        ]

        self.categorical_dim = sum(self.categorical_info)

        # If continuous_dim was not in metadata (legacy models), fall back
        if self.continuous_dim == 0 and self.num_continuous_features > 0:
            if self.continuous_info:
                self.continuous_dim = sum(
                    val + modes for val, modes in self.continuous_info
                )
            else:
                # Traditional min-max scaling (1 value per feature)
                self.continuous_dim = self.num_continuous_features

        self.total_feature_dim = self.continuous_dim + self.categorical_dim

        self.is_fitted = True

        # Return ALL persisted metadata keys, not just thresholds + type.
        # save() accepts arbitrary **extra_metadata; _master_save_kwargs has
        # grown over time (horizon added 2026-05-30, more likely later).
        # Whitelisting on load drops those keys before the validator sees
        # them, generating misleading 'missing key' warnings on metadata
        # that actually has them.
        return dict(metadata)

    def evaluate(
        self,
        real_data: pd.DataFrame,
        num_samples: Optional[int] = None,
        class_label: Optional[int] = None,
        class_probs: Optional[np.ndarray] = None,
        generated_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate generated samples against real data.

        Focuses on diversity and correlation preservation to detect overfitting.

        Args:
            real_data: Real dataframe to compare against
            num_samples: Number of samples to generate (default: eval_num_samples)
            class_label: Specific class to generate (if None, uses class_probs or uniform)
            class_probs: Probability distribution over classes
            generated_data: Optional pre-generated data. If provided, num_samples is ignored.

        Returns:
            Dictionary of evaluation metrics
        """
        # Check if models exist (can evaluate during training)
        if self.generator is None:
            raise ValueError("Generator model not created. Must call fit() first.")

        # Use provided generated data or generate new samples
        if generated_data is None:
            if num_samples is None:
                num_samples = self.eval_num_samples

            # Generate synthetic samples
            generated_data = self.generate(
                num_samples=num_samples,
                class_label=class_label,
                class_probs=class_probs,
            )

        # Ensure same columns and order
        real_data = real_data[self.column_order].copy()
        generated_data = generated_data[self.column_order].copy()

        return self.evaluate_with_dataframes(real_data, generated_data)


    def _compute_diversity_metrics(
        self, real_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute diversity metrics to detect mode collapse."""
        metrics = {}

        # Convert to numpy for distance calculations using the fitted transformation
        real_array = self._transform_data(real_data)
        gen_array = self._transform_data(generated_data)

        # Sample subset for efficiency (max 1000 samples, shuffled to avoid bias)
        n_real = min(len(real_array), 1000)
        n_gen = min(len(gen_array), 1000)

        # Shuffle before sampling to get representative data
        if len(real_array) > n_real:
            real_indices = np.random.RandomState(42).permutation(len(real_array))[
                :n_real
            ]
            real_sample = real_array[real_indices]
        else:
            real_sample = real_array

        if len(gen_array) > n_gen:
            gen_indices = np.random.RandomState(42).permutation(len(gen_array))[:n_gen]
            gen_sample = gen_array[gen_indices]
        else:
            gen_sample = gen_array

        # 1. Pairwise distances within generated samples (higher = more diverse)
        if len(gen_sample) > 1:
            gen_distances = pdist(gen_sample, metric="euclidean")
            metrics["gen_pairwise_distance_mean"] = float(np.mean(gen_distances))
            metrics["gen_pairwise_distance_std"] = float(np.std(gen_distances))
            metrics["gen_pairwise_distance_min"] = float(np.min(gen_distances))
        else:
            metrics["gen_pairwise_distance_mean"] = 0.0
            metrics["gen_pairwise_distance_std"] = 0.0
            metrics["gen_pairwise_distance_min"] = 0.0

        # 2. Pairwise distances within real samples (for comparison)
        if len(real_sample) > 1:
            real_distances = pdist(real_sample, metric="euclidean")
            metrics["real_pairwise_distance_mean"] = float(np.mean(real_distances))
        else:
            metrics["real_pairwise_distance_mean"] = 0.0

        # 3. Diversity ratio (gen/real) - should be close to 1.0
        if metrics["real_pairwise_distance_mean"] > 0:
            metrics["diversity_ratio"] = (
                metrics["gen_pairwise_distance_mean"]
                / metrics["real_pairwise_distance_mean"]
            )
        else:
            metrics["diversity_ratio"] = 0.0

        # 4. Unique value counts for categorical columns
        unique_counts = {}
        for col in self.categorical_columns:
            if col in generated_data.columns:
                real_unique = real_data[col].nunique()
                gen_unique = generated_data[col].nunique()
                unique_counts[col] = {
                    "real": int(real_unique),
                    "generated": int(gen_unique),
                    "ratio": (
                        float(gen_unique / real_unique) if real_unique > 0 else 0.0
                    ),
                }
        metrics["categorical_uniqueness"] = unique_counts

        # 5. Coverage metric (how well generated samples cover the value space)
        coverage_scores = []
        for col in self.continuous_columns:
            if col in generated_data.columns:
                real_min, real_max = real_data[col].min(), real_data[col].max()
                gen_min, gen_max = generated_data[col].min(), generated_data[col].max()
                real_range = real_max - real_min
                if real_range > 0:
                    # How much of the real range is covered by generated samples
                    coverage = min(1.0, (gen_max - gen_min) / real_range)
                    coverage_scores.append(coverage)
        metrics["value_space_coverage"] = (
            float(np.mean(coverage_scores)) if coverage_scores else 0.0
        )

        # 6. Nearest neighbor distances (generated to real)
        # If all generated samples are very close to real samples = overfitting
        if len(gen_sample) > 0 and len(real_sample) > 0:
            try:
                from sklearn.neighbors import NearestNeighbors

                nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
                nn.fit(real_sample)
                distances, _ = nn.kneighbors(gen_sample)
                metrics["nearest_real_distance_mean"] = float(np.mean(distances))
                metrics["nearest_real_distance_std"] = float(np.std(distances))
            except ImportError:
                # Fallback if sklearn not available
                metrics["nearest_real_distance_mean"] = 0.0
                metrics["nearest_real_distance_std"] = 0.0
        else:
            metrics["nearest_real_distance_mean"] = 0.0
            metrics["nearest_real_distance_std"] = 0.0

        return metrics

    def _compute_correlation_metrics(
        self, real_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute correlation preservation metrics."""
        metrics = {}

        # Only compute correlations for continuous columns
        continuous_cols = [
            col for col in self.continuous_columns if col in real_data.columns
        ]
        metrics["num_continuous_cols"] = len(continuous_cols)
        metrics["total_columns"] = len(real_data.columns)
        metrics["num_categorical_cols"] = len(self.categorical_columns)

        if len(continuous_cols) < 2:
            # Not enough continuous columns for correlation computation
            # Return 1.0 to indicate no issue (can't compute correlations)
            metrics["correlation_preservation"] = 1.0
            metrics["correlation_error"] = 0.0
            metrics["warning"] = (
                f"Only {len(continuous_cols)} continuous column(s), need >= 2 for correlation"
            )
            return metrics

        # Compute correlation matrices (handle potential constant columns)
        try:
            real_corr = real_data[continuous_cols].corr().values
            gen_corr = generated_data[continuous_cols].corr().values

            # Check for NaN or infinite values in correlations
            if np.any(np.isnan(real_corr)) or np.any(np.isnan(gen_corr)):
                # Some columns might be constant or have issues
                metrics["correlation_preservation"] = 0.0
                metrics["correlation_error"] = 1.0
                metrics["num_continuous_cols"] = len(continuous_cols)
                metrics["warning"] = "NaN values in correlation matrices"
                return metrics

            # Extract upper triangle (avoid diagonal and duplicates)
            mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
            real_corr_flat = real_corr[mask]
            gen_corr_flat = gen_corr[mask]

            # Correlation preservation score (1.0 = perfect, 0.0 = no correlation)
            if len(real_corr_flat) > 0:
                # Remove any NaN values that might have been introduced
                valid_mask = ~(np.isnan(real_corr_flat) | np.isnan(gen_corr_flat))
                if valid_mask.sum() > 0:
                    real_corr_flat = real_corr_flat[valid_mask]
                    gen_corr_flat = gen_corr_flat[valid_mask]

                    # Mean absolute error in correlations
                    corr_error = np.mean(np.abs(real_corr_flat - gen_corr_flat))
                    metrics["correlation_error"] = float(corr_error)

                    # Correlation of correlations (how well structure is preserved)
                    real_std = np.std(real_corr_flat)
                    gen_std = np.std(gen_corr_flat)
                    real_mean = np.mean(real_corr_flat)
                    gen_mean = np.mean(gen_corr_flat)

                    # Store diagnostic info
                    metrics["real_corr_std"] = float(real_std)
                    metrics["gen_corr_std"] = float(gen_std)
                    metrics["real_corr_mean"] = float(real_mean)
                    metrics["gen_corr_mean"] = float(gen_mean)
                    metrics["num_corr_pairs"] = len(real_corr_flat)

                    # Lower threshold for variance check (more lenient for large correlation matrices)
                    # With many features, correlations can be similar, so we use a more lenient threshold
                    variance_threshold = 1e-4

                    if real_std > variance_threshold and gen_std > variance_threshold:
                        # Both have variance, compute correlation of correlations
                        try:
                            corr_corr = np.corrcoef(real_corr_flat, gen_corr_flat)[0, 1]
                            if np.isnan(corr_corr):
                                # This can happen if arrays have zero variance after centering
                                # Check if they're nearly identical
                                if np.allclose(
                                    real_corr_flat, gen_corr_flat, atol=1e-3
                                ):
                                    metrics["correlation_preservation"] = 1.0
                                else:
                                    # Use alternative: mean absolute normalized difference
                                    # If correlations are similar in magnitude, that's still good
                                    mean_abs_diff = np.mean(
                                        np.abs(real_corr_flat - gen_corr_flat)
                                    )
                                    max_abs_corr = max(
                                        np.abs(real_corr_flat).max(),
                                        np.abs(gen_corr_flat).max(),
                                    )
                                    if max_abs_corr > 0:
                                        normalized_diff = mean_abs_diff / max_abs_corr
                                        # Convert error to score (lower error = higher score)
                                        metrics["correlation_preservation"] = max(
                                            0.0, 1.0 - normalized_diff
                                        )
                                    else:
                                        metrics["correlation_preservation"] = 0.0
                                    metrics["warning"] = (
                                        "NaN in corrcoef, using normalized diff instead"
                                    )
                            else:
                                metrics["correlation_preservation"] = float(corr_corr)
                        except Exception as e:
                            # Fallback: use mean absolute error as proxy
                            mean_abs_diff = np.mean(
                                np.abs(real_corr_flat - gen_corr_flat)
                            )
                            max_abs_corr = max(
                                np.abs(real_corr_flat).max(),
                                np.abs(gen_corr_flat).max(),
                            )
                            if max_abs_corr > 0:
                                normalized_diff = mean_abs_diff / max_abs_corr
                                metrics["correlation_preservation"] = max(
                                    0.0, 1.0 - normalized_diff * 2.0
                                )
                            else:
                                metrics["correlation_preservation"] = 0.0
                            metrics["warning"] = f"Error in corrcoef: {str(e)[:30]}"
                    elif (
                        real_std <= variance_threshold and gen_std <= variance_threshold
                    ):
                        # Both have very low/zero variance - check if correlation values are similar
                        # This happens when all correlations are very similar (e.g., all near 0 or all near 1)
                        mean_diff = abs(real_mean - gen_mean)
                        # Use a more lenient threshold for mean difference when variance is low
                        # Since correlations are all similar, we just need the means to be close
                        if mean_diff < 0.2:  # Correlations are similar (within 0.2)
                            metrics["correlation_preservation"] = 1.0
                        else:
                            # Still give some credit if means are reasonably close
                            metrics["correlation_preservation"] = max(
                                0.0, 1.0 - mean_diff
                            )
                            metrics["warning"] = (
                                f"Low variance in both (std<{variance_threshold:.1e}), "
                                f"means differ by {mean_diff:.3f}"
                            )
                    elif real_std <= variance_threshold:
                        # Real has low variance, generated has variance - structure changed
                        metrics["correlation_preservation"] = 0.0
                        real_std_str = f"{real_std:.2e}"
                        gen_std_str = f"{gen_std:.2e}"
                        metrics["warning"] = (
                            f"Real correlations have low variance ({real_std_str}), "
                            f"generated has variance ({gen_std_str})"
                        )
                    else:
                        # Generated has low variance, real has variance - structure lost
                        metrics["correlation_preservation"] = 0.0
                        gen_std_str = f"{gen_std:.2e}"
                        real_std_str = f"{real_std:.2e}"
                        metrics["warning"] = (
                            f"Generated correlations have low variance ({gen_std_str}), "
                            f"real has variance ({real_std_str})"
                        )

                    metrics["num_continuous_cols"] = len(continuous_cols)
                    metrics["num_corr_pairs"] = len(real_corr_flat)
                else:
                    # No valid correlation pairs
                    metrics["correlation_preservation"] = 0.0
                    metrics["correlation_error"] = 1.0
                    metrics["num_continuous_cols"] = len(continuous_cols)
                    metrics["warning"] = "No valid correlation pairs found"
            else:
                metrics["correlation_error"] = 0.0
                metrics["correlation_preservation"] = 1.0
                metrics["num_continuous_cols"] = len(continuous_cols)
        except Exception as e:
            # If correlation computation fails, return 0.0 but don't crash
            metrics["correlation_preservation"] = 0.0
            metrics["correlation_error"] = 1.0
            metrics["num_continuous_cols"] = len(continuous_cols)
            metrics["error"] = str(e)

        return metrics

    def _compute_statistical_metrics(
        self, real_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute statistical similarity metrics."""
        metrics = {}

        # For continuous columns: mean, std comparison
        continuous_stats = {}
        for col in self.continuous_columns:
            if col in real_data.columns:
                real_mean = float(real_data[col].mean())
                real_std = float(real_data[col].std())
                gen_mean = float(generated_data[col].mean())
                gen_std = float(generated_data[col].std())

                # Get data range for normalization (more stable than relative error)
                real_min = float(real_data[col].min())
                real_max = float(real_data[col].max())
                real_range = max(real_max - real_min, abs(real_mean), 1e-8)

                # Normalize errors by the range of the data (more stable than relative error)
                # This prevents exploding errors when values are close to zero
                mean_error = abs(real_mean - gen_mean) / real_range
                std_error = abs(real_std - gen_std) / max(real_range, 1e-8)

                continuous_stats[col] = {
                    "mean_error": mean_error,
                    "std_error": std_error,
                    "real_mean": real_mean,
                    "gen_mean": gen_mean,
                    "real_std": real_std,
                    "gen_std": gen_std,
                    "real_min": real_min,
                    "real_max": real_max,
                    "real_range": real_range,
                }
        metrics["continuous_statistics"] = continuous_stats

        # Overall mean/std error (with diagnostics)
        if continuous_stats:
            metrics["mean_error_avg"] = float(
                np.mean([s["mean_error"] for s in continuous_stats.values()])
            )
            metrics["std_error_avg"] = float(
                np.mean([s["std_error"] for s in continuous_stats.values()])
            )
            # Store max errors for diagnostics
            metrics["mean_error_max"] = float(
                max([s["mean_error"] for s in continuous_stats.values()])
            )
            metrics["std_error_max"] = float(
                max([s["std_error"] for s in continuous_stats.values()])
            )
            # Find worst column for diagnostics
            if continuous_stats:
                worst_mean_col = max(
                    continuous_stats.items(), key=lambda x: x[1]["mean_error"]
                )[0]
                worst_std_col = max(
                    continuous_stats.items(), key=lambda x: x[1]["std_error"]
                )[0]
                metrics["worst_mean_error_col"] = worst_mean_col
                metrics["worst_std_error_col"] = worst_std_col
        else:
            metrics["mean_error_avg"] = 0.0
            metrics["std_error_avg"] = 0.0
            metrics["mean_error_max"] = 0.0
            metrics["std_error_max"] = 0.0

        # For categorical columns: distribution comparison
        categorical_stats = {}
        for col in self.categorical_columns:
            if col in real_data.columns:
                real_counts = real_data[col].value_counts(normalize=True).sort_index()
                gen_counts = (
                    generated_data[col].value_counts(normalize=True).sort_index()
                )

                # Align indices
                all_values = sorted(set(real_counts.index) | set(gen_counts.index))
                real_probs = np.array([real_counts.get(v, 0.0) for v in all_values])
                gen_probs = np.array([gen_counts.get(v, 0.0) for v in all_values])

                # Total variation distance for discrete distributions
                if len(all_values) > 0:
                    tv_distance = np.sum(np.abs(real_probs - gen_probs)) / 2.0
                    categorical_stats[col] = {
                        "total_variation_distance": float(tv_distance),
                        "real_unique": int(real_data[col].nunique()),
                        "gen_unique": int(generated_data[col].nunique()),
                    }
        metrics["categorical_statistics"] = categorical_stats

        # Overall categorical error
        if categorical_stats:
            metrics["categorical_error_avg"] = float(
                np.mean(
                    [s["total_variation_distance"] for s in categorical_stats.values()]
                )
            )
        else:
            metrics["categorical_error_avg"] = 0.0

        return metrics




class CTABGANPlusEnhanced(CTABGANPlusEnhancedMixin, CTABGANPlus):
    """
    Enhanced CTAB-GAN+ with optional CNN, auxiliary model, and paper losses.

    Adds (all optional):
    - use_cnn: TableGAN-style CNN generator/discriminator.
    - use_auxiliary: Auxiliary classifier A (MLP 4x256) for downstream loss.
    - info_loss_weight: Match mean/std of real vs generated (default 0).
    - downstream_loss_weight: CE(labels, A(fake)) for semantic integrity (default 0).
    - generator_loss_weight: Same as downstream (condition matching); default 0.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        generator_layers: List[int] = [256, 256],
        discriminator_layers: List[int] = [256, 256],
        batch_size: int = 1024,
        epochs: int = 300,
        learning_rate: float = 2e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
        gp_weight: float = 10.0,
        verbose: bool = True,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-4,
        reduce_lr_patience: int = 10,
        reduce_lr_factor: float = 0.5,
        reduce_lr_min: float = 1e-6,
        pac: int = 10,
        monitor_metric: str = "eval_quality",
        eval_frequency: int = 5,
        eval_num_samples: int = 1000,
        random_seed: Optional[int] = 42,
        integer_columns: List[str] = [],
        n_critic: int = 5,
        use_cnn: bool = False,
        use_auxiliary: bool = False,
        info_loss_weight: float = 0.0,
        downstream_loss_weight: float = 0.0,
        generator_loss_weight: float = 0.0,
    ):
        """
        Extra args over CTABGANPlus:

        - use_cnn: Use CNN backbone for G/D when True.
        - use_auxiliary: Build auxiliary classifier A (trained on real data) for downstream loss.
        - info_loss_weight: Weight for mean/std matching loss (paper: information loss).
        - downstream_loss_weight: Weight for CE(labels, A(fake)) (paper: downstream loss).
        - generator_loss_weight: Weight for condition-matching (same CE term; paper: generator loss).
        """
        super().__init__(
            latent_dim=latent_dim,
            generator_layers=generator_layers,
            discriminator_layers=discriminator_layers,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            gp_weight=gp_weight,
            verbose=verbose,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_factor=reduce_lr_factor,
            reduce_lr_min=reduce_lr_min,
            pac=pac,
            monitor_metric=monitor_metric,
            eval_frequency=eval_frequency,
            eval_num_samples=eval_num_samples,
            random_seed=random_seed,
            integer_columns=integer_columns,
            n_critic=n_critic,
        )
        self.use_cnn = use_cnn
        self.use_auxiliary = use_auxiliary
        self.info_loss_weight = float(info_loss_weight)
        self.downstream_loss_weight = float(downstream_loss_weight)
        self.generator_loss_weight = float(generator_loss_weight)

    def _build_auxiliary_model(self):
        """Build auxiliary classifier A: encoded row -> class logits (paper: 4x256 MLP)."""
        inp = layers.Input(shape=(self.total_feature_dim,))
        x = inp
        for _ in range(4):
            x = layers.Dense(256, activation="relu")(x)
        logits = layers.Dense(self.num_classes, name="aux_logits")(x)
        self.auxiliary = Model(inp, logits, name="ctab_gan_auxiliary")
        self.auxiliary.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def _create_models(self):
        """
        Create generator and discriminator models.

        When `use_cnn` is False, this defers to the original MLP-based
        architecture from `CTABGANPlus`. When `use_cnn` is True, it builds
        a lightweight CNN backbone similar in spirit to TableGAN/CTAB-GAN+:

        - Generator: (z, y) -> dense -> reshape to square "image" -> Conv2D blocks
          -> flatten -> dense -> feature heads.
        - Discriminator: (x, y) -> dense -> reshape to square "image" -> Conv2D
          blocks -> flatten -> dense -> Wasserstein score.
        """
        if not getattr(self, "use_cnn", False):
            super()._create_models()
            if getattr(self, "use_auxiliary", False):
                self._build_auxiliary_model()
            return

        # --------------------
        # Generator (CNN-based)
        # --------------------
        noise_input = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(self.num_classes,))

        # Concatenate noise and label
        g_in = layers.Concatenate()([noise_input, label_input])

        # Project to a square feature map
        # Use total_feature_dim as a proxy for required spatial size
        side = int(np.ceil(np.sqrt(max(self.total_feature_dim, 4))))
        proj_dim = side * side

        x = layers.Dense(proj_dim, activation="linear")(g_in)
        x = layers.Reshape((side, side, 1))(x)

        # A couple of Conv2D blocks (TableGAN-style)
        for _ in range(2):
            x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        x = layers.Flatten()(x)

        # Final dense layer to match original hidden size before heads
        if self.generator_layers:
            x = layers.Dense(self.generator_layers[-1], activation="relu")(x)

        # Output layer splits into continuous branches (scalar + mode_probs) and categorical branches
        outputs = []
        for i, (val_dim, num_modes) in enumerate(self.continuous_info):
            # Scalar output (value within mode)
            scalar = layers.Dense(1, activation="tanh", name=f"cont_val_{i}")(x)
            outputs.append(scalar)

            if num_modes > 0:
                # Categorical output (which mode)
                mode_logits = layers.Dense(num_modes, name=f"cont_mode_logits_{i}")(x)
                mode_probs = GumbelSoftmax(
                    temperature=self.temperature, name=f"cont_mode_probs_{i}"
                )(mode_logits)
                outputs.append(mode_probs)

        cat_outputs = []
        for i, num_cats in enumerate(self.categorical_info):
            logits = layers.Dense(num_cats, name=f"cat_logits_{i}")(x)
            probs = GumbelSoftmax(temperature=self.temperature, name=f"cat_probs_{i}")(
                logits
            )
            cat_outputs.append(probs)

        if cat_outputs:
            all_outputs = layers.Concatenate()(outputs + cat_outputs)
        else:
            all_outputs = (
                layers.Concatenate()(outputs) if len(outputs) > 1 else outputs[0]
            )

        self.generator = Model(
            [noise_input, label_input], all_outputs, name="ctab_gan_generator_cnn"
        )

        # ------------------------
        # Discriminator (CNN-based)
        # ------------------------
        data_input = layers.Input(shape=(self.total_feature_dim * self.pac,))
        label_input_d = layers.Input(shape=(self.num_classes * self.pac,))

        # Concatenate packed data and labels
        d_in = layers.Concatenate()([data_input, label_input_d])

        # Project to square feature map
        side_d = int(np.ceil(np.sqrt(max(self.total_feature_dim * self.pac, 4))))
        proj_dim_d = side_d * side_d

        x_d = layers.Dense(proj_dim_d, activation="linear")(d_in)
        x_d = layers.Reshape((side_d, side_d, 1))(x_d)

        # A couple of Conv2D blocks with LeakyReLU
        for _ in range(2):
            x_d = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(x_d)
            x_d = layers.LeakyReLU(0.2)(x_d)

        x_d = layers.Flatten()(x_d)

        if self.discriminator_layers:
            x_d = layers.Dense(self.discriminator_layers[-1])(x_d)
            x_d = layers.LeakyReLU(0.2)(x_d)

        # Output: single value (Wasserstein distance)
        output_d = layers.Dense(1)(x_d)

        self.discriminator = Model(
            [data_input, label_input_d],
            output_d,
            name="ctab_gan_discriminator_cnn",
        )

        # Compile discriminator (same optimizer/loss as base class)
        if self.use_mixed_precision:
            optimizer_d = tf.keras.mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                )
            )
        else:
            optimizer_d = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
            )

        self.discriminator.compile(
            optimizer=optimizer_d,
            loss=self._wasserstein_loss,
        )

        # As in the base class, keep a dummy GAN wrapper only for optimizer tracking.
        self.discriminator.trainable = False

        # Dummy GAN inputs (not actually used for forward passes)
        gan_noise_input = layers.Input(shape=(self.pac * self.latent_dim,))
        gan_label_input = layers.Input(shape=(self.pac * self.num_classes,))
        _ = gan_noise_input, gan_label_input  # Silence unused variable warnings

        # Generator optimizer
        if self.use_mixed_precision:
            optimizer_g = tf.keras.mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                )
            )
        else:
            optimizer_g = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
            )

        self.generator.optimizer = optimizer_g
        self.discriminator.trainable = True

        if getattr(self, "use_auxiliary", False):
            self._build_auxiliary_model()


    @tf.function
    def _train_generator_step(self, labels, noise, real_data=None):
        """Generator step with optional information loss and downstream/generator loss."""
        with tf.GradientTape() as g_tape:
            fake_data = self.generator([noise, labels], training=True)

            pac_size = tf.shape(fake_data)[0] // self.pac
            fake_data_pac = tf.reshape(
                fake_data, [pac_size, self.pac * self.total_feature_dim]
            )
            labels_pac = tf.reshape(labels, [pac_size, self.pac * self.num_classes])

            fake_scores = self.discriminator([fake_data_pac, labels_pac], training=True)
            g_loss = -tf.reduce_mean(fake_scores)

            # Information loss: match mean and std of real vs fake (paper)
            if real_data is not None and getattr(self, "info_loss_weight", 0.0) > 0:
                # Ensure both statistics are computed in the same dtype (respect mixed precision)
                real_cast = tf.cast(real_data, fake_data.dtype)
                mean_r = tf.reduce_mean(real_cast, axis=0)
                mean_f = tf.reduce_mean(fake_data, axis=0)
                std_r = tf.math.reduce_std(real_cast, axis=0) + 1e-8
                std_f = tf.math.reduce_std(fake_data, axis=0) + 1e-8
                info_loss = tf.reduce_mean((mean_r - mean_f) ** 2) + tf.reduce_mean(
                    (std_r - std_f) ** 2
                )
                info_loss = tf.minimum(info_loss, 10.0)
                info_loss = tf.where(
                    tf.math.is_finite(info_loss), info_loss, tf.zeros_like(info_loss)
                )
                g_loss = (
                    g_loss + tf.cast(self.info_loss_weight, g_loss.dtype) * info_loss
                )

            # Downstream / generator loss: CE(labels, A(fake)) so generated data predicts the condition
            down_weight = getattr(self, "downstream_loss_weight", 0.0) + getattr(
                self, "generator_loss_weight", 0.0
            )
            if real_data is not None and down_weight > 0 and self.auxiliary is not None:
                pred = self.auxiliary(fake_data, training=False)
                down_loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(
                        labels, tf.cast(pred, g_loss.dtype), from_logits=True
                    )
                )
                down_loss = tf.minimum(down_loss, 10.0)
                down_loss = tf.where(
                    tf.math.is_finite(down_loss), down_loss, tf.zeros_like(down_loss)
                )
                g_loss = g_loss + tf.cast(down_weight, g_loss.dtype) * down_loss

            # If combined loss is non-finite, use only adversarial loss to avoid NaN step
            g_loss = tf.where(
                tf.math.is_finite(g_loss), g_loss, -tf.reduce_mean(fake_scores)
            )

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        g_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in g_grads]
        valid_grads = [
            (g, v)
            for g, v in zip(g_grads, self.generator.trainable_variables)
            if g is not None
        ]
        if valid_grads:
            self.generator.optimizer.apply_gradients(valid_grads)
        return g_loss

    def _post_train_diagnostics(
        self, train_data: np.ndarray, train_labels: np.ndarray
    ) -> None:
        """Print end-of-training diagnostics for class-conditioning health.

        Reports:
        - Auxiliary accuracy on real training data (sanity check that the
          auxiliary actually learned the class boundary).
        - For each class c: auxiliary accuracy on samples generated with
          class_label=c, plus the mean per-feature value of those samples.

        These four numbers tell us which of the failure modes is active:
        - aux-acc-on-real low → auxiliary not training to good accuracy.
        - aux-acc-on-real high but aux-acc-on-fake low → generator ignores
          class label (auxiliary CE term is too weak, or distribution shift
          between real and fake encoded space).
        - Per-class feature means equal across classes → generator output
          does not depend on class label at all (full mode collapse).
        """
        if self.auxiliary is None:
            return
        try:
            print("\n  --- CTAB-GAN+ end-of-training diagnostics ---")
            real_pred = self.auxiliary(train_data, training=False).numpy()
            real_true = np.argmax(train_labels, axis=1)
            real_pred_idx = np.argmax(real_pred, axis=1)
            aux_acc_real = float(np.mean(real_pred_idx == real_true))
            print(f"  Aux accuracy on REAL train data: {aux_acc_real:.3f}")

            n_per_class = 200
            for c in range(self.num_classes):
                noise = tf.random.normal(
                    (n_per_class, self.latent_dim), dtype=tf.float32
                )
                labels = np.tile(
                    np.eye(self.num_classes, dtype=np.float32)[c], (n_per_class, 1)
                )
                labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)
                fake_encoded = self.generator([noise, labels_t], training=False).numpy()
                fake_pred = self.auxiliary(fake_encoded, training=False).numpy()
                fake_pred_idx = np.argmax(fake_pred, axis=1)
                acc_c = float(np.mean(fake_pred_idx == c))
                pred_dist = np.bincount(fake_pred_idx, minlength=self.num_classes)
                gen_df = self.generate(n_per_class, class_label=c)
                feat_means = gen_df.values.astype("float32").mean(axis=0)
                fm_str = " ".join(f"{m:+.2f}" for m in feat_means)
                print(
                    f"  Class {c}: aux says {pred_dist.tolist()} "
                    f"(acc={acc_c:.3f}); feat means = [{fm_str}]"
                )
            print("  --- end diagnostics ---\n")
        except Exception as exc:
            print(f"  Diagnostics failed: {exc}")


