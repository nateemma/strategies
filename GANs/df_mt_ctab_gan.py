"""
CTAB-GAN+ for conditional tabular data generation with multi-task labels.

CTAB-GAN+ is specifically designed for tabular data with mixed data types
(continuous and categorical). It uses conditional generation to create
samples conditioned on multiple task labels simultaneously.

API:
    model = CTABGANPlusMT()
    model.fit(dataframe, labels_dict, categorical_columns=['col1', 'col2'])
    generated_samples, labels_dict = model.generate(num_samples=1000, task_labels={"trading": ...})
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Model
import os
import pickle
import warnings
from typing import List, Optional, Dict, Any
from scipy.spatial.distance import pdist
from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning


def _concatenate_task_labels(task_labels_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate all task labels into a single condition vector."""
    sorted_keys = sorted(task_labels_dict.keys())
    return np.concatenate([task_labels_dict[k] for k in sorted_keys], axis=1)


@keras.saving.register_keras_serializable()
class GumbelSoftmax(keras.layers.Layer):
    """Gumbel-Softmax layer for categorical feature generation.
    Returns sharp distributions during training while remaining differentiable,
    and returns standard softmax probabilities during inference."""

    def __init__(self, temperature=0.2, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, logits, training=None):
        if training:
            # Add Gumbel noise for sharp sampling (ensure dtype matches logits for mixed precision)
            uniform = tf.random.uniform(
                tf.shape(logits), minval=1e-5, maxval=1.0 - 1e-5, dtype=logits.dtype
            )
            gumbel_noise = -tf.math.log(-tf.math.log(uniform))
            temp = tf.cast(self.temperature, dtype=logits.dtype)
            return tf.nn.softmax((logits + gumbel_noise) / temp)
        else:
            # Standard softmax during inference
            return tf.nn.softmax(logits)

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


class CTABGANPlusMT:
    """
    Multi-Task CTAB-GAN+ model for conditional tabular data generation.

    Handles mixed data types (continuous and categorical) and generates
    samples conditioned on multiple task labels simultaneously.
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
    ):
        """
        Initialize Multi-Task CTAB-GAN+ model.

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
        self.task_label_dims: Dict[str, int] = {}  # {"trading": 3, "regime": 3, ...}
        self.total_cond_dim: int = 0  # Sum of all task label dimensions
        self.sorted_tasks: List[str] = []  # Sorted task names for consistent ordering
        self.num_features: int = 0
        self.vgm_models: Dict[str, BayesianGaussianMixture] = (
            {}
        )  # For Variational Gaussian Mixture
        self.column_info: Dict[str, Any] = {}
        self.continuous_info: List = (
            []
        )  # List of (1, num_modes) tuples per continuous column
        self.generator: Optional[Model] = None
        self.discriminator: Optional[Model] = None
        self.gan: Optional[Model] = None
        self.is_fitted = False

        # Enable mixed precision training for better GPU utilization (~2x speedup on modern GPUs)
        self.use_mixed_precision = True
        if self.use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                if self.verbose:
                    print("    Mixed precision training enabled (float16/float32)")
            except (AttributeError, RuntimeError):
                self.use_mixed_precision = False
                if self.verbose:
                    print("    Mixed precision not available, using float32")

        # Configure GPU for better utilization (must be called after verbose is set)
        self._configure_gpu()

    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility across all libraries."""
        # Set Python's built-in random seed
        import random

        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set TensorFlow random seed
        tf.random.set_seed(seed)

        # Set TensorFlow to use deterministic operations (may impact performance)
        # Note: Some GPU operations may still be non-deterministic
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            # Deterministic ops not available in this TensorFlow version
            pass

        if self.verbose:
            print(f"    Random seed set to {seed} for reproducibility")

    def _configure_gpu(self):
        """Configure GPU for optimal utilization and suggest batch size."""
        # Check available GPUs
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 0:
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Collect GPU info for batch size recommendations
                gpu_details = []
                for gpu in gpus:
                    memory_gb = None
                    device_name = gpu.name

                    try:
                        # Try to get device details (may include memory info)
                        device_details = tf.config.experimental.get_device_details(gpu)
                        if isinstance(device_details, dict):
                            if "device_name" in device_details:
                                device_name = device_details["device_name"]
                            # Some GPU drivers report memory in device details
                            if "memory" in device_details:
                                memory_gb = device_details["memory"] / (1024**3)
                    except (AttributeError, RuntimeError, KeyError):
                        # Device details not available - that's okay
                        pass

                    gpu_details.append({"name": device_name, "memory_gb": memory_gb})

                if self.verbose:
                    print(
                        f"    Configured {len(gpus)} GPU(s) with memory growth enabled"
                    )
                    for detail in gpu_details:
                        mem_str = (
                            f" ({detail['memory_gb']:.1f} GB)"
                            if detail["memory_gb"]
                            else ""
                        )
                        print(f"      GPU: {detail['name']}{mem_str}")

                    # Suggest batch size based on GPU memory (using powers of 2 for efficiency)
                    if gpu_details and gpu_details[0]["memory_gb"]:
                        memory_gb = gpu_details[0]["memory_gb"]
                        if memory_gb >= 24:
                            suggested_batch = 2048
                        elif memory_gb >= 16:
                            suggested_batch = (
                                2048  # Can handle 2048, or 1024 if memory constrained
                            )
                        elif memory_gb >= 8:
                            suggested_batch = 1024
                        else:
                            suggested_batch = 512

                        if self.batch_size < suggested_batch:
                            print(
                                f"    💡 Tip: Your GPU has {memory_gb:.1f} GB memory. "
                                f"Consider increasing batch_size to {suggested_batch} or higher "
                                f"for better GPU utilization (current: {self.batch_size})"
                            )
                        else:
                            print(
                                f"    Batch size {self.batch_size} is appropriate for "
                                f"{memory_gb:.1f} GB GPU memory"
                            )
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
        labels: Dict[str, np.ndarray],
        categorical_columns: List[str],
        validation_split: float = 0.1,
    ):
        """
        Fit the Multi-Task CTAB-GAN+ model to the provided dataframe.

        Args:
            dataframe: Input dataframe with mixed data types
            labels: Dictionary of task labels, each one-hot encoded: {"trading": (n, 3), "regime": (n, 3), ...}
            categorical_columns: List of column names that are categorical
            validation_split: Fraction of data to use for validation
        """
        if dataframe.empty:
            raise ValueError("Dataframe cannot be empty")

        # Validate labels
        if not isinstance(labels, dict):
            raise ValueError("labels must be a dictionary of task labels")

        if len(labels) == 0:
            raise ValueError("labels dictionary cannot be empty")

        # Validate all labels have same batch size
        batch_sizes = {task: arr.shape[0] for task, arr in labels.items()}
        if len(set(batch_sizes.values())) > 1:
            raise ValueError(
                f"All task labels must have the same batch size. Found: {batch_sizes}"
            )

        expected_batch_size = batch_sizes[list(labels.keys())[0]]
        if expected_batch_size != len(dataframe):
            raise ValueError(
                f"Dataframe length ({len(dataframe)}) doesn't match labels batch size ({expected_batch_size})"
            )

        # Store column information
        self.categorical_columns = categorical_columns
        self.continuous_columns = [
            col for col in dataframe.columns if col not in categorical_columns
        ]

        # Handle NA values and convert to float32
        dataframe = dataframe.copy()
        for col in self.continuous_columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
            dataframe[col] = dataframe[col].fillna(0.0)
            dataframe[col] = dataframe[col].astype(np.float32)

        # Basic NaN handling for categorical
        for col in self.categorical_columns:
            dataframe[col] = dataframe[col].fillna("unknown")

        # Log column distribution
        if self.verbose:
            print(
                f"    Data columns: {len(dataframe.columns)} total, "
                f"{len(self.categorical_columns)} categorical, "
                f"{len(self.continuous_columns)} continuous"
            )
            if self.continuous_columns:
                print("    Fitting VGM models for continuous columns...")

        # Process and validate task labels
        labels_processed = {}
        for task, task_labels in labels.items():
            arr = np.asarray(task_labels)
            if arr.ndim == 1:
                num_classes = int(arr.max()) + 1
                labels_processed[task] = np.eye(num_classes, dtype=np.float32)[
                    arr.astype(int)
                ]
            elif arr.ndim == 2:
                labels_processed[task] = arr.astype(np.float32)
            else:
                raise ValueError(f"Task '{task}' labels must be 1D or 2D array")

        # Store task label dimensions
        self.task_label_dims = {
            task: arr.shape[1] for task, arr in labels_processed.items()
        }
        self.total_cond_dim = sum(self.task_label_dims.values())
        self.sorted_tasks = sorted(list(self.task_label_dims.keys()))

        self.num_features = len(dataframe.columns)

        # Analyze columns — fit VGM for continuous, one-hot map for categorical
        self.column_info = {}
        self.vgm_models = {}
        categorical_info = []
        continuous_info = []
        self.column_order = list(dataframe.columns)

        for col in dataframe.columns:
            if col in categorical_columns:
                unique_vals = sorted(dataframe[col].unique())
                num_categories = len(unique_vals)
                cat_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                idx_to_cat = unique_vals
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
                        "vgm_components": 0,
                    }
                    continuous_info.append((1, 0))
                    continue

                # Fit Variational Gaussian Mixture Model
                bgm = BayesianGaussianMixture(
                    n_components=10,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )
                clean_data = dataframe[col].dropna().values.reshape(-1, 1)

                if self.verbose:
                    progress = len(self.vgm_models) + 1
                    print(
                        f"        [{progress}/{len(self.continuous_columns)}] {col}...",
                        end=" ",
                        flush=True,
                    )

                if len(np.unique(clean_data)) > 1:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        bgm.fit(clean_data)
                    if self.verbose:
                        print("Done")
                else:
                    if self.verbose:
                        print("Constant (Skipped)")
                    bgm.means_ = np.array([[clean_data[0, 0]]])
                    bgm.covariances_ = np.array([[[1e-4]]])
                    bgm.weights_ = np.array([1.0])
                    bgm.n_components = 1
                    bgm.predict_proba = lambda x: np.ones((len(x), 1))

                self.vgm_models[col] = bgm
                self.column_info[col] = {
                    "type": "continuous",
                    "min": col_min,
                    "max": col_max,
                    "mean": col_mean,
                    "std": col_std,
                    "vgm_components": bgm.n_components,
                }
                continuous_info.append((1, bgm.n_components))

        # Store column info
        self.categorical_info = categorical_info
        self.continuous_info = continuous_info
        self.num_categorical_features = len(categorical_columns)
        self.num_continuous_features = len(continuous_info)

        total_features = self.num_categorical_features + self.num_continuous_features
        if total_features != len(dataframe.columns):
            raise ValueError(
                f"Feature count mismatch: categorical ({self.num_categorical_features}) + "
                f"continuous ({self.num_continuous_features}) = {total_features}, "
                f"but dataframe has {len(dataframe.columns)} columns"
            )

        # Expanded feature dimensions (VGM scalar + mode one-hot per continuous col)
        self.categorical_dim = sum(self.categorical_info)
        self.continuous_dim = sum(val + modes for val, modes in self.continuous_info)
        self.total_feature_dim = self.continuous_dim + self.categorical_dim

        # Create models
        self._create_models()

        if self.verbose:
            print("\n    Pre-processing complete. Starting GAN training...")

        # Prepare training data using VGM transform
        train_data = self._transform_data(dataframe)
        train_labels_dict = labels_processed

        # Split validation if needed
        if validation_split > 0:
            split_idx = int(len(train_data) * (1 - validation_split))
            train_data, val_data = train_data[:split_idx], train_data[split_idx:]
            train_labels_dict = {
                task: arr[:split_idx] for task, arr in train_labels_dict.items()
            }
            val_labels_dict = {
                task: arr[split_idx:] for task, arr in labels_processed.items()
            }
        else:
            val_data = None
            val_labels_dict = None

        # Store original dataframe for evaluation
        self._original_dataframe = dataframe.copy()

        # Train the model
        self._train(
            train_data,
            train_labels_dict,
            val_data,
            val_labels_dict,
            original_dataframe=dataframe,
        )

        self.is_fitted = True

    def _transform_data(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe to numpy array using VGM encoding.
        Continuous data produces scalar + one-hot mode per column (VGM).
        Categorical data is one-hot encoded."""

        continuous_data_list = []
        for col in self.continuous_columns:
            if col in self.vgm_models and self.vgm_models[col] is not None:
                bgm = self.vgm_models[col]
                data = df[col].values.reshape(-1, 1)
                probs = bgm.predict_proba(data)
                modes = np.argmax(probs, axis=1)
                means = bgm.means_.reshape(1, -1)
                stds = np.sqrt(bgm.covariances_).reshape(1, -1)
                chosen_means = means[0, modes]
                chosen_stds = stds[0, modes]
                normalized_value = (df[col].values - chosen_means) / (
                    4 * chosen_stds + 1e-8
                )
                normalized_value = np.clip(normalized_value, -0.99, 0.99).reshape(-1, 1)
                num_modes = bgm.n_components
                one_hot_mode = np.eye(num_modes, dtype=np.float32)[modes]
                continuous_data_list.append(normalized_value)
                continuous_data_list.append(one_hot_mode)
            else:
                # Fallback min-max normalization
                info = self.column_info[col]
                val = df[col].values
                norm = 2 * (val - info["min"]) / (info["max"] - info["min"] + 1e-8) - 1
                continuous_data_list.append(norm.reshape(-1, 1))

        if continuous_data_list:
            continuous_data = np.concatenate(continuous_data_list, axis=1)
        else:
            continuous_data = np.zeros((len(df), 0), dtype=np.float32)

        categorical_data_list = []
        for col in self.categorical_columns:
            info = self.column_info[col]
            cat_to_idx = info["cat_to_idx"]
            num_cats = info["num_categories"]
            indices = np.array([cat_to_idx.get(val, 0) for val in df[col]])
            one_hot = np.eye(num_cats, dtype=np.float32)[indices]
            categorical_data_list.append(one_hot)

        if categorical_data_list:
            categorical_data = np.concatenate(categorical_data_list, axis=1)
            if continuous_data.shape[1] > 0:
                return np.concatenate([continuous_data, categorical_data], axis=1)
            else:
                return categorical_data
        return continuous_data

    def generate(
        self,
        num_samples: int,
        task_labels: Optional[Dict[str, np.ndarray]] = None,
    ) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Generate synthetic samples with specified task labels.

        Args:
            num_samples: Number of samples to generate
            task_labels: Dictionary of one-hot encoded labels for each task.
                        If None, generates with uniform distribution for all tasks.

        Returns:
            (generated_dataframe, task_labels_dict)
        """
        if self.generator is None:
            raise ValueError("Generator model not created. Must call fit() first.")

        # Prepare task labels
        if task_labels is None:
            task_labels = {}
            if self.random_seed is not None:
                rng = np.random.RandomState(self.random_seed + 2000)
            else:
                rng = np.random
            for task in self.sorted_tasks:
                num_classes = self.task_label_dims[task]
                classes = rng.randint(0, num_classes, size=num_samples)
                task_labels[task] = np.eye(num_classes, dtype=np.float32)[classes]
        else:
            if set(task_labels.keys()) != set(self.task_label_dims.keys()):
                raise ValueError(
                    f"task_labels must contain all tasks: {list(self.task_label_dims.keys())}"
                )
            for task in self.sorted_tasks:
                if task_labels[task].shape != (num_samples, self.task_label_dims[task]):
                    raise ValueError(
                        f"Task '{task}' labels shape {task_labels[task].shape} doesn't match "
                        f"expected ({num_samples}, {self.task_label_dims[task]})"
                    )

        # Concatenate all task labels into condition vector
        cond_vector = _concatenate_task_labels(task_labels)

        noise = tf.random.normal((num_samples, self.latent_dim), dtype=tf.float32)
        cond_vector_t = tf.convert_to_tensor(cond_vector, dtype=tf.float32)

        generated = self.generator([noise, cond_vector_t], training=False)
        generated = generated.numpy().astype(np.float32)

        # Split output: continuous portion first (VGM encoded), then categorical
        continuous_output = generated[:, : self.continuous_dim]
        categorical_output = generated[:, self.continuous_dim :]

        # Decode continuous columns using VGM inverse transform
        continuous_values = {}
        cont_offset = 0
        for idx, col in enumerate(self.continuous_columns):
            info = self.column_info[col]
            vgm_components = info["vgm_components"]

            scalar = continuous_output[:, cont_offset]
            offset_mode = cont_offset + 1
            mode_probs = continuous_output[
                :, offset_mode : offset_mode + vgm_components
            ]
            cont_offset += 1 + vgm_components

            if vgm_components > 0:
                modes = np.argmax(mode_probs, axis=1)
                bgm = self.vgm_models[col]
                means = bgm.means_.reshape(1, -1)
                stds = np.sqrt(bgm.covariances_).reshape(1, -1)
                chosen_means = means[0, modes]
                chosen_stds = stds[0, modes]
                denormalized = (scalar * 4 * chosen_stds) + chosen_means
            else:
                # Simple min-max reverse scaling for integer/linear columns
                denormalized = (
                    0.5 * (scalar + 1) * (info["max"] - info["min"]) + info["min"]
                )

            continuous_values[col] = np.clip(denormalized, info["min"], info["max"])

        # Decode categorical columns (argmax from one-hot probabilities)
        categorical_values = {}
        cat_offset = 0
        for idx, col in enumerate(self.categorical_columns):
            info = self.column_info[col]
            num_cats = info["num_categories"]
            probs = categorical_output[:, cat_offset : cat_offset + num_cats]
            cat_offset += num_cats
            category_indices = np.argmax(probs, axis=1)
            idx_to_cat = info["idx_to_cat"]
            categorical_values[col] = np.array(
                [idx_to_cat[int(cat_idx)] for cat_idx in category_indices]
            )

        # Assemble dataframe in original column order
        data_dict = {}
        for col in self.column_order:
            if col in categorical_values:
                data_dict[col] = categorical_values[col]
            else:
                data_dict[col] = continuous_values[col]

        generated_df = pd.DataFrame(data_dict, columns=self.column_order)
        return generated_df, task_labels

    def _create_models(self):
        """Create generator and discriminator models with VGM-aware output and PacGAN discriminator."""
        # Generator: noise + condition vector -> VGM-encoded continuous + one-hot categorical
        noise_input = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(self.total_cond_dim,))

        x = layers.Concatenate()([noise_input, label_input])

        for layer_size in self.generator_layers:
            x = layers.Dense(layer_size)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        # Build per-column output branches (same structure as single-task CTABGANPlus)
        outputs = []
        for i, (val_dim, num_modes) in enumerate(self.continuous_info):
            scalar = layers.Dense(1, activation="tanh", name=f"cont_val_{i}")(x)
            outputs.append(scalar)
            if num_modes > 0:
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
            [noise_input, label_input], all_outputs, name="mt_ctab_gan_generator"
        )

        # Discriminator: PacGAN — receives pac samples concatenated together
        data_input = layers.Input(shape=(self.total_feature_dim * self.pac,))
        label_input_d = layers.Input(shape=(self.total_cond_dim * self.pac,))

        x_d = layers.Concatenate()([data_input, label_input_d])

        for layer_size in self.discriminator_layers:
            x_d = layers.Dense(layer_size)(x_d)
            x_d = layers.LeakyReLU(0.2)(x_d)

        output_d = layers.Dense(1)(x_d)

        self.discriminator = Model(
            [data_input, label_input_d], output_d, name="mt_ctab_gan_discriminator"
        )

        # Compile with mixed-precision loss scaling if enabled
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

        # Generator optimizer (stored directly — training loop is manual)
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

    def _wasserstein_loss(self, y_true, y_pred):
        """Wasserstein loss for WGAN."""
        return tf.reduce_mean(y_pred)

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

        # PacGAN reshaping for gradient penalty
        pac_size = tf.shape(interpolated)[0] // self.pac
        num_to_keep = pac_size * self.pac
        interpolated = interpolated[:num_to_keep]
        labels = labels[:num_to_keep]

        interpolated_pac = tf.reshape(
            interpolated, [pac_size, self.pac * self.total_feature_dim]
        )
        labels_pac = tf.reshape(labels, [pac_size, self.pac * self.total_cond_dim])

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_pac)
            pred = self.discriminator([interpolated_pac, labels_pac], training=True)
            # Force float32 for mixed precision stability during gradient calculation
            pred = tf.cast(pred, tf.float32)

        grads = gp_tape.gradient(pred, [interpolated_pac])[0]
        # Compute gradient norm in float32 space to prevent float16 overflow/nan
        grads = tf.cast(grads, tf.float32)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
        gp = tf.reduce_mean(tf.square(norm - 1.0))
        return gp

    @tf.function  # Graph compilation for better GPU utilization
    def _train_discriminator_step(self, real_data, real_labels, noise):
        """Compiled training step for discriminator."""
        with tf.GradientTape() as d_tape:
            # Generate fake data
            fake_data = self.generator([noise, real_labels], training=True)

            # PacGAN reshaping
            pac_size = tf.shape(real_data)[0] // self.pac
            num_to_keep = pac_size * self.pac
            real_data = real_data[:num_to_keep]
            fake_data = fake_data[:num_to_keep]
            real_labels = real_labels[:num_to_keep]

            real_data_pac = tf.reshape(
                real_data, [pac_size, self.pac * self.total_feature_dim]
            )
            fake_data_pac = tf.reshape(
                fake_data, [pac_size, self.pac * self.total_feature_dim]
            )
            real_labels_pac = tf.reshape(
                real_labels, [pac_size, self.pac * self.total_cond_dim]
            )

            # Discriminator scores in float32 to prevent overflows
            real_scores = tf.cast(
                self.discriminator([real_data_pac, real_labels_pac], training=True),
                tf.float32,
            )
            fake_scores = tf.cast(
                self.discriminator([fake_data_pac, real_labels_pac], training=True),
                tf.float32,
            )

            # Wasserstein loss
            d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)

            # Gradient penalty
            gp = self._gradient_penalty(real_data, fake_data, real_labels)

            # Use bounded loss combination
            gp_weight_t = tf.cast(self.gp_weight, tf.float32)
            d_loss = d_loss + gp_weight_t * gp

            # Prevent hard NaN collapse
            d_loss = tf.where(tf.math.is_finite(d_loss), d_loss, tf.zeros_like(d_loss))

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

    @tf.function  # Graph compilation for better GPU utilization
    def _train_generator_step(self, labels, noise, real_data=None):
        """Compiled training step for generator. real_data optional for subclasses (info/downstream loss)."""
        with tf.GradientTape() as g_tape:
            fake_data = self.generator([noise, labels], training=True)

            # PacGAN reshaping
            pac_size = tf.shape(fake_data)[0] // self.pac
            num_to_keep = pac_size * self.pac
            fake_data = fake_data[:num_to_keep]
            labels = labels[:num_to_keep]

            fake_data_pac = tf.reshape(
                fake_data, [pac_size, self.pac * self.total_feature_dim]
            )
            labels_pac = tf.reshape(labels, [pac_size, self.pac * self.total_cond_dim])

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
        train_labels_dict: Dict[str, np.ndarray],
        val_data: Optional[np.ndarray] = None,
        val_labels_dict: Optional[Dict[str, np.ndarray]] = None,
        original_dataframe: Optional[pd.DataFrame] = None,
    ):
        """Train the Multi-Task CTAB-GAN+ model."""
        n_samples = len(train_data)
        steps_per_epoch = n_samples // self.batch_size

        # Early stopping and LR reduction tracking
        # Determine if we're maximizing (eval metrics) or minimizing (losses)
        is_maximizing = self.monitor_metric.startswith("eval_")
        best_metric = float("-inf") if is_maximizing else float("inf")
        patience_counter = 0
        lr_patience_counter = 0
        best_generator_weights = None
        best_discriminator_weights = None
        best_epoch = 0
        divergence_recovery_count = 0
        DIVERGENCE_D_LOSS_THRESHOLD = -12.0
        DIVERGENCE_G_LOSS_THRESHOLD = 12.0

        # Inform user about evaluation frequency
        if is_maximizing and self.verbose:
            print(
                f"    Using '{self.monitor_metric}' for best epoch selection. "
                f"Evaluation will run every epoch to ensure no best epoch is missed."
            )

        for epoch in range(self.epochs):
            # Shuffle data (use seeded random state for reproducibility)
            if self.random_seed is not None:
                rng = np.random.RandomState(self.random_seed + epoch)
                indices = rng.permutation(n_samples)
            else:
                indices = np.random.permutation(n_samples)
            train_data_shuffled = train_data[indices]
            train_labels_shuffled = {
                task: arr[indices] for task, arr in train_labels_dict.items()
            }

            d_losses = []
            g_losses = []

            for step in range(steps_per_epoch):
                # Get batch
                start_idx = step * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_data = train_data_shuffled[start_idx:end_idx]
                batch_labels_dict = {
                    task: arr[start_idx:end_idx]
                    for task, arr in train_labels_shuffled.items()
                }
                batch_size_actual = len(batch_data)

                # Concatenate task labels for condition vector
                batch_cond = _concatenate_task_labels(batch_labels_dict)

                # Train discriminator using compiled step
                noise = tf.random.normal(
                    (batch_size_actual, self.latent_dim), dtype=tf.float32
                )
                batch_data_t = tf.convert_to_tensor(batch_data, dtype=tf.float32)
                batch_cond_t = tf.convert_to_tensor(batch_cond, dtype=tf.float32)

                d_loss = self._train_discriminator_step(
                    batch_data_t, batch_cond_t, noise
                )
                d_losses.append(float(d_loss))

                # Optional: train auxiliary on real batch (used by CTABGANPlusMTEnhanced)
                if (
                    hasattr(self, "auxiliary")
                    and self.auxiliary is not None
                    and hasattr(self, "_train_auxiliary_step")
                ):
                    self._train_auxiliary_step(batch_data_t, batch_cond_t)

                # Train generator using compiled step (real_data passed for info/downstream loss in Enhanced)
                noise = tf.random.normal(
                    (batch_size_actual, self.latent_dim), dtype=tf.float32
                )
                g_loss = self._train_generator_step(batch_cond_t, noise, batch_data_t)
                g_losses.append(float(g_loss))

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
                    # Sample subset of training data for evaluation
                    eval_sample_size = min(len(train_data), self.eval_num_samples)
                    if self.random_seed is not None:
                        rng = np.random.RandomState(self.random_seed + epoch + 1000)
                        eval_indices = rng.choice(
                            len(train_data), eval_sample_size, replace=False
                        )
                    else:
                        eval_indices = np.random.choice(
                            len(train_data), eval_sample_size, replace=False
                        )

                    # Create evaluation dataframe from original dataframe
                    eval_real = original_dataframe.iloc[eval_indices].copy()

                    eval_labels_dict = {
                        task: train_labels_dict[task][eval_indices]
                        for task in train_labels_dict.keys()
                    }

                    # Evaluate
                    eval_metrics = self.evaluate(
                        real_data=eval_real,
                        num_samples=self.eval_num_samples,
                        task_labels=eval_labels_dict,
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
                        print(
                            f"    Restoring best model from epoch {best_epoch} "
                            f"({self.monitor_metric}={best_metric:.4f})"
                        )
                # Restore best weights
                if best_generator_weights is not None:
                    self.generator.set_weights(best_generator_weights)
                if best_discriminator_weights is not None:
                    self.discriminator.set_weights(best_discriminator_weights)
                break

            # Display training progress
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
                    if is_using_eval_metric:
                        eval_str = (
                            f", eval: quality={overall:.3f} [best: {best_metric:.3f}], "
                            f"div={diversity:.3f} (ratio={div_ratio:.3f}), "
                            f"corr={corr:.3f}, stat={stat_score:.3f}, valid={validity:.3f}"
                        )
                    else:
                        eval_str = (
                            f", eval: quality={overall:.3f}, div={diversity:.3f} (ratio={div_ratio:.3f}), "
                            f"corr={corr:.3f}, stat={stat_score:.3f}, valid={validity:.3f}"
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

    def evaluate(
        self,
        real_data: pd.DataFrame,
        num_samples: Optional[int] = None,
        task_labels: Optional[Dict[str, np.ndarray]] = None,
        generated_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate generated samples against real data.

        Args:
            real_data: Real dataframe to compare against
            num_samples: Number of samples to generate (default: eval_num_samples)
            task_labels: Optional task labels for generation
            generated_data: Optional pre-generated data. If provided, num_samples is ignored.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.generator is None:
            raise ValueError("Generator model not created. Must call fit() first.")

        # Use provided generated data or generate new samples
        if generated_data is None:
            if num_samples is None:
                num_samples = self.eval_num_samples

            # Generate synthetic samples
            generated_data, _ = self.generate(
                num_samples=num_samples,
                task_labels=task_labels,
            )

        return self.evaluate_with_dataframes(real_data, generated_data)

    def evaluate_with_dataframes(
        self, real_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate generated samples against real data.

        Args:
            real_data: Real dataframe to compare against
            generated_data: Generated dataframe to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure same columns and order
        real_data = real_data[self.column_order].copy()
        generated_data = generated_data[self.column_order].copy()

        metrics = {}

        # 1. DIVERSITY METRICS
        metrics["diversity"] = self._compute_diversity_metrics(
            real_data, generated_data
        )

        # 2. CORRELATION PRESERVATION
        metrics["correlation"] = self._compute_correlation_metrics(
            real_data, generated_data
        )

        # 3. STATISTICAL SIMILARITY
        metrics["statistics"] = self._compute_statistical_metrics(
            real_data, generated_data
        )

        # 4. VALIDITY CHECKS
        metrics["validity"] = self._compute_validity_metrics(generated_data)

        # 5. OVERALL SCORE
        metrics["overall_score"] = self._compute_overall_score(metrics)

        return metrics

    def _compute_diversity_metrics(
        self, real_data: pd.DataFrame, generated_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute diversity metrics to detect mode collapse."""
        metrics = {}

        # Convert to numpy for distance calculations
        real_array = real_data.values.astype(np.float32)
        gen_array = generated_data.values.astype(np.float32)

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

        # 1. Pairwise distances within generated samples
        if len(gen_sample) > 1:
            gen_distances = pdist(gen_sample, metric="euclidean")
            metrics["gen_pairwise_distance_mean"] = float(np.mean(gen_distances))
            metrics["gen_pairwise_distance_std"] = float(np.std(gen_distances))
            metrics["gen_pairwise_distance_min"] = float(np.min(gen_distances))
        else:
            metrics["gen_pairwise_distance_mean"] = 0.0
            metrics["gen_pairwise_distance_std"] = 0.0
            metrics["gen_pairwise_distance_min"] = 0.0

        # 2. Pairwise distances within real samples
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

        # 5. Coverage metric
        coverage_scores = []
        for col in self.continuous_columns:
            if col in generated_data.columns:
                real_min, real_max = real_data[col].min(), real_data[col].max()
                gen_min, gen_max = generated_data[col].min(), generated_data[col].max()
                real_range = real_max - real_min
                if real_range > 0:
                    coverage = min(1.0, (gen_max - gen_min) / real_range)
                    coverage_scores.append(coverage)
        metrics["value_space_coverage"] = (
            float(np.mean(coverage_scores)) if coverage_scores else 0.0
        )

        # 6. Nearest neighbor distances
        if len(gen_sample) > 0 and len(real_sample) > 0:
            try:
                from sklearn.neighbors import NearestNeighbors

                nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
                nn.fit(real_sample)
                distances, _ = nn.kneighbors(gen_sample)
                metrics["nearest_real_distance_mean"] = float(np.mean(distances))
                metrics["nearest_real_distance_std"] = float(np.std(distances))
            except ImportError:
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

        if len(continuous_cols) < 2:
            metrics["correlation_preservation"] = 1.0
            metrics["correlation_error"] = 0.0
            return metrics

        # Compute correlation matrices
        real_corr = real_data[continuous_cols].corr().values
        gen_corr = generated_data[continuous_cols].corr().values

        # Extract upper triangle (avoid diagonal and duplicates)
        mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
        real_corr_flat = real_corr[mask]
        gen_corr_flat = gen_corr[mask]

        # Correlation preservation score
        if len(real_corr_flat) > 0:
            corr_error = np.mean(np.abs(real_corr_flat - gen_corr_flat))
            metrics["correlation_error"] = float(corr_error)

            if np.std(real_corr_flat) > 0 and np.std(gen_corr_flat) > 0:
                corr_corr = np.corrcoef(real_corr_flat, gen_corr_flat)[0, 1]
                metrics["correlation_preservation"] = (
                    float(corr_corr) if not np.isnan(corr_corr) else 0.0
                )
            else:
                metrics["correlation_preservation"] = 0.0
        else:
            metrics["correlation_error"] = 0.0
            metrics["correlation_preservation"] = 1.0

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

        # Overall mean/std error
        if continuous_stats:
            metrics["mean_error_avg"] = float(
                np.mean([s["mean_error"] for s in continuous_stats.values()])
            )
            metrics["std_error_avg"] = float(
                np.mean([s["std_error"] for s in continuous_stats.values()])
            )
        else:
            metrics["mean_error_avg"] = 0.0
            metrics["std_error_avg"] = 0.0

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

                # Total variation distance
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

    def _compute_validity_metrics(self, generated_data: pd.DataFrame) -> Dict[str, Any]:
        """Check validity of generated samples."""
        metrics = {}

        # Check continuous columns are in valid ranges
        continuous_valid = {}
        for col in self.continuous_columns:
            if col in generated_data.columns and col in self.column_info:
                info = self.column_info[col]
                col_min = info["min"]
                col_max = info["max"]

                out_of_range = (
                    (generated_data[col] < col_min) | (generated_data[col] > col_max)
                ).sum()
                total = len(generated_data)

                continuous_valid[col] = {
                    "out_of_range_count": int(out_of_range),
                    "out_of_range_pct": (
                        float(out_of_range / total) if total > 0 else 0.0
                    ),
                }
        metrics["continuous_validity"] = continuous_valid

        # Check categorical columns have valid values
        categorical_valid = {}
        for col in self.categorical_columns:
            if col in generated_data.columns and col in self.column_info:
                info = self.column_info[col]
                valid_values = set(info["unique_values"])
                gen_values = set(generated_data[col].unique())

                invalid_values = gen_values - valid_values
                categorical_valid[col] = {
                    "invalid_value_count": len(invalid_values),
                    "invalid_values": list(invalid_values) if invalid_values else [],
                }
        metrics["categorical_validity"] = categorical_valid

        # Overall validity score
        total_invalid = sum(
            s["out_of_range_count"] for s in continuous_valid.values()
        ) + sum(s["invalid_value_count"] for s in categorical_valid.values())
        total_samples = len(generated_data)
        metrics["overall_validity_score"] = (
            1.0 - (total_invalid / total_samples) if total_samples > 0 else 1.0
        )

        return metrics

    def _compute_overall_score(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall quality score from all metrics."""
        score = {}

        # Diversity score (0-1, higher is better)
        diversity_ratio = metrics["diversity"].get("diversity_ratio", 0.0)
        diversity_score = 1.0 - abs(1.0 - diversity_ratio) * 0.5
        diversity_score = max(0.0, min(1.0, diversity_score))
        score["diversity_score"] = diversity_score

        # Correlation preservation score (0-1, higher is better)
        corr_preservation = metrics["correlation"].get("correlation_preservation", 0.0)
        score["correlation_score"] = max(0.0, min(1.0, corr_preservation))

        # Statistical similarity score (0-1, higher is better)
        mean_error = metrics["statistics"].get("mean_error_avg", 1.0)
        std_error = metrics["statistics"].get("std_error_avg", 1.0)
        cat_error = metrics["statistics"].get("categorical_error_avg", 1.0)
        stat_score = 1.0 / (1.0 + mean_error + std_error + cat_error)
        score["statistical_score"] = max(0.0, min(1.0, stat_score))

        # Validity score (0-1, higher is better)
        validity_score = metrics["validity"].get("overall_validity_score", 0.0)
        score["validity_score"] = validity_score

        # Weighted overall score
        score["overall_quality"] = (
            diversity_score * 0.4
            + corr_preservation * 0.4
            + stat_score * 0.15
            + validity_score * 0.05
        )

        return score

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
            "column_info": self.column_info,
            "column_order": self.column_order,
            "task_label_dims": self.task_label_dims,
            "num_categorical_features": self.num_categorical_features,
            "num_continuous_features": self.num_continuous_features,
            "categorical_info": self.categorical_info,
            "continuous_info": self.continuous_info,
            "categorical_dim": self.categorical_dim,
            "continuous_dim": self.continuous_dim,
            "total_feature_dim": self.total_feature_dim,
            "vgm_models": self.vgm_models,
            "latent_dim": self.latent_dim,
            "generator_layers": self.generator_layers,
            "discriminator_layers": self.discriminator_layers,
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
        if not os.path.exists(filepath):
            raise ValueError(f"Model directory does not exist: {filepath}")

        # Load models
        self.generator = tf.keras.models.load_model(
            os.path.join(filepath, "generator.keras")
        )
        self.discriminator = tf.keras.models.load_model(
            os.path.join(filepath, "discriminator.keras"), compile=False
        )

        # Load metadata
        metadata_path = os.path.join(filepath, "metadata.pkl")
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata file does not exist: {metadata_path}")

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.categorical_columns = metadata["categorical_columns"]
        self.continuous_columns = metadata["continuous_columns"]
        self.column_info = metadata["column_info"]
        self.column_order = metadata["column_order"]
        self.task_label_dims = metadata["task_label_dims"]
        self.total_cond_dim = sum(self.task_label_dims.values())
        self.sorted_tasks = sorted(list(self.task_label_dims.keys()))
        self.num_categorical_features = metadata["num_categorical_features"]
        self.num_continuous_features = metadata["num_continuous_features"]
        self.categorical_info = metadata["categorical_info"]
        self.continuous_info = metadata.get(
            "continuous_info", [(1, 0)] * self.num_continuous_features
        )
        self.categorical_dim = metadata.get(
            "categorical_dim", sum(self.categorical_info)
        )
        self.continuous_dim = metadata.get(
            "continuous_dim", self.num_continuous_features
        )
        self.total_feature_dim = metadata.get(
            "total_feature_dim", self.continuous_dim + self.categorical_dim
        )
        self.vgm_models = metadata.get("vgm_models", {})
        self.latent_dim = metadata["latent_dim"]
        self.generator_layers = metadata["generator_layers"]
        self.discriminator_layers = metadata["discriminator_layers"]

        # Recompile discriminator
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
            ),
            loss=self._wasserstein_loss,
        )

        # Re-attach optimizer to generator (no GAN wrapper needed — training is manual)
        optimizer_g = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.optimizer = optimizer_g
        self.discriminator.trainable = True

        self.is_fitted = True

        # Return thresholds and training_type if they exist in metadata (for strategies to use)
        return {
            "min_buy_gain_threshold": metadata.get("min_buy_gain_threshold"),
            "min_sell_loss_threshold": metadata.get("min_sell_loss_threshold"),
            "training_type": metadata.get("training_type"),
        }


class CTABGANPlusMTEnhanced(CTABGANPlusMT):
    """
    Enhanced Multi-Task CTAB-GAN+ with optional CNN, auxiliary model, and paper losses.

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
        use_cnn: bool = False,
        use_auxiliary: bool = False,
        info_loss_weight: float = 0.0,
        downstream_loss_weight: float = 0.0,
        generator_loss_weight: float = 0.0,
    ):
        """
        Extra args over CTABGANPlusMT:

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
        )
        self.use_cnn = use_cnn
        self.use_auxiliary = use_auxiliary
        self.info_loss_weight = float(info_loss_weight)
        self.downstream_loss_weight = float(downstream_loss_weight)
        self.generator_loss_weight = float(generator_loss_weight)
        self.auxiliary: Optional[Model] = None

    def _build_auxiliary_model(self):
        """Build auxiliary classifier A: encoded row -> task logits."""
        inp = layers.Input(shape=(self.total_feature_dim,))
        x = inp
        for _ in range(4):
            x = layers.Dense(256, activation="relu")(x)

        # For multi-task, we output a single dense layer matching the total condition vector size.
        logits = layers.Dense(self.total_cond_dim, name="aux_logits")(x)
        self.auxiliary = Model(inp, logits, name="mt_ctab_gan_auxiliary")

        # Binary cross-entropy effectively treats each task's class bits independently
        self.auxiliary.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def _create_models(self):
        """
        Create generator and discriminator models.
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
        label_input = layers.Input(shape=(self.total_cond_dim,))

        # Concatenate noise and label
        g_in = layers.Concatenate()([noise_input, label_input])

        # Project to a square feature map
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

        if self.generator_layers:
            x = layers.Dense(self.generator_layers[-1], activation="relu")(x)

        # Output layer splits into continuous branches (scalar + mode_probs) and categorical branches
        outputs = []
        for i, (val_dim, num_modes) in enumerate(self.continuous_info):
            scalar = layers.Dense(1, activation="tanh", name=f"cont_val_{i}")(x)
            outputs.append(scalar)
            if num_modes > 0:
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
            [noise_input, label_input], all_outputs, name="mt_ctab_gan_generator_cnn"
        )

        # ------------------------
        # Discriminator (CNN-based)
        # ------------------------
        data_input = layers.Input(shape=(self.total_feature_dim * self.pac,))
        label_input_d = layers.Input(shape=(self.total_cond_dim * self.pac,))

        d_in = layers.Concatenate()([data_input, label_input_d])

        side_d = int(np.ceil(np.sqrt(max(self.total_feature_dim * self.pac, 4))))
        proj_dim_d = side_d * side_d

        x_d = layers.Dense(proj_dim_d, activation="linear")(d_in)
        x_d = layers.Reshape((side_d, side_d, 1))(x_d)

        for _ in range(2):
            x_d = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(x_d)
            x_d = layers.LeakyReLU(0.2)(x_d)

        x_d = layers.Flatten()(x_d)

        if self.discriminator_layers:
            x_d = layers.Dense(self.discriminator_layers[-1])(x_d)
            x_d = layers.LeakyReLU(0.2)(x_d)

        output_d = layers.Dense(1)(x_d)

        self.discriminator = Model(
            [data_input, label_input_d],
            output_d,
            name="mt_ctab_gan_discriminator_cnn",
        )

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

        self.discriminator.trainable = False

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

    def _train_auxiliary_step(
        self, real_data: tf.Tensor, real_labels: tf.Tensor
    ) -> None:
        """Train auxiliary classifier on real batch (one step)."""
        if self.auxiliary is None:
            return
        self.auxiliary.train_on_batch(real_data, real_labels)

    @tf.function
    def _train_generator_step(self, labels, noise, real_data=None):
        """Generator step with optional information loss and downstream/generator loss."""
        with tf.GradientTape() as g_tape:
            fake_data = self.generator([noise, labels], training=True)

            pac_size = tf.shape(fake_data)[0] // self.pac
            num_to_keep = pac_size * self.pac
            fake_data_trunc = fake_data[:num_to_keep]
            labels_trunc = labels[:num_to_keep]

            fake_data_pac = tf.reshape(
                fake_data_trunc, [pac_size, self.pac * self.total_feature_dim]
            )
            labels_pac = tf.reshape(
                labels_trunc, [pac_size, self.pac * self.total_cond_dim]
            )

            # Use float32 to prevent mixed precision overflow
            fake_scores = tf.cast(
                self.discriminator([fake_data_pac, labels_pac], training=True),
                tf.float32,
            )
            g_loss = -tf.reduce_mean(fake_scores)

            # Start loss accumulating in float32 space initially
            g_loss_float32 = g_loss

            # Information loss (Mean/Std matching)
            if real_data is not None and getattr(self, "info_loss_weight", 0.0) > 0:
                # Calculate in float32 to prevent float16 overflow/NaN
                real_cast = tf.cast(real_data, tf.float32)
                fake_cast = tf.cast(fake_data, tf.float32)

                mean_r = tf.reduce_mean(real_cast, axis=0)
                mean_f = tf.reduce_mean(fake_cast, axis=0)

                # IMPORTANT: Add epsilon inside sqrt to prevent NaN gradients when variance is exactly 0
                var_r = tf.math.reduce_variance(real_cast, axis=0)
                var_f = tf.math.reduce_variance(fake_cast, axis=0)
                std_r = tf.sqrt(var_r + 1e-8)
                std_f = tf.sqrt(var_f + 1e-8)

                info_loss = tf.reduce_mean((mean_r - mean_f) ** 2) + tf.reduce_mean(
                    (std_r - std_f) ** 2
                )
                info_loss = tf.cast(info_loss, tf.float32)

                # Hard cap to prevent exploding loss
                info_loss = tf.minimum(info_loss, 50.0)
                g_loss_float32 = (
                    g_loss_float32
                    + tf.cast(self.info_loss_weight, tf.float32) * info_loss
                )

            # Downstream / generator loss
            down_weight = getattr(self, "downstream_loss_weight", 0.0) + getattr(
                self, "generator_loss_weight", 0.0
            )
            if real_data is not None and down_weight > 0 and self.auxiliary is not None:
                pred = self.auxiliary(fake_data, training=False)
                # Compute BinaryCrossentropy in float32 to prevent exp() overflow mapping to NaN
                bce_loss = tf.keras.losses.binary_crossentropy(
                    tf.cast(labels, tf.float32),
                    tf.cast(pred, tf.float32),
                    from_logits=True,
                )
                down_loss = tf.reduce_mean(bce_loss)
                down_loss = tf.cast(down_loss, tf.float32)
                down_loss = tf.minimum(down_loss, 50.0)
                g_loss_float32 = (
                    g_loss_float32 + tf.cast(down_weight, tf.float32) * down_loss
                )

            # Fallback to pure g_loss if auxiliary parts are NaN
            g_loss_float32 = tf.where(
                tf.math.is_finite(g_loss_float32), g_loss_float32, g_loss
            )
            # Cast back to original precision for tape gradient
            g_loss = tf.cast(g_loss_float32, g_loss.dtype)

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

    def save(
        self,
        filepath: str,
        min_buy_gain_threshold: Optional[float] = None,
        min_sell_loss_threshold: Optional[float] = None,
        training_type: Optional[int] = None,
    ):
        """Save generator, discriminator, and auxiliary (if present)."""
        super().save(
            filepath,
            min_buy_gain_threshold=min_buy_gain_threshold,
            min_sell_loss_threshold=min_sell_loss_threshold,
            training_type=training_type,
        )
        if self.auxiliary is not None:
            self.auxiliary.save(os.path.join(filepath, "auxiliary.keras"))

    def load(self, filepath: str) -> Dict[str, Optional[float]]:
        """Load generator, discriminator, and auxiliary if saved."""
        result = super().load(filepath)
        aux_path = os.path.join(filepath, "auxiliary.keras")
        if os.path.exists(aux_path):
            self.auxiliary = tf.keras.models.load_model(aux_path)
        return result
