# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
TrainingEngine - training data-prep + orchestration mixin.

Owns the core single-task training pipeline extracted verbatim from
BaseNNStrategy: prepare_training_data (per-pair scale/split/window/one-hot,
aggregating across pairs), get_training_class_weights, and train_model (the
orchestration that feeds classifier.train() — feature-count guard, preprocess,
class weights, markov matrix, seeded shuffle, fit, persist).

It also owns GAN augmentation: enhance_training_data (single-task row-level) and
preprocess_training_data (multi-task tensor-level) plus their GAN _resolve_*/
metadata/balance helpers, and the markov-smoothing helpers.

This is a mixin designed to be composed into an NN strategy (listed first in
BaseNNStrategy's bases). It calls collaborators via self — scale_dataframe /
get_normalized_size / normalise_for_gan / gan_scaler_a (FeatureNormalizer),
get_storage_location / get_model_path / debug_print (BaseStrategy),
dataframeUtils, and strategy config (gan_type, gan_augment, gan_augment_seed,
gan_synth_* knobs). It introduces no literal references to its host class.

The deterministic boundary this produces is pinned by
Framework/tests/test_training_pipeline_characterization.py so a continued
extraction can be validated in seconds.
"""

from typing import Optional, List, Any, Dict, Tuple

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle

from Predictors.KerasBasePredictor import KerasBasePredictor
from Framework.BaseStrategy import GANType


class TrainingEngine:
    """Training data-prep + orchestration. Stateless beyond what it reads via
    self; intended to be mixed into a BaseNNStrategy-like host."""

    def get_training_class_weights(self, train_labels=None, validation_labels=None):
        """Get the class weights for the training data"""
        if self.augment_training_data and train_labels is not None:
            labels_to_use = train_labels
        else:
            labels_to_use = (
                validation_labels if validation_labels is not None else train_labels
            )

        if labels_to_use is None:
            return [1.0, 1.0, 1.0]

        # Handle multi-task case (dictionary)
        if isinstance(labels_to_use, dict):
            class_weights_dict = {}
            for task_name, task_labels in labels_to_use.items():
                if task_labels.ndim == 2:
                    class_indices = np.argmax(task_labels, axis=1)
                else:
                    class_indices = task_labels.astype(int)

                unique_classes, class_counts = np.unique(
                    class_indices, return_counts=True
                )
                total_samples = len(class_indices)
                num_classes = len(np.unique(class_indices))

                balanced_weights = np.zeros(num_classes)
                for i, count in zip(unique_classes, class_counts):
                    if count > 0:
                        balanced_weights[i] = total_samples / (num_classes * count)
                    else:
                        balanced_weights[i] = 0.0

                class_weights_dict[task_name] = balanced_weights.tolist()

            return class_weights_dict

        # Handle single-task case (numpy array)
        else:
            labels_array = np.asarray(labels_to_use)

            if labels_array.ndim == 2:
                class_indices = np.argmax(labels_array, axis=1)
            else:
                class_indices = labels_array.astype(int)

            class_counts = np.bincount(class_indices, minlength=3)
            total_samples = len(class_indices)
            num_classes = 3

            class_weights = np.zeros(num_classes)
            for i in range(num_classes):
                if class_counts[i] > 0:
                    class_weights[i] = total_samples / (num_classes * class_counts[i])
                else:
                    class_weights[i] = 0.0

            return class_weights.tolist()
    def prepare_training_data(
        self,
        dataframe: List[DataFrame],
        labels: List[Any],
        norm: bool = True,
        pair_names: Optional[List[str]] = None,
    ):
        """Prepare the training data"""

        if len(dataframe) < 1:
            raise ValueError("No dataframes passed in")

        num_pairs = len(dataframe)
        num_rows = np.shape(dataframe[0])[0]
        max_index = num_pairs * num_rows

        if max_index <= 1:
            raise ValueError(
                f"Insufficient data for training: max_index={max_index} "
                f"(num_pairs={num_pairs}, num_rows={num_rows}). "
                f"Need at least 2 total rows across all pairs."
            )

        aggr_tsr_train = None
        aggr_tsr_test = None
        aggr_train_labels = None
        aggr_test_labels = None

        # Optional per-sample P&L-magnitude weights, carried in the "%pnl_weight"
        # column when a strategy opts in (get_training_labels sets it). Accumulated
        # in parallel with train_labels through the SAME slicing so alignment
        # holds. have_weights stays False → returns None (default: no weighting).
        train_weights_parts = []
        have_weights = True

        for i in range(num_pairs):

            pair_labels = np.asarray(labels[i])
            pair_weights_full = None
            if "%pnl_weight" in dataframe[i].columns:
                pair_weights_full = np.asarray(
                    dataframe[i]["%pnl_weight"].values, dtype=np.float64
                )
            if norm:
                df_norm = self.scale_dataframe(dataframe[i])
            else:
                df_norm = dataframe[i]

            split_idx = int(self.TRAIN_DATA_SPLIT * len(df_norm))

            buffer_size = self.seq_len - 1

            train_end = split_idx - buffer_size
            train_df = df_norm[:train_end]

            test_start = train_end
            test_df = df_norm[test_start:]

            train_labels = pair_labels[:train_end]
            test_labels = pair_labels[test_start:]
            train_weights = (
                pair_weights_full[:train_end]
                if pair_weights_full is not None
                else None
            )

            pair_name = (
                pair_names[i]
                if pair_names is not None and i < len(pair_names)
                else None
            )
            train_df, train_labels = self.enhance_training_data(
                train_df, train_labels, pair_name=pair_name
            )

            train_labels = self.dataframeUtils.one_hot_encode(train_labels, 3)
            test_labels = self.dataframeUtils.one_hot_encode(test_labels, 3)

            tsr_train = self.dataframeUtils.df_to_tensor(
                train_df, self.seq_len, method=self.tensor_method
            )
            tsr_test = self.dataframeUtils.df_to_tensor(
                test_df, self.seq_len, method=self.tensor_method
            )

            if self.tensor_method == 0:
                offset = self.seq_len - 1
            else:
                offset = 0

            train_labels = train_labels[offset:]
            test_labels = test_labels[offset:]

            if train_weights is not None:
                train_weights = train_weights[offset:]
                train_weights_parts.append(train_weights)
            else:
                have_weights = False

            if aggr_tsr_train is None:
                aggr_tsr_train = tsr_train
                aggr_tsr_test = tsr_test
                aggr_train_labels = train_labels
                aggr_test_labels = test_labels
            else:
                aggr_tsr_train = np.concatenate([aggr_tsr_train, tsr_train], axis=0)
                aggr_tsr_test = np.concatenate([aggr_tsr_test, tsr_test], axis=0)
                aggr_train_labels = np.concatenate(
                    [aggr_train_labels, train_labels], axis=0
                )
                aggr_test_labels = np.concatenate(
                    [aggr_test_labels, test_labels], axis=0
                )

        aggr_train_weights = (
            np.concatenate(train_weights_parts, axis=0)
            if have_weights and train_weights_parts
            else None
        )

        return (
            aggr_tsr_train,
            aggr_tsr_test,
            aggr_train_labels,
            aggr_test_labels,
            aggr_train_weights,
        )

    def train_model(
        self,
        dataframes: [DataFrame],
        labels: [Any],
        classifier: KerasBasePredictor,
        pair_names: Optional[List[str]] = None,
    ):
        """Train the model - default implementation"""

        tsr_train, tsr_test, train_labels, test_labels, train_weights = (
            self.prepare_training_data(
                dataframes, labels, pair_names=pair_names,
            )
        )

        if tsr_train is not None and len(tsr_train.shape) >= 2:
            actual_features = tsr_train.shape[-1]
            if hasattr(classifier, "model") and classifier.model is not None:
                model_input_shape = classifier.model.input_shape
                if (
                    isinstance(model_input_shape, (list, tuple))
                    and len(model_input_shape) > 0
                ):
                    input_shape = (
                        model_input_shape[0]
                        if isinstance(model_input_shape, list)
                        else model_input_shape
                    )
                else:
                    input_shape = model_input_shape

                if input_shape and len(input_shape) >= 2:
                    expected_features = input_shape[-1]
                    if actual_features != expected_features:
                        expected_size = (
                            self.get_normalized_size(dataframes[0])
                            if dataframes
                            else "unknown"
                        )
                        raise ValueError(
                            f"Feature count mismatch: Model expects {expected_features} features, "
                            f"but training data has {actual_features} features after augmentation.\n"
                            f"  Expected normalized size: {expected_size}\n"
                            f"  This usually means the GAN model was trained with a different feature set.\n"
                            f"  Please retrain the GAN model with the current feature set."
                        )

        tsr_train, tsr_test, train_labels, test_labels = self.preprocess_training_data(
            dataframes[0], tsr_train, tsr_test, train_labels, test_labels
        )

        class_weights = self.get_training_class_weights(
            train_labels=train_labels, validation_labels=test_labels
        )

        if self.use_markov_smoothing:
            label_seq = self._labels_to_class_indices(test_labels)
            self.markov_transition_matrix = self._compute_markov_transition_matrix(
                label_seq, num_classes=3
            )

        if self.shuffle_train_data:
            # df_to_tensor with method=3 (Apple-Silicon default) returns
            # mlx.core.array, which sklearn.utils.shuffle can't index
            # ("Cannot index mlx array using the given type yet").
            # Coerce to numpy before shuffling; classifier.train()
            # converts back to mlx internally if it wants.  Same
            # workaround NNPredict applies inside its prepare_training_data.
            tsr_train = np.asarray(tsr_train)
            if isinstance(train_labels, dict):
                train_labels = {k: np.asarray(v) for k, v in train_labels.items()}
                rng = np.random.RandomState(42)
                indices = rng.permutation(len(tsr_train))
                tsr_train = tsr_train[indices]
                train_labels = {
                    key: value[indices] for key, value in train_labels.items()
                }
            else:
                train_labels = np.asarray(train_labels)
                if train_weights is not None:
                    # Shuffle weights in lockstep so P&L weighting stays aligned.
                    tsr_train, train_labels, train_weights = shuffle(
                        tsr_train, train_labels, train_weights, random_state=42
                    )
                else:
                    tsr_train, train_labels = shuffle(
                        tsr_train, train_labels, random_state=42
                    )

        # Pass sample_weights only when present, so classifiers whose train()
        # doesn't accept the kwarg (Keras, other backends) are unaffected. Only
        # the opt-in P&L-weighted path (MLXClassifierNary) ever receives it.
        train_kwargs = {"class_weights": class_weights}
        if train_weights is not None:
            train_kwargs["sample_weights"] = train_weights
        classifier.train(
            tsr_train, tsr_test, train_labels, test_labels, **train_kwargs
        )

        if self.use_markov_smoothing and self.markov_transition_matrix is not None:
            markov_path = self.get_markov_matrix_path()
            markov_dir = os.path.dirname(markov_path)
            if not os.path.exists(markov_dir):
                os.makedirs(markov_dir)
            np.save(markov_path, self.markov_transition_matrix)

        return None

    # ====================================================================
    # Markov smoothing helpers
    # ====================================================================

    def _labels_to_class_indices(self, labels) -> np.ndarray:
        """Normalize label formats to 1D class index array."""
        if isinstance(labels, dict):
            if "trading" in labels:
                labels = labels["trading"]
            else:
                labels = next(iter(labels.values()))

        arr = np.asarray(labels)
        if arr.ndim > 1:
            return np.argmax(arr, axis=1).astype(int)
        return arr.astype(int)
    @staticmethod
    def _compute_markov_transition_matrix(
        label_seq: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Compute transition probabilities P(next_state | current_state)."""
        if label_seq is None or len(label_seq) < 2:
            return np.eye(num_classes, dtype=float)

        counts = np.zeros((num_classes, num_classes), dtype=float)
        prev = label_seq[:-1]
        nxt = label_seq[1:]

        for a, b in zip(prev, nxt):
            if 0 <= a < num_classes and 0 <= b < num_classes:
                counts[int(a), int(b)] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return counts / row_sums

    # ====================================================================
    # Training trigger (called from the populate_indicators hook)
    # ====================================================================

    def maybe_train(self, dataframe: DataFrame, labels, curr_pair: str) -> DataFrame:
        """Lazily build the classifier and run the training trigger.

        Multi-pair aggregation path: buffer each pair's (frame, labels); once the
        last whitelist pair is seen, build the sequential index and train on the
        full set (when training is needed and no model exists yet), then return
        the current pair's indexed frame. Single-pair path: train if no model
        exists, then return the frame. Relocated verbatim from
        populate_indicators so the engine — not the freqtrade hook — owns the
        training trigger; all collaborators are resolved via self."""
        # set up the classifier if it doesn't already exist
        if self.classifier is None:
            num_features = self.get_normalized_size(dataframe)
            self.classifier_type = self.get_classifier_type()
            self.classifier = self.get_classifier(
                self.classifier_type, self.curr_pair, self.seq_len, num_features
            )
            self.classifier.set_model_path(self.get_model_path())
            self.classifier.set_batch_size(self.batch_size)

        if self.aggregate_pairs:
            whitelist = self.dp.current_whitelist()

            self.df_array.append(dataframe)
            self.label_array.append(labels)
            self.pair_count += 1

            if self.pair_count == len(whitelist):
                self.df_array = self.add_sequential_index(self.df_array)

                if self.training_needed and not self.model_exists():
                    self.debug_print(f"    Training model on {self.pair_count} pairs")
                    self.train_model(
                        self.df_array,
                        self.label_array,
                        self.classifier,
                        pair_names=list(whitelist),
                    )

            if self.pair_count == len(whitelist):
                pair_index = (
                    whitelist.index(curr_pair) if curr_pair in whitelist else -1
                )
                if pair_index >= 0 and pair_index < len(self.df_array):
                    dataframe = self.df_array[pair_index]
                else:
                    dataframe = self.df_array[-1]

        else:
            if not self.model_exists():
                self.train_model(
                    [dataframe], [labels], self.classifier,
                    pair_names=[self.curr_pair],
                )

            self.dbg_curr_df = dataframe
            dataframe = self.add_debug_indicators(dataframe)

        return dataframe

    # ====================================================================
    # GAN augmentation (single-task row-level + multi-task tensor-level)
    # ====================================================================

    _MULTI_TASK_GAN_TYPES = frozenset(
        {GANType.MT_WGAN, GANType.MT_CTAB_GAN, GANType.MT_DDPM}
    )

    def _resolve_neural_discriminator_path(self) -> Optional[str]:
        """Return the configured path or the conventional default under
        the strategy's storage location. Returns None when the filter
        is disabled (both reject_pct and threshold inactive)."""
        rej = float(getattr(self, "gan_synth_neural_discrim_reject_pct", 0.0))
        thr = getattr(self, "gan_synth_neural_discrim_threshold", None)
        if rej <= 0.0 and thr is None:
            return None
        explicit = getattr(self, "gan_synth_neural_discrim_model_path", None)
        if explicit:
            return str(explicit)
        try:
            return os.path.join(
                self.get_storage_location(), "Discriminators", "realness"
            )
        except Exception:
            return None
    def _resolve_realsignal_root(self) -> Optional[str]:
        """Return the configured path or the conventional default for
        the per-class realsignal classifiers. Returns None when the
        filter is disabled (both reject_pct and threshold inactive)."""
        rej = float(getattr(self, "gan_synth_realsignal_reject_pct", 0.0))
        thr = getattr(self, "gan_synth_realsignal_threshold", None)
        if rej <= 0.0 and thr is None:
            return None
        explicit = getattr(self, "gan_synth_realsignal_model_root", None)
        if explicit:
            return str(explicit)
        try:
            return os.path.join(
                self.get_storage_location(), "Discriminators", "realsignal"
            )
        except Exception:
            return None
    def _resolve_autoencoder_root(self) -> Optional[str]:
        """Return the configured path or the conventional default for
        the per-class autoencoders. Returns None when the filter is
        disabled (both reject_pct and threshold inactive)."""
        rej = float(getattr(self, "gan_synth_autoencoder_reject_pct", 0.0))
        thr = getattr(self, "gan_synth_autoencoder_threshold", None)
        if rej <= 0.0 and thr is None:
            return None
        explicit = getattr(self, "gan_synth_autoencoder_model_root", None)
        if explicit:
            return str(explicit)
        try:
            return os.path.join(
                self.get_storage_location(), "Discriminators", "autoencoder"
            )
        except Exception:
            return None
    def _apply_gan_inference_overrides(self, interface) -> None:
        """Push strategy-level sampling knobs onto the loaded model.

        Both attributes are best-effort — if the model doesn't have the
        named attribute (e.g. WGAN doesn't have a guidance scale), we
        skip silently rather than raise. That way the same hook is safe
        to call from every GAN load path.
        """
        model = getattr(interface, "_model", None)
        if model is None:
            return
        steps = getattr(self, "gan_inference_sample_steps", None)
        if steps is not None and hasattr(model, "num_sample_steps"):
            model.num_sample_steps = int(steps)
        scale = getattr(self, "gan_inference_guidance_scale", None)
        if scale is not None and hasattr(model, "guidance_scale"):
            model.guidance_scale = float(scale)
        clip = getattr(self, "gan_inference_zscore_clip", None)
        if clip is not None and hasattr(model, "_ZSCORE_CLIP"):
            model._ZSCORE_CLIP = float(clip)
    def _format_for_gan_scaler(self, array_2d):
        if isinstance(array_2d, pd.DataFrame):
            return array_2d
        columns = []
        if hasattr(self, "_get_gan_feature_columns"):
            columns = self._get_gan_feature_columns()
        if columns and array_2d.shape[1] == len(columns):
            return pd.DataFrame(array_2d, columns=columns)
        return array_2d
    def _resolve_gan_passthrough_indices(
        self,
        train_minmax: Any = None,
        train_df: Any = None,
    ) -> Optional[List[int]]:
        """Resolve ``gan_passthrough_columns`` to integer indices using
        the most reliable column-order reference available.

        Priority order:
          1. ``train_minmax`` if it's a DataFrame (post-normalisation
             frame whose columns match what the GAN saw at training).
          2. The GAN scaler's ``feature_names_in_`` attribute.
          3. ``train_df`` columns as a last resort.

        Returns ``None`` when the config is empty or no resolved
        indices land in the resolved column order — that signals the
        callers to skip the swap entirely.
        """
        configured = getattr(self, "gan_passthrough_columns", None)
        if not configured:
            return None

        feature_names: Optional[List[str]] = None
        if isinstance(train_minmax, pd.DataFrame):
            feature_names = list(train_minmax.columns)
        else:
            scaler = getattr(self, "gan_scaler_a", None)
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                try:
                    feature_names = list(scaler.feature_names_in_)
                except Exception:
                    feature_names = None
            if feature_names is None and isinstance(train_df, pd.DataFrame):
                feature_names = list(train_df.columns)

        if not feature_names:
            return None

        from GANs.passthrough import resolve_column_indices  # noqa: E402
        indices = resolve_column_indices(configured, feature_names)
        return indices or None
    def _invoke_balance_multi_task(
        self,
        interface: Any,
        data: np.ndarray,
        labels: Dict[str, np.ndarray],
        *,
        dataframe: Any = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Single dispatch point for multi-task GAN augmentation.

        Resolves feature names, passthrough indices, and AE-filter config
        from instance state, then calls ``GANs.balance.balance_multi_task``.
        All the multi-task augmentation call sites funnel through here so any
        future kwarg addition lands in one place.
        """
        from GANs.balance import balance_multi_task  # noqa: E402

        feature_names: Optional[List[str]] = None
        scaler = getattr(self, "gan_scaler_a", None)
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            try:
                feature_names = list(scaler.feature_names_in_)
            except Exception:
                feature_names = None

        passthrough_indices = self._resolve_gan_passthrough_indices(
            train_minmax=None, train_df=dataframe
        )

        return balance_multi_task(
            interface=interface,
            data=data,
            labels=labels,
            target_ratios=self.gan_target_ratio,
            log=print,
            debug_log=self.debug_print,
            diagnostics=bool(getattr(self, "gan_run_diagnostics", False)),
            feature_names=feature_names,
            passthrough_columns=passthrough_indices,
            autoencoder_threshold=getattr(
                self, "gan_synth_autoencoder_threshold", None
            ),
            autoencoder_model_root=self._resolve_autoencoder_root(),
            seed=self.gan_augment_seed,
        )
    def enhance_training_data(
        self,
        train_df: DataFrame,
        train_labels,
        pair_name: Optional[str] = None,
    ) -> Tuple[DataFrame, Any]:
        """Augment the per-pair training set with the configured GAN.

        Pure dispatcher — single source of truth for "load the GAN,
        validate its metadata against the strategy, balance the
        classes, hand back the augmented frame".  The strategy never
        sees GAN-type-specific code.

        Behaviour:
          * ``gan_type == NONE`` or ``gan_augment is False`` → pass-through.
          * Single-task type (WGAN, CTAB_GAN, CGAN) with ndarray labels →
            normalise → load with strict metadata validation →
            ``balance_single_task`` → denormalise.
          * Multi-task type (MT_WGAN, MT_CTAB_GAN) with dict labels →
            **handed off to ``preprocess_training_data``** because
            multi-task GANs operate on the windowed 3-D tensors produced
            after ``prepare_training_data``, not on the pre-windowed 2-D
            DataFrame seen here.  This method returns the inputs
            unchanged for that case; the real work happens in
            ``BaseNNMTStrategy.preprocess_training_data``.
          * Mismatched gan_type / label shape → pass-through (a noisy
            warning is logged so the misconfiguration is visible).

        Metadata validation: a strict ``GANInterface.load(expected=…)``
        comparison.  If the GAN was trained with different thresholds /
        training_type / num_features than the strategy currently
        declares, ``GANMetadataMismatchError`` is raised so the
        operator must explicitly retrain or update — silently using
        stale metadata corrupts training (the bug we fixed).

        Returns the augmented ``(train_df, train_labels)`` in the same
        types and shapes as the inputs.
        """
        if self.gan_type == GANType.NONE or not self.gan_augment:
            return train_df, train_labels

        is_multi_task_labels = isinstance(train_labels, dict)
        is_multi_task_type = self.gan_type in self._MULTI_TASK_GAN_TYPES

        if is_multi_task_labels and not is_multi_task_type:
            # Misconfiguration — log loudly and skip rather than crash.
            print(
                f"    enhance_training_data: gan_type={self.gan_type.name} "
                f"and label shape ({'dict' if is_multi_task_labels else 'ndarray'}) "
                f"disagree on multi-task — skipping augmentation"
            )
            return train_df, train_labels

        # Multi-task augmentation has to happen on the windowed 3-D tensor,
        # not on this pre-windowed DataFrame -- pass through and let
        # BaseNNMTStrategy.preprocess_training_data run the balance against
        # the tensor shape the GAN was actually trained on.
        if is_multi_task_labels:
            return train_df, train_labels

        if is_multi_task_type:
            # Single-task strategy with MT GAN type — defer augmentation to
            # preprocess_training_data, which operates on the windowed 3D tensor
            # produced by prepare_training_data, matching the shape the GAN was
            # trained on.
            return train_df, train_labels

        if train_df is None or len(train_df) == 0:
            return train_df, train_labels
        if len(train_labels) == 0:
            return train_df, train_labels

        # Lazy imports — the GAN stack pulls in TF / MLX which we don't
        # want to import for strategies that never enable augmentation.
        from GANs.balance import balance_single_task  # noqa: E402
        from GANs.GANInterface import GANInterface, GANMetadataMismatchError  # noqa: E402
        from GANs.paths import gan_save_path  # noqa: E402

        save_path = gan_save_path(
            self.get_storage_location(),
            self.gan_type,
            use_pca=bool(getattr(self, "use_pca_reduction", False)),
            post_gan_scaling=bool(getattr(self, "use_post_gan_scaling", False)),
        )
        interface = GANInterface(self.gan_type, save_path=save_path)

        expected = self._gan_expected_metadata(train_df)
        try:
            interface.load(expected=expected)
        except GANMetadataMismatchError:
            # Already self-explanatory — propagate as-is so the
            # operator sees the per-key diff.
            raise
        except FileNotFoundError as load_err:
            raise RuntimeError(
                f"GAN model not found at {save_path}. "
                f"Train it first via the corresponding Create* strategy "
                f"(gan_type={self.gan_type.name}). "
                f"Underlying error: {load_err}"
            ) from load_err
        except Exception as load_err:
            raise RuntimeError(
                f"Failed to load GAN model at {save_path}: {load_err}"
            ) from load_err
        self._apply_gan_inference_overrides(interface)

        # Normalise to GAN training space.  Keep the DataFrame view —
        # the balance helpers use it for column-aware passthrough and
        # the post-balance denormalise step below also expects it.
        train_minmax = self.normalise_for_gan(train_df)
        if not isinstance(train_minmax, pd.DataFrame):
            train_minmax = self._format_for_gan_scaler(train_minmax)
            if not isinstance(train_minmax, pd.DataFrame):
                # Last resort — synthesise column names so passthrough
                # and concatenation can still match.
                train_minmax = pd.DataFrame(
                    np.asarray(train_minmax),
                    columns=list(train_df.columns),
                )

        passthrough = self._resolve_gan_passthrough_for_dispatcher(
            train_minmax, train_df,
        )

        # Multi-task GANs are handled in BaseNNMTStrategy.preprocess_training_data
        # against the 3-D tensor; we return early above for is_multi_task_labels
        # so we only reach here for the single-task path.
        aug_minmax, aug_labels = balance_single_task(
            interface=interface,
            data=train_minmax,
            labels=train_labels,
            target_ratio=self.gan_target_ratio,
            log=print,
            debug_log=self.debug_print,
            diagnostics=bool(self.gan_run_diagnostics),
            feature_names=list(train_df.columns),
            density_reject_pct=float(
                getattr(self, "gan_synth_density_reject_pct", 0.0)
            ),
            density_n_components=int(
                getattr(self, "gan_synth_density_components", 8)
            ),
            discrim_reject_pct=float(
                getattr(self, "gan_synth_discrim_reject_pct", 0.0)
            ),
            neural_discrim_reject_pct=float(
                getattr(self, "gan_synth_neural_discrim_reject_pct", 0.0)
            ),
            neural_discrim_model_path=self._resolve_neural_discriminator_path(),
            neural_discrim_threshold=getattr(
                self, "gan_synth_neural_discrim_threshold", None
            ),
            realsignal_reject_pct=float(
                getattr(self, "gan_synth_realsignal_reject_pct", 0.0)
            ),
            realsignal_model_root=self._resolve_realsignal_root(),
            realsignal_threshold=getattr(
                self, "gan_synth_realsignal_threshold", None
            ),
            mahalanobis_reject_pct=float(
                getattr(self, "gan_synth_mahalanobis_reject_pct", 0.0)
            ),
            mahalanobis_threshold=getattr(
                self, "gan_synth_mahalanobis_threshold", None
            ),
            autoencoder_reject_pct=float(
                getattr(self, "gan_synth_autoencoder_reject_pct", 0.0)
            ),
            autoencoder_model_root=self._resolve_autoencoder_root(),
            autoencoder_threshold=getattr(
                self, "gan_synth_autoencoder_threshold", None
            ),
            passthrough_columns=passthrough,
            pair_name=pair_name,
            seed=self.gan_augment_seed,
        )

        # Denormalise back to the strategy's input space and restore
        # the original column order — some backends emit columns in
        # their own training order which would otherwise drift.
        if isinstance(aug_minmax, np.ndarray):
            aug_minmax = self._format_for_gan_scaler(aug_minmax)
        aug_normalized = self.denormalise_from_gan(aug_minmax)
        if isinstance(aug_normalized, pd.DataFrame):
            aug_df = aug_normalized.reset_index(drop=True)
        else:
            aug_df = pd.DataFrame(
                np.asarray(aug_normalized),
                columns=list(train_df.columns),
            )
        if list(aug_df.columns) != list(train_df.columns):
            # Reorder rather than raise — the GAN's column order is a
            # superset/permutation, not a real mismatch (a mismatch
            # would have failed metadata validation above).
            aug_df = aug_df[train_df.columns]

        return aug_df, aug_labels
    def _gan_expected_metadata(self, train_df: DataFrame) -> Dict[str, Any]:
        """Metadata fields the strategy expects to find in the saved GAN.

        ``GANInterface.load(expected=…)`` compares each key against the
        persisted metadata and raises ``GANMetadataMismatchError`` on
        any drift.  Strategy is the source of truth: if the GAN was
        trained with different thresholds, the *operator* decides
        whether to retrain or update the strategy — we don't decide
        for them.

        Subclasses can extend this (e.g. CTAB-GAN-aware strategies can
        add ``column_order`` or ``num_features``) without touching the
        dispatcher itself.
        """
        return {
            "min_buy_gain_threshold": float(self.MIN_BUY_GAIN_THRESHOLD),
            "min_sell_loss_threshold": float(self.MIN_SELL_LOSS_THRESHOLD),
            "training_type": int(self.TRAINING_TYPE),
            "horizon": int(self.HORIZON),
        }
    def _resolve_gan_passthrough_for_dispatcher(
        self,
        train_minmax: Any,
        train_df: DataFrame,
    ) -> Optional[List[Any]]:
        """Resolve ``gan_passthrough_columns`` to a form the balance
        helpers can consume.

        For DataFrame inputs we return the configured names filtered to
        what's present (``swap_passthrough_columns`` accepts names).
        For ndarray inputs we delegate to the existing index resolver.
        """
        configured = getattr(self, "gan_passthrough_columns", None)
        if not configured:
            return None
        if isinstance(train_minmax, pd.DataFrame):
            present = [c for c in configured if c in train_minmax.columns]
            return present or None
        return self._resolve_gan_passthrough_indices(train_minmax, train_df)
    def preprocess_training_data(
        self, dataframe: DataFrame, train_data, test_data, train_labels, test_labels
    ):
        """Tensor-level GAN augmentation for the single-task + MT GAN case.

        The standard single-task GAN path runs at DataFrame row level inside
        enhance_training_data.  When the configured GAN is a multi-task tensor
        backend (MT_DDPM, MT_WGAN, MT_CTAB_GAN) and labels are a plain ndarray
        (i.e. this is a single-task classifier), we must augment at tensor level
        instead — otherwise sliding-window sequence construction mixes real and
        iid synthetic rows in unnatural ways.

        Pass-through for: GAN disabled; non-MT GAN type; labels already a dict
        (handled by BaseNNMTStrategy override); non-3D train_data; empty data.
        """
        # Guards — return inputs unchanged for anything that isn't the new case.
        if self.gan_type == GANType.NONE or not getattr(self, "gan_augment", False):
            return train_data, test_data, train_labels, test_labels
        if self.gan_type not in self._MULTI_TASK_GAN_TYPES:
            return train_data, test_data, train_labels, test_labels
        if isinstance(train_labels, dict):
            # MT classifier path — BaseNNMTStrategy handles this in its override.
            return train_data, test_data, train_labels, test_labels
        if not isinstance(train_data, np.ndarray) or train_data.ndim != 3:
            return train_data, test_data, train_labels, test_labels
        if train_data.shape[0] == 0:
            return train_data, test_data, train_labels, test_labels

        # Wrap single-task labels as {"trading": one_hot}.
        # Coerce to 2-D one-hot if labels arrived as 1-D class indices.
        labels_arr = np.asarray(train_labels)
        if labels_arr.ndim == 1:
            num_classes = int(labels_arr.max()) + 1
            one_hot = np.eye(num_classes, dtype=np.float32)[labels_arr.astype(int)]
        else:
            one_hot = labels_arr.astype(np.float32)
        wrapped_labels = {"trading": one_hot}

        # Lazy imports — keep GAN stack out of strategies that never use it.
        from GANs.GANInterface import GANInterface, GANMetadataMismatchError  # noqa: E402
        from GANs.paths import gan_save_path  # noqa: E402
        from GANs.mt_label_wrappers import (  # noqa: E402
            _PadMissingTaskLabelsWrapper,
            _UnflattenedGenerateWrapper,
        )

        save_path = gan_save_path(
            self.get_storage_location(),
            self.gan_type,
            use_pca=bool(getattr(self, "use_pca_reduction", False)),
            post_gan_scaling=bool(getattr(self, "use_post_gan_scaling", False)),
        )
        interface = GANInterface(self.gan_type, save_path=save_path)
        expected = self._gan_expected_metadata(dataframe)
        try:
            interface.load(expected=expected)
        except GANMetadataMismatchError:
            raise
        except FileNotFoundError as load_err:
            raise RuntimeError(
                f"GAN model not found at {save_path}. Train it first via the "
                f"corresponding Create* strategy (gan_type={self.gan_type.name}). "
                f"Underlying error: {load_err}"
            ) from load_err
        except Exception as load_err:
            raise RuntimeError(
                f"Failed to load GAN model at {save_path}: {load_err}"
            ) from load_err
        self._apply_gan_inference_overrides(interface)

        T, F = int(train_data.shape[1]), int(train_data.shape[2])
        wrapped_interface = _UnflattenedGenerateWrapper(interface, T=T, F=F)

        # Single-task strategy + multi-task GAN: pad missing task labels with
        # uniform-random one-hots so the model sees its training conditioning
        # regime. Without this, lag-1 autocorrelation collapses to negative
        # values for high-AR features (the GAN was trained with N tasks
        # summed in the label embedding; passing only "trading" is OOD).
        gan_model = getattr(interface, "_model", None)
        gan_task_dims = getattr(gan_model, "task_label_dims", None)
        if gan_task_dims:
            wrapped_interface = _PadMissingTaskLabelsWrapper(
                wrapped_interface, expected_task_label_dims=gan_task_dims
            )

        aug_train_data, aug_labels_dict = self._invoke_balance_multi_task(
            wrapped_interface,
            train_data,
            wrapped_labels,
            dataframe=dataframe,
        )

        # Unwrap labels back to ndarray for the single-task classifier.
        aug_train_labels = aug_labels_dict["trading"]

        # Post-GAN scaling path: apply polymorphic tensor scaler to the
        # combined real+synth tensor so the classifier sees scaled data.
        if getattr(self, "use_post_gan_scaling", False):
            from utils.Scalers import load_scaler  # noqa: E402
            from Framework.FeatureScaler import FeatureScaler  # noqa: E402
            tensor_scaler = load_scaler(self.get_storage_location(), "main_tensor_scaler")
            aug_train_data = tensor_scaler.transform(aug_train_data)
            if test_data is not None and test_data.size > 0:
                test_data = tensor_scaler.transform(test_data)

        return aug_train_data, test_data, aug_train_labels, test_labels
