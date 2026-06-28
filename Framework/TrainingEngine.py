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

This is a mixin designed to be composed into an NN strategy (listed first in
BaseNNStrategy's bases). It calls collaborators via self — scale_dataframe /
get_normalized_size (FeatureNormalizer), enhance_training_data /
preprocess_training_data and the markov + path helpers (still on BaseNNStrategy,
pending the later GAN-augmentation increment), debug_print (BaseStrategy),
dataframeUtils. It introduces no literal references to its host class.

The deterministic boundary this produces is pinned by
Framework/tests/test_training_pipeline_characterization.py so a continued
extraction can be validated in seconds.
"""

from typing import Optional, List, Any

import os
import numpy as np
from pandas import DataFrame
from sklearn.utils import shuffle

from Predictors.KerasBasePredictor import KerasBasePredictor


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

        for i in range(num_pairs):

            pair_labels = np.asarray(labels[i])
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

        return aggr_tsr_train, aggr_tsr_test, aggr_train_labels, aggr_test_labels
    def train_model(
        self,
        dataframes: [DataFrame],
        labels: [Any],
        classifier: KerasBasePredictor,
        pair_names: Optional[List[str]] = None,
    ):
        """Train the model - default implementation"""

        tsr_train, tsr_test, train_labels, test_labels = self.prepare_training_data(
            dataframes, labels, pair_names=pair_names,
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
                tsr_train, train_labels = shuffle(
                    tsr_train, train_labels, random_state=42
                )

        classifier.train(
            tsr_train, tsr_test, train_labels, test_labels, class_weights=class_weights
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
