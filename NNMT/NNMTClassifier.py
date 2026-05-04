# Multi-task Neural Network Classifier Factory
# Base class for multi-task learning with different architectural variants

import tensorflow as tf
from enum import Enum
from Predictors.KerasClassifierMultiTask import KerasClassifierMultiTask
from utils.PositionalEncoding import PositionalEncoding
from utils.PCALayer import TrainablePCALayer
from sklearn.decomposition import PCA


class NNMTClassifier_Base(KerasClassifierMultiTask):
    """
    Base class for Multi-Task Neural Network classifiers.
    Handles the common parts: regime, volatility, and risk prediction.
    Subclasses implement the specific architectural approach (LSTM, Transformer, etc.)
    """

    MIN_FILTER_SIZE = 32
    # MIN_FILTER_SIZE = 64
    DROPOUT_RATE = 0.2
    L2_REGULARIZATION = 1e-3

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag=tag)
        # Override the name set by the parent class
        self.name = self.__class__.__name__

    def create_model(self, seq_len, num_features, pca_W=None, pca_M=None):
        """
        Override the base create_model to implement the multi-task architecture.
        This method creates the input/output layers and calls the subclass-specific architecture.
        """

        self.seq_len = seq_len

        # Input layer
        inputs = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="input_layer"
        )

        # num_filters = 64
        num_filters = max(
            self.MIN_FILTER_SIZE, self.nearest_power_of_2(num_features - 1)
        )
        num_components = num_filters

        # 1. Expand Feature Space (Learned Initial Mapping)
        # resized_input = tf.keras.layers.Dense(
        #     num_filters, activation="linear", name="resized_input"
        # )(inputs)
        resized_input = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=2,  # Capture patterns across 2 consecutive time steps (~10 min window)
                    activation="relu",
                    padding="same",  # Preserve sequence length
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name="resized_input",
        )(inputs)

        # 3. Create the model-specific architecture (implemented by subclasses)
        # This should return tensor of shape (batch_size, features) - no sequence dimension
        # common_layer = self.create_common_layer(resized_input, seq_len, num_filters)
        common_layer = self.create_common_layer(resized_input, seq_len, num_components)

        # skip_connection = tf.keras.layers.GlobalAveragePooling1D(name="skip_connection")(pca_output)
        # common_last_step = tf.keras.layers.GlobalAveragePooling1D(name="common_last_step")(common_layer)
        # common_layer = tf.keras.layers.Concatenate(name="common_layer")(
        #     [skip_connection, common_last_step]
        # )

        # make sure the common_layer and resized_inputs match shapes
        if (len(common_layer.shape) == 2) and (len(resized_input.shape) == 3):
            resized_input = tf.keras.layers.GlobalAveragePooling1D(name="reshape")(
                resized_input
            )
        elif (len(common_layer.shape) == 3) and (len(resized_input.shape) == 2):
            common_layer = tf.keras.layers.GlobalAveragePooling1D(name="reshape")(
                common_layer
            )

        # common_layer = tf.keras.layers.Concatenate(name="common_layer_concat")(
        #     [resized_input, common_layer]
        # )
        common_layer = tf.keras.layers.Add(name="residual")(
            [resized_input, common_layer]
        )

        # update size in case it changed
        last_dim = common_layer.shape[-1]  # TensorShape or None
        if last_dim is not None:
            num_components = max(self.MIN_FILTER_SIZE, int(last_dim))

        # Multi-task output heads (x is already flattened by subclasses)
        # 1. Trading actions classification (3 classes: Sell/Hold/Buy) - PRIMARY OUTPUT
        trading_head = self.create_task_head(
            common_layer, num_components, "trading_head"
        )
        trading_output = tf.keras.layers.Dense(3, activation="softmax", name="trading")(
            trading_head
        )

        # 2. Market regime classification (3 classes: Bull/Bear/Sideways) - AUXILIARY
        regime_head = self.create_task_head(common_layer, num_components, "regime_head")
        regime_output = tf.keras.layers.Dense(3, activation="softmax", name="regime")(
            regime_head
        )

        # 3. Risk classification (tri-state: LOW=0, MEDIUM=1, HIGH=2) - AUXILIARY
        risk_head = self.create_task_head(common_layer, num_components, "risk_head")
        risk_output = tf.keras.layers.Dense(3, activation="softmax", name="risk")(
            risk_head
        )

        # 4. Momentum prediction (continuous) - AUXILIARY
        momentum_head = self.create_task_head(
            common_layer, num_components, "momentum_head"
        )
        momentum_output = tf.keras.layers.Dense(
            3, activation="softmax", bias_initializer="zeros", name="momentum"
        )(momentum_head)

        # 5. Flow regression (VFI indicator) - AUXILIARY
        flow_head = self.create_task_head(common_layer, num_components, "flow_head")
        flow_output = tf.keras.layers.Dense(
            3, activation="softmax", bias_initializer="zeros", name="flow"
        )(flow_head)

        # 6. Profit regression (continuous) - AUXILIARY
        profit_head = self.create_task_head(common_layer, num_components, "profit_head")
        profit_output = tf.keras.layers.Dense(
            3, activation="softmax", bias_initializer="zeros", name="profit"
        )(profit_head)

        # Create the multi-output model
        model = tf.keras.Model(
            inputs=inputs,
            outputs={
                "trading": trading_output,
                "regime": regime_output,
                "momentum": momentum_output,
                "risk": risk_output,
                "flow": flow_output,
                "profit": profit_output,
            },
            name=self.name,
        )

        return model

    def create_custom_common_layer(self, inputs, seq_len, num_features):

        # default is GRU for temporal patterns
        x = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    num_features,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(self.L2_REGULARIZATION),
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name="custom_common_layer",
        )(inputs)

        return x

    def create_common_layer(self, inputs, seq_len, num_features):
        "Create a common layer with multiple CNNs and a cutom layer"

        # num_filters = num_features // 2
        num_filters = max(
            self.MIN_FILTER_SIZE, self.nearest_power_of_2(num_features // 2 + 1)
        )

        common_layer = tf.keras.Sequential(
            [
                # initial convolutional layer
                tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=5,
                    activation="elu",
                    padding="same",
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                # second convolutional layer
                tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=5,
                    activation="elu",
                    padding="same",
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                # insert the class-specific coomon layer
            ],
            name="common_layer",
        )(inputs)

        # insert the class-specific coomon layer
        common_layer = self.create_custom_common_layer(
            common_layer, seq_len, num_filters
        )

        return common_layer

    def create_task_head(self, inputs, num_filters, name):

        l2_strength = 1e-2
        nf = max(self.MIN_FILTER_SIZE, self.nearest_power_of_2((num_filters // 2) + 1))

        task_head = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(
                    nf,
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                ),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                tf.keras.layers.Dense(
                    16,
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                ),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name=name,
        )(inputs)

        return task_head

    # utility to define wavenet block (used in several places)
    def wavenetBlock(self, n_filters, filter_size, rate):
        def f(input_):
            # Gated activation (tanh and sigmoid gates)
            tanh_out = tf.keras.layers.Convolution1D(
                n_filters,
                filter_size,
                padding="causal",
                dilation_rate=rate,
                activation="tanh",
            )(input_)
            sigmoid_out = tf.keras.layers.Convolution1D(
                n_filters,
                filter_size,
                padding="causal",
                dilation_rate=rate,
                activation="sigmoid",
            )(input_)

            # Gating mechanism
            x = tf.keras.layers.Multiply()([tanh_out, sigmoid_out])
            x = tf.keras.layers.LayerNormalization()(x)

            # Skip connection (1x1 conv on gated output)
            skip_x = tf.keras.layers.Convolution1D(
                n_filters,
                1,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            )(x)

            # Residual connection (1x1 conv on input, then add to gated output)
            res_x = tf.keras.layers.Convolution1D(
                n_filters,
                1,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            )(input_)
            res_x = tf.keras.layers.Add()([x, res_x])

            return res_x, skip_x

        return f


class NNMTClassifier_LSTM(NNMTClassifier_Base):
    """Multi-task classifier using LSTM architecture"""

    def create_custom_common_layer(self, inputs, seq_len, num_features):
        # LSTM architecture
        x = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    num_features,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(self.L2_REGULARIZATION),
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name="custom_common_layer",
        )(inputs)

        return x


class NNMTClassifier_Transformer(NNMTClassifier_Base):
    """Multi-task classifier using Transformer architecture"""

    def create_custom_common_layer(self, inputs, seq_len, num_features):
        # Transformer architecture with proper normalization
        x = tf.keras.layers.Dense(128)(inputs)
        x = tf.keras.layers.LayerNormalization()(x)

        # Multiple transformer layers for better representation learning
        for i in range(3):  # 3 transformer layers
            # Multi-head attention with residual connection and layer norm
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=16
            )(x, x)
            x = tf.keras.layers.Add()([x, attention_output])
            x = tf.keras.layers.LayerNormalization()(x)

            # Feed-forward network with residual connection and layer norm
            ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(256, activation="relu"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(128),
                ]
            )
            x = tf.keras.layers.Add()([x, ffn(x)])
            x = tf.keras.layers.LayerNormalization()(x)

        return x


class NNMTClassifier_CNN(NNMTClassifier_Base):
    """Multi-task classifier using CNN architecture"""

    def create_custom_common_layer(self, inputs, seq_len, num_features):
        # CNN architecture
        x = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    num_features, 3, activation="elu", padding="same"
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name="custom_common_layer",
        )(inputs)

        return x


class NNMTClassifier_GRU(NNMTClassifier_Base):
    """Multi-task classifier using GRU architecture"""

    def create_custom_common_layer(self, inputs, seq_len, num_features):
        # GRU for temporal patterns
        x = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    num_features,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(self.L2_REGULARIZATION),
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name="gru_common_layer",
        )(inputs)

        return x


class NNMTClassifier_Attention(NNMTClassifier_Base):
    """Multi-task classifier using Self-Attention architecture"""

    def create_custom_common_layer(self, inputs, seq_len, num_features):
        # Self-Attention architecture with proper normalization
        x = inputs

        # Self-attention mechanism with residual connection and layer norm
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(
            x, x
        )
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization()(x)

        return x


class NNMTClassifier_Wavenet(NNMTClassifier_Base):
    """Multi-task classifier using (full) Wavenet architecture"""

    # Note that this is notoriously slow to train!

    def create_custom_common_layer(self, inputs, seq_len, num_features):

        x = tf.keras.layers.Convolution1D(
            num_features, 2, padding="causal", dilation_rate=1
        )(inputs)

        # A, B = self.wavenetBlock(64, 2, 1)(inputs)

        skip_connections = []
        for i in range(1, 3):
            rate = 1
            for j in range(1, 10):
                x, skip = self.wavenetBlock(num_features, 2, rate)(x)
                skip_connections.append(skip)
                rate = 2 * rate

            x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.Add()(skip_connections)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Convolution1D(
            num_features,
            1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(self.L2_REGULARIZATION),
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Convolution1D(
            num_features,
            1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(self.L2_REGULARIZATION),
        )(x)

        x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)

        return x


class NNMTClassifier_Wavenet_Fast(NNMTClassifier_Base):
    """Multi-task classifier using optimized Wavenet architecture (faster training)"""

    def create_custom_common_layer(self, inputs, seq_len, num_features):
        # Use fewer filters for speed (cap at 64)
        n_filters = min(num_features, 64)

        x = tf.keras.layers.Convolution1D(
            n_filters, 2, padding="causal", dilation_rate=1
        )(inputs)

        skip_connections = []
        # Optimized: 1 block instead of 2, 6 rates instead of 9
        for i in range(1, 2):  # Only 1 block
            rate = 1
            for j in range(1, 7):  # Only 6 rates (1,2,4,8,16,32)
                x, skip = self.wavenetBlock(n_filters, 2, rate)(x)
                skip_connections.append(skip)
                rate = min(2 * rate, 32)  # Cap at 32 instead of 256

            x = tf.keras.layers.LayerNormalization()(x)

        # Use only last 3 skip connections for speed
        skip_connections = skip_connections[-3:]
        x = tf.keras.layers.Add()(skip_connections)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Convolution1D(
            n_filters,
            1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(self.L2_REGULARIZATION),
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Project back to num_features to match task head expectations
        x = tf.keras.layers.Convolution1D(
            num_features,
            1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(self.L2_REGULARIZATION),
        )(x)

        x = tf.keras.layers.Dropout(self.DROPOUT_RATE)(x)

        return x


# --------------------------------------------------------------
# Multi-* architecture

# these use more advanced types of layers on each head.
# They override the create_model method (not create_custom_common_layer)

# --------------------------------------------------------------


class NNMTClassifier_Multi_LSTM(NNMTClassifier_Base):
    """Multi-task classifier using Multi-LSTM architecture"""

    def create_task_head(self, inputs, num_filters, name):

        # Set the L2 strength (this is a hyperparameter to tune)
        l2_strength = 1e-3
        # nf = num_filters // 2
        nf = max(self.MIN_FILTER_SIZE, self.nearest_power_of_2((num_filters // 2) + 1))

        # add a CNN layer to extract features
        head_cnn = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=nf,
                    kernel_size=5,
                    activation="elu",
                    padding="same",
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name=f"{name}_cnn",
        )(inputs)
        # head_cnn = inputs

        # # concatenate the CNN layer with the inputs (feed forward)
        # head_cnn = tf.keras.layers.Concatenate(name=f"{name}_concat")(
        #     [head_cnn, inputs]
        # )

        # the rest of the task head
        task_head = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    nf,
                    activation="tanh",
                    return_sequences=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                # tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(
                    nf,
                    activation="elu",
                    # activation="tanh",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                ),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                # tf.keras.layers.Dense(16, activation=activation),
                # tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name=name,
        )(head_cnn)

        return task_head


class NNMTClassifier_Multi_GRU(NNMTClassifier_Base):
    """Multi-task classifier using Multi-GRU architecture"""

    def create_task_head(self, inputs, num_filters, name):

        task_head = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=2,
                    activation="tanh",
                    padding="causal",
                ),
                tf.keras.layers.GRU(
                    num_filters // 2, return_sequences=True, activation="tanh"
                ),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(num_filters // 2, activation="tanh"),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                tf.keras.layers.Dense(16, activation="tanh"),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name=name,
        )(inputs)

        return task_head


class NNMTClassifier_Multi_CNN(NNMTClassifier_Base):
    """Multi-task classifier using Multi-CNN architecture"""

    def create_task_head(self, inputs, num_filters, name):

        # only create wavenet for the continuous tasks (momentum, flow, profit)
        if any(task in name for task in ["momentum", "flow", "profit"]):
            # simplified Wavenet
            task_head = inputs
            for i, rate in enumerate((1, 2, 4, 8) * 2):  # type: ignore
                task_head = tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=2,
                    padding="causal",
                    activation="tanh",
                    dilation_rate=rate,
                    name=f"{name}_conv1d_{rate}_{i+1}",
                )(task_head)
                task_head = tf.keras.layers.Dropout(
                    0.2,
                    name=f"{name}_dropout_{rate}_{i+1}",
                )(task_head)

            task_head = tf.keras.Sequential(
                [
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(num_filters // 2, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(16, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                ],
                name=f"{name}_downsize",
            )(task_head)
        else:
            # just create an LSTM layer/head
            task_head = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(
                        num_filters, return_sequences=True, activation="tanh"
                    ),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(num_filters // 2, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(16, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                ],
                name=name,
            )(inputs)

        return task_head


class NNMTClassifier_Multi_Wavenet(NNMTClassifier_Base):
    """Multi-task classifier using Multi-CNN architecture"""

    # NOTE: currently not working well

    def simpleWavenetBlock(self, n_filters, filter_size, rate):
        def f(input_):
            # Simple Conv1D without gating
            x = tf.keras.layers.Convolution1D(
                n_filters,
                filter_size,
                padding="causal",
                dilation_rate=rate,
                activation="relu",  # Simple ReLU instead of gating
            )(input_)

            # Skip connection
            skip_x = tf.keras.layers.Convolution1D(n_filters, 1, padding="same")(x)

            # Residual connection
            res_x = tf.keras.layers.Convolution1D(n_filters, 1, padding="same")(input_)
            res_x = tf.keras.layers.Add()([x, res_x])

            return res_x, skip_x

        return f

    def create_task_head(self, inputs, num_filters, name):

        # only create wavenet for the continuous tasks (momentum, flow, profit)
        if any(task in name for task in ["momentum", "flow", "profit"]):

            num_layers = 8
            layer_norm_freq = 1

            wavenet = inputs

            # (greatly)simplified Wavenet with task-specific adjustments
            wavenet = inputs
            rates = (1, 2, 4, 8) * (num_layers // 4)  # Adjust rates based on num_layers

            for i, rate in enumerate(rates):
                residual = wavenet
                wavenet = tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=2,
                    padding="causal",
                    activation="relu",
                    dilation_rate=rate,
                )(wavenet)

                # Apply layer normalization less frequently for complex tasks
                if i % layer_norm_freq == layer_norm_freq - 1:
                    wavenet = tf.keras.layers.LayerNormalization()(wavenet)

                # Add residual connection every 2 layers
                if i % 2 == 1:
                    wavenet = tf.keras.layers.Add()([wavenet, residual])

            # Concatenate average and max pooling
            wavenet_pooled = tf.keras.layers.Concatenate(name=f"{name}_pooled")(
                [
                    tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_avg")(wavenet),
                    tf.keras.layers.GlobalMaxPooling1D(name=f"{name}_max")(wavenet),
                ]
            )

            # Flow might need different approach
            dense_layers = [
                tf.keras.layers.Dense(2 * num_filters, activation="tanh"),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                tf.keras.layers.Dense(16, activation="tanh"),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ]

            task_head = tf.keras.Sequential(dense_layers, name=name)(wavenet_pooled)

        else:
            # just create an LSTM layer/head
            task_head = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(
                        num_filters, return_sequences=True, activation="tanh"
                    ),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(num_filters, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(16, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                ],
                name=name,
            )(inputs)

        return task_head


class NNMTClassifier_Multi_Attention(NNMTClassifier_Base):
    """Multi-task classifier using Multi-Head Attention architecture"""

    def create_task_head(self, inputs, num_filters, name):

        if any(task in name for task in ["momentum", "flow", "profit"]):
            head_ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(64, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(128, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                ]
            )

            # task head
            task_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=16, name=f"{name}_attention"
            )(inputs, inputs)
            task_head = tf.keras.layers.Add()([task_attention, task_attention])
            task_head = tf.keras.layers.LayerNormalization()(task_head)
            task_head = tf.keras.layers.Add()([task_head, head_ffn(task_head)])
            task_head = tf.keras.layers.LayerNormalization()(task_head)

            task_head = tf.keras.Sequential(
                [
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(num_filters // 2, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(16, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                ],
                name=name,
            )(task_head)

        else:
            # just create an LSTM layer/head
            task_head = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(
                        num_filters, return_sequences=True, activation="tanh"
                    ),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(num_filters // 2, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(16, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                ],
                name=name,
            )(inputs)

        return task_head


class NNMTClassifier_Multi_MLP(NNMTClassifier_Base):
    """Multi-task classifier using MLP architecture - simple but effective for regression"""

    def create_task_head(self, inputs, num_filters, name):

        task_head = tf.keras.Sequential(
            [
                # # second convolutional layer
                # tf.keras.layers.Conv1D(
                #     filters=num_filters,
                #     kernel_size=5,
                #     activation="relu",
                #     padding="same",
                # ),
                # tf.keras.layers.LayerNormalization(),
                # tf.keras.layers.Dropout(self.DROPOUT_RATE),
                # # GRU for temporal patterns
                # tf.keras.layers.GRU(
                #     num_filters,
                #     return_sequences=True,
                #     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                # ),
                # tf.keras.layers.LayerNormalization(),
                # tf.keras.layers.Dropout(self.DROPOUT_RATE),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(num_filters, activation="tanh"),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
                tf.keras.layers.Dense(16, activation="tanh"),
                tf.keras.layers.Dropout(self.DROPOUT_RATE),
            ],
            name=name,
        )(inputs)

        return task_head


class NNMTClassifier_Multi_Transformer(NNMTClassifier_Base):
    """Multi-task classifier using full Transformer architecture"""

    def create_task_head(self, inputs, num_filters, name):

        # For regression tasks, use full transformer blocks
        if any(task in name for task in ["momentum", "flow", "profit"]):

            num_layers = 4
            num_heads = 8
            key_dim = 16
            ff_dim = 64

            x = PositionalEncoding(self.seq_len, num_filters)(inputs)

            # Multiple transformer blocks
            for i in range(num_layers):
                # Multi-head attention
                attention_output = tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=key_dim, name=f"{name}_attention_{i}"
                )(x, x)

                # Add & Norm
                x = tf.keras.layers.Add(name=f"{name}_add1_{i}")([x, attention_output])
                x = tf.keras.layers.LayerNormalization(name=f"{name}_norm1_{i}")(x)

                # Feed-forward network
                ffn = tf.keras.Sequential(
                    [
                        # tf.keras.layers.Dense(ff_dim, activation="relu"),
                        tf.keras.layers.Dense(ff_dim, activation="tanh"),
                        tf.keras.layers.Dropout(self.DROPOUT_RATE),
                        tf.keras.layers.Dense(num_filters),
                        tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    ],
                    name=f"{name}_ffn_{i}",
                )

                # Add & Norm
                ffn_output = ffn(x)
                x = tf.keras.layers.Add(name=f"{name}_add2_{i}")([x, ffn_output])
                x = tf.keras.layers.LayerNormalization(name=f"{name}_norm2_{i}")(x)

            # Global pooling
            x = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)

            # Final dense layers
            task_head = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(num_filters // 2, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    # tf.keras.layers.Dense(64, activation="tanh"),
                    # tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    # tf.keras.layers.Dense(32, activation="tanh"),
                    # tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(16, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                ],
                name=f"{name}_final",
            )(x)

        else:
            # Catorgical tasks, use LSTM
            task_head = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(
                        num_filters, return_sequences=True, activation="tanh"
                    ),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(num_filters // 2, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                    tf.keras.layers.Dense(16, activation="tanh"),
                    tf.keras.layers.Dropout(self.DROPOUT_RATE),
                ],
                name=name,
            )(inputs)

        return task_head


# --------------------------------------------------------------
# Classifier Type Enum
# --------------------------------------------------------------


class ClassifierType(Enum):
    LSTM = NNMTClassifier_LSTM
    Transformer = NNMTClassifier_Transformer
    CNN = NNMTClassifier_CNN
    GRU = NNMTClassifier_GRU
    Wavenet = NNMTClassifier_Wavenet
    Wavenet_Fast = NNMTClassifier_Wavenet_Fast
    Attention = NNMTClassifier_Attention
    Multi_LSTM = NNMTClassifier_Multi_LSTM
    Multi_GRU = NNMTClassifier_Multi_GRU
    Multi_Attention = NNMTClassifier_Multi_Attention
    Multi_CNN = NNMTClassifier_Multi_CNN
    Multi_Wavenet = NNMTClassifier_Multi_Wavenet
    Multi_MLP = NNMTClassifier_Multi_MLP
    Multi_Transformer = NNMTClassifier_Multi_Transformer


# --------------------------------------------------------------
# Factory function
# --------------------------------------------------------------


def create_classifier(clf_type: ClassifierType, pair, nfeatures, seq_len, tag=""):
    """
    Factory function to create multi-task classifier based on type.

    Args:
        clf_type: Type of classifier (LSTM, Transformer, etc.)
        pair: Trading pair
        nfeatures: Number of input features
        seq_len: Sequence length
        tag: Optional tag for the model

    Returns:
        tuple: (classifier_instance, classifier_name)
    """
    clf_name = str(clf_type).split(".")[-1]
    clf = clf_type.value(pair, seq_len, nfeatures, tag=tag)
    # clf_name = clf.__class__.__name__

    return clf, clf_name
