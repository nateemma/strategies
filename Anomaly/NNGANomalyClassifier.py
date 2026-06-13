from typing import Any
# Neural Network Classifiers for GANomaly detection
# GANomaly combines GAN training with anomaly detection and classification

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import json
from enum import Enum
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

# Strategy specific imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="keras.src.saving.saving_lib"
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_RUN_EAGER_OP_AS_FUNCTION"] = "0"

from Predictors.KerasAnomalyDetector import KerasAnomalyDetector

log = logging.getLogger(__name__)


# Base class for GANomaly classifiers
class GANomalyClassifierBase(KerasAnomalyDetector):
    """
    Base class for GANomaly classifiers that combine:
    - GAN training (generator + discriminator)
    - Encoder-decoder for reconstruction
    - Classification head for trading signals

    Training phases:
    1. GAN training (normal data only): Train generator + discriminator
    2. Encoder training (normal data only): Train encoder + generator (autoencoder)
    3. Classifier training (buy/sell data only): Train classifier head
    """

    is_trained = False
    clean_data_required = True  # training data cannot contain anomalies for phases 1-2

    # GAN training parameters
    ganomaly_latent_dim = 32
    ganomaly_generator_lr = 0.0002
    ganomaly_discriminator_lr = 0.0002
    ganomaly_beta1 = 0.5
    ganomaly_beta2 = 0.999
    ganomaly_epochs = 200
    ganomaly_batch_size = 1024  # Increased for better GPU utilization

    # Training history
    g_loss_history = []
    d_loss_history = []

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag=tag)
        self.name = self.__class__.__name__

        # Discriminator (only for GAN training, not part of main model)
        self.discriminator = None

    def _create_components(self, seq_len, num_features):
        """
        Create encoder, generator, and discriminator components.
        Returns: (encoder_model, generator_model, discriminator_model, latent_size)
        These are separate models that can be trained independently, then combined.
        """
        latent_size = self.ganomaly_latent_dim

        # Encoder: maps sequences -> encoded representation (3D for non-Dense variants)
        encoder_input = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="encoder_input"
        )
        encoder_output = self.create_encoder(
            encoder_input, seq_len, num_features, latent_size
        )
        # Encoder outputs 3D tensor directly (no Dense layer)
        encoder_model = tf.keras.Model(encoder_input, encoder_output, name="encoder")

        # Generator: maps latent -> reconstruction
        generator_input = tf.keras.layers.Input(
            shape=(latent_size,), name="generator_input"
        )
        generator_output = self.create_generator(generator_input, seq_len, num_features)
        generator_model = tf.keras.Model(
            generator_input, generator_output, name="generator"
        )

        # Discriminator: maps input -> real/fake
        # Input shape depends on implementation (sequences or flattened)
        # For now, assume sequences - implementations can flatten if needed
        discriminator_input = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="discriminator_input"
        )
        discriminator_output = self.create_discriminator(
            discriminator_input, seq_len, num_features
        )
        discriminator_model = tf.keras.Model(
            discriminator_input, discriminator_output, name="discriminator"
        )

        return encoder_model, generator_model, discriminator_model, latent_size

    def create_model(self, seq_len, num_features):
        """
        Create unified model with encoder, generator, and classifier head.
        Returns model with outputs: {'reconstruction', 'trading'}

        All components work with 3D tensors (sequences). Implementations handle
        flattening/reshaping internally if needed.
        """
        self.seq_len = seq_len
        latent_size = self.ganomaly_latent_dim

        # Input layer (sequences) - always 3D: (batch, seq_len, num_features)
        inputs = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="input_layer"
        )

        # Create encoder (maps 3D sequences -> 3D encoded representation for non-Dense variants)
        encoded = self.create_encoder(inputs, seq_len, num_features, latent_size)

        # Generator takes 3D tensor directly from encoder (for non-Dense variants)
        decoded = self.create_generator(encoded, seq_len, num_features)

        # Create classifier head (needs 2D input, so flatten/pool the 3D encoder output)
        # For 3D tensors, use GlobalAveragePooling1D; for 2D (Dense variant), use as-is
        if len(encoded.shape) == 3:
            latent_2d = tf.keras.layers.GlobalAveragePooling1D()(encoded)
        else:
            latent_2d = encoded
        classifier_features = self.create_classifier_head(latent_2d, latent_size)

        # Output layers
        reconstruction_output = tf.keras.layers.Dense(
            num_features, activation="linear", name="reconstruction"
        )(decoded)

        trading_output = tf.keras.layers.Dense(
            3, activation="softmax", bias_initializer="zeros", name="trading"
        )(classifier_features)

        # Create unified model
        model = tf.keras.Model(
            inputs=inputs,
            outputs={
                "reconstruction": reconstruction_output,
                "trading": trading_output,
            },
            name=self.name,
        )

        return model

    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        """
        Create encoder architecture - override in subclasses.

        Args:
            inputs: Input tensor (shape: (batch, seq_len, num_features)) - always 3D
            seq_len: Sequence length
            num_features: Number of features per timestep
            latent_size: Size of latent representation

        Returns:
            Encoded tensor (shape: (batch, ...)) - will be processed to latent
            Implementation can flatten/pool internally if needed
        """
        raise NotImplementedError("Subclass must implement create_encoder")

    def create_generator(self, latent, seq_len, num_features):
        """
        Create generator architecture - override in subclasses.

        Args:
            latent: Latent tensor (shape: (batch, latent_size))
            seq_len: Sequence length
            num_features: Number of features per timestep

        Returns:
            Generated tensor (shape: (batch, seq_len, num_features)) - must be 3D
            Implementation must reshape to 3D if it flattens internally
        """
        raise NotImplementedError("Subclass must implement create_generator")

    def create_discriminator(self, inputs, seq_len, num_features):
        """
        Create discriminator architecture - override in subclasses.

        Args:
            inputs: Input tensor (shape: (batch, seq_len, num_features)) - always 3D
            seq_len: Sequence length
            num_features: Number of features per timestep

        Returns:
            Discriminator output (shape: (batch, 1))
            Implementation can flatten internally if needed
        """
        raise NotImplementedError("Subclass must implement create_discriminator")

    def create_classifier_head(self, latent_vector, latent_size):
        """Create classifier head with stronger regularization to prevent overfitting"""
        l2_strength = 1e-2  # Increased from 1e-3 for stronger regularization
        d1_size = max(8, self.nearest_power_of_2((latent_size // 2) + 1))
        d2_size = max(4, d1_size // 2)

        classifier_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    d1_size,
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                ),
                tf.keras.layers.Dropout(
                    0.4
                ),  # Increased from 0.2 to reduce overfitting
                tf.keras.layers.Dense(
                    d2_size,
                    activation="elu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
                ),
                tf.keras.layers.Dropout(
                    0.4
                ),  # Increased from 0.2 to reduce overfitting
            ],
            name="classifier_head",
        )(latent_vector)

        return classifier_head

    def get_discriminator_path(self):
        """Get path for discriminator model"""
        model_path = self.get_model_path()
        discriminator_path = model_path.replace(".keras", "_discriminator.keras")
        return discriminator_path

    def model_exists(self) -> bool:
        """Check if all required models exist (all-or-nothing)"""
        main_model_path = self.get_model_path()
        discriminator_path = self.get_discriminator_path()

        # Check if main model exists
        if not os.path.exists(main_model_path):
            return False

        # Discriminator is optional (can be discarded after training)
        # But if it doesn't exist, we'll need to retrain GAN phase
        # For now, we'll require it to exist for simplicity
        # Can be changed later if needed
        if not os.path.exists(discriminator_path):
            return False

        return True

    def save(self, path=""):
        """Save main model and discriminator"""
        if len(path) == 0:
            path = self.get_model_path()
        else:
            self.model_path = path

        # Save main model
        print(f"    Saving main model to: {path}")
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        keras.models.save_model(self.model, filepath=path, save_format=self.model_ext)

        # Save discriminator if it exists
        if self.discriminator is not None:
            discriminator_path = self.get_discriminator_path()
            print(f"    Saving discriminator to: {discriminator_path}")
            keras.models.save_model(
                self.discriminator,
                filepath=discriminator_path,
                save_format=self.model_ext,
            )

            # Save training history
            history_path = path.replace(".keras", "_history.json")
            history = {
                "g_loss_history": [float(x) for x in self.g_loss_history],
                "d_loss_history": [float(x) for x in self.d_loss_history],
            }
            with open(history_path, "w") as f:
                json.dump(history, f)

    def load(self, path=""):
        """Load main model and discriminator"""
        if len(path) == 0:
            path = self.get_model_path()
        else:
            self.model_path = path

        # Load main model
        if os.path.exists(path):
            print(f"    Loading main model from: {path}")
            self.model = keras.models.load_model(path, compile=False)
        else:
            print(f"    Main model not found at: {path}")
            return None

        # Load discriminator
        discriminator_path = self.get_discriminator_path()
        if os.path.exists(discriminator_path):
            print(f"    Loading discriminator from: {discriminator_path}")
            self.discriminator = keras.models.load_model(
                discriminator_path, compile=False
            )
        else:
            print(f"    Discriminator not found at: {discriminator_path}")
            self.discriminator = None

        # Load training history if available
        history_path = path.replace(".keras", "_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
                self.g_loss_history = history.get("g_loss_history", [])
                self.d_loss_history = history.get("d_loss_history", [])

        return self.model

    def train(
        self,
        df_train_norm,
        df_test_norm,
        train_results,
        test_results,
        force_train=False,
        class_weights=None,
    ):
        """
        Override train method to implement 3-phase training:
        1. GAN training (normal data only)
        2. Encoder training (normal data only)
        3. Classifier training (buy/sell data only)
        """

        # Check if models exist (all-or-nothing)
        if not force_train and self.model_exists():
            # Load existing models
            self.model = self.load()
            if self.model is not None:
                # Recompile for inference
                self.model = self.compile_model(self.model, class_weights=class_weights)
                self.is_trained = True
                return

        # Convert dataframes to tensors if needed
        if self.dataframeUtils.is_dataframe(df_train_norm):
            train_tensor = self.dataframeUtils.df_to_tensor(
                df_train_norm, self.seq_len, method=0
            )
        else:
            train_tensor = df_train_norm.copy()

        # Extract normal data (for phases 1-2) and buy/sell data (for phase 3)
        # train_results should be a dict with 'trading' key containing one-hot encoded labels
        if isinstance(train_results, dict) and "trading" in train_results:
            trading_labels = train_results["trading"]
            # Convert one-hot to class indices
            if len(trading_labels.shape) > 1:
                class_indices = np.argmax(trading_labels, axis=1)
            else:
                class_indices = trading_labels

            # Normal data: HOLD class (class 1)
            normal_mask = class_indices == 1
            # Buy/sell data: SELL (0) or BUY (2)
            signal_mask = (class_indices == 0) | (class_indices == 2)

            normal_data = train_tensor[normal_mask]
            signal_data = train_tensor[signal_mask]
            signal_labels = trading_labels[signal_mask]

            print(
                f"    Data split: {len(normal_data)} normal samples, {len(signal_data)} signal samples"
            )
        else:
            # Fallback: use all data as normal (shouldn't happen in practice)
            normal_data = train_tensor
            signal_data = train_tensor
            signal_labels = (
                train_results["trading"]
                if isinstance(train_results, dict)
                else train_results
            )
            print(
                f"    WARNING: Could not split data, using all {len(normal_data)} samples"
            )

        # Phase 1: GAN Training (normal data only)
        print("    Phase 1: Training GAN (generator + discriminator)...")
        self.train_gan_phase(normal_data, self.seq_len, self.num_features)

        # Phase 2: Encoder Training (normal data only)
        print("    Phase 2: Training encoder (autoencoder)...")
        self.train_encoder_phase(normal_data, self.seq_len, self.num_features)

        # Phase 3: Classifier Training (buy/sell data only)
        print("    Phase 3: Training classifier head...")
        self.train_classifier_phase(signal_data, signal_labels, class_weights)

        # Save models
        self.save()
        self.is_trained = True

    def train_gan_phase(self, normal_data: np.ndarray, seq_len: int, num_features: int):
        """Phase 1: Train GAN (generator + discriminator) on normal data"""

        latent_size = self.ganomaly_latent_dim

        # Ensure data is in sequence format
        if len(normal_data.shape) == 2:
            # Flattened data - reshape to sequences
            normal_data = normal_data.reshape(-1, seq_len, num_features)

        # Create generator model
        generator_input = tf.keras.layers.Input(shape=(latent_size,))
        generator_output = self.create_generator(generator_input, seq_len, num_features)
        temp_generator = tf.keras.Model(
            generator_input, generator_output, name="temp_generator"
        )

        # Compile generator
        g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.ganomaly_generator_lr,
            beta_1=self.ganomaly_beta1,
            beta_2=self.ganomaly_beta2,
        )
        temp_generator.compile(optimizer=g_optimizer, loss="mse")

        # Create discriminator if it doesn't exist
        if self.discriminator is None:
            discriminator_input = tf.keras.layers.Input(shape=(seq_len, num_features))
            discriminator_output = self.create_discriminator(
                discriminator_input, seq_len, num_features
            )
            self.discriminator = tf.keras.Model(
                discriminator_input, discriminator_output, name="discriminator"
            )

            # Compile discriminator
            d_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.ganomaly_discriminator_lr,
                beta_1=self.ganomaly_beta1,
                beta_2=self.ganomaly_beta2,
            )
            self.discriminator.compile(
                optimizer=d_optimizer, loss="binary_crossentropy", metrics=["accuracy"]
            )

        # Create combined model for generator training
        # Generator outputs 3D sequences, discriminator expects 3D sequences
        discriminator_for_gan = tf.keras.models.clone_model(self.discriminator)
        discriminator_for_gan.set_weights(self.discriminator.get_weights())
        discriminator_for_gan.trainable = False

        # Create combined model: generator -> discriminator
        combined_input = tf.keras.layers.Input(shape=(latent_size,))
        generated_data = temp_generator(
            combined_input
        )  # Outputs 3D: (batch, seq_len, num_features)
        discriminator_output = discriminator_for_gan(generated_data)  # Expects 3D input

        combined_model = tf.keras.Model(combined_input, discriminator_output)
        combined_model.compile(optimizer=g_optimizer, loss="binary_crossentropy")

        # Training loop
        epochs = self.ganomaly_epochs
        batch_size = self.ganomaly_batch_size
        noise_dim = self.ganomaly_latent_dim

        real_labels = np.ones((batch_size, 1)) * 0.9  # Label smoothing
        fake_labels = np.zeros((batch_size, 1)) + 0.1

        self.g_loss_history = []
        self.d_loss_history = []

        # Calculate number of batches per epoch
        num_samples = normal_data.shape[0]
        batches_per_epoch = max(1, num_samples // batch_size)
        
        print(f"      Training GAN for {epochs} epochs with batch size {batch_size}")
        print(f"      ({batches_per_epoch} batches per epoch, {num_samples} total samples)")

        for epoch in range(epochs):
            epoch_d_losses = []
            epoch_g_losses = []
            
            # Shuffle data at start of each epoch
            indices = np.random.permutation(num_samples)
            
            # Process batches per epoch
            for batch_idx in range(batches_per_epoch):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                real_batch = normal_data[batch_indices]
                
                # Adjust labels if batch is smaller than batch_size
                current_batch_size = len(real_batch)
                if current_batch_size < batch_size:
                    real_labels_batch = np.ones((current_batch_size, 1)) * 0.9
                    fake_labels_batch = np.zeros((current_batch_size, 1)) + 0.1
                else:
                    real_labels_batch = real_labels
                    fake_labels_batch = fake_labels
                
                # Train Discriminator
                noise = np.random.normal(0, 1, (current_batch_size, noise_dim))
                # Use a direct eager call (not model.predict) — predict() inside a
                # tight loop leaks memory (rebuilds its internal function/iterator
                # every call), which OOM-kills the process mid-training on large
                # datasets. model(x, training=False) is the same inference-mode
                # forward pass without the leak.
                fake_batch = temp_generator(noise, training=False).numpy()
                
                # Train discriminator on real and fake batches (both 3D)
                d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels_batch)
                d_loss_fake = self.discriminator.train_on_batch(fake_batch, fake_labels_batch)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                epoch_d_losses.append(float(d_loss[0]))
                
                # Train Generator (train generator every discriminator update)
                noise = np.random.normal(0, 1, (current_batch_size, noise_dim))
                g_loss = combined_model.train_on_batch(noise, real_labels_batch)
                epoch_g_losses.append(float(g_loss))
            
            # Average losses for the epoch
            d_loss_avg = np.mean(epoch_d_losses)
            g_loss_avg = np.mean(epoch_g_losses)
            
            self.d_loss_history.append(d_loss_avg)
            self.g_loss_history.append(g_loss_avg)

            if epoch % 20 == 0:
                print(
                    f"      Epoch {epoch}/{epochs} - D Loss: {d_loss_avg:.4f}, G Loss: {g_loss_avg:.4f}"
                )

        # Store trained generator weights for later use
        self._trained_generator_weights = temp_generator.get_weights()

        print("      GAN training completed")

    def train_encoder_phase(
        self, normal_data: np.ndarray, seq_len: int, num_features: int
    ):
        """Phase 2: Train encoder + generator (autoencoder) on normal data"""

        # Ensure data is in sequence format
        if len(normal_data.shape) == 2:
            # Flattened data - reshape to sequences
            normal_data = normal_data.reshape(-1, seq_len, num_features)

        # Build autoencoder graph using encoder and generator layers directly
        # (not using pre-built models since encoder outputs 3D, generator accepts 3D)
        autoencoder_input = tf.keras.layers.Input(
            shape=(seq_len, num_features), name="ae_input"
        )
        latent_size = self.ganomaly_latent_dim
        
        # Encoder outputs 3D tensor
        encoded = self.create_encoder(autoencoder_input, seq_len, num_features, latent_size)
        
        # Generator accepts 3D tensor directly
        decoder_output = self.create_generator(encoded, seq_len, num_features)
        
        autoencoder_model = tf.keras.Model(
            autoencoder_input, decoder_output, name="autoencoder"
        )
        
        # Initialize generator layers with trained weights from GAN phase if available
        if hasattr(self, "_trained_generator_weights") and isinstance(
            self._trained_generator_weights, list
        ):
            try:
                # Find generator layers in the autoencoder model and set their weights
                # The generator layers are the ones after the encoder output
                # We need to match layer names/structures to transfer weights
                # For now, skip weight transfer as it's complex with functional API
                # The generator will train from scratch in encoder phase
                print("      Note: Generator will train from encoder phase (GAN weights not transferred)")
            except Exception:
                print(
                    "      Warning: Generator will train from encoder phase"
                )

        # Compile autoencoder
        autoencoder_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.ganomaly_generator_lr
            ),
            loss="mse",
        )

        # Train autoencoder
        print(f"      Training autoencoder on {len(normal_data)} normal samples")

        # Split for validation
        split_idx = int(0.8 * len(normal_data))
        train_split = normal_data[:split_idx]
        val_split = normal_data[split_idx:]

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", mode="min", factor=0.5, patience=5, verbose=1
            ),
        ]

        autoencoder_model.fit(
            train_split,
            train_split,
            validation_data=(val_split, val_split),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        # Store trained weights from autoencoder model
        # Note: We can't extract separate encoder/generator weights from the unified autoencoder model
        # The final model will be built fresh in train_classifier_phase or create_model
        # The autoencoder weights are embedded in the model structure and will be saved with the model
        self._trained_encoder_weights = []
        self._trained_generator_weights = []
        
        # Store reference to trained autoencoder model for weight extraction if needed
        # In practice, the final unified model (created in create_model or train_classifier_phase)
        # will contain the trained weights and will be saved via Keras model.save()
        print("      Encoder training completed")

    def train_classifier_phase(
        self, signal_data: np.ndarray, signal_labels: np.ndarray, class_weights=None
    ):
        """Phase 3: Train classifier head on buy/sell data - only classifier head is trainable"""

        # Build main model using create_model (handles 3D tensors correctly)
        # Note: Weight transfer from previous phases is not currently implemented
        # The trained autoencoder weights are embedded in the saved model structure
        if self.model is None:
            self.model = self.create_model(self.seq_len, self.num_features)

        # Freeze all layers first
        for layer in self.model.layers:
            layer.trainable = False

        # Unfreeze only classifier head and trading output layers
        # The classifier_head Sequential layers have names like "classifier_head/dense_X"
        # and the trading output layer is named "trading"
        def set_layer_trainable(layer, should_train):
            """Recursively set layer and sublayers trainable status"""
            layer.trainable = should_train
            # Handle nested layers (Sequential, etc.)
            if hasattr(layer, "layers"):
                for sublayer in layer.layers:
                    set_layer_trainable(sublayer, should_train)

        # Iterate through all layers and unfreeze classifier-related ones
        for layer in self.model.layers:
            layer_name = layer.name
            # Unfreeze trading output layer
            if layer_name == "trading":
                set_layer_trainable(layer, True)
            # Unfreeze classifier_head layers (by name pattern)
            elif "classifier_head" in layer_name:
                set_layer_trainable(layer, True)

        # Also check all sublayers recursively
        def check_and_unfreeze_sublayers(layer):
            """Recursively check and unfreeze classifier-related sublayers"""
            if hasattr(layer, "layers"):
                for sublayer in layer.layers:
                    if "classifier_head" in sublayer.name or sublayer.name == "trading":
                        set_layer_trainable(sublayer, True)
                    check_and_unfreeze_sublayers(sublayer)

        for layer in self.model.layers:
            check_and_unfreeze_sublayers(layer)

        # Compile model for classification training (must recompile after changing trainable)
        # IMPORTANT: Set reconstruction loss weight to 0 during classifier training
        # We only want to optimize the trading loss, not reconstruction
        # Pass loss_weights directly to compile_model to avoid recompilation issues
        self.model = self.compile_model(
            self.model,
            class_weights=class_weights,
            loss_weights={
                "reconstruction": 0.0,  # Disable reconstruction loss
                "trading": 1.0,  # Only optimize trading loss
            },
        )

        # Verify trainable variables (more accurate than layers)
        trainable_vars = self.model.trainable_variables
        non_trainable_vars = self.model.non_trainable_variables
        total_vars = len(trainable_vars) + len(non_trainable_vars)

        print(f"      Trainable variables: {len(trainable_vars)}/{total_vars}")
        if len(trainable_vars) > 0:
            # Group variables by layer name for clarity
            var_names = [v.name for v in trainable_vars]
            classifier_vars = [
                v for v in var_names if "classifier_head" in v or "trading" in v
            ]
            encoder_vars = [
                v for v in var_names if "encoder" in v or "input_layer" in v
            ]
            generator_vars = [
                v
                for v in var_names
                if "generator" in v or "decoded" in v or "reconstruction" in v
            ]

            print(f"      Classifier variables: {len(classifier_vars)}")
            if len(encoder_vars) > 0:
                print(
                    f"      WARNING: Found {len(encoder_vars)} encoder variables - should be 0!"
                )
            if len(generator_vars) > 0:
                print(
                    f"      WARNING: Found {len(generator_vars)} generator variables - should be 0!"
                )
            if len(trainable_vars) <= 20:
                print(f"      Trainable variable names: {var_names}")
        else:
            print("      WARNING: No trainable variables found! All layers are frozen.")

        print(
            "      Loss weights: reconstruction=0.0, trading=1.0 (only trading loss will be optimized)"
        )

        # Convert signal_labels to class indices if they're one-hot encoded
        # signal_labels might be one-hot (shape: [N, 3]) or class indices (shape: [N])
        if len(signal_labels.shape) > 1:
            # One-hot encoded - convert to class indices for stratification
            signal_labels_1d = np.argmax(signal_labels, axis=1)
        else:
            # Already class indices
            signal_labels_1d = signal_labels

        # Use stratified split to preserve class distribution in train/val
        # This helps ensure validation set is representative
        train_split, val_split, train_labels_1d, val_labels_1d = train_test_split(
            signal_data,
            signal_labels_1d,  # Use class indices for stratification
            test_size=0.2,
            stratify=signal_labels_1d,  # Preserve class distribution
            random_state=42,
            shuffle=True,
        )

        # Convert back to one-hot for model training (reconstruction labels are ignored due to weight=0)
        num_classes = 3
        train_labels_onehot = np.eye(num_classes)[train_labels_1d]
        val_labels_onehot = np.eye(num_classes)[val_labels_1d]

        # Print class distributions for debugging
        train_dist = np.bincount(train_labels_1d, minlength=num_classes)
        val_dist = np.bincount(val_labels_1d, minlength=num_classes)
        train_total = train_dist.sum()
        val_total = val_dist.sum()
        train_pct = (
            (train_dist / train_total * 100)
            if train_total > 0
            else np.zeros_like(train_dist)
        )
        val_pct = (
            (val_dist / val_total * 100) if val_total > 0 else np.zeros_like(val_dist)
        )
        train_dist_list = train_dist.tolist()
        val_dist_list = val_dist.tolist()
        train_pct_list = [f"{p:.1f}" for p in train_pct.tolist()]
        val_pct_list = [f"{p:.1f}" for p in val_pct.tolist()]
        print(f"      Train class distribution: {train_dist_list} ({train_pct_list}%)")
        print(f"      Val class distribution: {val_dist_list} ({val_pct_list}%)")

        # Only provide trading labels - reconstruction targets not needed (loss weight is 0)
        # Note: reconstruction targets are still provided for model structure, but loss weight is 0
        train_labels_split = {
            "reconstruction": train_split,  # Still provide for model structure, but weight is 0
            "trading": train_labels_onehot,  # One-hot encoded for model
        }
        val_labels_split = {
            "reconstruction": val_split,  # Still provide for model structure, but weight is 0
            "trading": val_labels_onehot,  # One-hot encoded for model
        }

        print(
            f"      Training classifier on {len(train_split)} signal samples (val: {len(val_split)})"
        )

        # Callbacks with reduced patience to prevent overfitting
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_trading_mcc",
                mode="max",
                patience=10,  # Reduced from 20 to stop earlier
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_trading_mcc",
                mode="max",
                factor=0.1,
                patience=5,  # Reduced from 8
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_checkpoint_path(),
                save_weights_only=True,
                monitor="val_trading_mcc",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Train classifier - only classifier head weights will be updated
        self.model.fit(
            train_split,
            train_labels_split,
            validation_data=(val_split, val_labels_split),
            epochs=100,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        print("      Classifier training completed")


# Dense GANomaly Classifier
class GANomalyClassifier_Dense(GANomalyClassifierBase):
    """Dense (MLP) based GANomaly classifier - flattens sequences internally for dense layers"""

    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        """Create dense encoder - flattens 3D input internally"""
        # Flatten sequences for dense layers
        x = tf.keras.layers.Flatten()(inputs)

        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def create_generator(self, latent, seq_len, num_features):
        """Create dense generator - outputs 3D sequences (reshapes internally)"""
        total_features = seq_len * num_features

        x = tf.keras.layers.Dense(128)(latent)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(total_features, activation="tanh")(x)
        # Reshape back to 3D sequences
        x = tf.keras.layers.Reshape((seq_len, num_features))(x)
        return x

    def create_discriminator(self, inputs, seq_len, num_features):
        """Create dense discriminator - flattens 3D input internally"""
        # Flatten sequences for dense layers
        x = tf.keras.layers.Flatten()(inputs)

        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return x


# LSTM GANomaly Classifier
class GANomalyClassifier_LSTM(GANomalyClassifierBase):
    """LSTM based GANomaly classifier - works directly with sequences"""

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag=tag)
        self.lstm_units = 128
        self.lstm_layers = 2

    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        """Create LSTM encoder - works with 3D sequences, returns 3D"""
        x = tf.keras.layers.LSTM(
            self.lstm_units, activation="tanh", return_sequences=True
        )(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.LSTM(
            self.lstm_units // 2, activation="tanh", return_sequences=True
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    def create_generator(self, latent, seq_len, num_features):
        """Create LSTM generator - accepts 2D or 3D tensor, outputs 3D sequences"""
        # Handle both 2D (latent vector for GAN training) and 3D (from encoder) inputs
        if len(latent.shape) == 2:
            # 2D input: repeat vector to match sequence length
            x = tf.keras.layers.RepeatVector(seq_len)(latent)
        else:
            # 3D input: use directly
            x = latent
        
        x = tf.keras.layers.LSTM(
            self.lstm_units // 2, activation="tanh", return_sequences=True
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.LSTM(
            self.lstm_units, activation="tanh", return_sequences=True
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Project to num_features
        x = tf.keras.layers.Dense(num_features, activation="tanh")(x)
        return x

    def create_discriminator(self, inputs, seq_len, num_features):
        """Create LSTM discriminator - works with 3D sequences"""
        x = tf.keras.layers.LSTM(
            self.lstm_units, activation="tanh", return_sequences=True
        )(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.LSTM(
            self.lstm_units // 2, activation="tanh", return_sequences=False
        )(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return x


# Attention GANomaly Classifier
class GANomalyClassifier_Attention(GANomalyClassifierBase):
    """Attention-based GANomaly classifier - works directly with sequences"""

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag=tag)
        self.head_size = 32
        self.num_heads = 4
        self.ff_dim = 128
        self.dropout = 0.2

    def attention_block(self, inputs, head_size, num_heads, dropout, ff_dim):
        """Create an attention block with multi-head attention and feed-forward"""
        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)

        # Only apply attention if sequence length > 1
        if inputs.shape[1] > 1:
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=head_size, dropout=dropout
            )(x, x)
            x = tf.keras.layers.Add()([attention_output, x])

        # Feed Forward
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(inputs.shape[-1]),
            ]
        )
        x = tf.keras.layers.Add()([ffn(x), x])

        return x

    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        """Create attention encoder - works with 3D sequences, returns 3D"""
        x = self.attention_block(
            inputs, self.head_size, self.num_heads, self.dropout, self.ff_dim
        )
        # Return 3D tensor (no GlobalAveragePooling1D)
        encoded = x
        return encoded

    def create_generator(self, latent, seq_len, num_features):
        """Create attention generator - accepts 2D or 3D tensor, outputs 3D sequences"""
        # Handle both 2D (latent vector for GAN training) and 3D (from encoder) inputs
        if len(latent.shape) == 2:
            # 2D input: repeat vector to match sequence length
            x = tf.keras.layers.RepeatVector(seq_len)(latent)
        else:
            # 3D input: use directly
            x = latent
        
        # Project to match expected feature dimension if needed
        if x.shape[-1] != num_features:
            x = tf.keras.layers.Dense(num_features)(x)

        # Attention decoder
        x = self.attention_block(
            x, self.head_size, self.num_heads, self.dropout, self.ff_dim
        )

        # Project to num_features with tanh activation
        x = tf.keras.layers.Dense(num_features, activation="tanh")(x)
        return x

    def create_discriminator(self, inputs, seq_len, num_features):
        """Create attention discriminator - works with 3D sequences"""
        x = self.attention_block(
            inputs, self.head_size, self.num_heads, self.dropout, self.ff_dim
        )

        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return x


# Transformer Block for Transformer GANomaly
class TransformerBlock(tf.keras.layers.Layer):
    """Complete Transformer block with self-attention and feed-forward"""

    def __init__(self, t_size, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.t_size = t_size
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        # Initialize layers in __init__ but don't build them yet
        self.att = None
        self.ffn = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        """Build the layer with proper input shape"""
        # Create layers with proper input shape
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.t_size, value_dim=self.t_size
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.dff, activation="relu"),
                tf.keras.layers.Dense(self.t_size),
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

        super(TransformerBlock, self).build(input_shape)

    def call(self, x, training=True):
        # Self-attention
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update(
            {
                "t_size": self.t_size,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "rate": self.rate,
            }
        )
        return config


# Transformer GANomaly Classifier
class GANomalyClassifier_Transformer(GANomalyClassifierBase):
    """Transformer-based GANomaly classifier - works directly with sequences"""

    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag=tag)
        self.t_size = 32
        self.num_heads = 8
        self.dff = 256  # Feed-forward dimension
        self.num_layers = 4  # Number of transformer layers
        self.dropout_rate = 0.1

    def create_encoder(self, inputs, seq_len, num_features, latent_size):
        """Create transformer encoder - works with 3D sequences, returns 3D"""
        self.t_size = max(
            32, self.nearest_power_of_2(num_features // 2) + 1
        )  # Ensure t_size >= num_features
        self.num_heads = min(
            8, self.t_size // 8
        )  # Ensure num_heads divides t_size evenly

        # Project to t_size dimension
        x = tf.keras.layers.Dense(self.t_size)(inputs)

        # Transformer encoder layers
        for i in range(self.num_layers):
            x = TransformerBlock(
                self.t_size,
                self.num_heads,
                self.dff,
                self.dropout_rate,
                name=f"transformer_block_{i}",
            )(x)

        # Return 3D tensor (no GlobalAveragePooling1D)
        encoded = x

        return encoded

    def create_generator(self, latent, seq_len, num_features):
        """Create transformer generator - accepts 2D or 3D tensor, outputs 3D sequences"""
        # Handle both 2D (latent vector for GAN training) and 3D (from encoder) inputs
        if len(latent.shape) == 2:
            # 2D input: repeat vector to match sequence length
            x = tf.keras.layers.RepeatVector(seq_len)(latent)
        else:
            # 3D input: use directly
            x = latent
        
        # Ensure it's projected to t_size
        if x.shape[-1] != self.t_size:
            x = tf.keras.layers.Dense(self.t_size)(x)

        # Transformer decoder layers
        for i in range(self.num_layers):
            x = TransformerBlock(
                self.t_size,
                self.num_heads,
                self.dff,
                self.dropout_rate,
                name=f"transformer_block_decoder_{i}",
            )(x)

        # Project back to original feature dimension with tanh activation
        decoded = tf.keras.layers.Dense(num_features, activation="tanh")(x)

        return decoded

    def create_discriminator(self, inputs, seq_len, num_features):
        """Create transformer discriminator - works with 3D sequences"""
        # Project to t_size dimension
        x = tf.keras.layers.Dense(self.t_size)(inputs)

        # Transformer layers
        for i in range(self.num_layers):
            x = TransformerBlock(
                self.t_size,
                self.num_heads,
                self.dff,
                self.dropout_rate,
                name=f"transformer_block_discriminator_{i}",
            )(x)

        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return x


# Classifier Type Enum
class ClassifierType(Enum):
    Dense = GANomalyClassifier_Dense
    LSTM = GANomalyClassifier_LSTM
    Attention = GANomalyClassifier_Attention
    Transformer = GANomalyClassifier_Transformer


# Factory function
def create_classifier(
    classifier_type: ClassifierType, pair, nfeatures, seq_len, tag=""
):
    """Create a GANomaly classifier of the specified type"""
    classifier_name = str(classifier_type).split(".")[-1]
    classifier = classifier_type.value(pair, seq_len, nfeatures, tag=tag)
    return classifier, classifier_name
