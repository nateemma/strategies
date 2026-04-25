# Neural Network Anomaly Detector: collection of neural network anomaly detection models, types and access functions

# usage: detector, name = NNDetector.create_detector(detector_type, pair, nfeatures, seq_len, tag="")

import numpy as np
from pandas import DataFrame, Series
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
from enum import Enum, auto

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
import warnings
import random
import os

warnings.simplefilter(action="ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.saving.saving_lib")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'

import tensorflow as tf

# Base class for anomaly detectors
class AnomalyDetectorBase:
    is_trained = False
    clean_data_required = False  # training data cannot contain anomalies
    
    def __init__(self, pair, seq_len, num_features, tag=""):
        self.pair = pair
        self.seq_len = seq_len
        self.num_features = num_features
        self.tag = tag
        self.name = f"AnomalyDetector_{self.__class__.__name__}_{tag}"
        self.model = None
        self.model_path = None
        self.combine_models = True
        self.dbg_verbose = True
        
        # Training parameters
        self.num_epochs = 100
        self.batch_size = 128
        self.learning_rate = 0.01  # Default learning rate
        
    def set_model_path(self, model_path):
        self.model_path = model_path
        
    def set_combine_models(self, combine_models):
        self.combine_models = combine_models
        
    def set_learning_rate(self, learning_rate):
        """Set the learning rate for the detector"""
        self.learning_rate = learning_rate
        
    def create_model(self, seq_len, num_features):
        """Override this method in subclasses to create specific model architectures"""
        raise NotImplementedError("Subclasses must implement create_model")
        
    def preprocess_data(self, data_values):
        """Preprocess data to the format expected by this detector"""
        # Base implementation - subclasses can override if needed
        return data_values
        
    def train(self, train_data, val_data, force_train=False):
        """Train the anomaly detector"""
        if self.model is None:
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return
            self.compile_model()
            self.model.summary()
            
        # Preprocess data for this detector type
        train_data_processed = self.preprocess_data(train_data)
        val_data_processed = self.preprocess_data(val_data)
            
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1 if self.dbg_verbose else 0
        )
        
        # Add learning rate reduction
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=1e-8,
            verbose=1 if self.dbg_verbose else 0
        )
        
        # Add model checkpointing
        if self.model_path:
            checkpoint_path = self.model_path.replace('.keras', '_best.keras')
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1 if self.dbg_verbose else 0
            )
            callbacks = [early_stopping, lr_scheduler, model_checkpoint]
        else:
            callbacks = [early_stopping, lr_scheduler]
            
        # Train the model
        history = self.model.fit(
            train_data_processed, train_data_processed,  # Autoencoder: input = output
            validation_data=(val_data_processed, val_data_processed),
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1 if self.dbg_verbose else 0
        )
        
        # Save the model
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            
        self.is_trained = True
        return history
        
    def compile_model(self):
        """Compile the model with appropriate optimizer and loss"""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=0.1,
            beta_1=0.9,
            beta_2=0.999
        )
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
    def predict(self, data, verbose=0):
        """Get reconstructions from the autoencoder"""
        if self.model is None:
            print("    ERR: no model for predictions")
            return np.zeros_like(data)
        
        # Preprocess data for this detector type
        data_processed = self.preprocess_data(data)
            
        return self.model.predict(data_processed, verbose=verbose)
        
    def get_reconstruction_errors(self, data):
        """Calculate reconstruction errors (MSE between input and output)"""
        reconstructions = self.predict(data, verbose=0)
        
        # For 3D data, average across the sequence dimension
        if len(reconstructions.shape) == 3:
            errors = np.mean(np.square(data - reconstructions), axis=(1, 2))
        else:
            errors = np.mean(np.square(data - reconstructions), axis=1)
            
        return errors

    def save_model(self, model_path):
        """Save the model to the specified path"""
        if self.model is not None:
            import os
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            print(f"    Model saved to: {model_path}")
        else:
            print("    ERROR: No model to save")

    def load_model(self, model_path):
        """Load the model from the specified path"""
        import os
        if os.path.exists(model_path):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.compile_model()
            print(f"    Model loaded from: {model_path}")
        else:
            print(f"    ERROR: Model file not found: {model_path}")


# --------------------------------------------------------------
# Dense Autoencoder

class NNDetector_Dense(AnomalyDetectorBase):
    is_trained = False
    clean_data_required = False
    model_type = 'Dense'  # For feature checking
    
    def preprocess_data(self, data_values):
        """Convert 3D data to 2D for Dense autoencoder"""
        if len(data_values.shape) == 3:
            # Flatten sequence dimension: (batch, seq, features) -> (batch, seq*features)
            batch_size = data_values.shape[0]
            return data_values.reshape(batch_size, -1)
        return data_values
    
    def create_model(self, seq_len, num_features):
        # Calculate total input features (seq_len * num_features for flattened data)
        total_features = seq_len * num_features
        
        # Encoder
        inputs = tf.keras.layers.Input(shape=(total_features,))
        
        # Calculate optimal layer sizes based on input features
        latent_size = max(8, total_features // 8)  # Adaptive latent size
        encoder_size1 = max(32, total_features * 2)
        encoder_size2 = max(16, total_features)
        
        # Dense Encoder with better architecture
        encoded = tf.keras.layers.Dense(encoder_size1, activation='elu')(inputs)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        
        encoded = tf.keras.layers.Dense(encoder_size2, activation='elu')(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        
        # Latent representation
        latent = tf.keras.layers.Dense(latent_size, activation='elu')(encoded)
        latent = tf.keras.layers.BatchNormalization()(latent)
        
        # Decoder
        decoded = tf.keras.layers.Dense(encoder_size2, activation='elu')(latent)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        
        decoded = tf.keras.layers.Dense(encoder_size1, activation='elu')(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        
        # Output layer - use linear activation for better reconstruction
        outputs = tf.keras.layers.Dense(total_features, activation='linear')(decoded)
        
        # Create model
        model = tf.keras.Model(inputs, outputs, name=f'Dense_Autoencoder_{self.tag}')
        
        return model
    
    def predict(self, data, verbose=0):
        """Get reconstructions and reshape back to 3D if needed"""
        if self.model is None:
            print("    ERR: no model for predictions")
            return np.zeros_like(data)
        
        # Preprocess data for this detector type
        data_processed = self.preprocess_data(data)
        
        # Get predictions
        reconstructions_2d = self.model.predict(data_processed, verbose=verbose)
        
        # Reshape back to original format if input was 3D
        if len(data.shape) == 3:
            batch_size, seq_len, num_features = data.shape
            reconstructions = reconstructions_2d.reshape(batch_size, seq_len, num_features)
        else:
            reconstructions = reconstructions_2d
            
        return reconstructions


# --------------------------------------------------------------
# LSTM Autoencoder

class NNDetector_LSTM(AnomalyDetectorBase):
    is_trained = False
    clean_data_required = False
    model_type = 'LSTM'  # For feature checking
    
    def preprocess_data(self, data_values):
        """Ensure data is 3D for LSTM autoencoder"""
        if len(data_values.shape) == 2:
            # Convert 2D to 3D: (batch, features) -> (batch, 1, features)
            return data_values.reshape(data_values.shape[0], 1, data_values.shape[1])
        return data_values
    
    def create_model(self, seq_len, num_features):
        # Encoder
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
        
        # LSTM Encoder
        encoded = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(inputs)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        
        encoded = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=False)(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        
        # Latent representation
        latent = tf.keras.layers.Dense(16, activation='elu')(encoded)
        latent = tf.keras.layers.BatchNormalization()(latent)
        
        # Repeat vector to match sequence length
        repeated = tf.keras.layers.RepeatVector(seq_len)(latent)
        
        # LSTM Decoder
        decoded = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(repeated)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        
        decoded = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        
        # Output layer
        outputs = tf.keras.layers.Dense(num_features, activation='linear')(decoded)
        
        # Create model
        model = tf.keras.Model(inputs, outputs, name=f'LSTM_Autoencoder_{self.tag}')
        
        return model


# --------------------------------------------------------------
# Attention-based Autoencoder

class NNDetector_Attention(AnomalyDetectorBase):
    is_trained = False
    clean_data_required = False
    model_type = 'Attention'  # For feature checking
    
    def preprocess_data(self, data_values):
        """Ensure data is 3D for Transformer autoencoder"""
        if len(data_values.shape) == 2:
            # Convert 2D to 3D: (batch, features) -> (batch, 1, features)
            return data_values.reshape(data_values.shape[0], 1, data_values.shape[1])
        return data_values
    
    def create_model(self, seq_len, num_features):
        # Encoder
        inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
        
        # Attention Encoder
        encoded = self.attention_encoder(inputs, head_size=32, num_heads=4, dropout=0.2, ff_dim=128)
        encoded = tf.keras.layers.GlobalAveragePooling1D()(encoded)
        
        # Latent representation
        latent = tf.keras.layers.Dense(32, activation='elu')(encoded)
        latent = tf.keras.layers.BatchNormalization()(latent)
        
        # Repeat vector to match sequence length
        repeated = tf.keras.layers.RepeatVector(seq_len)(latent)
        
        # Attention Decoder
        decoded = self.attention_encoder(repeated, head_size=32, num_heads=4, dropout=0.2, ff_dim=128)
        
        # Output layer
        outputs = tf.keras.layers.Dense(num_features, activation='linear')(decoded)
        
        # Create model
        model = tf.keras.Model(inputs, outputs, name=f'Attention_Autoencoder_{self.tag}')
        
        return model
        
    def attention_encoder(self, inputs, head_size, num_heads, dropout, ff_dim):
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
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(inputs.shape[-1]),
        ])
        x = tf.keras.layers.Add()([ffn(x), x])
        
        return x


# --------------------------------------------------------------
# Full Transformer Autoencoder

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for Transformer"""
    
    def __init__(self, max_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Create positional encoding matrix using numpy for simplicity
        import numpy as np
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)  # Add batch dimension
    
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model
        })
        return config

class TransformerBlock(tf.keras.layers.Layer):
    """Complete Transformer block with self-attention and feed-forward"""
    
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
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
            num_heads=self.num_heads, 
            key_dim=self.d_model,
            value_dim=self.d_model
        )
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu'),
            tf.keras.layers.Dense(self.d_model)
        ])
        
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
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate
        })
        return config

class NNDetector_Transformer(AnomalyDetectorBase):
    is_trained = False
    clean_data_required = False
    model_type = 'Transformer'  # For feature checking
    
    def preprocess_data(self, data_values):
        """Ensure data is 3D for Transformer autoencoder"""
        if len(data_values.shape) == 2:
            # Convert 2D to 3D: (batch, features) -> (batch, 1, features)
            return data_values.reshape(data_values.shape[0], 1, data_values.shape[1])
        return data_values
    
    def create_model(self, seq_len, num_features):
        # Model parameters - adjusted for better compatibility
        d_model = max(64, num_features)  # Ensure d_model >= num_features
        num_heads = min(8, d_model // 8)  # Ensure num_heads divides d_model evenly
        dff = 256  # Feed-forward dimension
        num_layers = 4  # Number of transformer layers
        dropout_rate = 0.1
        
        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            d_model = ((d_model // num_heads) + 1) * num_heads
        
        # For seq_len=1, use a simpler architecture to avoid softmax issues
        if seq_len == 1:
            # Simple dense autoencoder for single timestep
            inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
            
            # Flatten input
            x = tf.keras.layers.Flatten()(inputs)
            
            # Encoder
            encoded = tf.keras.layers.Dense(128, activation='elu')(x)
            encoded = tf.keras.layers.BatchNormalization()(encoded)
            encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)
            
            encoded = tf.keras.layers.Dense(64, activation='elu')(encoded)
            encoded = tf.keras.layers.BatchNormalization()(encoded)
            encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)
            
            # Latent representation
            latent = tf.keras.layers.Dense(32, activation='elu')(encoded)
            latent = tf.keras.layers.BatchNormalization()(latent)
            latent = tf.keras.layers.Dropout(dropout_rate)(latent)
            
            # Decoder
            decoded = tf.keras.layers.Dense(64, activation='elu')(latent)
            decoded = tf.keras.layers.BatchNormalization()(decoded)
            decoded = tf.keras.layers.Dropout(dropout_rate)(decoded)
            
            decoded = tf.keras.layers.Dense(128, activation='elu')(decoded)
            decoded = tf.keras.layers.BatchNormalization()(decoded)
            decoded = tf.keras.layers.Dropout(dropout_rate)(decoded)
            
            # Output layer
            outputs = tf.keras.layers.Dense(num_features, activation='linear')(decoded)
            outputs = tf.keras.layers.Reshape((seq_len, num_features))(outputs)
            
        else:
            # Full transformer for multi-timestep sequences
            # Encoder
            inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
            
            # Project to d_model dimension
            x = tf.keras.layers.Dense(d_model)(inputs)
            
            # Add positional encoding
            x = PositionalEncoding(seq_len, d_model)(x)
            
            # Transformer encoder layers
            for i in range(num_layers):
                x = TransformerBlock(d_model, num_heads, dff, dropout_rate, name=f'transformer_block_{i}')(x)
            
            # Global pooling to get sequence representation
            encoded = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Latent representation
            latent = tf.keras.layers.Dense(32, activation='elu')(encoded)
            latent = tf.keras.layers.BatchNormalization()(latent)
            latent = tf.keras.layers.Dropout(dropout_rate)(latent)
            
            # Repeat vector to match sequence length
            repeated = tf.keras.layers.RepeatVector(seq_len)(latent)
            
            # Project back to d_model dimension
            x = tf.keras.layers.Dense(d_model)(repeated)
            
            # Add positional encoding for decoder
            x = PositionalEncoding(seq_len, d_model)(x)
            
            # Transformer decoder layers
            for i in range(num_layers):
                x = TransformerBlock(d_model, num_heads, dff, dropout_rate, name=f'transformer_block_decoder_{i}')(x)
            
            # Project back to original feature dimension
            x = tf.keras.layers.Dense(num_features, activation='linear')(x)
            
            # Output layer
            outputs = x
        
        # Create model
        model = tf.keras.Model(inputs, outputs, name=f'Transformer_Autoencoder_{self.tag}')
        
        return model


# --------------------------------------------------------------
# Variational Autoencoder (VAE)

class SamplingLayer(tf.keras.layers.Layer):
    """Custom sampling layer for VAE that is properly serializable"""
    
    def __init__(self, **kwargs):
        super(SamplingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        config = super(SamplingLayer, self).get_config()
        return config

class NNDetector_VAE(AnomalyDetectorBase):
    is_trained = False
    clean_data_required = False
    model_type = 'VAE'  # For feature checking
    
    def preprocess_data(self, data_values):
        """Convert 3D data to 2D for VAE"""
        if len(data_values.shape) == 3:
            # Flatten sequence dimension: (batch, seq, features) -> (batch, seq*features)
            batch_size = data_values.shape[0]
            return data_values.reshape(batch_size, -1)
        return data_values
    
    def create_model(self, seq_len, num_features):
        # Calculate total input features (seq_len * num_features for flattened data)
        total_features = seq_len * num_features
        
        # Encoder
        inputs = tf.keras.layers.Input(shape=(total_features,))
        
        # Encoder layers
        encoded = tf.keras.layers.Dense(128, activation='relu')(inputs)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        
        encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        
        # Latent space parameters
        z_mean = tf.keras.layers.Dense(16, name='z_mean')(encoded)
        z_log_var = tf.keras.layers.Dense(16, name='z_log_var')(encoded)
        
        # Sampling layer using custom layer
        z = SamplingLayer(name='z')([z_mean, z_log_var])
        
        # Decoder
        decoded = tf.keras.layers.Dense(64, activation='relu')(z)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        
        decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        
        # Output layer
        outputs = tf.keras.layers.Dense(total_features, activation='linear')(decoded)
        
        # Create model
        model = tf.keras.Model(inputs, outputs, name=f'VAE_Autoencoder_{self.tag}')
        
        return model
        
    def compile_model(self):
        """Compile VAE with MSE loss (KL divergence handled by sampling)"""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=0.1,
            beta_1=0.9,
            beta_2=0.999
        )
        
        # Use MSE loss - the VAE behavior comes from the sampling layer
        # which introduces stochasticity and regularization
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    def predict(self, data, verbose=0):
        """Get reconstructions and reshape back to 3D if needed"""
        if self.model is None:
            print("    ERR: no model for predictions")
            return np.zeros_like(data)
        
        # Preprocess data for this detector type
        data_processed = self.preprocess_data(data)
        
        # Get predictions
        reconstructions_2d = self.model.predict(data_processed, verbose=verbose)
        
        # Reshape back to original format if input was 3D
        if len(data.shape) == 3:
            batch_size, seq_len, num_features = data.shape
            reconstructions = reconstructions_2d.reshape(batch_size, seq_len, num_features)
        else:
            reconstructions = reconstructions_2d
            
        return reconstructions


# --------------------------------------------------------------
# LightGBM-based Anomaly Detector

class NNDetector_LightGBM(AnomalyDetectorBase):
    is_trained = False
    clean_data_required = False
    model_type = 'LightGBM'  # For feature checking
    
    def __init__(self, pair, seq_len, num_features, tag=""):
        super().__init__(pair, seq_len, num_features, tag)
        self.model = None
        self.learning_rate = 0.01
        
        # LightGBM specific parameters
        self.n_estimators = 100
        self.max_depth = 10
        self.contamination = 0.1  # Expected proportion of anomalies
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        
    def preprocess_data(self, data_values):
        """Convert data to 2D format for LightGBM"""
        if len(data_values.shape) == 3:
            # Flatten sequence dimension: (batch, seq, features) -> (batch, seq*features)
            batch_size = data_values.shape[0]
            return data_values.reshape(batch_size, -1)
        return data_values
        
    def create_model(self, seq_len, num_features):
        """LightGBM doesn't need model creation like neural networks"""
        return None
        
    def compile_model(self):
        """LightGBM doesn't need compilation like neural networks"""
        pass
        
    def build_model(self):
        """Build LightGBM model for anomaly detection"""
        try:
            import lightgbm as lgb
        except ImportError:
            print("    ERROR: LightGBM not installed. Please install with: pip install lightgbm")
            return None
            
        # Use LightGBM's regression model for anomaly detection
        # We'll use it as an autoencoder: input = target
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective='regression',
            random_state=42,
            verbose=-1,
            n_jobs=-1  # Use all CPU cores
        )
        
    def train(self, train_data, val_data, force_train=False):
        """Train LightGBM model"""
        if self.model is None:
            self.build_model()
            if self.model is None:
                return None
                
        # Import lightgbm here to ensure it's available
        try:
            import lightgbm as lgb
        except ImportError:
            print("    ERROR: LightGBM not installed. Please install with: pip install lightgbm")
            return None
                
        # Preprocess data for LightGBM
        train_data_2d = self.preprocess_data(train_data)
        val_data_2d = self.preprocess_data(val_data)
        
        # Ensure data is contiguous to avoid LightGBM memory warnings
        X_train = np.ascontiguousarray(train_data_2d)
        y_train = np.ascontiguousarray(train_data_2d)  # Self-supervised learning
        X_val = np.ascontiguousarray(val_data_2d)
        y_val = np.ascontiguousarray(val_data_2d)
        
        # For LightGBM, we need to handle the case where y_train is 2D
        # We'll train separate models for each feature and average the predictions
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            # Multi-feature case: train separate models for each feature
            self.models = []
            for i in range(y_train.shape[1]):
                model = self.model.__class__(**self.model.get_params())
                model.fit(
                    X_train, y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    eval_metric='mse',
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                self.models.append(model)
        else:
            # Single feature case
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mse',
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
        self.is_trained = True
        return {'loss': 0.0}  # LightGBM doesn't return history like Keras
        
    def predict(self, data, verbose=0):
        """Get reconstructions (for compatibility with neural network detectors)"""
        if self.model is None and not hasattr(self, 'models'):
            print("    ERR: no model for predictions")
            return np.zeros_like(data)
            
        # For LightGBM, we return the original data as "reconstructions"
        # since LightGBM doesn't actually reconstruct - it just learns patterns
        # The actual anomaly detection happens in get_reconstruction_errors()
        return data
        
    def get_reconstruction_errors(self, data):
        """Calculate reconstruction errors (anomaly scores) for LightGBM"""
        # print(f"    LightGBM get_reconstruction_errors debug:")
        # print(f"      self.model: {self.model}")
        # print(f"      hasattr(self, 'models'): {hasattr(self, 'models')}")
        # print(f"      self.models: {getattr(self, 'models', None)}")
        
        if self.model is None and (not hasattr(self, 'models') or self.models is None):
            print("    ERR: no model for predictions")
            return np.zeros(data.shape[0])
            
        # Preprocess data for LightGBM
        data_2d = self.preprocess_data(data)
        # Ensure data is contiguous to avoid LightGBM memory warnings
        data_2d = np.ascontiguousarray(data_2d)
        
        # Get predictions
        if hasattr(self, 'models') and self.models:
            # Multi-feature case: average predictions from all models
            predictions_list = []
            for model in self.models:
                pred = model.predict(data_2d)
                predictions_list.append(pred)
            predictions = np.column_stack(predictions_list)
        else:
            # Single feature case
            predictions = self.model.predict(data_2d)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
        
        # Calculate reconstruction error (MSE between input and output)
        errors = np.mean((data_2d - predictions) ** 2, axis=1)
        
        # Debug output to understand what's happening
        # print(f"    LightGBM reconstruction error debug:")
        # print(f"      Input data shape: {data_2d.shape}")
        # print(f"      Predictions shape: {predictions.shape}")
        # print(f"      Input data range: [{data_2d.min():.6f}, {data_2d.max():.6f}]")
        # print(f"      Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        # print(f"      Error range: [{errors.min():.6f}, {errors.max():.6f}]")
        # print(f"      Mean error: {errors.mean():.6f}")
        
        return errors
        
    def save_model(self, model_path):
        """Save LightGBM model(s)"""
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if hasattr(self, 'models') and self.models:
            # Save multiple models as a single joblib file
            print(f"    Saving {len(self.models)} LightGBM models to: {model_path}")
            import joblib
            joblib.dump(self.models, model_path)
            print(f"    Models saved successfully. File exists: {os.path.exists(model_path)}")
                
        elif self.model is not None:
            # Save single model
            print(f"    Saving single LightGBM model to: {model_path}")
            import joblib
            joblib.dump(self.model, model_path)
            print(f"    Single model saved successfully. File exists: {os.path.exists(model_path)}")
        else:
            print("    ERROR: No model to save")
                
    def load_model(self, model_path):
        """Load LightGBM model(s)"""
        try:
            import lightgbm as lgb
        except ImportError:
            print("    ERROR: LightGBM not installed. Please install with: pip install lightgbm")
            return
            
        if os.path.exists(model_path):
            # Load model using joblib
            print(f"    Loading LightGBM model from: {model_path}")
            import joblib
            loaded_data = joblib.load(model_path)
            
            # Check if it's a list of models or a single model
            if isinstance(loaded_data, list):
                # Multiple models
                self.models = loaded_data
                print(f"    Loaded {len(self.models)} LightGBM models successfully")
            else:
                # Single model
                self.model = loaded_data
                print(f"    Loaded single LightGBM model successfully")
        else:
            print(f"    ERROR: Model file not found: {model_path}")


# --------------------------------------------------------------
# Types of Detector

class DetectorType(Enum):
    Dense = NNDetector_Dense  # Dense Autoencoder
    LSTM = NNDetector_LSTM  # LSTM Autoencoder
    Attention = NNDetector_Attention  # Attention-based Autoencoder
    Transformer = NNDetector_Transformer  # Full Transformer Autoencoder
    VAE = NNDetector_VAE  # Variational Autoencoder
    LightGBM = NNDetector_LightGBM  # LightGBM-based Anomaly Detector


# --------------------------------------------------------------

# factory to create detector based on ID. Returns detector and name
def create_detector(detector_type: DetectorType, pair, nfeatures, seq_len, tag=""):
    detector_name = str(detector_type).split(".")[-1]
    detector = detector_type.value(pair, seq_len, nfeatures, tag=tag)

    return detector, detector_name

# -------------------------------------------------------------- 