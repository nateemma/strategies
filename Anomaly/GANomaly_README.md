# GANomaly Implementation for Freqtrade

## Overview

GANomaly (Generative Adversarial Network for Anomaly Detection) is a powerful approach that combines the strengths of GANs with anomaly detection. This implementation provides a complete GANomaly strategy for detecting market anomalies in cryptocurrency trading.

## Key Features

### 1. **Adversarial Training**
- **Generator**: Learns to generate normal market patterns
- **Discriminator**: Learns to distinguish between real and generated data
- **Encoder**: Maps real data to latent space for anomaly detection

### 2. **Anomaly Detection**
- High reconstruction error indicates anomalies
- Normalized anomaly scores for robust detection
- Rolling window statistics for adaptive thresholds

### 3. **Multiple Architectures**
- **NNGANomaly**: Standard dense architecture
- **NNGANomaly_LSTM**: LSTM-based architecture for temporal patterns

## Architecture

### Generator Network
```
Input (latent_dim) → Dense → LeakyReLU → BatchNorm → Dense → LeakyReLU → BatchNorm → Dense → Tanh → Output
```

### Discriminator Network
```
Input (features) → Dense → LeakyReLU → Dropout → Dense → LeakyReLU → Dropout → Dense → Sigmoid → Output
```

### Encoder Network
```
Input (features) → Dense → LeakyReLU → BatchNorm → Dense → LeakyReLU → BatchNorm → Dense → Tanh → Latent
```

## Strategy Parameters

### GANomaly-specific Parameters
```python
ganomaly_latent_dim = 32          # Latent space dimension
ganomaly_generator_lr = 0.0002    # Generator learning rate
ganomaly_discriminator_lr = 0.0002 # Discriminator learning rate
ganomaly_beta1 = 0.5              # Adam optimizer parameter
ganomaly_beta2 = 0.999            # Adam optimizer parameter
ganomaly_epochs = 100             # Training epochs
ganomaly_batch_size = 32          # Batch size for training
```

### Anomaly Detection Parameters
```python
anomaly_threshold = 0.1           # Threshold for anomaly detection
anomaly_window = 20               # Rolling window for anomaly scores
```

## Usage

### Basic Usage
```bash
# Run with standard GANomaly
freqtrade backtesting --strategy NNGANomaly --config config.json

# Run with LSTM variant
freqtrade backtesting --strategy NNGANomaly_LSTM --config config.json
```

### Hyperopt
```bash
# Optimize GANomaly parameters
freqtrade hyperopt --strategy NNGANomaly --hyperopt-loss SharpeHyperOptLoss --epochs 100
```

## How It Works

### 1. **Training Phase**
1. **Data Preparation**: Extract and normalize market features
2. **Adversarial Training**: Train generator and discriminator in alternating fashion
3. **Encoder Training**: Train encoder to map real data to latent space
4. **Model Saving**: Save all three components (generator, discriminator, encoder)

### 2. **Inference Phase**
1. **Feature Extraction**: Extract market features from current data
2. **Encoding**: Map features to latent space using encoder
3. **Decoding**: Reconstruct features using generator
4. **Error Calculation**: Compute reconstruction error as anomaly score
5. **Signal Generation**: Generate entry/exit signals based on anomaly thresholds

### 3. **Anomaly Detection**
- **Reconstruction Error**: Measures how well the model can reconstruct normal patterns
- **Normalization**: Uses rolling statistics to normalize anomaly scores
- **Thresholding**: Applies threshold to identify anomalies

## Advantages

### 1. **Better Feature Learning**
- Adversarial training forces the generator to learn more realistic patterns
- Discriminator provides strong supervision for feature learning

### 2. **Robust Anomaly Detection**
- Reconstruction error is more reliable than distance-based methods
- Adaptive normalization handles changing market conditions

### 3. **Flexible Architecture**
- Easy to extend with different architectures (LSTM, Transformer, etc.)
- Modular design allows for easy experimentation

## Comparison with Other Methods

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| **GANomaly** | Better feature learning, robust detection | More complex, longer training |
| **Autoencoder** | Simple, fast training | Limited feature learning |
| **VAE** | Probabilistic, uncertainty estimation | Complex training, mode collapse |
| **Isolation Forest** | Fast, no training required | Limited feature interactions |

## Performance Considerations

### 1. **Training Time**
- GANomaly requires more training time than simple autoencoders
- LSTM variant adds additional computational overhead
- Consider using GPU acceleration for faster training

### 2. **Memory Usage**
- Three separate models (generator, discriminator, encoder)
- LSTM layers require more memory than dense layers
- Monitor memory usage during training

### 3. **Hyperparameter Tuning**
- Learning rates are critical for stable training
- Batch size affects training stability
- Number of epochs depends on data complexity

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce learning rates
   - Increase batch size
   - Add gradient clipping

2. **Mode Collapse**
   - Adjust discriminator training frequency
   - Use different learning rates for generator/discriminator
   - Add noise to training data

3. **Poor Anomaly Detection**
   - Check feature normalization
   - Adjust anomaly threshold
   - Increase training epochs

### Debug Tips

1. **Monitor Losses**: Track generator and discriminator losses
2. **Visualize Results**: Plot anomaly scores over time
3. **Feature Analysis**: Check which features contribute most to anomalies

## Future Enhancements

1. **Attention Mechanisms**: Add attention layers for better feature selection
2. **Conditional GANs**: Use market conditions as conditioning variables
3. **Multi-modal GANs**: Incorporate multiple data sources (price, volume, sentiment)
4. **Online Learning**: Implement incremental training for adaptive models

## References

- Akcay, S., et al. "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training." arXiv preprint arXiv:1805.06725 (2018).
- Goodfellow, I., et al. "Generative Adversarial Nets." NIPS 2014.
- Schlegl, T., et al. "Unsupervised Anomaly Detection with Generative Adversarial Networks." MICCAI 2017. 