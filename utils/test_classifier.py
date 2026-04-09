#!/usr/bin/env python3

"""
Test script for ClassifierKerasBinary to debug prediction issues
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from ClassifierKerasBinary import ClassifierKerasBinary
from DataframeUtils import DataframeUtils

def test_classifier():
    """Test the classifier with synthetic data"""
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    seq_len = 20
    
    # Create synthetic features
    features = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels (imbalanced)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Create dataframe
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    df['labels'] = labels
    
    print(f"Data shape: {df.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Initialize classifier
    classifier = ClassifierKerasBinary("TEST/USDT", seq_len, n_features)
    classifier.batch_size = 32
    classifier.num_epochs = 5
    classifier.dataframeUtils = DataframeUtils()
    
    # Split data
    train_size = int(0.8 * n_samples)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    train_labels = df_train['labels'].values
    test_labels = df_test['labels'].values
    
    df_train = df_train.drop('labels', axis=1)
    df_test = df_test.drop('labels', axis=1)
    
    # Adjust labels to match tensor size (sequences reduce sample count)
    train_tensor = classifier.dataframeUtils.df_to_tensor(df_train, seq_len)
    test_tensor = classifier.dataframeUtils.df_to_tensor(df_test, seq_len)
    
    # Adjust labels to match tensor size
    train_labels = train_labels[seq_len-1:len(train_labels)]
    test_labels = test_labels[seq_len-1:len(test_labels)]
    
    print(f"Adjusted train labels shape: {train_labels.shape}")
    print(f"Adjusted test labels shape: {test_labels.shape}")
    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Test tensor shape: {test_tensor.shape}")
    
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    
    # Train the model
    print("\nTraining model...")
    classifier.train(train_tensor, test_tensor, train_labels, test_labels, force_train=True)
    
    # Test predictions
    print("\nTesting predictions...")
    predictions = classifier.predict(df_test)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions unique: {np.unique(predictions, return_counts=True)}")
    
    # Debug predictions
    print("\nDebug predictions...")
    raw_preds = classifier.model.predict(classifier.dataframeUtils.df_to_tensor(df_test, seq_len), verbose=0)
    raw_preds = np.array(raw_preds)
    classifier.debug_predictions(raw_preds, predictions)
    
    if raw_preds is not None:
        print(f"Raw predictions shape: {raw_preds.shape}")
        print(f"Binary predictions shape: {predictions.shape}")
        print(f"Binary predictions unique: {np.unique(predictions, return_counts=True)}")
    
    # Test optimal threshold
    print("\nFinding optimal threshold...")
    optimal_threshold = classifier.find_optimal_threshold(df_test, test_labels)
    print(f"Optimal threshold: {optimal_threshold}")
    
    # Test with optimal threshold
    print("\nTesting with optimal threshold...")
    optimal_predictions = classifier.predict(df_test)
    print(f"Optimal predictions unique: {np.unique(optimal_predictions, return_counts=True)}")

if __name__ == "__main__":
    test_classifier() 