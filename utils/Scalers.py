"""
Utilities for saving and loading scalers
"""

import pickle
import os
from pathlib import Path


def get_scaler_path(location, name):
    """Get the path to a scaler"""
    return Path(f"{location}/{name}.pkl")


def scaler_exists(location, name):
    """Check if a scaler exists"""
    return get_scaler_path(location, name).exists()


def save_scaler(scaler, location, name):
    """Save a scaler to a file"""
    path = get_scaler_path(location, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Saving scaler to {path}")
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def remove_scaler(location, name):
    """Remove a scaler from a file"""
    path = get_scaler_path(location, name)
    path.unlink(missing_ok=True)


def load_scaler(location, name):
    """Load a scaler from a file"""
    path = get_scaler_path(location, name)
    # print(f"    Loading scaler from {path}")
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler
