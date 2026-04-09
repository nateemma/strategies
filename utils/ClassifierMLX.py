from __future__ import annotations

# Base class for MLX-based classifiers.
# Mirrors ClassifierKeras.py API exactly so callers can swap backends.
#
# Key differences from ClassifierKeras:
#   - Models are mlx.nn.Module subclasses (not keras.Model)
#   - Weights are saved as .safetensors (not .keras)
#   - Architecture is always reconstructed from code on load (no full model serialisation)
#   - No TF/Keras dependency

import os
import sys
import logging
import warnings
import random
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
mx.random.seed(SEED)


# -----------------------------------------------------------------------

class ClassifierMLX:
    """
    Base class for MLX-based classifier/encoder models.
    Provides model lifecycle management (load/save/path) with the same
    external API as ClassifierKeras so calling code is interchangeable.
    """

    # --- class-level defaults (mirrors ClassifierKeras) ---
    model: Optional[nn.Module] = None
    is_trained: bool = False
    model_path: str = ""
    model_ext: str = "safetensors"
    name: str = ""
    root_dir: str = ""
    checkpoint_path: str = "/tmp/mlx_model.safetensors"

    seq_len: int = 8
    num_features: int = 64
    batch_size: int = 256

    default_max_epochs: int = 256
    num_epochs: int = default_max_epochs
    default_learning_rate: float = 0.0001
    learning_rate: float = default_learning_rate

    clean_data_required: bool = False
    model_per_pair: bool = False
    new_model: bool = False

    requires_dataframes: bool = False   # accept tensors (numpy arrays)
    prescale_dataframe: bool = True
    single_prediction: bool = False
    combine_models: bool = False

    dataframeUtils = None  # lazy-loaded in __init__ to avoid TF import at module level

    # -----------------------------------------------------------------------

    def __init__(self, pair: str, seq_len: int, num_features: int, tag: str = ""):
        super().__init__()

        self.loaded_from_file = False
        self.seq_len = seq_len
        self.num_features = num_features

        # Lazy import so that TF/Keras is NOT imported at module load time
        if self.dataframeUtils is None:
            from DataframeUtils import DataframeUtils
            self.dataframeUtils = DataframeUtils()

        # Build name from class name + optional pair/tag suffix
        pair_suffix = ("_" + pair.split("/")[0]) if self.model_per_pair else ""
        tag_suffix  = ("_" + tag) if tag else ""
        self.category = self.__class__.__name__
        self.name     = self.category + pair_suffix + tag_suffix

    # -----------------------------------------------------------------------
    # Path management — same convention as ClassifierKeras
    # -----------------------------------------------------------------------

    def set_model_path(self, path: str):
        # Prevent freqtrade upstream strategies from forcing .keras extensions on MLX models
        if path.endswith(".keras"):
            path = path[:-6] + f".{self.model_ext}"
        elif path.endswith(".h5"):
            path = path[:-3] + f".{self.model_ext}"

        self.model_path = path
        filename = path.split("/")[-1]
        self.name = filename.split(".")[0]
        self.root_dir = os.path.dirname(path)
        os.makedirs(self.root_dir, exist_ok=True)
        self.checkpoint_path = self.get_checkpoint_path()

    def get_model_root_dir(self) -> str:
        if self.root_dir:
            return self.root_dir
        log.error("MLX model path not set")
        return ""

    def get_model_path(self) -> str:
        return self.model_path

    def get_checkpoint_path(self) -> str:
        checkpoint_dir = f"/tmp/{self.name}/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, f"checkpoint.{self.model_ext}")

    # -----------------------------------------------------------------------
    # Setters (same API as ClassifierKeras)
    # -----------------------------------------------------------------------

    def set_combine_models(self, flag: bool):
        self.combine_models = flag

    def set_target_column(self, target_column: str):
        pass

    def set_num_epochs(self, num_epochs: int | None = None):
        self.num_epochs = num_epochs if num_epochs is not None else self.default_max_epochs

    def set_learning_rate(self, rate: float | None = None):
        self.learning_rate = max(rate, 1e-5) if rate is not None else self.default_learning_rate

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    # -----------------------------------------------------------------------
    # Model creation (subclasses override this to return an nn.Module)
    # -----------------------------------------------------------------------

    def create_model(self, seq_len: int, num_features: int) -> Optional[nn.Module]:
        """Subclasses must override this and return an mlx.nn.Module."""
        log.warning("create_model() should be defined by the subclass")
        return None

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, path: str = ""):
        if not path:
            path = self.model_path
        else:
            self.model_path = path

        if self.model is None:
            log.warning("save() called but model is None")
            return

        save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)

        print(f"    Saving MLX model to: {path}")
        self.model.save_weights(path)

    def load(self, path: str = "") -> nn.Module | None:
        if not path:
            path = self.model_path
        else:
            self.model_path = path

        if not os.path.exists(path):
            print(f"    MLX model not found ({path})...")
            self.new_model = True
            self.is_trained = False
            return None

        print(f"    Loading existing MLX model ({path})...")
        try:
            # Reconstruct architecture first, then load weights
            model = self.create_model(self.seq_len, self.num_features)
            if model is None:
                raise RuntimeError("create_model() returned None — cannot load weights")
                
            model.input_shape = (None, self.seq_len, self.num_features)
            model.load_weights(path)
            mx.eval(model.parameters())
            
            self.is_trained = True
            self.new_model  = False
            self.loaded_from_file = True
            self.model = model
            
            print(f"    MLX model loaded: {path}")
            return model
        except Exception as e:
            print(f"    Error loading MLX model from {path}: {e}")
            print(f"    Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load MLX model from {path}. Error: {e}") from e

    def update_model_weights(self):
        """Reload best checkpoint into self.model (mirrors ClassifierKeras)."""
        cp = self.get_checkpoint_path()
        if os.path.exists(cp) and self.model is not None:
            print(f"    Loading MLX checkpoint ({cp})...")
            self.model.load_weights(cp)
            mx.eval(self.model.parameters())
        else:
            print(f"    MLX checkpoint not found ({cp})")

    # -----------------------------------------------------------------------
    # State queries (same API as ClassifierKeras)
    # -----------------------------------------------------------------------

    def model_exists(self) -> bool:
        return os.path.exists(self.model_path)

    def model_is_trained(self) -> bool:
        if self.model is not None and self.is_trained:
            if not (self.combine_models and self.new_model):
                return True
        return False

    def new_model_created(self) -> bool:
        return ClassifierMLX.new_model

    def needs_clean_data(self) -> bool:
        return self.clean_data_required

    def needs_dataframes(self) -> bool:
        return self.requires_dataframes

    def prescale_data(self) -> bool:
        return self.prescale_dataframe

    def returns_single_prediction(self) -> bool:
        return self.single_prediction

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def nearest_power_of_2(self, n: int) -> int:
        return 2 ** n.bit_length()

    # Kept for API compatibility with ClassifierKeras
    def backtest(self, data):
        return self.predict(data)

    def mad_score(self, points):
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)
        return 0.6745 * ad / mad
