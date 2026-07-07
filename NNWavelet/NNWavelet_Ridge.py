# type: ignore
# pylint: disable=import-error
"""NNWavelet_Ridge — NNWavelet with the multi-output Ridge forecaster.

RECOMMENDED predictor for this family. At the tuned g5 gate (z3.0/p0.97) it
matched-or-beat the MLX MLP on walk-forward OOS (Ridge +0.65% mean, 3/4 windows
vs MLX +0.55%, 2/4) on fewer trades, and — being closed-form — it is
deterministic and reproducible where the MLX fit is a per-retrain lottery
(+1.11% vs +0.55% across two fits of the same config). The nonlinear model does
not beat the linear floor here, consistent with the wider NNPredict family."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from NNWaveletStrategy import NNWaveletStrategy
from WaveletForecaster import WaveletRegressorType


class NNWavelet_Ridge(NNWaveletStrategy):
    def get_classifier_type(self):
        return WaveletRegressorType.RIDGE
