# type: ignore
# pylint: disable=import-error
"""NNWavelet — the MLX multi-output MLP wavelet-reconstruction strategy (the
"neural" reference). Thin leaf over NNWaveletStrategy so the family has a class
named after the dir. NOTE: NNWavelet_Ridge is the RECOMMENDED predictor — it
matched/beat this MLX model on walk-forward OOS and is deterministic; the MLP
does not beat the linear floor here."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from NNWaveletStrategy import NNWaveletStrategy

import utils.Wavelets as Wavelets


class NNWavelet_DWTA(NNWaveletStrategy):
    
    wavelet_type = Wavelets.WaveletType.DWTA
