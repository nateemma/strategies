"""Ablation rung config for the no-GAN include_list minimization study.

Selects the rung's include_list via the ABLATION_RUNG env var so the
experimental strategy (NNNC_Ablate) and its scaler builder
(CreateScalersAblate) share a single source of truth. Throwaway /
experimental — NOT part of the production strategy set.

Rungs (drops are cumulative), from the 2026-06-30 signal x redundancy
screening across XRP/SOL/LINK at gbb / H=48 / thr=0.007:
  BASELINE : current 24-feature production include_list
  A        : drop redundant oscillators (dups of cci_scaled / rsi_scaled)
  B        : A + low-signal tail (cg_ss, fast_diff, fisher_ss)
  C        : B + soft-cluster trim (keep atr_norm, gain_norm, ema_fast_norm)
"""

from __future__ import annotations

import os

# Current production active include_list (24 features) = BASELINE.
_BASELINE = [
    "adx_scaled", "aroonosc_scaled", "atr_norm", "bb_position", "bb_width",
    "cci_scaled", "di_diff_scaled", "ema_fast_norm", "fast_diff",
    "fastk_scaled", "fisher_ss", "cg_ss", "gain_norm", "guard_metric_pos",
    "guard_metric_neg", "macd_pos", "macd_neg", "macdhist_norm", "mfi_scaled",
    "rsi_scaled", "sar_ratio", "spread_ma", "vwap_pos", "vwap_neg",
]

_DROP_A = ["fastk_scaled", "bb_position"]
_DROP_B = _DROP_A + ["cg_ss", "fast_diff", "fisher_ss"]
_DROP_C = _DROP_B + ["spread_ma", "macdhist_norm"]
# Rung D: also drop the 3 features whose only barrier was the over-strict
# check_columns_included guard (atr_norm/di_diff_scaled/aroonosc_scaled).
# Their derived regime/flow/momentum/risk labels still compute from the
# dataframe; this only removes them as direct model input. Moderate signal
# (0.22-0.24), so unlike the earlier free drops this may cost val_mcc.
_DROP_D = _DROP_C + ["atr_norm", "di_diff_scaled", "aroonosc_scaled"]


def _drop(base: list[str], drop: list[str]) -> list[str]:
    return [c for c in base if c not in drop]


RUNGS = {
    "BASELINE": list(_BASELINE),
    "A": _drop(_BASELINE, _DROP_A),
    "B": _drop(_BASELINE, _DROP_B),
    "C": _drop(_BASELINE, _DROP_C),
    "D": _drop(_BASELINE, _DROP_D),
}


class SkipColumnCheck:
    """Neutralise the over-strict check_columns_included guard for the study.

    get_risk_level/get_flow/get_momentum read their source column from the
    dataframe (which keeps every populated column), so requiring the column
    to also be in include_list is a spurious assertion — it only blocks
    dropping a feature from MODEL INPUT. Mixed into the throwaway ablation
    classes so production BaseNNStrategy is untouched.
    """

    def check_columns_included(self, required_columns, function_name):
        return None


ABLATE_STORAGE_SUBDIR = "_ablate"


class AblateGANMixin(SkipColumnCheck):
    """Isolation for the GAN-chain phase.

    Overrides get_storage_location to a dedicated subdir so the scaler, GAN,
    discriminators and model all land under saved_data/_ablate/ — the GAN
    save/load path has no name hook (only type under the storage dir), so
    redirecting storage is the only clean way to avoid clobbering the global
    production tab_ddpm GAN. Production saved_data/ is untouched.
    """

    def get_storage_location(self) -> str:
        base = super().get_storage_location()
        path = os.path.join(base, ABLATE_STORAGE_SUBDIR) + os.sep
        os.makedirs(path, exist_ok=True)
        return path


def current_rung() -> str:
    return os.environ.get("ABLATION_RUNG", "BASELINE").upper()


def current_include_list() -> list[str]:
    rung = current_rung()
    if rung not in RUNGS:
        raise ValueError(f"Unknown ABLATION_RUNG={rung}; expected {list(RUNGS)}")
    return list(RUNGS[rung])
