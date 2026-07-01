"""CreateScalersAblate — builds isolated scalers for the NNNC_Ablate study.

Same scaler-creation pipeline as CreateScalers but with the per-rung
include_list (ABLATION_RUNG) and isolated scaler names, so the production
global main_scaler / main_tensor_scaler are never overwritten.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.CreateScalers import CreateScalers
from _ablation_config import current_include_list, SkipColumnCheck


class CreateScalersAblate(SkipColumnCheck, CreateScalers):
    include_list = current_include_list()

    # rolling_dataframe_normalise uses main_scaler_name; CreateScalers'
    # tensor/PCA save paths use tensor_scaler_name. Override both so the
    # experimental scalers land in their own files.
    main_scaler_name = "exp_scaler"
    scaler_name = "exp_scaler"
    tensor_scaler_name = "exp_tensor_scaler"
    pca_name = "exp_pca_data"
