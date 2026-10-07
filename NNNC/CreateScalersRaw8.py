# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore

"""CreateScalersRaw8 — builds the isolated main_scaler_fwdmae_raw8 scaler for
NNNC_ForwardMAE_Raw.

Same scaler-creation pipeline as Framework.CreateScalers but with the
raw8-only include_list and isolated scaler/tensor/PCA names, so the shared
production main_scaler (saved_data/main_scaler.pkl) is never overwritten.
Uses the SAME _raw8_features.add_raw8_features() helper as the strategy, so
the scaler and the strategy's model input can never drift apart.

`CreateScalers.populate_indicators()` unconditionally removes the OLD scaler
file for BOTH `self.main_scaler_name` and `self.gan_scaler_a_name` before
fitting. Every name this class could delete is therefore overridden below --
including `gan_scaler_a_name`, which this arm does not use and has no reason
to destroy. Other `CreateScalers` subclasses leave it at the default and so
delete the shared `saved_data/gan_scaler_a.pkl` as a side effect; that is
existing behaviour elsewhere, not a reason to repeat it here.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from pandas import DataFrame

from Framework.CreateScalers import CreateScalers
from _raw8_features import add_raw8_features, RAW8_COLUMNS


class CreateScalersRaw8(CreateScalers):
    include_list = list(RAW8_COLUMNS)

    # rolling_dataframe_normalise uses main_scaler_name; CreateScalers' own
    # tensor/PCA save paths use tensor_scaler_name/pca_name -- override all
    # so nothing lands in a shared production scaler file.
    main_scaler_name = "main_scaler_fwdmae_raw8"
    scaler_name = "main_scaler_fwdmae_raw8"
    tensor_scaler_name = "fwdmae_raw8_tensor_scaler"
    pca_name = "fwdmae_raw8_pca_data"
    # Not used by this arm. Overridden purely so the parent's unconditional
    # remove_old_scaler(self.gan_scaler_a_name) cannot delete the shared
    # saved_data/gan_scaler_a.pkl, which the GAN-based NNNC/NNMT arms need.
    gan_scaler_a_name = "fwdmae_raw8_gan_scaler_a_unused"

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Add the raw8 features instead of the default regime/risk/flow/
        momentum set (see NNNC_ForwardMAE_Raw.add_additional_indicators for
        why skipping those is safe here)."""
        return add_raw8_features(dataframe)
