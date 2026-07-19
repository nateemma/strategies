# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
PonderStrategyMixin — forwards ``ponder_steps`` onto the classifier the strategy
builds, so the N sweep is driven by a strategy class attribute (edit one attr,
retrain under a distinct name).

Unlike NoisyCoconutStrategyMixin, this does NOT override get_model_path: the
ponder model has new params and must train its own weights (it cannot reuse the
production weights). The shared TabDDPM GAN + scalers are reused automatically
(they live at the shared saved_data/ root), so the only variable vs production is
the ponder head. Mix in *before* the NNNC base:

    class NNNC_DDPM_MLX_Ponder(PonderStrategyMixin, NNNC_DDPM_MLX):
        classifier_type = ClassifierTypeMLX.LSTM_PONDER
        ponder_steps = 3
"""


class PonderStrategyMixin:
    # Number of shared refinement steps; ponder_steps=0 == production arch.
    ponder_steps = 3
    # Optional training-seed override for robustness sweeps (None = default).
    train_seed = None

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        clf = super().get_classifier(classifier_type, pair, seq_len, num_features)
        if clf is not None:
            clf.ponder_steps = self.ponder_steps
            clf.train_seed = self.train_seed
        return clf
