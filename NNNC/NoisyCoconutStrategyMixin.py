# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NoisyCoconutStrategyMixin — stamps NoisyCoconut inference params onto the
classifier the strategy builds, so the (sigma, K, seed) sweep is driven by
strategy class attributes (edit one attr, rerun) rather than a class explosion
of predictors. Mix in *before* the NNNC base so this get_classifier wins:

    class NNNC_DDPM_MLX_Noisy(NoisyCoconutStrategyMixin, NNNC_DDPM_MLX):
        classifier_type = ClassifierTypeMLX.LSTM_NOISY
        noisy_sigma = 0.1
"""


class NoisyCoconutStrategyMixin:
    # Swept via edit-and-rerun; forwarded onto the classifier instance below.
    noisy_sigma = 0.1
    noisy_k = 16
    noisy_seed = 42

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        clf = super().get_classifier(classifier_type, pair, seq_len, num_features)
        if clf is not None:
            clf.noisy_sigma = self.noisy_sigma
            clf.noisy_k = self.noisy_k
            clf.noisy_seed = self.noisy_seed
        return clf
