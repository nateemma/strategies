# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
TrainSeedStrategyMixin — stamps a training seed onto the classifier so a strategy
can be retrained deterministically at different seeds (for seed-robust A/B).
Mix in *before* the NNNC base:

    class NNNC_DDPM_MLX_s1(TrainSeedStrategyMixin, NNNC_DDPM_MLX):
        train_seed = 1
"""


class TrainSeedStrategyMixin:
    train_seed = None

    # Feature 030: the model key (Framework/ModelKey.py). This class declares the names it
    # owns; the lists merge along the MRO. ATTRIBUTES/METHODS shape trained weights;
    # IGNORED act only after the model. The audit test keeps every name classified.
    MODEL_KEY_ATTRIBUTES = (
        "train_seed",
    )

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        clf = super().get_classifier(classifier_type, pair, seq_len, num_features)
        if clf is not None:
            clf.train_seed = self.train_seed
        return clf
