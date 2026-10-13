# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore

"""
NoisyCoconutStrategyMixin — wires a NoisyCoconut A/B strategy to REUSE an
already-trained model (no retrain) and forwards the (sigma, K, seed) sweep
params onto the classifier it builds, so the sweep is driven by strategy class
attributes (edit one attr, rerun) rather than a class explosion of predictors.

Mix in *before* the NNNC base so these overrides win the MRO:

    class NNNC_DDPM_MLX_Noisy(NoisyCoconutStrategyMixin, NNNC_DDPM_MLX):
        classifier_type = ClassifierTypeMLX.LSTM_NOISY
        reuse_model_from = "NNNC_DDPM_MLX"
        noisy_sigma = 0.1

``get_model_path`` normally keys off the strategy class name
(``saved_data/<class>/<class>.keras``). We redirect it to ``reuse_model_from``
so the wrapper loads the production weights instead of retraining a fresh model
under its own name. Scalers live at the shared ``saved_data/`` root
(``get_storage_location``) so they resolve unchanged; the GAN chain is
training-time only and is never invoked on a load-and-predict backtest.
"""


class NoisyCoconutStrategyMixin:
    # Name of the trained strategy whose weights this wrapper reuses.
    reuse_model_from = "NNNC_DDPM_MLX"

    # Swept via edit-and-rerun; forwarded onto the classifier instance below.
    noisy_sigma = 0.1
    noisy_k = 16
    noisy_seed = 42

    # Feature 030: the model key (Framework/ModelKey.py). This class declares the names it
    # owns; the lists merge along the MRO. ATTRIBUTES/METHODS shape trained weights;
    # IGNORED act only after the model. The audit test keeps every name classified.
    MODEL_KEY_IGNORED = (
        "noisy_k", "noisy_sigma", "noisy_seed", "reuse_model_from",
    )

    def get_model_path(self) -> str:
        # Reuse the production model dir instead of this class's own name.
        root_dir = self.get_storage_location()
        name = self.reuse_model_from
        return root_dir + name + "/" + name + ".keras"

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        clf = super().get_classifier(classifier_type, pair, seq_len, num_features)
        if clf is not None:
            clf.noisy_sigma = self.noisy_sigma
            clf.noisy_k = self.noisy_k
            clf.noisy_seed = self.noisy_seed
        return clf
