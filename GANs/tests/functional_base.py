"""
functional_base — no longer used.

All GAN types now use the fit() / generate() / save() / load() lifecycle.
The FitGenSuiteConfig factories in test_functional_suite.py supersede the
OutputContractMixin, SaveLoadMixin, and IntegrationMixin that lived here.

Kept as an empty module so existing imports do not break during the transition.
"""

# Training constants kept for backwards compatibility (unused internally).
FAST_EPOCHS   = 2
FAST_BATCH    = 16
FAST_N_CRITIC = 1
