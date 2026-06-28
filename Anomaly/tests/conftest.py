"""
pytest configuration for Anomaly/tests/.

Overrides the freqtrade root pyproject.toml which sets
    addopts = "--dist loadscope"
Without an explicit -n count, xdist collects 0 items for tests that are not
skip-wrapped at import time. Disabling distribution here lets the smoke test
collect and run normally.
"""


def pytest_configure(config):
    if hasattr(config, "workerinput"):
        return   # already inside an xdist worker — leave alone
    try:
        config.option.dist = "no"
    except AttributeError:
        pass   # xdist not installed; nothing to do
