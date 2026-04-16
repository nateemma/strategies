"""
pytest configuration for GANs/tests/.

Overrides the freqtrade root pyproject.toml which sets
    addopts = "--dist loadscope"
That activates pytest-xdist distribution mode.  Without an explicit -n
count, xdist collects 0 items for any test that isn't skip-wrapped at
import time — which breaks the RUN_SLOW_TESTS quality test classes.

Setting dist = "no" here disables distribution for this subdirectory so
both the fast functional tests and the slow quality tests run correctly.
"""


def pytest_configure(config):
    # Disable xdist distribution for this test directory.
    # The GAN functional and quality tests are sequential by design:
    # each class clears TF state in setUpClass / setUp.  Running them
    # across xdist workers would break that isolation.
    if hasattr(config, "workerinput"):
        return   # already inside an xdist worker — leave alone
    try:
        config.option.dist = "no"
    except AttributeError:
        pass   # xdist not installed; nothing to do
