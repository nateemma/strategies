# utility to print out current environment

import multiprocessing
import sys
import platform
import freqtrade

# Note that we have to surround import with try/except since not all strategies require all of these packages

try:
    import tensorflow as tf
    tf_installed = True
except ModuleNotFoundError:
    tf_installed = False

try:
    #import keras
    keras_installed = True
except ModuleNotFoundError:
    keras_installed = False

try:
    import sklearn
    sklearn_installed = True
except ModuleNotFoundError:
    sklearn_installed = False

try:
    import torch
    torch_installed = True
except ModuleNotFoundError:
    torch_installed = False

try:
    import pytorch_lightning
    lightning_installed = True
except ModuleNotFoundError:
    lightning_installed = False

try:
    import darts
    darts_installed = True
except ModuleNotFoundError:
    darts_installed = False


def print_environment():

    NOT_INSTALLED = "(not installed)"

    # freqtrade
    freqtrade_version = freqtrade.__version__

    # OS info
    os_type = sys.platform
    if os_type.lower() == "darwin":
        os_type = os_type + " (MacOS)"
    os_version = platform.platform()
    num_cpus = multiprocessing.cpu_count()

    # Python
    python_version = sys.version.split('\n')

    # sklearn
    if sklearn_installed:
        sklearn_version = sklearn.__version__
    else:
        sklearn_version = NOT_INSTALLED

    # Tensorflow
    if tf_installed:
        tf_version = tf.__version__
        tf_devices = tf.config.get_visible_devices()
    else:
        tf_version = NOT_INSTALLED

    # keras
    if keras_installed:
        keras_version = tf.keras.__version__
    else:
        keras_version = NOT_INSTALLED

    # pytorch
    if torch_installed:
        torch_version = torch.__version__
    else:
        torch_version = NOT_INSTALLED

    # pytorch lightning
    if lightning_installed:
        lightning_version = pytorch_lightning.__version__
    else:
        lightning_version = NOT_INSTALLED

    # darts
    if darts_installed:
        darts_version = darts.__version__
    else:
        darts_version = NOT_INSTALLED

    print("")
    print("Software Environment:")
    print("")
    print(f"    freqtrade:  {freqtrade_version}")
    print(f"    OS Type:    {os_type}, Version: {os_version}")
    print(f"    python:     {python_version}")
    print(f"    sklearn:    {sklearn_version}")
    print(f"    tensorflow: {tf_version}, devices:{tf_devices}")
    print(f"    keras:      {keras_version}")
    print(f"    pytorch:    {torch_version}")
    print(f"    lightning:  {lightning_version}")
    print(f"    darts:      {darts_version}")
    print("")


# checks whether a package is installed
def package_installed(package) -> bool:
    try:
        dist = pkg_resources.get_distribution(package)
        installed = True
    except pkg_resources.DistributionNotFound:
        installed = False
    return installed
