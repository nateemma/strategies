# utility class to print out current environment

import multiprocessing
import sys
import platform

import pkg_resources
import freqtrade

class Environment:
        
    tf_installed = False
    keras_installed = False
    sklearn_installed = False
    torch_installed = False
    lightning_installed = False
    darts_installed = False


    def __init__(self):

        # Note that we have to surround import with try/except since not all strategies require all of these packages

        try:
            import tensorflow as tf
            self.tf_installed = True
        except ModuleNotFoundError:
            self.tf_installed = False

        try:
            import keras
            self.keras_installed = True
        except ModuleNotFoundError:
            self.keras_installed = False

        try:
            import sklearn
            self.sklearn_installed = True
        except ModuleNotFoundError:
            self.sklearn_installed = False

        try:
            import torch
            self.torch_installed = True
        except ModuleNotFoundError:
            self.torch_installed = False

        try:
            import pytorch_lightning
            self.lightning_installed = True
        except ModuleNotFoundError:
            self.lightning_installed = False

        try:
            import darts
            self.darts_installed = True
        except ModuleNotFoundError:
            self.darts_installed = False


    def print_environment(self):

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
        if self.sklearn_installed:
            import sklearn
            sklearn_version = sklearn.__version__
        else:
            sklearn_version = NOT_INSTALLED

        # Tensorflow
        if self.tf_installed:
            import tensorflow as tf
            tf_version = tf.__version__
            tf_devices = tf.config.get_visible_devices()
        else:
            tf_version = NOT_INSTALLED

        # keras
        if self.keras_installed:
            import keras
            keras_version = keras.__version__
        else:
            keras_version = NOT_INSTALLED

        # pytorch
        if self.torch_installed:
            import torch
            torch_version = torch.__version__
        else:
            torch_version = NOT_INSTALLED

        # pytorch lightning
        if self.lightning_installed:
            import pytorch_lightning
            lightning_version = pytorch_lightning.__version__
        else:
            lightning_version = NOT_INSTALLED

        # darts
        if self.darts_installed:
            import darts
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
    def package_installed(self, package) -> bool:
        try:
            dist = pkg_resources.get_distribution(package)
            installed = True
        except pkg_resources.DistributionNotFound:
            installed = False
        return installed
