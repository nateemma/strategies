# Mac M1 Notes

Installing on an M1 machine can be a real pain in the rear end. Below are some notes that will hopefully help if you are
trying to do this. Note that _pip_ means _pip3_ (I have _pip_ aliased to _pip3_)

Also, there is a script _install_packages.sh_ in _user_data/strategies/scripts_, that will do all of this for you

*WARNING*: If you are running on an Apple processor (M1 etc) then there will likely be some dependency issues. The Mac M1 versions of libraries are generally only available via conda, and they do not keep up wit the pace of other packages such as python and tensorflow. In partuclar, be careful about upgrading python. At the time of writing, the latest version of python is 3.12, but the M1-based version of tensorflow-deps and tensorflow-macos only support python 3.10

## Installing Anaconda
You should only need to do this once
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```

## General Packages
Run _pip(3) install_ on the following packages (these are not Mac-specific):

- finta
- prettytable
- PyWavelets
- simdkalman
- pykalman
- scipy
- scikit-learn

Of course, there may be package conflicts. Right now, there is a conflict with _numba_ and _numpy_, so I have to do this:

```
  conda install numba; pip3 uninstall numba
  pip3 install numpy<1.24 # obviously check version if packages update
```

## Installing tensorflow

```

conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
conda install -c conda-forge -y pandas jupyter
```

## Installing pytorch

```
conda install pytorch torchvision -c pytorch
```

## Installing darts

```
pip install darts
```

For some reason, conda does not work. Probably a mac package incompatibility

##talib
Often, updating freqtrade fails with an error building _talib_. If that happens, try this:

```
pip3 uninstall talib
brew install talib
pip3 install talib
```

(it may be _ta-lib_)