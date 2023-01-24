# Mac M1 Notes

## Installing Anaconda
> curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
> sh Miniconda3-latest-MacOSX-arm64.sh

## Installing tensorflow

> conda install -c apple tensorflow-deps
> pip install tensorflow-macos
> pip install tensorflow-metal
> conda install -c conda-forge -y pandas jupyter


## Installing pytorch

> conda install pytorch torchvision -c pytorch

## Installing darts

> pip install darts

For some reason, conda does not work. Probably a mac package incompatibility