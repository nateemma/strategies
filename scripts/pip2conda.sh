#!/bin/zsh

# script to 'move' packages from pip to conda
# I need to do this because of spotty Mac M1 support on pip channels

# 1. replace pip/pypi packages in the conda environment
for lib in `conda list | grep 'pypi' | cut -f 1 -d ' '`; \
do
  echo "*****checking $lib*****"; \
  conda install -y -c conda-forge $lib && \
  pip uninstall -y $lib; \
done


#2. Replace outdated pip packages with conda equivalents
for lib in `pip list -o | grep '.' | cut -f 1 -d ' '`; \
do
  echo "*****checking $lib*****"; \
  conda install -y -c conda-forge $lib && \
  pip uninstall -y $lib; \
done

#3. anything left in pip, update in pip
echo ""
echo "deactivate your environment then re-enter
echo "run pip list -o"
echo "run pip install ,package> for anything in the list""