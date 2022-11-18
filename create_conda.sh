#!/usr/bin/env bash

set -e # Exit if anything fails

# Usage: `./create_conda.sh env_name env_name cuda_version(optional)`

# Handle arguments
if [ $# -lt 1 ] || [ $# -gt 2 ]
then
  echo "Incorrect arguments. Usage: ./create_conda.sh env_name cuda_version(optional)"
  exit 1
fi

if [ $# -gt 1 ]
then
  echo -e "\n#### Building 3DML environment with GPU support for CUDA version $2 ####\n"
  build="GPU"
else
  echo -e "\n#### Building 3DML environment for CPU only ####\n"
  build="CPU"
fi

if [ -d /home/$USER/miniconda3 ]; then
  source /home/$USER/miniconda3/etc/profile.d/conda.sh
elif [ -d /home/$USER/anaconda3 ]; then
  source /home/$USER/anaconda3/etc/profile.d/conda.sh
else
  echo "No valid conda install found!"
  exit 1
fi


conda activate base
conda create -n $1 -y python=3.9
conda activate $1

if [ $build == "CPU" ]
then
  export USE_CUDA=0
  conda install "pytorch==1.12.0" cpuonly -c pytorch -y
else
  export USE_CUDA=1
  export FORCE_CUDA=1
  conda install "pytorch>=1.12.0" "torchvision>=0.13.0" "cudatoolkit=$2" pyg pytorch-scatter -c pytorch -c pyg -y
fi

## other requirements
#pip install -r pip-requirements.txt

## Install OpenPoints stuff
## https://github.com/guochengqian/PointNeXt/blob/de51947c5ec64e9922801631b4af85a6ec0a6049/install.sh#L33-L54
#cd openpoints/cpp/pointnet2_batch
#python setup.py install
#cd ..
#
## grid_subsampling library. necessary only if interested in S3DIS_sphere
#cd subsampling
#python setup.py build_ext --inplace
#cd ..
#
#
## # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
#cd pointops/
#python setup.py install
#cd ..
#
## Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
#cd chamfer_dist
#python setup.py install --user
#cd ../emd
#python setup.py install --user

echo "The conda environment $1 was created successfully."