#!/usr/bin/env bash

set -e # Exit if anything fails

# Install OpenPoints stuff
# https://github.com/guochengqian/PointNeXt/blob/de51947c5ec64e9922801631b4af85a6ec0a6049/install.sh#L33-L54
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ..

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
python setup.py build_ext --inplace
cd ..


# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
python setup.py install
cd ..

# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
