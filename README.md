<div align="center">

# 3D-ML

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

What it does

## Installation

```bash
# clone project
git clone https://github.com/CCInc/3d-ml.git && cd 3d-ml


# Create a fresh conda environment
conda activate base
conda create -n 3dml_env -y python=3.9
conda activate 3dml_env

# For GPU support
nvidia-smi # To make sure you have drivers and CUDA installed, also gives your CUDA version

# Install pytorch
# Pick a compatible combination of pytorch and cuda version from https://pytorch.org/
# Below is an example for CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y

# Install pyg
conda install pyg pytorch-scatter -c pyg -y

# install other requirements
pip install -r requirements.txt

# install openpoints
./install_openpoints.sh
```

## Run tests

## Run a simple training

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu task=segmentation

# train on GPU
python src/train.py trainer=gpu task=segmentation
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.datamodule.batch_size=64
```

## Contribute
