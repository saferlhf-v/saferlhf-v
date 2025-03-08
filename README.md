## Introduction

This project is built on top of the [align-anything](https://github.com/PKU-Alignment/align-anything) framework. We introduce new features through the Safe RLHF-V method, enhancing the safety and performance of RLHF multi-modal training.

## Quick Start

### Easy Installation

```bash
# clone the repository
git clone git@github.com:saferlhf-v/saferlhf-v.git
cd saferlhf-v

# create virtual env
conda create -n saferlhf-v python==3.11
conda activate saferlhf-v
```

- **`[Optional]`** We recommend installing [CUDA](https://anaconda.org/nvidia/cuda) in the conda environment and set the environment variable.

```bash
# We tested on the H800 computing cluster, and this version of CUDA works well.
# You can adjust this version according to the actual situation of the computing cluster.

conda install nvidia/label/cuda-12.2.0::cuda
export CUDA_HOME=$CONDA_PREFIX
```

> If your CUDA installed in a different location, such as `/usr/local/cuda/bin/nvcc`, you can set the environment variables as follows:

```bash
export CUDA_HOME="/usr/local/cuda"
```

Finally, install `saferlhf-v` by:

```bash
# We prepare quick installation for training, you can use the following command:
pip install -e .
```


### Training

We provide some scripts for quick start, you can find them in the `./scripts` directory. These scripts would automatically download the model and dataset, and run the training or evaluation.

For example, `scripts/safe_rlhf_v.sh` is the script for Safe RLHF-V training, you can run it by:

```bash
cd scripts
bash safe_rlhf_v.sh
```


## Wandb Logger

We support `wandb` logging. By default, it is set to offline. If you need to view wandb logs online, you can specify the environment variables of `WANDB_API_KEY` before starting the training:

```bash
export WANDB_API_KEY="..."  # your W&B API key here
```


## Report Issues

If you have any questions in the process of using saferlhf-v, don't hesitate to ask your questions on [the GitHub issue page](https://github.com/saferlhf-v/saferlhf-v/issues/new/choose), we will reply to you in 2-3 working days.

# License

saferlhf-v is released under Apache License 2.0.
