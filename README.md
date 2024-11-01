<div align="center">

# FusionBench-LLM

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/fusion-bench/fusion-bench-project-template"><img alt="Template" src="https://img.shields.io/badge/-FusionBench--Project--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

A Collection of examples for large language model fusion, based on [FusionBench](https://github.com/tanganke/fusion_bench/).

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -n myenv python=3.12

# activate conda environment
conda activate myenv
```

## How to run

run my method

```shell
fusion_bench \
    --config-path $PWD/config --config-name main \
    # method=...
    # method.option_1=...
    # modelpool=...
    # ...
```

or

```shell
bash scripts/run_experiments.sh
```
