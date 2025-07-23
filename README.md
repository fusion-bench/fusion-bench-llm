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
git clone https://github.com/fusion-bench/fusion-bench-llm
cd fusion-bench-llm

# [OPTIONAL] create conda environment
conda create -n fbllm python=3.12
conda activate fbllm

# install pytorch according to instructions
# https://pytorch.org/get-started/

# 1. install fusion-bench
pip install fusion-bench
# or install fusion-bench in edit mode (recommended for development)
git clone https://github.com/tanganke/fusion_bench third_party/fusion_bench
pip install -e third_party/fusion_bench

# 2. install fusion-bench-llm
cd .. # go back to fusion-bench-llm directory
pip install -e .
```

## How to run

```shell
fusion_bench_llm \
    # method=...
    # method.option_1=...
    # modelpool=...
    # ...
```
