#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-Math-7B"
  "Qwen/Qwen2.5-Coder-7B"
)

evaluate_all_models
