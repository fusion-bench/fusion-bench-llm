#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "Qwen/Qwen2.5-1.5B"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-Math-1.5B"
  "Qwen/Qwen2.5-Coder-1.5B"
)

evaluate_all_models
