#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mlfoundations-dev/seed_math_allenai_math"
  "MergeBench/Llama-3.1-8B_coding"
  "MergeBench/Llama-3.1-8B_multilingual"
)

evaluate_all_models
