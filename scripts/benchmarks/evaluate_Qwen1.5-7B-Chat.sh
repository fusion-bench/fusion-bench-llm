#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "Qwen/Qwen1.5-7B-Chat"
  "abacusai/Liberated-Qwen1.5-7B"
  "YeungNLP/firefly-qwen1.5-en-7b"
)

LM_EVAL_ARGS="--apply_chat_template"

evaluate_all_models
