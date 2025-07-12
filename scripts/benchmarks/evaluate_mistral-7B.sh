#! /bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "mistralai/Mistral-7B-v0.1"
  "meta-math/MetaMath-Mistral-7B"
  "cognitivecomputations/dolphin-2.1-mistral-7b"
  "uukuguy/speechless-code-mistral-7b-v1.0"
)

evaluate_all_models
