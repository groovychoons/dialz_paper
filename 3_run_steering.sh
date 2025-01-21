#!/bin/bash

# List of models
MODELS=(
  "mistralai/Mistral-7B-Instruct-v0.1"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

# Loop through each model and run the Python script
for MODEL in "${MODELS[@]}"; do
  echo "Running script for model: $MODEL"
  python 3_steering_optimisation.py "$MODEL"
done