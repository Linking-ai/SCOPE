#!/bin/bash

# Set default values
K=60
PROMPT_PATH="data/LongGenBench_GSM8K_prompt/LongGenBench_prompt_8demos.txt"
OUTPUT_DIR="data/LongGenBench/gsm8k"
QUESTION_LIMIT=1260

# Run the Python script with the specified arguments
python longgenbench_GSM8K.py \
    --k $K \
    --prompt_path "$PROMPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --question_limit $QUESTION_LIMIT \
