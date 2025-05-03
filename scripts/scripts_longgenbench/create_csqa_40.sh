#!/bin/bash

# Set default values
K=40
PROMPT_PATH="data/LongGenBench_CSQA_prompt/LongGenBench_prompt_5demos.txt"
OUTPUT_DIR="data/LongGenBench/csqa"
QUESTION_LIMIT=1200

# Run the Python script with the specified arguments
python longgenbench_CSQA.py \
    --k $K \
    --prompt_path "$PROMPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --question_limit $QUESTION_LIMIT \
