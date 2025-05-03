#!/bin/bash

# Set default values
K=30
DATA_DIR="data/MMLU"
OUTPUT_DIR="data/LongGenBench/mmlu/"

# Run the Python script with the specified arguments
python longgenbench_MMLU.py \
    --k $K \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
