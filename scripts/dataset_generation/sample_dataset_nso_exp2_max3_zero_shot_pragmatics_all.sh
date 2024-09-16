#!/bin/bash

python src/dataset_generation/generate_boxes_data.py \
    --num_samples 2200 \
    --output_dir data/boxes_nso_exp2_max3_zero_shot_pragmatics \
    --num_operations 12 \
    --expected_num_items_per_box 2 \
    --max_items_per_box 3 \
    --include_modifiers always  \
    --omit_modifiers_in_ops always \
    --zero_shot



