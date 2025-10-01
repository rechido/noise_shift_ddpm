#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_dir "./20250714_bernoulli_seperate_0.05" \
    --checkpoint_dir "checkpoint-10" \
    --output_dir "generated_images" \
    --prior_mean_file_name "prior_means/bernoulli_seperateCH_3072dim_10point_constant0.05.pt" \
    --batch_size 10 \
    --num_images 10
