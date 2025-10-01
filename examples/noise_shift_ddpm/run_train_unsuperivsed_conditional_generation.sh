output_dir='./20250308'

export CUDA_VISIBLE_DEVICES="1"

accelerate launch --num_processes=1 train_unsuperivsed_conditional_generation.py \
    --dataset_name=cifar10 \
    --logger="wandb" \
    --checkpointing_steps=99999999 \
    --resolution=32 \
    --train_batch_size=64 \
    --eval_batch_size=10 \
    --num_epochs=99999999 \
    --save_images_epochs=10 \
    --save_model_epochs=10 \
    --output_dir=$output_dir \
    --checkpointing_steps=10 \
    --resume_from_checkpoint="latest" \
    --use_ema \
    --prior_mean_file_name="prior_means/bernoulli_seperateCH_3072dim_10point_constant0.05.pt"
