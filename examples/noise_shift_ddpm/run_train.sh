output_dir='./20250714_bernoulli_seperate_0.05'

export CUDA_VISIBLE_DEVICES="1"

accelerate launch --num_processes=1 train.py \
    --dataset_name=cifar10 \
    --logger="wandb" \
    --resolution=32 \
    --train_batch_size=64 \
    --eval_batch_size=10 \
    --num_epochs=99999999 \
    --save_images_epochs=1 \
    --save_model_epochs=99999999 \
    --output_dir=$output_dir \
    --checkpointing_steps=10 \
    --resume_from_checkpoint="latest" \
    --use_ema \
    --prior_mean_file_name="prior_means/bernoulli_seperateCH_3072dim_10point_constant0.05.pt"

