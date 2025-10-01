#!/usr/bin/env python3

import argparse
import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets import load_dataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from evaluate_util import compute_fid, compute_prdcf1


def get_augmentations(args):
    return transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])


def transform_images(augmentations):
    def _transform(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}
    return _transform


def prepare_dataset(args, split):
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, split=split)
    else:
        data_dir = args.train_data_dir if split == "train" else args.test_data_dir
        dataset = load_dataset("imagefolder", data_dir=data_dir, cache_dir=args.cache_dir, split=split)
    
    aug = get_augmentations(args)
    dataset = dataset.with_transform(transform_images(aug))
    return dataset



def get_real_images(loader, num_to_use, device):
    """
    Collect real images from the loader. The loader yields images in [-1,1].
    Returns exactly num_to_use images as a tensor of shape (N, 3, H, W) in [-1,1].
    """
    real_list = []
    count = 0
    for batch in loader:
        images = batch["input"].to(device)
        needed = num_to_use - count
        if images.shape[0] > needed:
            images = images[:needed]
        real_list.append(images)
        count += images.shape[0]
        if count >= num_to_use:
            break
    return torch.cat(real_list, dim=0)


def denormalize_minus1_to_0_1(images: torch.Tensor):
    """
    Convert images from [-1,1] to [0,1].
    """
    return (images + 1) / 2


def main(args):
    device = args.device
    
    evaluation_dir = os.path.join(args.model_dir, args.output_dir)
    os.makedirs(evaluation_dir, exist_ok=True)
    
    train_dataset = prepare_dataset(args, split="train")
    test_dataset = prepare_dataset(args, split="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.dataloader_num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.dataloader_num_workers)

    max_fake = max(args.num_generated_train, args.num_generated_test)
    
    fake_dir = os.path.join(args.model_dir, args.output_dir)
    fake_path = os.path.join(fake_dir, "fake_images.pt")
    
    print(f"\nLoading fake images from {fake_path} ...")
    
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake images file not found at {fake_path}")
    with open(f"{fake_path}", "rb") as f:
        fake_images = torch.load(f, map_location=device)
    fake_images = fake_images[:max_fake].to(device)
    fake_01 = fake_images[:args.num_generated_train]

    real_train = get_real_images(train_loader, args.num_generated_train, device)
    real_test  = get_real_images(test_loader, args.num_generated_test, device)
    real_train_01 = denormalize_minus1_to_0_1(real_train)
    real_test_01  = denormalize_minus1_to_0_1(real_test)
    
    
    # -------------------------------------------------------------------------------------------
    ### Evaluate FID ###
    # -------------------------------------------------------------------------------------------

    print(f"Computing FID scores for variant '{fake_path}' on TRAIN set...")
    fid_train = compute_fid(real_train_01, fake_01, device=device)
    print(f"Computing FID scores for variant '{fake_path}' on TEST set...")
    fid_test = compute_fid(real_test_01, fake_01, device=device)

    train_output_file = os.path.join(evaluation_dir, f"fid_train_fake{args.num_generated_train}.txt")
    with open(train_output_file, "w") as f:
        f.write(f"{fid_train:.6f}\n")
    print(f"\nFID (TRAIN) results saved to {train_output_file}")

    test_output_file = os.path.join(evaluation_dir, f"fid_test_fake{args.num_generated_train}.txt")
    with open(test_output_file, "w") as f:
        f.write(f"{fid_test:.6f}\n")
    print(f"FID (TEST) results saved to {test_output_file}")
    
    # -------------------------------------------------------------------------------------------
    ### Evaluate prdcf1 ###
    # -------------------------------------------------------------------------------------------
    k_set = [3]
    train_output_file = os.path.join(evaluation_dir, f"prdcf1_train_fake{args.num_generated_train}.txt")
    compute_prdcf1(real_train_01, fake_01, k_values=k_set, output_file=train_output_file)
            
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FID for MNIST using a pretrained diffusion model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory (as saved with pipeline.save_pretrained)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory (relative to model_dir) to save results")
    parser.add_argument("--dataset_name", type=str, default="cifar10",
                        help="Name of the dataset to use (loaded via Hugging Face datasets); set to None if using imagefolder")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Dataset config name if applicable")
    parser.add_argument("--train_data_dir", type=str, default=None,
                        help="Path to training images if not using a named dataset")
    parser.add_argument("--test_data_dir", type=str, default=None,
                        help="Path to test images if not using a named dataset")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for loading the dataset")
    parser.add_argument("--resolution", type=int, default=32,
                        help="Image resolution used during training")
    parser.add_argument("--center_crop", action="store_true",
                        help="Use center crop (otherwise random crop)")
    parser.add_argument("--random_flip", action="store_true",
                        help="Apply random horizontal flip")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for evaluation")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of workers for the dataloader")
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                        help="Number of denoising steps for image generation")
    parser.add_argument("--num_generated_train", type=int, default=50000,
                        help="Desired number of images to use from the training set for FID evaluation")
    parser.add_argument("--num_generated_test", type=int, default=10000,
                        help="Desired number of images to use from the test set for FID evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    args = parser.parse_args()
    main(args)
