import argparse
import os
import torch
from safetensors.torch import load_file as safe_load_file

from diffusers import DDPMNoiseShiftPipeline as DDPMPipeline


def main(args):
    device = args.device
    
    prior_means = torch.load(args.prior_mean_file_name)
    num_prior = prior_means.shape[0]
    
    total_num_images = args.num_images * num_prior
    
    # Load the pipeline from the pretrained model directory
    pipeline = DDPMPipeline.from_pretrained(args.model_dir).to(device)
    ema_state_dict = safe_load_file(
        os.path.join(args.model_dir, args.checkpoint_dir, "unet_ema", "diffusion_pytorch_model.safetensors")
    )
    pipeline.unet.load_state_dict(ema_state_dict)
    
    # Define file path for the full precision tensor file
    save_dir = os.path.join(args.model_dir, args.checkpoint_dir, args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fake_images.pt")
    
    fake_images_list = []
    
    for i in range(num_prior):
        
        fake_per_prior = None
        
        while True:
            print(f"Generating batch of {args.batch_size} images for prior index {i}.")
            prior_labels = torch.full((args.batch_size,), i)
            output = pipeline(
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                output_type="np",
                prior_labels=prior_labels,
                prior_means=prior_means,
            )
            # Convert numpy output (B, H, W, 3) to tensor (B, 3, H, W)
            new_batch = torch.from_numpy(output.images).permute(0, 3, 1, 2).to(device).float()
            
            if fake_per_prior == None:
                fake_per_prior = new_batch
            else:
                fake_per_prior = torch.cat([fake_per_prior, new_batch], dim=0)
                
            print(f"Generated ({fake_per_prior.shape[0]} / {args.num_images}) images")
            
            if fake_per_prior.shape[0] >= args.num_images:
                fake_images_list.append(fake_per_prior)
                break
            
    fake_images_tensor = torch.cat(fake_images_list, dim=0)
            
    with open(save_path, "wb") as f:
        torch.save(fake_images_tensor.cpu(), f)
        
    print(f"Total {fake_images_tensor.shape[0]} Images generation complete.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fake images using a specified prior index and save as a full precision tensor file."
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Subfolder inside model_dir containing the checkpoint (if any)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory (relative to model_dir) to save results")
    parser.add_argument("--num_images", type=int, default=10000,
                        help="number of images to generate per class")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for image generation")
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                        help="Number of denoising steps for image generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    parser.add_argument(
        "--prior_mean_file_name",
        type=str,
        default=None,
        help="The name of the pt file that the prior mean tensor is saved in."
    )
    args = parser.parse_args()
    main(args)