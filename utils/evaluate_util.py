from tqdm import tqdm

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from torcheval.metrics import FrechetInceptionDistance as TorchevalFID

from precision_and_recall.prdc.prdc import compute_prdc

# Set a default batch size for FID updates
FID_BATCH_SIZE = 100

def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor, device: str = "cpu") -> float:
    """
    Compute FID using torcheval.metrics.FrechetInceptionDistance in batches.
    Expects images as float32 in [0,1] of shape (N, 3, H, W).
    """
    fid_metric = TorchevalFID().to(device)
    
    num_real = real_images.shape[0]
    for i in tqdm(range(0, num_real, FID_BATCH_SIZE), desc="Torcheval (real)", leave=False):
        batch = real_images[i:i+FID_BATCH_SIZE].to(device, dtype=torch.float32).clamp(0, 1)
        fid_metric.update(batch, is_real=True)
    
    num_fake = fake_images.shape[0]
    for i in tqdm(range(0, num_fake, FID_BATCH_SIZE), desc="Torcheval (fake)", leave=False):
        batch = fake_images[i:i+FID_BATCH_SIZE].to(device, dtype=torch.float32).clamp(0, 1)
        fid_metric.update(batch, is_real=False)
    
    print("Torcheval: Starting fid_metric.compute()...")
    return float(fid_metric.compute().item())           
            

def extract_features(images: torch.Tensor, model: nn.Module, input_size: int, batch_size: int = 64) -> torch.Tensor:
    """
    Extract features from images in batches with a progress bar.
    
    Args:
        images (torch.Tensor): Tensor of images, shape (N, C, H, W), normalized to [0,1].
        model (nn.Module): Pretrained feature extractor.
        input_size (int): The spatial resolution required by the model (e.g. 299 for Inception, 224 for VGG16).
        batch_size (int): Batch size for processing.
    
    Returns:
        torch.Tensor: Extracted features of shape (N, feature_dim).
    """
    # Move images to CPU first to avoid loading everything on GPU
    images = images.cpu()
    n_images = images.size(0)
    features_list = []
    
    # Prepare normalization tensors on the device of the model
    device = next(model.parameters()).device
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    for i in tqdm(range(0, n_images, batch_size), desc="Extracting features for PRDC computation"):
        batch = images[i:i+batch_size].to(device)
        # Resize if necessary
        if batch.shape[-1] != input_size or batch.shape[-2] != input_size:
            batch = F.interpolate(batch, size=(input_size, input_size),
                                  mode='bilinear', align_corners=False)
        # Normalize using ImageNet statistics
        batch = (batch - imagenet_mean) / imagenet_std
        with torch.no_grad():
            feats = model(batch)
        # Flatten features to 2D [batch_size, feature_dim]
        feats = feats.view(feats.size(0), -1)
        features_list.append(feats.cpu())
    
    return torch.cat(features_list, dim=0)

def compute_prdcf1(real_images: torch.Tensor, 
                               fake_images: torch.Tensor, 
                               k_values: list = list(range(1, 11)), 
                               feature_extractors: list = ['vgg16'], 
                               output_file: str = 'prdc_metrics.txt', 
                               batch_size: int = 64):
    """
    Compute PRDC (Precision, Recall, Density, Coverage) and F1 scores for each k in k_values 
    and for each feature extractor, re-using the extracted features.
    
    The results are saved in one file with the following format (tab-separated):
        model	k	precision	recall	density	coverage	f1
        inception_v3	1	0.0	0.0	0.0	0.0	0.0
        inception_v3	2	...
        vgg16	        1	...
        etc.
    
    Args:
        real_images (torch.Tensor): Real images normalized to [0,1], shape (N, C, H, W).
        fake_images (torch.Tensor): Generated images normalized to [0,1], shape (M, C, H, W).
        k_values (list): List of k values (for nearest neighbors) over which to compute metrics.
        feature_extractors (list): List of feature extractor names; options: 'inception' or 'vgg16'.
        output_file (str): Path to the output file where metrics are saved.
        batch_size (int): Batch size for feature extraction.
    """
    results = []
    results.append("model\tk\tprecision\trecall\tdensity\tcoverage\tf1\n")
    
    for feat_ext in feature_extractors:
        print(f"\nProcessing feature extractor: {feat_ext}")
        if feat_ext not in ['inception', 'vgg16']:
            raise ValueError(f"Invalid feature_extractor='{feat_ext}'. Choose 'inception' or 'vgg16'.")
        
        # Set up the model and determine input resolution
        if feat_ext == 'inception':
            model = models.inception_v3(pretrained=True, transform_input=False)
            model.fc = nn.Identity()  # Use the global average pooling features (2048-d)
            input_size = 299
            model_name = "inception_v3"
        else:  # 'vgg16'
            model = models.vgg16(pretrained=True)
            model.classifier = model.classifier[:-1]  # Use fc2 layer (4096-d features)
            input_size = 224
            model_name = "vgg16"
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Extract features once for real and fake images
        real_feats = extract_features(real_images, model, input_size, batch_size=batch_size)
        fake_feats = extract_features(fake_images, model, input_size, batch_size=batch_size)
        real_features_np = real_feats.numpy()
        fake_features_np = fake_feats.numpy()
        
        # Iterate over each k value
        for k in k_values:
            print(f"  Computing PRDC for k = {k} ...")
            metrics = compute_prdc(
                real_features=real_features_np,
                fake_features=fake_features_np,
                nearest_k=k,
                distance_metric='euclidean'
            )
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            density = metrics.get('density', 0.0)
            coverage = metrics.get('coverage', 0.0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results.append(f"{model_name}\t{k}\t{precision}\t{recall}\t{density}\t{coverage}\t{f1}\n")
    
    # Write all results into one output file
    with open(output_file, "w") as f:
        f.writelines(results)
    
    print(f"\nSaved metrics to {output_file}")