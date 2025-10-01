# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import numpy as np
from typing import List, Optional, Tuple, Union

import torch

from ...utils import is_torch_xla_available
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class DDPMNoiseShiftPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        prior_labels: Union[int, torch.Tensor],
        prior_means: torch.Tensor,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            prior_label (`int`, *optional*, defaults to 0):
                If using a shifted-noise scheduler, the index of the prior mean to use.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images and the second element is a numpy
                array of mean values for each timestep.
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
            
        image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)
            
        # Make mask for addition of mean shift term to designated element for each sample.
        # --------------------------------------------------------------------------------
        B, C, H, W = image_shape
        # If prior_labels is a scalar, convert it to a tensor of shape (B,)
        if isinstance(prior_labels, int):
            prior_labels = torch.full((B,), prior_labels, dtype=torch.long, device=image.device)
        # Ensure that prior_labels has shape (B,)
        if prior_labels.dim() != 1 or prior_labels.shape[0] != B:
            raise ValueError("prior_labels must be a tensor of shape (B,), one label per sample.")
        # Check that all labels are within the valid range [0, num_prior - 1]
        if (prior_labels < 0).any() or (prior_labels >= self.scheduler.num_prior).any():
            raise ValueError("Each prior label must be in the range [0, num_prior - 1].")
        # --------------------------------------------------------------------------------

        # If using our shifted noise scheduler, start with noise having the corresponding prior mean.
        prior_means = prior_means.to(prior_labels.device) # [10, 3072]
        mean_shift_T = prior_means[prior_labels] # [B, 3072]
        mean_shift_T = mean_shift_T.view(B, C, H, W) # [B, 3, 32, 32]
        mean_shift_T = mean_shift_T.to(self.device)
        
        # Start from the shifted noise priors
        image = image + mean_shift_T
        
        nu_ts, mu_ts = self.scheduler.compute_mu_t(prior_means)
        mean_shift_forward = self.scheduler.compute_mean_shift_forward(prior_means)
        mean_shift_reverse = self.scheduler.compute_mean_shift_reverse(prior_means, mu_ts)
        
        prior_labels = prior_labels.to(self.device)
        mean_shift_forward = mean_shift_forward.to(self.device)
        mean_shift_reverse = mean_shift_reverse.to(self.device)
        

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, prior_labels=prior_labels, mean_shift_forward=mean_shift_forward, mean_shift_reverse=mean_shift_reverse, generator=generator).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
