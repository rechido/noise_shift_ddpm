## Diffusion Models With Implicit Conditions Driven by Latent Shifts

This code is based on Hugging Face diffusers library from https://github.com/huggingface/diffusers.git

Author: Da Eun Lee, Nakamura Kensuke, Byung-Woo Hong

Paper: https://ieeexplore.ieee.org/abstract/document/11142693

## Usage

1. clone https://github.com/huggingface/diffusers.git to your local
2. install the diffusers library following its tutorial
3. check if you can run the unconditional generation example. This code is the baseline of our experiment.
4. When you make up your environment, clone our github.
5. goto examples/noise_shift_ddpm/
6. install the packages in requirements.txt by "python -m pip install -r requirements.txt"
7. try "bash run_train.sh" to start training. In order to run the training codes, you are required to prepare pre-computed prior_means pt file. 
8. During training, checkpoint models will be saved.
9. Using the checkpoints, try "bash run_inference.sh" to generate fake images and save them.
10. Using the fake images, try "bash run_evaluate.sh" to measure fid, precision, and recall.
11. Try checking examples/prior_generation folder to prepare prior_means pytorch tensor file for training.

## Preliminary: Shifted Diffusion

Shifted Diffusion is one of previous works for non-zero mean Gaussian prior distributions $p(x_T)$. 

One step forward process of the Shifted Diffusion is defined as

$q(z_t|z_{t-1}) = \mathcal{N}(z_t; \sqrt{\alpha_t} z_{t-1} + s_t, \beta_t \Sigma), where \ s_t = (1 - \alpha_t) \mu$

and multi-step forward process is

$q(z_t|z_0) = \mathcal{N}\left(z_t; \sqrt{\overline{\alpha}_t} z_0 + (1 - \sqrt{\overline{\alpha}_t}) \mu, (1 - \overline{\alpha}_t) \Sigma \right)$

Then reverse process can be derived as

$p_{\theta} (z_{t-1}|z_t) = \mathcal{N}(z_{t-1}; \nu_{\theta}, \Lambda),$

where

$\nu_{\theta} = \gamma (z_t - s_t) + \eta ẑ_0 + \tau (1 - \sqrt{\bar{\alpha} _{t-1}}) \mu$, $\Lambda = \tilde{\beta} _t \Sigma$, 

$\gamma = \frac{\sqrt{\alpha_t} \bar{\beta} _{t-1}}{\bar{\beta}_t}$,  $\eta = \frac{\sqrt{\alpha _{t-1}} \beta_t}{\bar{\beta}_t}$, $\tau = \frac{\beta_t}{\bar{\beta}_t}$, $\hat{z}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (z_t - \sqrt{\bar{\beta}_t} \epsilon _\theta)$

Finally, the training objective function is given as

$L(\theta) = \mathbb{E}_{t, z_0, \epsilon} \left[ \| \epsilon - \epsilon _{\theta} ( \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t )  \|^2 \right]$


where the neural network is trained to predict the standard Gaussian noise.



## Interpretation of Shifted Diffusion as DDPM with shifted data

Applying reparameterization trick to the forward process of Shifted Diffusion leads to the following equation.

$z_t = \sqrt{\bar{\alpha}_t} z_0 + (1 - \sqrt{\bar{\alpha}_t}) \mu + \sqrt{\bar{\beta}_t} \epsilon$, where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$

This equation can be reformulated as

$z'_t = \sqrt{\bar{\alpha}_t} z'_0 + \sqrt{\bar{\beta}_t} \epsilon$, where $z'_t = z_t - \mu$ and $z'_0 = z_0 - \mu$,

which is effectively equivalent to the forward process of vanilla DDPMs for shifted data.



## Method: DDPM with shifted noise

Our motivation for formulating the generalized forward process with a non-zero mean Gaussian prior in DDPM is to target the shift in noise rather than data.

The forward process is proposed as

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{\bar{\beta}_t} \epsilon$, where $\epsilon \sim \mathcal{N}(\mu, \mathbf{I})$

The proposed method could be more intuitive way incorporating non-zero mean Gaussian priors.

To derive reverse process of the proposed method, we use generalized formulation for the forward process as

$x_t = \sqrt{\bar{\alpha}_t} x_0 +  \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(\mu_t, \bar{\beta}_t \mathbf{I})$, $\mu_0 = 0, \mu_T = \mu$.

Then the one step forward process $q(x_t|x_{t-1})$ can be represented as

$x_t = \sqrt{\alpha_t} x_{t-1} +  \xi_t$, $\xi_t \sim \mathcal{N}(\nu_t, \beta_t \mathbf{I})$, 

where

$v_1 = \mu_1$, $v_t = \mu_t - \sqrt{\bar{\alpha}_t} \sum _{t=1} ^{t-1} \frac{v_i}{\sqrt{\bar{\alpha}_i}}$, ($t>1$)

From this equation and the given $\mu_t = \sqrt{1 - \bar{\alpha}_t} \mu$, we can compute $\nu_t$.

The reverse process of our method can then be written as

$x _{t-1} = \gamma x_t + \eta \hat{x}_0 + \frac{\eta}{\sqrt{\bar{\alpha} _{t-1}}} \mu _{t-1} - \gamma \nu_t + \sqrt{\tilde{\beta}_t} z$,

where

$\gamma = \frac{\sqrt{\alpha_t} \bar{\beta} _{t-1}}{\bar{\beta}_t}$, $\eta = \frac{\sqrt{\alpha _{t-1}} \beta_t}{\bar{\beta}_t}$, $\tilde{\beta}_t = \frac{\bar{\beta} _{t-1}}{\bar{\beta}_t} \beta_t$, $z \sim \mathcal{N}(0, \mathbf{I})$, $\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \mu_t - \sqrt{\bar{\beta}_t} \epsilon _\theta)$.

Finally, the training objective of our method is given as

$L(\theta) = \mathbb{E} _{t, x_0, \epsilon} \left[ \| \frac{1}{\sqrt{\bar{\beta}_t}} ( x_t - \sqrt{\bar{\alpha}_t} x_0 - \mu_t) - \epsilon _\theta (\sqrt{\bar{\alpha}_t} x_0 + \sqrt{\bar{\beta}_t} \epsilon, t) \|^2 \right]$,

where the network $\epsilon _\theta$ learns the target $\epsilon_t - \mu_t$.



## Derivation of $\nu_t$

Let one-step and multi-step forward process be

$q(x_t|x_{t-1}) = \mathcal{N}(x_t | \sqrt{\alpha_t} x_0 + \nu_t, \beta_t \mathbf{I})$,

$q(x_t|x_0) = \mathcal{N}(x_t | \sqrt{\bar{\alpha}_t} x_0 + \mu_t, \bar{\beta}_t \mathbf{I})$.

Applying the reparameterization trick, the two equations become

$x_t = \sqrt{\alpha_t} x_{t-1} + \xi_t$, where $\xi_t \sim \mathcal{N}(\nu_t, (1 - \alpha_t) \mathbf{I})$,

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(\mu_t, (1 - \bar{\alpha}_t) \mathbf{I})$.

From the one-step forward process, we can obtain one-step process for the time step $t-1$.

$x_{t-1} = \sqrt{\alpha_{t-1}} x_{t-2} +  \xi_{t-1}$, where $\xi_{t-1} \sim \mathcal{N}(\nu_{t-1}, (1 - \alpha_{t-1}) \mathbf{I})$

Substituting $x_{t-1}$ into one-step process for the time step $t$ leads to

$x_t = \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} +  \xi_{t-1}) + \xi_t$,

$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} +  (\sqrt{\alpha_t} \xi_{t-1} + \xi_t)$.


Following the same technique of vanilla DDPMs, we merge the two Gaussian noises as

$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \eta_{t-1}$, where $\eta_{t-1} \sim \mathcal{N}(\sqrt{\alpha_t} \nu_{t-1} + \nu_t, (1 - \alpha_t \alpha_{t-1}) \mathbf{I})$

Similarly, repeating the same calculation upto the time step of $t=1$ results in

$x_t = \sqrt{\alpha_t \cdots \alpha_1} x_0 + (\xi_t + \sqrt{\alpha_t} \xi_{t-1} + \sqrt{\alpha_t \alpha_{t-1}} \xi_{t-2} + \cdots + \sqrt{\alpha_t \cdots \alpha_3} \xi_2 + \sqrt{\alpha_t \cdots \alpha_2} \xi_1)$,

which can be rewritten as

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \epsilon_t$, where $\epsilon _t \sim \mathcal{N} (\nu _t + \sqrt{\alpha _t} \nu _{t-1} + \sqrt{\alpha _t \alpha _{t-1}} \nu _{t-2} + \cdots + \sqrt{\alpha_t \cdots \alpha_3} \nu_2 + \sqrt{\alpha_t \cdots \alpha_2} \nu_1, (1 - \bar{\alpha}_t) \mathbf{I})$

or

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(\sqrt{\bar{\alpha} _t} \sum _{i=1} ^t \frac{\nu_i}{\sqrt{\bar{\alpha}_i}}, (1 - \bar{\alpha}_t) \mathbf{I})$

Therefore, we obtain the following result:

$\mu_t = \sqrt{\bar{\alpha} _t} \sum _{i=1}^t \frac{\nu_i}{\sqrt{\bar{\alpha}_i}}$ for $t > 1$

When $t=1$, it is straightforward that $\mu_1 = \nu_1$.

From the above relation, we can compute $\nu_t$ for the all time steps by the following iterative algorithm.

<p align="center">
  <img src="resource/algorithm1.png" />
</p>

 

## Derivation of the reverse process

We construct the reverse process of our model by first deriving the conditional posterior of the forward process $q(x_{t-1} | x_t, x_0)$ by using Bayes' theorem.

$q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1}, x_0) \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}$,

where 

$q(x_t|x_{t-1}, x_0) = q(x_t|x_{t-1}) = \mathcal{N}(x_t | \sqrt{\alpha_t} x_{t-1} + \nu_t, \beta_t \mathbf{I})$, (by Markov assumption)

$q(x _{t-1}|x_0) = \mathcal{N}(x _{t-1} | \sqrt{\bar{\alpha} _{t-1}} x_0 + \mu _{t-1}, \bar{\beta} _{t-1} \mathbf{I})$,

$q(x_t|x_0) = \mathcal{N}(x_t | \sqrt{\bar{\alpha}_t} x_0 + \mu_t, \bar{\beta}_t \mathbf{I})$.

Multiplying three normal distributions gives

$q(x_{t-1} \mid x_t, x_0) \propto \exp \bigg( -\frac{1}{2} \left( \frac{x_t - \left(\sqrt{\alpha_t} x _{t-1} + \nu_t\right)}{\sqrt{\beta_t}} \right)^2 - \frac{1}{2} \left( \frac{x _{t-1} - \left(\sqrt{\bar{\alpha} _{t-1}} x_0 + \mu _{t-1}\right)}{\sqrt{\bar{\beta} _{t-1}}} \right)^2 + \frac{1}{2} \left( \frac{x_t - \left(\sqrt{\bar{\alpha}_t} x_0 + \mu_t\right)}{\sqrt{\bar{\beta}_t}} \right)^2  \bigg)$

$q(x_{t-1} \mid x_t, x_0) \propto \exp \bigg( -\frac{1}{2} \left( \left( \frac{x_t - \sqrt{\alpha_t} x_{t-1} - \nu_t}{\sqrt{\beta_t}} \right)^2 + \left( \frac{x _{t-1} - \sqrt{\bar{\alpha} _{t-1}} x_0 - \mu _{t-1}}{\sqrt{\bar{\beta} _{t-1}}} \right)^2 \right) + C \bigg)$

We treated every terms unrelated to $x _{t-1}$ as constants. We now rearrage the first two terms,

$\frac{(x_t - \sqrt{\alpha_t} x_{t-1} - \nu_t)^2}{\beta_t} = \frac{\alpha_t}{\beta_t} x_{t-1}^2 - 2 \frac{\sqrt{\alpha_t}}{\beta_t} (x_t - \nu_t) x_{t-1} + C$

$\frac{(x _{t-1} - \sqrt{\bar{\alpha} _{t-1}} x_0 - \mu _{t-1})^2}{\bar{\beta} _{t-1}} = \frac{1}{\bar{\beta} _{t-1}} x _{t-1} ^2 - 2 \frac{\sqrt{\bar{\alpha} _{t-1}} x_0 + \mu _{t-1}}{\bar{\beta} _{t-1}} x _{t-1} + C$

Then

$q(x_{t-1} \mid x_t, x_0) \propto \exp \bigg( -\frac{1}{2} \left( A_t x _{t-1} ^2 - 2 B_t x _{t-1} + C \right) \bigg)$,

where

$A_t = \frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta} _{t-1}}$, $B_t = \frac{\sqrt{\alpha_t}}{\beta_t} (x_t - \nu_t) + \frac{\sqrt{\bar{\alpha} _{t-1}} x_0 + \mu _{t-1}}{\bar{\beta} _{t-1}}$.

$q(x_{t-1} \mid x_t, x_0) \propto \exp \left( -\frac{1}{2} A_t \left(x_{t-1} - \frac{B_t}{A_t}\right)^2 + C \right)$

Therefore, we end up with the fact that the conditional posterior is also normal distribution which can be described as

$q(x _{t-1} \mid x_t, x_0) \propto \exp \left( -\frac{1}{2} \frac{(x _{t-1} - \tilde{\mu}_t)^2}{\tilde{\beta}_t} \right) = \mathcal{N}(x _{t-1} | \tilde{\mu}_t, \tilde{\beta}_t \mathbf{I})$,

where 

$\tilde{\mu}_t = \frac{B_t}{A_t}, \quad \tilde{\beta}_t = \frac{1}{A_t}$.

As a next step, we design the reverse process of our model as

$p_{\theta}(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1} | \tilde{\mu}_{\theta} (x_t, t), \tilde{\beta}_t \mathbf{I})$,

where

$\tilde{\mu} _{\theta} = \frac{B _{\theta}}{A_t}$, $\tilde{\beta}_t = \frac{1}{A_t}$, $B _{\theta} = \frac{\sqrt{\alpha_t}}{\beta_t} (x_t - \nu_t) + \frac{\sqrt{\bar{\alpha} _{t-1}} \hat{x}_0 + \mu _{t-1}}{\bar{\beta} _{t-1}}$, $\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \mu_t - \sqrt{\bar{\beta}_t} \epsilon _\theta)$,

which simplifies the denoising matching loss $\mathbb{E} _{x_t \sim q(x_t | x_0)} [D _{KL} (q(x _{t-1} | x_t, x_0) \parallel p _\theta (x _{t-1} | x_t))]$.

Now we obtain the update equation of the reverse process by applying reparameterization trick.

$x _{t-1} = \mu _\theta(x_t, t) + \sqrt{\tilde{\beta}_t} z, \ z \sim \mathcal{N}(0, \mathbf{I})$,

$x _{t-1} = \frac{B _{\theta}}{A_t} + \sqrt{\frac{1}{A_t}} z$,

$x _{t-1} = \frac{\frac{\sqrt{\alpha_t}}{\beta_t} (x_t - \nu_t) + \frac{\sqrt{\bar{\alpha} _{t-1}} \hat{x} _0 + \mu _{t-1}}{\bar{\beta} _{t-1}}}{\frac{\alpha_t}{\beta_t} 
    + \frac{1}{\bar{\beta} _{t-1}}} + \sqrt{\frac{1}{\frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta} _{t-1}}}} z$,

$x _{t-1} = \frac{\frac{\sqrt{\alpha_t}}{\beta_t}}{\frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta} _{t-1}}} x_t 
    + \frac{\frac{\sqrt{\bar{\alpha} _{t-1}}}{\bar{\beta} _{t-1}}}{\frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta} _{t-1}}} \hat{x} _0 
    + (\frac{\frac{\mu _{t-1}}{\bar{\beta} _{t-1}} - \frac{\sqrt{\alpha_t}}{\beta_t} \nu_t}{\frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta} _{t-1}}}) 
    + \sqrt{\frac{1}{\frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta} _{t-1}}}} z$,

$x _{t-1} = \frac{\sqrt{\alpha_t} \bar{\beta} _{t-1}}{\bar{\beta}_t} x_t 
    + \frac{\sqrt{\bar{\alpha} _{t-1}} \beta_t}{\bar{\beta}_t} \hat{x} _0 
    + \frac{\beta_t}{\bar{\beta}_t} \mu _{t-1}
    + \frac{\bar{\beta} _{t-1} \sqrt{\alpha_t}}{\bar{\beta}_t} \nu_t
    + \sqrt{\frac{\bar{\beta} _{t-1}}{\bar{\beta}_t} \beta_t} z$,

or

$x _{t-1} = \gamma x_t + \eta \hat{x} _0 + \frac{\eta}{\sqrt{\bar{\alpha} _{t-1}}} \mu _{t-1} - \gamma \nu_t + \sqrt{\tilde{\beta}_t} z$,

where

$\gamma = \frac{\sqrt{\alpha_t} \bar{\beta} _{t-1}}{\bar{\beta}_t}$, $\eta = \frac{\sqrt{\alpha _{t-1}} \beta_t}{\bar{\beta}_t}$, $\tilde{\beta}_t = \frac{\bar{\beta} _{t-1}}{\bar{\beta}_t} \beta_t$, $z \sim \mathcal{N}(0, \mathbf{I})$, $\hat{x} _0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \mu_t - \sqrt{\bar{\beta}_t} \epsilon _\theta)$.



# Derivation of the training objective function

We start from the denoising matching loss $\mathbb{E} _{x_t \sim q(x_t | x_0)} [D _{KL} (q(x _{t-1} | x_t, x_0) \parallel p _\theta (x _{t-1} | x_t))]$. Because both $q(x _{t-1} | x_t, x_0)$ and $p _\theta (x _{t-1} | x_t)$ are normal distributions and from the KL divergence formula of two multivariate Gaussian distribution, the denoising matching loss can be simplified as follows:

$L _{simple} \propto \mathbb{E} _{x_t \sim q(x_t | x_0)} \left[ \| \tilde{\mu}_t(x_t, x_0) - \mu _\theta (x_t, t) \|^2 \right]$,

$L _{simple} \propto \mathbb{E}_q \left[ \| \frac{B_t}{A_t} - \frac{B _{\theta}}{A_t} \|^2 \right]$,

$L _{simple} \propto \mathbb{E}_q \left[ \| B_t - B _{\theta} \|^2 \right]$,

$L _{simple} \propto \mathbb{E}_q \left[ \| \frac{\sqrt{\alpha_t}}{\beta_t} (x_t - \nu_t) + \frac{\sqrt{\bar{\alpha} _{t-1}} x_0 + \mu _{t-1}}{\bar{\beta} _{t-1}} - (\frac{\sqrt{\alpha_t}}{\beta_t} (x_t - \nu_t) + \frac{\sqrt{\bar{\alpha} _{t-1}} \hat{x} _0 + \mu _{t-1}}{\bar{\beta} _{t-1}}) \|^2 \right]$,

$L _{simple} \propto \mathbb{E}_q \left[ \| x_0 - \hat{x} _0 \|^2 \right]$,

$L _{simple} \propto \mathbb{E}_q \left[ \| x_0 - \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \mu_t - \sqrt{\bar{\beta}_t} \epsilon _\theta) \|^2 \right]$,

$L _{simple} \propto \mathbb{E}_q \left[ \| \frac{1}{\sqrt{\bar{\beta}_t}} ( x_t - \sqrt{\bar{\alpha}_t} x_0 - \mu_t) - \epsilon _\theta \|^2 \right]$,

which is the same form as we proposed before.
            
 
## References

PRDC codes are cited by the paper "https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf"

Related Works

DDPM: "https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf"

PriorGrad: "https://arxiv.org/pdf/2106.06406"

Shifted Diffusion: "https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Shifted_Diffusion_for_Text-to-Image_Generation_CVPR_2023_paper.pdf"

ShiftDDPMs: "https://ojs.aaai.org/index.php/AAAI/article/download/25465/25237"
