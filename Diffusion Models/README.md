# Overview

This notebook presents a comprehensive walkthrough of diffusion models, a cutting-edge generative modeling framework used to synthesize high-quality images. The implementation draws inspiration from seminal papers like DDPM (Denoising Diffusion Probabilistic Models) and Denoising Score Matching, using PyTorch for modular and transparent code.

# Libraries used

| **Import**                                         | **Alias / Component**          | **Description / Purpose**                                                                                                                                                                                                                                               |
| -------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `import torch`                                     | —                              | Core PyTorch library. Provides tensors, GPU computation, autograd, and neural network training functionality.                                                                                                                                                           |
| `import torchvision`                               | —                              | Library of datasets, model architectures, and image transformations for computer vision in PyTorch.                                                                                                                                                                     |
| `from torch import nn`                             | `nn`                           | High-level neural network API from PyTorch. Provides essential building blocks like `Linear`, `Conv2d`, `BatchNorm`, `Sequential`, etc.                                                                                                                                 |
| `from torch.nn import functional as F`             | `F`                            | Functional interface to neural network operations, e.g., `F.relu`, `F.cross_entropy`, `F.mse_loss`. Often used for more fine-grained control than `nn.Module` layers.                                                                                                   |
| `from torch.utils.data import DataLoader`          | `DataLoader`                   | Loads data in mini-batches with features like shuffling, multiprocessing, and batching. Used for both training and testing loops.                                                                                                                                       |
| `from diffusers import DDPMScheduler, UNet2DModel` | `DDPMScheduler`, `UNet2DModel` | Imported from Hugging Face's `diffusers` library. <br>• `DDPMScheduler`: Handles the noise scheduling in a Denoising Diffusion Probabilistic Model (DDPM). <br>• `UNet2DModel`: A U-Net based architecture used to predict noise at each step of the denoising process. |
| `from matplotlib import pyplot as plt`             | `plt`                          | Used for plotting and visualizing images (e.g., original vs. denoised) and metrics during training or inference.                                                                                                                                                        |

# Dataset Loading & Preparation

* **MNIST Dataset**:
  Loads the standard MNIST dataset containing 28×28 grayscale images of handwritten digits (0–9).

* **Automatic Download**:
  The dataset is downloaded and saved locally if not already present.

* **Tensor Conversion**:
  Images are transformed to PyTorch tensors using `ToTensor()`, which also scales pixel values to \[0, 1].

* **Mini-Batch Creation**:
  A `DataLoader` is used to create batches of 8 images with shuffling enabled for training randomness.

* **Batch Format**:
  A single batch `x` has shape `[8, 1, 28, 28]` representing 8 grayscale images; `y` contains the corresponding digit labels.

* **Visualization**:
  Uses `make_grid()` and `matplotlib` to display the batch of images in a single composite view.

# The Corruption Process

**Purpose:**
To apply controlled noise to an image tensor, simulating partial corruption—useful in diffusion models.

1. **Input Parameters:**

   * `x`: A batch of input images (tensor of shape `[B, C, H, W]`).
   * `amount`: A 1D tensor of length `B` indicating the corruption level for each image.

2. **Generate Noise:**

   * `torch.rand_like(x)` creates random noise with the same shape as `x` and values between 0 and 1.

3. **Reshape `amount`:**

   * `.view(-1, 1, 1, 1)` reshapes `amount` so it can be broadcasted across the full image dimensions.

4. **Mix Clean and Noisy Images:**

   * Formula: `x * (1 - amount) + noise * amount`
   * When `amount = 0`: output is the clean image.
   * When `amount = 1`: output is pure noise.
   * Intermediate values give partially noisy images.

5. **Result:**

   * Returns a tensor of corrupted images where each image has a different noise level based on `amount`.

# The   Basic Model

| **Component**           | **Type / Structure**            | **Description / Functionality**                                                                                         |
| ----------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Input**               | Tensor of shape `[B, 1, H, W]`  | Grayscale image input (e.g., from MNIST), where `B` is batch size, `H` and `W` are height and width.                    |
| **Down Layers**         | `ModuleList` of 3 Conv2D layers | Sequential convolution layers that encode features. Each uses `kernel_size=5` and `padding=2` to preserve spatial size. |
| Layer 1                 | `Conv2d(1, 32, 5, padding=2)`   | Converts 1-channel image to 32-channel feature map.                                                                     |
| Layer 2                 | `Conv2d(32, 64, 5, padding=2)`  | Increases depth to 64 channels.                                                                                         |
| Layer 3                 | `Conv2d(64, 64, 5, padding=2)`  | Keeps same number of channels (64), deeper feature extraction.                                                          |
| **Downscaling**         | `MaxPool2d(2)`                  | Applied after first two down layers to reduce spatial size (H, W → H/2, W/2).                                           |
| **Skip Connections**    | `h.append(x)` / `x += h.pop()`  | Stores intermediate outputs before downscaling and reuses them in up path (for feature reuse and better gradient flow). |
| **Activation Function** | `SiLU()` (a.k.a. Swish)         | Applied after every Conv2D layer. Smooth, non-monotonic activation known to outperform ReLU in many deep networks.      |
| **Up Layers**           | `ModuleList` of 3 Conv2D layers | Reconstructs image features back to original dimensions.                                                                |
| Layer 4                 | `Conv2d(64, 64, 5, padding=2)`  | Applied first after the final down layer.                                                                               |
| Layer 5                 | `Conv2d(64, 32, 5, padding=2)`  | Reduces feature depth from 64 to 32.                                                                                    |
| Layer 6                 | `Conv2d(32, 1, 5, padding=2)`   | Outputs a single-channel image again.                                                                                   |
| **Upscaling**           | `Upsample(scale_factor=2)`      | Doubles spatial resolution before skip connection. Used after each up layer except the first.                           |
| **Output**              | Tensor of shape `[B, 1, H, W]`  | Reconstructed (e.g., denoised) image with the same shape as the input.                                                  |


# Training

* **Batch Size**: 128 samples per training step (`batch_size=128`).
* **Epochs**: Model is trained for 3 full passes over the dataset (`n_epochs=3`).
* **Model**: Uses `BasicUNet`, moved to GPU/CPU as defined by `device`.
* **Loss Function**: Mean Squared Error (`nn.MSELoss`) to compare predicted and original (clean) images.
* **Optimizer**: Adam with a learning rate of `1e-3`.
* **Corruption**: Each input batch `x` is corrupted with random Gaussian noise using `corrupt(x, noise_amount)`.
* **Training Step**:

  * Pass corrupted image `noise_x` through the model.
  * Calculate loss between prediction and clean image.
  * Backpropagate and update model weights.
* **Loss Tracking**: Losses are stored, and average loss per epoch is printed.
* **Plotting**: Training loss is visualized over time using `plt.plot(losses)`.

# Sampling

In a **diffusion model**, the generation of an image is done by starting from **pure random noise** and then progressively **denoising it step by step**. This stepwise process of cleaning up the noise is what we call **sampling**. Each step in sampling aims to bring the noisy image closer to a realistic, clean image — much like reverse engineering the degradation process.

---
 
You start with an **image made of random noise**, which has no recognizable pattern (like static on a TV screen). Then, a neural network (such as a U-Net) is used to **predict what a "clean" image might look like** given the noisy input.

This prediction is not applied fully in one go. Instead, the model **gradually blends** the prediction into the original noise — increasing the weight of the prediction more with each step. So, the image becomes **increasingly clean over several iterations**, mimicking the diffusion process in reverse.

---

The **mixing factor** determines **how much influence** the model's prediction has at each step. In the early stages, you only trust the model a little, blending a small amount of the prediction into the current noisy image. In later steps, you trust it more, so the image becomes more like the model's guess.

This gradual blending:

* Prevents abrupt jumps in image appearance.
* Simulates the multi-step denoising behavior of real diffusion samplers.
* Allows the model to refine its prediction based on partially cleaned images.

---

At each step of the process, you store the current version of the image and the prediction made by the model. These snapshots are then **displayed side by side**, so you can **see the progression from noise to image**.

* The **first column** shows the noisy input at each stage.
* The **second column** shows the model’s predicted clean image at that stage.

Together, they tell a visual story of how the noise is sculpted into a digit-like image — which is especially helpful for educational and debugging purposes.

---

* Sampling is the **core generation step** in diffusion models.
* It begins from **pure noise** and ends in a **structured, clean image**.
* The process is done **step-by-step**, where each step predicts and mixes in a cleaner version.
* This mimics how actual diffusion models denoise an image from a known noise schedule.
* Visualization allows us to **observe how the model incrementally refines the image**.






