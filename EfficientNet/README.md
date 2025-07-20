# Overview

EfficientNet is a family of convolutional neural networks developed by Google that:

1. Balances model depth, width, and resolution

2. Achieves high accuracy with fewer parameters and FLOPs

3. Uses compound scaling: scaling all dimensions together instead of arbitrarily

The EfficientNet backbone builds mostly from MBConv blocks, which were introduced in MobileNetV2/V3.

# ConvBlock class

A custom convolutional block that combines three common layers used in CNNs:

1. Conv2D – learns spatial features

2. BatchNorm2D – normalizes activations to speed up training

3. SiLU activation – adds non-linearity (also known as Swish)

* It's reusable across different CNN architectures

* Encapsulates a standard "Conv → Norm → Activation" pattern

* Supports **depthwise separable convolutions** via the groups parameter

# Standard Convolution Vs Depthwise Convolution

1. **Standard Convolution**

This is the regular 2D convolution used in most CNNs (like ResNet, VGG, etc.).

How It Works:

* Every **input channel** is connected to **every output channel** via a **3D filter** (with depth equal to the number of input channels).
* The operation **mixes spatial and cross-channel (depth) information** in one step.

Example:

If you have:

* Input: `C_in = 3` (e.g., RGB image)
* Output: `C_out = 64`
* Kernel size: `3×3`

Then:

* Each of the 64 filters is of shape `3×3×3`
* Total multiply-add operations: `C_out × C_in × k × k`

Pros:

* Powerful, captures both spatial and cross-channel relationships

Cons:

* Computationally expensive and memory-heavy

 2. **Depthwise Convolution**

Depthwise convolution is part of **depthwise separable convolution**, used in lightweight models like MobileNet.

How It Works:

* Each **input channel is convolved separately** using its own 2D filter
* Does **not mix** information between channels — only spatial filtering

This is followed by:

* A **pointwise (1×1) convolution** to combine channel information (called `Pointwise Conv`)

 Example:

If you have:

* Input: `C_in = 3`
* Depthwise kernel: each `3×3` → one for each channel

Then:

* Only 3 filters (not 3×64 like standard conv)
* Much fewer computations


 Comparison Table

| Feature             | Standard Convolution                   | Depthwise Convolution                               |
| ------------------- | -------------------------------------- | --------------------------------------------------- |
| Mixes channel info? | ✅ Yes                                  | ❌ No (until pointwise conv)                         |
| Computation         | 💥 High                                | ⚡ Low                                               |
| Filters per output  | One 3D filter per output channel       | One 2D filter per input channel                     |
| Use case            | Accuracy-focused models (ResNet, etc.) | Efficiency-focused models (MobileNet, EfficientNet) |

Intuition

* **Standard Conv** = learns *what* and *where* features are by mixing channel + spatial data in one go
* **Depthwise Conv** = learns *where* in each channel (spatial info), then **pointwise conv** tells *what* features to combine
  
# Squeeze and Excitation

The SE block helps the network learn which channels are important and should be emphasized or suppressed. It does this by learning channel-wise attention weights — boosting informative features and reducing noise.

**How it works : **

1. **Squeeze(Global info) :**  Compress each feature map into a single value (channel descriptor) using global average pooling.Reduces each C×H×W feature map to a single number per channel → output: C×1×1.Captures global context of each feature channel.

2. **Excitation(Learn Attention) :** Pass the descriptors through a small neural network (two FC layers or convs) to produce a weight (0–1) for each channel.

    nn.Conv2d(in_channels, reduced_dim, 1)
    nn.SiLU()
    nn.Conv2d(reduced_dim, in_channels, 1)

This acts as a small 2-layer MLP.Reduces the channel dimension (in_channels → reduced_dim) and then restores it back.Learns a nonlinear function that decides which channels are important.

    nn.Sigmoid()

Squashes the values to a range of [0, 1] → these are the attention weights

3. **Reweighting :** Multiply each channel of the input tensor by its learned weight.Emphasizes informative channels, suppresses others.


# Mobile Bottleneck Convolution Block

The MBBlock is crafted to efficiently extract and refine features from input images using a blend of:

  Channel expansion
  
  Depthwise convolution
  
  Squeeze-and-Excitation (SE) attention
  
  Channel projection

Its main goal is to balance performance and efficiency. It achieves this by separating the convolution process into lightweight stages that focus on spatial and channel information independently.

** Functionality:**

* **Conditional channel expansion:** If the number of input channels is smaller than a predefined multiple (called the expansion ratio), the block increases the number of channels before applying the core operations. This lets the model learn richer features while still being computationally efficient.

* **Depthwise Convolution:** Instead of using a standard convolution that operates across all input channels jointly, the MBBlock uses a depthwise convolution, which applies a separate filter to each channel. This reduces the number of operations significantly, especially for high-resolution inputs, while still capturing spatial features in each channel.

* **Squeeze and Excitation:** This attention mechanism helps the block to adaptively recalibrate channel-wise features. It “squeezes” spatial information into a global descriptor, then “excites” the channels by emphasizing the most important ones. This leads to better focus on informative features and helps boost performance with minimal extra cost.

* **Projection:** After depthwise processing, the block compresses the features back to the desired number of output channels using a 1×1 convolution, also known as pointwise convolution. This step acts like a bottleneck that removes redundancy and controls the dimensionality of output features.

# Feature Extractor

The feature_extractor method constructs the backbone of the EfficientNet model by sequentially stacking convolutional blocks and Mobile Inverted Bottleneck (MBBlock) layers. This is where most of the feature learning happens—where the raw input image is transformed into high-level feature representations that capture the structure, texture, and semantics necessary for the model’s final classification task.

The `feature_extractor` method constructs the **backbone** of the EfficientNet model by sequentially stacking convolutional blocks and Mobile Inverted Bottleneck (MBBlock) layers. This is where most of the **feature learning** happens—where the raw input image is transformed into high-level feature representations that capture the structure, texture, and semantics necessary for the model’s final classification task.

1. **Initial Convolution Layer**

The first layer increases the number of channels from 3 (RGB) to a larger base number scaled by `width_factor`. This initial expansion allows the model to start processing more meaningful patterns right from the beginning. A stride of 2 is applied here to **reduce the spatial resolution**, balancing performance and computational cost.

2. **Width and Depth Scaling**

EfficientNet introduces compound scaling via two hyperparameters: `width_factor` and `depth_factor`. These scale the **channel size** and **number of layers**, respectively, based on the model variant (e.g., B0 to B7). The method uses these to adjust:

* The number of output channels (`out_channels`) per block.
* The number of times each block is repeated (`num_layers`).

This allows one function to create EfficientNets of varying sizes and capacities, adapting to different computational budgets.

 3. **MBBlock Stacking with Controlled Downsampling**

The method loops over predefined parameters that define each stage in the network. Each stage uses MBBlocks with:

* A specific expansion ratio (how much to widen channels temporarily),
* A fixed kernel size,
* A defined number of repeats (scaled with `depth_factor`),
* And a **stride**, which is 2 only for the first layer in each stage to enable **downsampling** (i.e., reducing spatial resolution progressively).

The use of a **larger stride in the first MBBlock** of a stage allows the network to gradually reduce image dimensions (like going from 112×112 → 56×56), which saves memory and increases abstraction. Remaining blocks in the stage use a stride of 1 to preserve resolution while deepening the model.


 4. **Channel Alignment with Multiples of 4**

Output channels are rounded to the nearest multiple of 4 using the expression `4 * ceil(...)`. This is done for **hardware efficiency**—many GPUs and TPUs are optimized for memory access and computation when the number of channels aligns with hardware vector sizes (4, 8, 16, etc.).

5. **Final Convolution Layer**

After all MBBlocks are stacked, a final 1×1 convolution is added to transform the last feature map to a fixed number of channels (`last_channels`). This acts as a **bridge** between the feature extractor and the classifier head, standardizing the output for the final stages of the network.

# Putting it all together : EfficientNet Class

**How Everything Comes Together: **

When an input image passes through the EfficientNet model:

1. **Feature Extraction** (defined separately via `self.feature_extractor()`):

   * A sequence of `ConvBlock` and `MBBlock` layers processes the image.
   * These extract increasingly abstract patterns and reduce the spatial dimensions.

2. **Projection to Final Channels**:

   * The final block’s output is resized to `last_channels` (e.g., 1280).

3. **Global Pooling and Flattening**:

   * Average pooling compresses spatial data.
   * Flattening reshapes it for classification.

4. **Classification**:

   * The classifier produces a score for each class.

Together, this modular architecture gives EfficientNet its key strengths:

* **Scalability** across model sizes,
* **Efficiency** via depthwise separable convolutions and squeeze-excitation,
* **High performance** on tasks like image classification with fewer parameters and FLOPs than older models.


