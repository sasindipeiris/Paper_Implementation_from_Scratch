# Pytorch implementation of ResNet

This repository provides a clean and educational implementation of ResNet (Residual Network) in PyTorch, built entirely from scratch. It includes both the standard residual block and the more efficient bottleneck residual block used in deeper architectures like ResNet-50 and above. The design is modular, flexible, and suitable for learning, experimentation, and customization.

# Architecture overview

The core idea of ResNet is to allow the network to learn residual functions, i.e., to learn the difference between input and output rather than the full transformation. This is achieved using skip connections (shortcut paths) that allow information to bypass certain layers, improving training stability and enabling the training of very deep networks.

The architecture consists of:

1.An initial convolution + batch normalization.

2.Multiple stages of residual or bottleneck blocks (defined by n_blocks and n_channels).

3.Each stage may reduce spatial resolution via stride and increase feature depth.

4.A global average pooling at the end to convert feature maps to fixed-size vector outputs.

# ShortcutProjection

This module handles dimension alignment in skip connections. If the spatial size or number of channels changes between the input and output, a 1×1 convolution is applied with stride and batch normalization to ensure the tensors can be added safely.

# Residual Block

implements the classic ResNet block with:

1.Two 3×3 convolutions.

2.Batch normalization and ReLU activations.

3.A skip connection, either identity or projected.

Used in shallow networks like ResNet-18 and ResNet-34.

**Why Residual Blocks are used?**

Residual blocks are designed to address the **degradation problem** in deep neural networks—where adding more layers actually worsens performance. This occurs because very deep networks struggle to learn identity mappings and propagate gradients effectively, leading to vanishing or exploding gradients.

By introducing a **skip connection**, a residual block allows the network to learn only the **residual (difference)** between the input and the desired output, rather than the full transformation. This greatly simplifies the learning process and improves gradient flow during training.

**Mathematically**, if the true underlying mapping we want to learn is:

H(x)


the residual block helps the model instead learn:


F(x) = H(x) − x


and outputs:

F(x) + x


This structure makes it easier for the network to **approximate identity mappings** or small refinements, especially in deep architectures, enabling much deeper models to be trained effectively.


# BottleneckResidualBlock

Used in deeper networks, this block includes:

1.1×1 conv (to reduce dimensions),

2.3×3 conv (for actual computation),

3.1×1 conv (to restore dimensions),

Batch norm and ReLU activations throughout.
More efficient in deeper models like ResNet-50/101/152.

**Why Bottleneck Blocks are used?**

1.Efficiency in Deep Networks:
In deep networks with many layers, 3×3 convolutions across high-dimensional tensors are computationally expensive. The bottleneck structure reduces the input channels (with a 1×1 conv), applies the heavy 3×3 operation on a smaller dimension, and then expands the result back. This saves computation and memory.

2.Enables Deeper Networks:
Bottleneck blocks allow architectures like ResNet-50 and beyond to go much deeper (e.g., 50, 101, 152 layers) without exploding in size or runtime cost.

3.Improved Gradient Flow:
Like regular residual blocks, they preserve gradient flow during training, helping to avoid vanishing gradient problems.



# ResnetBase

The full backbone model:

1.Accepts user-defined number of blocks per stage via n_blocks.

2.Allows using either ResidualBlock or BottleneckResidualBlock by passing a bottlenecks list.

3.Initial convolution uses a kernel size of 7 and stride 2 to reduce spatial resolution early.

4.Assembles all blocks using nn.Sequential.

5.Applies global average pooling (x.view(...).mean(dim=-1)) to convert final feature maps to a [batch_size, channels] vector.

Stride = 2 is used at the start of a stage to downsample the spatial resolution (e.g., from 56×56 to 28×28).

Padding = kernel_size // 2 ensures that the feature map size remains proportional after convolution when stride is 1.

Global Average Pooling replaces fully connected layers, improving parameter efficiency and translation invariance.


The ResNetBase class is a flexible, modular implementation of a residual network base that supports both standard residual blocks and bottleneck residual blocks. The architecture begins with a convolutional layer (with large kernel size) and batch normalization, followed by a series of blocks where the number of channels and blocks per stage are customizable. The use of stride=2 in only the first block of each stage enables spatial downsampling, which is critical in reducing computational load and increasing receptive field. The bottleneck blocks allow deeper architectures to be more efficient by reducing and restoring dimensions around the computationally expensive 3×3 convolution.

Global average pooling at the end replaces fully connected layers and reduces overfitting by summarizing spatial information into a single value per channel. This design is common in classification and representation learning tasks and forms the basis for many powerful ResNet-style architectures.

This modular setup allows easy adaptation of the ResNet design by changing the number of stages, depth per stage, and whether bottlenecks are used—making it ideal for both research and production use.























