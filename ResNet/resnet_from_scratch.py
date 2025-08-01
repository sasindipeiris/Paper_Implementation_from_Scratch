# -*- coding: utf-8 -*-
"""ResNet from scratch

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hSbqu0YFDf6jusngQzapGolWJ5AeBq-z
"""

! pip install labml_helpers

from typing import List,Optional

import torch
from torch import nn

from labml_helpers.module import Module

"""# Linear projections for shortcut connection

The `ShortcutProjection` class is a crucial component in deep residual networks like ResNet, specifically used to ensure that the shortcut (or skip) connection matches the shape of the main path's output when the dimensions differ. In standard residual blocks, the input is directly added to the output of a series of convolutional layers. However, this addition requires both tensors to have the same shape—same number of channels and spatial resolution. When the output of a block changes its feature map size (due to striding) or number of channels (due to convolution filters), a simple addition is no longer possible. To address this, `ShortcutProjection` uses a 1×1 convolution, which is a lightweight linear projection that can increase or reduce the number of channels and also apply spatial downsampling via stride. This operation ensures that the shortcut path aligns with the dimensions of the transformed features from the main path. After the convolution, batch normalization is applied to maintain training stability and help the model converge faster. This technique preserves the ability to add residual connections even in deeper and more complex architectures, enabling ResNets to scale efficiently.
"""

class ShortcutProjection(Module):
  def __init__(self,in_channels:int,out_channels:int,stride:int):
    super().__init__()
    self.conv=nn.Conv2d()(in_channels,out_channels,kernel_size=1,stride=stride)
    self.bn=nn.BatchNorm2d(out_channels)

  def forward(self,x:torch.Tensor):
    # Apply 1x1 convolution followed by batch normalization to input x
    return self.bn(self.conv(x))

"""# Residual Block"""

class ResidualBlock(Module):
  def __init__(self,in_channels:int,out_channels:int,stride:int):
    super().__init__()

    self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
    self.bn1=nn.BatchNorm2d(out_channels)
    self.act1=nn.ReLu()

    self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
    self.bn2=nn.BatchNorm2d(out_channels)

    if stride !=1 or in_channels !=out_channels:
      self.shortcut = ShortcutProjection(in_channels,out_channels,stride)
    else:
      self.shortcut=nn.Identity()

    self.act2=nn.ReLu()

  def forward(self,x:torch.Tensor):
    shortcut=self.shortcut(x)
    x=self.act1(self.bn1(self.conv1(x)))
    x=self.bn2(self.conv2(x))
    return self.act2(x+shortcut)

"""The condition if stride != 1 or in_channels != out_channels: is used to decide whether a shortcut connection (skip connection) in a Residual Block needs to transform the input before adding it to the main path output.

This logic ensures that the shortcut (skip connection) has the same shape as the output of the convolutional path, allowing them to be added element-wise. Without matching shapes, the addition would raise an error or produce invalid results.

So, the condition ensures:

Use identity shortcut when input and output shapes match.

Use a projection (1×1 conv + batch norm) shortcut when they differ in size or channels.

# Bottleneck Residual Block

| Feature                      | ResidualBlock                       | BottleneckResidualBlock                      |
| ---------------------------- | ----------------------------------- | -------------------------------------------- |
| Number of Convs              | 2                                   | 3                                            |
| Conv Types                   | 3×3 → 3×3                           | 1×1 → 3×3 → 1×1                              |
| Dimensionality Reduction     | ❌ No                                | ✅ Yes, using first 1×1 convolution           |
| Efficiency for Deep Networks | ❌ Less efficient for very deep nets | ✅ More efficient due to bottleneck structure |
| Typical Use                  | ResNet-18, ResNet-34                | ResNet-50, ResNet-101, ResNet-152            |
"""

class BottleneckResidualBlock(Module):
  def __init__(self,in_channels:int,bottleneck_channels:int,out_channels:int,stride:int):
    super().__init__()

    self.conv1=nn.Conv2d(in_channels,bottleneck_channels,kernel_size=1,stride=1)
    self.bn1=nn.BatchNorm2d(bottleneck_channels)
    self.act1=nn.ReLu()

    self.conv2=nn.Conv2d(bottleneck_channels,bottleneck_channels,kernel_size=3,stride=stride,padding=1)
    self.bn2=nn.BatchNorm2d(bottleneck_channels)
    self.act2=nn.ReLu()

    self.conv3=nn.Conv2d(bottleneck_channels,out_channels,kernel_size=1,stride=1)
    self.bn3=nn.BatchNorm2d(out_channels)

    if stride !=1 or in_channels!=out_channels:
      self.shortcut=ShortcutProjection(in_channels,out_channels,stride)
    else:
      self.shortcut=nn.Identity()

    self.act3=nn.ReLu()

  def forward(self,x:torch.Tensor):

    shortcut=self.shortcut(x)
    x=self.act1(self.bn1(self.conv1(x)))
    x=self.act2(self.bn2(self.conv2(x)))
    x=self.bn3(self.conv3(x))

    return self.act3(x+shortcut)

"""# ResNet Model

n_blocks: Number of residual/bottleneck blocks per feature map size(per stage with fixed spatial resolution and channel depth.)

n_channels: Output channels for each block group.

bottlenecks: Optional list specifying bottleneck channel sizes (if using bottleneck blocks).

img_channels: Number of input image channels (e.g., 3 for RGB).

first_kernel_size: Size of the initial convolution kernel (typically 7 in classic ResNets).
"""

class ResNetBase(Module):
  def __init__(self,n_blocks: List[int],n_channels : List[int],bottlenecks: Optional[List[int]]=None,img_channels: int=3,first_kernel_size: int =7):
    super().__init__()

    assert len(n_blocks) ==len(n_channels)
    assert bottlenecks is None or len(bottlenecks)==len(n_channels)

    self.conv=nn.Conv2d(img_channels,n_channels[0],kernel_size=first_kernel_size,stride=2,padding=first_kernel_size//2)
    self.bn=nn.BatchNorm2d(n_channels[0])

    blocks=[]
    prev_channels=n_channels[0]

    for i,channels in enumerate(n_channels):
      stride =2 if len(blocks)==0 else 1

      if bottlenecks is None:
        blocks.append(ResidualBlock(prev_channels,channels,stride=stride))
      else:
        blocks.append(BottleneckResidualBlock(prev_channels,bottlenecks[i],channels,stride=stride))

      prev_channels=channels

      for _ in range(n_blocks[i]-1):
        if bottlenecks is None:
          blocks.append(ResidualBlock(channels,channels,stride=1))
        else:
          blocks.append(BottleneckResidualBlock(channels,bottlenecks[i],channels,stride=1))

    self.blocks=nn.Sequential(*blocks)

  def forward(self,x:torch.Tensor):

    x=self.bn(self.conv(x))
    x=self.blocks(x)
    x=x.view(x.shape[0],x.shape[1],-1)
    return x.mean(dim=-1)