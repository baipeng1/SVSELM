# -*- coding: utf-8 -*-

'''
@Time    : 23/4/12 17:07
@Author  : Kevin BAI
@FileName: convolution.py
@Software: PyCharm
 
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import torch

"""ConvolutionModule definition."""

from torch import nn
from math import log
from Conformer.eca_module import eca_layer

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation
        #self.eca = eca_layer()

    def EfficientChannelAttention(self, x, gamma=2, b=1): #x = torch.randn(4, 1, 2461, 256)
        N, W, H, C = x.size() #4x1x2461x256
        c = log(C, 2) + b
        t = int(abs(c / gamma))
        k = t if t % 2 else t + 1
        # k = k*5
        avg_pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False).to(x.device)
        y = avg_pool(x).to(x.device)
        #print(y.dtype)
        y = conv(y.squeeze(-1).transpose(-1, -2)).to(x.device)
        y = y.transpose(-1, -2).unsqueeze(-1).to(x.device)
        return x * y.expand_as(x)

    # def AdaptiveK(self, x, gamma=2, b=1): #x = torch.randn(4, 1, 2461, 256)
    #     N, C, H, W = x.size() #4x1x2461x256
    #     c = log(H, 2) + b
    #     t = int(abs(c / gamma))
    #     k = t if t % 2 else t + 1
    #     return k
    def AdaptiveKdepthwise_conv(self, x, gamma=2, b=1): #x = torch.randn(4, 1, 2461, 256)
        N, W, H, C = x.size() #4x1x2461x256
        c = log(H, 2) + b
        t = int(abs(c / gamma))
        k = t if t % 2 else t + 1
        k = k*6
        #判断k
        if k%2==0:
            k=k+1
        # avg_pool = nn.AdaptiveAvgPool2d(1)
        x= x.squeeze(dim=1).transpose(1, 2)
        print(k)
        conv = nn.Conv1d(C, C, kernel_size=k, stride=1,padding=(k-1)//2, groups=C,bias=False).to(x.device)
        x = conv(x)
        #y = avg_pool(x).to(x.device)
        #print(y.dtype)
        #y = conv(y.squeeze(-1).transpose(-1, -2)).to(x.device)
        #y = y.transpose(-1, -2).unsqueeze(-1).to(x.device)
        #return x * y.expand_as(x)
        return x


    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).  例子：4x2461x256

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  #4x256x2461

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim) #4x512x2461
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim) #4x256x2461
        #print(x.dtype)
        # xadaptinput = torch.unsqueeze(x.transpose(1, 2),dim=1).to(x.device)  #4x1x2461x256
        #print(xadaptinput.dtype)
        # 1D Depthwise Conv
        # ks = self.AdaptiveK(xadaptinput)

        x = self.depthwise_conv(x)  # 4x256x2461  channel=2461

        #xadaptoutput1 = self.EfficientChannelAttention(xadaptinput).to(xadaptinput.device)
        # xadaptoutput1 = self.AdaptiveKdepthwise_conv(xadaptinput)
        #xadaptoutput = xadaptoutput1.squeeze(dim=1).transpose(1, 2).to(xadaptinput.device)
        #还是这样？
        #xadaptoutput = self.eca(xadaptinput)
        #x = (x + xadaptoutput)/2  #固定感受野与自适应感受野的特征图相加
        # x = xadaptoutput1.to(xadaptinput.device)
        x = self.activation(self.norm(x)) #4x256x2461

        x = self.pointwise_conv2(x) #4x256x2461

        return x.transpose(1, 2) #4x2461x256