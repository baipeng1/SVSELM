# -*- coding: utf-8 -*-

'''
@Time    : 23/4/21 22:33
@Author  : Kevin BAI
@FileName: eca_module.py
@Software: PyCharm
 
'''
import torch
from torch import nn
from torch.nn.parameter import Parameter
from math import log
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self):
        super(eca_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x ,gamma=2, b=1):

        N, C, H, W = x.size() #4x1x2461x256
        t= int(abs((log(C,2)+b)/gamma))
        k = t if t % 2 else t + 1
        avg_pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        y = avg_pool(x)
        y = conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        z = x * y.expand_as(x)
        z = z.squeeze(dim=1).transpose(1, 2)
        return z
        # # feature descriptor on the global spatial information
        # y = self.avg_pool(x)
        #
        # # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        #
        # # Multi-scale information fusion
        # y = self.sigmoid(y)
        #
        # return x * y.expand_as(x)