# -*- coding: utf-8 -*-

'''
@Time    : 23/4/26 16:52
@Author  : Kevin BAI
@FileName: keshihuaattentionjuzhen.py
@Software: PyCharm
 
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def show_heatmaps(xuhao, matrices, xlabel, ylabel, titles=None, figsize=(30, 30), cmap='Reds'):
    """显示矩阵热图"""

    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    fig.savefig(str(xuhao)+".png")  # 保存图片 注意 在show()之前  不然show会重新创建新的 图片
    # fig.savefig("img\\1.png")  # 保存图片 注意 在show()之前  不然show会重新创建新的 图片
    fig.show()

def show_heatmaps2(matrices, xlabel, ylabel, titles=None, figsize=(30, 30), cmap='Reds'):
    """显示矩阵热图"""

    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    fig.savefig("img\\2.png")  # 保存图片 注意 在show()之前  不然show会重新创建新的 图片
    fig.show()

if __name__ == '__main__':
    # q = torch.randn(2461, 77)
    # k = q
    # v = q
    # att = q @ (k.T)
    # show_heatmaps(att.reshape(1, 1, 2461, 2461), 'x', 'y')



    # att = torch.randn(4,2461,2461)
    # print(att.shape[0])
    # for i in range(att.shape[0]):
    #     print(att[i].size())
    #     show_heatmaps(att[i].reshape(1, 1, att.shape[1], att.shape[1]), 'x', 'y')


    att = torch.randn(4,4,2461,2461)
    for i in range(att.shape[0]):
        print(att[i].shape)
        att[i][0]
    # print(att.shape[0])
    # for i in range(att.shape[0]):
    #     print(att[i].size())
    #     show_heatmaps(att[i].reshape(1, 1, att.shape[1], att.shape[1]), 'x', 'y')



