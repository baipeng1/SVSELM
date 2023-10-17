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
import numpy as np

durations = np.load("phduration.npy")[-1]
a = open("0010_ph.txt").readlines()[0]
cc = a.split()
b = [i if i != '|' else ' ' for i in a.split()]
ph = ' '.join(b).split()
a = 1


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(30, 30), cmap='Reds'):
    """显示矩阵热图"""

    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    a = np.sum(durations)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    plt.subplot(111)
    plt.plot([0, a], [0, a], linewidth=1)
    st = 0

    for t in range(len(ph)):
        l = durations[t]
        if l == 0:
            break
        else:
            plt.text(st, st, ph[t])

        st = st + l

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

    # st = 0
    # for t in range(len(durations)):
    #     l = durations[t]
    #     if l == 0:
    #         continue
    #     ax.add_patch(plt.Rectangle((st,st),l,l,linewidth=1,fill=False))
    #     # ax.add_patch(plt.text(st,st, ph[t]))
    #
    #     st = st + l

    st = 0
    l1 = durations[0]
    ax.add_patch(plt.Rectangle((st, st), l1, l1, linewidth=1, fill=False))

    l2 = durations[1]
    ax.add_patch(plt.Rectangle((st, st), l1 + l2, l1, linewidth=1, fill=False))

    st = st + l1
    for t in range(1, len(durations) - 1):
        l0 = durations[t - 1]
        l1 = durations[t]
        l2 = durations[t + 1]

        if l1 == 0:
            continue
        ax.add_patch(plt.Rectangle((st - l0, st), l0 + l1 + l2, l1, linewidth=1, fill=False))

        # out[b, st:st + l1, st - l0:st + l1 + l2] = 0
        st = st + l1

    fig.savefig("11111111.png")  # 保存图片 注意 在show()之前  不然show会重新创建新的 图片
    fig.show()
    a = 1


if __name__ == '__main__':
    att = np.load("fs2weight_attentionnew.npy")
    # att = torch.randn(4,2461,2461)
    print(att.shape[0])
    # for i in range(att.shape[0]):
    # print(att[i].shape())
    show_heatmaps(att[0].reshape(1, 1, att.shape[1], att.shape[1]), 'x', 'y')



