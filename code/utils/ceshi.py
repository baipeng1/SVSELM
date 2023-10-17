# -*- coding: utf-8 -*-

'''
@Time    : 23/5/3 11:51
@Author  : Kevin BAI
@FileName: ceshi.py
@Software: PyCharm
 
'''
import torch
# (batch, head, time1, time2)
import torch
if __name__ == '__main__':
    # att = torch.randn(2, 4, 2461, 2461)
    #
    # # print(kong.shape)
    # pingjunattention = torch.zeros(att.shape[0],att.shape[2],att.shape[3])
    # # print(kong.shape)
    # for i in range(att.shape[0]):
    #     # print(att[i].shape)
    #     kong = torch.zeros(att.shape[2], att.shape[3])
    #     for j in range(att.shape[1]):
    #         # print(att[i][j].shape)
    #         print(j)
    #         kong=kong+att[i][j]
    #     pingjun = kong/att.shape[1]
    #     print(pingjun.shape)
    #     pingjunattention[i]=pingjun
    # print(pingjunattention.shape)
    scores = torch.randn(2,2000,256)
    # print(scores)
    # print(scores.shape)

    # scores[0][0] = float("-inf")
    # scores[0][0] = scores[0][0]+0.003
    # scores[0][1]=0
    # scores[0][2]=0
    # print(scores)
    scores2=torch.randn(2,2000,256)
    a = torch.cat([scores, scores2],dim=-1)
    # a=torch.softmax(scores,dim=0)
    print(a.shape)
