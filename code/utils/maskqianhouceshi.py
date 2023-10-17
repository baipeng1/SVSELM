# -*- coding: utf-8 -*-

'''
@Time    : 23/5/30 16:02
@Author  : Kevin BAI
@FileName: maskqianhouceshi.py
@Software: PyCharm
 
'''
import torch

durations =torch.tensor([[2,1,2,1,1],[1,2,1,2,1]])
def collate_masks2(durations):
  """Convert a list of 1d tensors into a padded 2d tensor."""
  out_lens = durations.sum(dim=1)
  max_len = out_lens.max()
  bsz, seq_len = durations.size()
  out = durations.new_ones((bsz, max_len, max_len)) * float("-inf")

  for b in range(bsz):
    st = 0
    l1 = durations[b, 0]
    out[b, st:st + l1, st:st + l1] = 0
    l2 = durations[b, 1]
    out[b, st:st + l1, st:st + l1 + l2] = 0
    st = st + l1
    for t in range(1, seq_len - 1):
      l0 = durations[b, t - 1]
      l1 = durations[b, t]
      l2 = durations[b, t + 1]

      if l1 == 0:
        continue
      out[b, st:st + l1, st - l0:st + l1 + l2] = 0
      st = st + l1

    l0 = durations[b, seq_len - 2]
    l1 = durations[b, seq_len - 1]
    out[b, st:st + l1, st - l0:st + l1] = 0
    st = st + l1

    out[b, st:, :] = 0

  # out = out*float("-inf")
  return out

  def collate_masks3(durations):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    num = 2
    out_lens = durations.sum(dim=1)
    max_len = out_lens.max()
    bsz, seq_len = durations.size()
    out = durations.new_ones((bsz, max_len, max_len)) * float("-inf")

    for b in range(bsz):
      st = 0
      l1 = durations[b, 0]
      out[b, st:st + l1, st:st + l1] = 0
      l2 = 0
      for i in range(num):
        l2 = l2 + durations[b, 1]
      # l2 = durations[b, 1]
      out[b, st:st + l1, st:st + l1 + l2] = 0
      st = st + l1
      for t in range(1, seq_len - 1):
        l0 = durations[b, t - 1]
        l1 = durations[b, t]
        l2 = 0
        for i in range(num):
          l2 = l2 + durations[b, 1]

        if l1 == 0:
          continue
        out[b, st:st + l1, st - l0:st + l1 + l2] = 0
        st = st + l1

      l0 = durations[b, seq_len - 2]
      l1 = durations[b, seq_len - 1]
      out[b, st:st + l1, st - l0:st + l1] = 0
      st = st + l1

      out[b, st:, :] = 0

    # out = out*float("-inf")
    return out

def collate_masks3(durations): #前1后2
    """Convert a list of 1d tensors into a padded 2d tensor."""
    num = 2
    out_lens = durations.sum(dim=1)
    max_len = out_lens.max()
    bsz, seq_len = durations.size()
    out = durations.new_ones((bsz, max_len, max_len)) * float("-inf")

    for b in range(bsz):
      st = 0
      l1 = durations[b, 0]
      out[b, st:st + l1, st:st + l1] = 0
      l2 = 0
      for i in range(1,num+1):
        l2 = l2 + durations[b, i]
      # l2 = durations[b, 1]
      out[b, st:st + l1, st:st + l1 + l2] = 0
      st = st + l1
      for t in range(1, seq_len - 1):
        l0 = durations[b, t - 1]
        l1 = durations[b, t]
        l2 = 0
        for i in range(num):
          l2 = l2 + durations[b, t+i]

        if l1 == 0:
          continue
        out[b, st:st + l1, st - l0:st + l1 + l2] = 0
        st = st + l1

      l0 = durations[b, seq_len - 2]
      l1 = durations[b, seq_len - 1]
      out[b, st:st + l1, st - l0:st + l1] = 0
      st = st + l1

      out[b, st:, :] = 0

    return out


def collate_masks3qian1hou2(durations):  # 前1后2新的代码
  """Convert a list of 1d tensors into a padded 2d tensor."""
  num = 2  # 后面mask数
  out_lens = durations.sum(dim=1)  # [6,5]
  max_len = out_lens.max()  # 6
  bsz, seq_len = durations.size()  # bsz=2 seq_len=4
  out = durations.new_ones((bsz, max_len, max_len)) * float("-inf")  # 新建一个全是负无穷大的张量

  for b in range(bsz):  # 每一个样本遍历
    st = 0  # 开始数字
    l1 = durations[b, 0]  # 遍历第一个样本的第一个时长 1
    out[b, st:st + l1, st:st + l1] = 0
    l2 = 0
    for i in range(1, num + 1):  # 1,2
      l2 = l2 + durations[b, i]  # l2获取到当前token之后两个数字长度和 当前为1 之后就是2+1=3
    # l2 = durations[b, 1]
    out[b, st:st + l1, st:st + l1 + l2] = 0  # 以上只解决第一行
    st = st + l1
    for t in range(1, seq_len - 1):
      l0 = durations[b, t - 1]  # 前一个token帧数
      l1 = durations[b, t]  # 当前token帧数
      l2 = 0
      for i in range(num):
        # t=t+1#问题估计在这里
        # if t+i<=seq_len-1:
        # print("t")
        # print(t)
        # print("t+i")
        # print(t+i)
        if t + i < seq_len - 1:
          l2 = l2 + durations[b, t + i + 1]

      if l1 == 0:
        continue
      out[b, st:st + l1, st - l0:st + l1 + l2] = 0
      st = st + l1

    l0 = durations[b, seq_len - 2]
    l1 = durations[b, seq_len - 1]
    out[b, st:st + l1, st - l0:st + l1] = 0
    st = st + l1

    out[b, st:, :] = 0

  return out

def collate_masks3qian2hou1(durations):  # 前2后1新的代码
  """Convert a list of 1d tensors into a padded 2d tensor."""
  num = 1  # 后面mask数
  num2=2
  out_lens = durations.sum(dim=1)  # [6,5]
  max_len = out_lens.max()  # 6
  bsz, seq_len = durations.size()  # bsz=2 seq_len=4
  out = durations.new_ones((bsz, max_len, max_len)) * float("-inf")  # 新建一个全是负无穷大的张量

  for b in range(bsz):  # 每一个样本遍历
    st = 0  # 开始数字
    l1 = durations[b, 0]  # 遍历第一个样本的第一个时长 1
    out[b, st:st + l1, st:st + l1] = 0
    l2 = 0
    for i in range(1, num + 1):  # 1,2
      l2 = l2 + durations[b, i]  # l2获取到当前token之后两个数字长度和 当前为1 之后就是2+1=3
    # l2 = durations[b, 1]
    out[b, st:st + l1, st:st + l1 + l2] = 0  # 以上只解决第一行
    st = st + l1 #第二个数字
    #以上只处理了第一行

    for t in range(1, seq_len - 1): #t是每一个元素的下标
      # for i in range(num2): #i=0/1
        # t=t+1#问题估计在这里
        # if t+i<=seq_len-1:
        # print("t")
        # print(t)
        # print("t+i")
        # print(t+i)
      # l0 = 0  #l0表示的是前面的帧数
      a=t-2
      if a>=0:#前面有两个元素
        c=durations[b, t-1]
        d=durations[b, t-2]
        l0 = c+d
      if a==-1: #前面有一个元素
        l0 = durations[b, t-1]
      # else:
      #   l0=0
      #           + i < seq_len - 1:
      #     l2 = l2 + durations[b, t + i + 1]
      # l0 = durations[b, t - 1]  # 前一个token帧数
      l1 = durations[b, t]  # 当前token帧数
      l2 = 0
      for i in range(num):
        # t=t+1#问题估计在这里
        # if t+i<=seq_len-1:
        # print("t")
        # print(t)
        # print("t+i")
        # print(t+i)
        if t + i < seq_len - 1:
          l2 = l2 + durations[b, t + i + 1]

      if l1 == 0:
        continue
      out[b, st:st + l1, st - l0:st + l1 + l2] = 0
      st = st + l1

    l0 = durations[b, seq_len - 2]
    l1 = durations[b, seq_len - 1]
    out[b, st:st + l1, st - l0:st + l1] = 0
    st = st + l1

    out[b, st:, :] = 0
  #以上是前1后1，我们再次进行遍历，讲每一个token前前一个置为0，达到前2后1的效果

  return out
# a = collate_masks3qian1hou2(durations)
# print(a)
def collate_masks3qian2hou2(durations):  # 前2后1新的代码  最后一行有点问题 时间关系 没修改 直接用了
  """Convert a list of 1d tensors into a padded 2d tensor."""
  num = 2  # 后面mask数
  num2=2
  out_lens = durations.sum(dim=1)  # [6,5]
  max_len = out_lens.max()  # 6
  bsz, seq_len = durations.size()  # bsz=2 seq_len=4
  out = durations.new_ones((bsz, max_len, max_len)) * float("-inf")  # 新建一个全是负无穷大的张量

  for b in range(bsz):  # 每一个样本遍历
    st = 0  # 开始数字
    l1 = durations[b, 0]  # 遍历第一个样本的第一个时长 1
    out[b, st:st + l1, st:st + l1] = 0
    l2 = 0
    for i in range(1, num + 1):  # 1,2
      l2 = l2 + durations[b, i]  # l2获取到当前token之后两个数字长度和 当前为1 之后就是2+1=3
    # l2 = durations[b, 1]
    out[b, st:st + l1, st:st + l1 + l2] = 0  # 以上只解决第一行
    st = st + l1 #第二个数字
    #以上只处理了第一行

    for t in range(1, seq_len - 1): #t是每一个元素的下标
      # for i in range(num2): #i=0/1
        # t=t+1#问题估计在这里
        # if t+i<=seq_len-1:
        # print("t")
        # print(t)
        # print("t+i")
        # print(t+i)
      # l0 = 0  #l0表示的是前面的帧数
      a=t-2
      if a>=0:#前面有两个元素
        c=durations[b, t-1]
        d=durations[b, t-2]
        l0 = c+d
      if a==-1: #前面有一个元素
        l0 = durations[b, t-1]
      # else:
      #   l0=0
      #           + i < seq_len - 1:
      #     l2 = l2 + durations[b, t + i + 1]
      # l0 = durations[b, t - 1]  # 前一个token帧数
      l1 = durations[b, t]  # 当前token帧数
      l2 = 0
      for i in range(num):
        # t=t+1#问题估计在这里
        # if t+i<=seq_len-1:
        # print("t")
        # print(t)
        # print("t+i")
        # print(t+i)
        if t + i < seq_len - 1:
          l2 = l2 + durations[b, t + i + 1]

      if l1 == 0:
        continue
      out[b, st:st + l1, st - l0:st + l1 + l2] = 0
      st = st + l1

    l0 = durations[b, seq_len - 2]
    l1 = durations[b, seq_len - 1]
    out[b, st:st + l1, st - l0:st + l1] = 0
    st = st + l1

    out[b, st:, :] = 0
  #以上是前1后1，我们再次进行遍历，讲每一个token前前一个置为0，达到前2后1的效果

  return out

def collate_masks3qian1hou0(durations):  # 前2后1新的代码  最后一行有点问题 时间关系 没修改 直接用了
  """Convert a list of 1d tensors into a padded 2d tensor."""
  num = 0  # 后面mask数
  # num2=1
  out_lens = durations.sum(dim=1)  # [6,5]
  max_len = out_lens.max()  # 6
  bsz, seq_len = durations.size()  # bsz=2 seq_len=4
  out = durations.new_ones((bsz, max_len, max_len)) * float("-inf")  # 新建一个全是负无穷大的张量

  for b in range(bsz):  # 每一个样本遍历
    st = 0  # 开始数字
    l1 = durations[b, 0]  # 遍历第一个样本的第一个时长 1
    out[b, st:st + l1, st:st + l1] = 0
    l2 = 0
    for i in range(1, num + 1):  # 1,2
      l2 = l2 + durations[b, i]  # l2获取到当前token之后两个数字长度和 当前为1 之后就是2+1=3
    # l2 = durations[b, 1]
    out[b, st:st + l1, st:st + l1 + l2] = 0  # 以上只解决第一行
    st = st + l1 #第二个数字
    #以上只处理了第一行

    for t in range(1, seq_len - 1): #t是每一个元素的下标
      # for i in range(num2): #i=0/1
        # t=t+1#问题估计在这里
        # if t+i<=seq_len-1:
        # print("t")
        # print(t)
        # print("t+i")
        # print(t+i)
      # l0 = 0  #l0表示的是前面的帧数
      a=t-1
      if a>=0:#前面有1个元素
        l0=durations[b, t-1]

      if a==-1: #前面有0个元素
        l0 = 0
      # else:
      #   l0=0
      #           + i < seq_len - 1:
      #     l2 = l2 + durations[b, t + i + 1]
      # l0 = durations[b, t - 1]  # 前一个token帧数
      l1 = durations[b, t]  # 当前token帧数
      l2 = 0
      for i in range(num):
        # t=t+1#问题估计在这里
        # if t+i<=seq_len-1:
        # print("t")
        # print(t)
        # print("t+i")
        # print(t+i)
        if t + i < seq_len - 1:
          l2 = l2 + durations[b, t + i + 1]

      if l1 == 0:
        continue
      out[b, st:st + l1, st - l0:st + l1 + l2] = 0
      st = st + l1

    l0 = durations[b, seq_len - 2]
    l1 = durations[b, seq_len - 1]
    out[b, st:st + l1, st - l0:st + l1] = 0
    st = st + l1

    out[b, st:, :] = 0
  #以上是前1后1，我们再次进行遍历，讲每一个token前前一个置为0，达到前2后1的效果

  return out

a = collate_masks3qian1hou0(durations)
print(a)