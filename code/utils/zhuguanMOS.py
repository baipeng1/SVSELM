# -*- coding: utf-8 -*-

'''
@Time    : 23/5/23 9:27
@Author  : Kevin BAI
@FileName: zhuguanMOS.py
@Software: PyCharm
 
'''
import os

import hashlib
import random
from tokenize import String

'''
读取文件夹下所有文件的名字并把他们用列表存起来
'''

path = "F:\\MOS\\gezixi\\conformerfocalronghebianmahou"
path2 = "F:\\MOS\\gezixi\\all"
datanames = os.listdir(path)
print(datanames)
for i in range(len(datanames)):
    # print(datanames[i])
    nameold = path+'\\'+datanames[i]
    print(nameold)
    name=datanames[i][:-4]
    alphabet = 'abcdefghijklmnopqrstuvwxyz!@#$%^&*()'
    char = random.choice(alphabet)
    char2 = random.choice(alphabet)
    name=name+char+char2+str(i)
    # print(name)
    hash_object = hashlib.sha256()
    hash_object.update(name.encode("gb2312"))
    hash_value = hash_object.hexdigest()
    # print(hash_value+".wav")
    namenew=path2+'\\'+str(i)+'\\'+hash_value+".wav"
    print(namenew)
    os.rename(nameold, namenew)

# alphabet = 'abcdefghijklmnopqrstuvwxyz!@#$%^&*()'
# char = random.choice(alphabet)
# print(char)