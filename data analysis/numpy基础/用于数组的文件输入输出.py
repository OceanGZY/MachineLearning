#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''

import numpy as np
import os

# 将数组以二进制格式保存到磁盘
# np.save ，默认情况下，是以 未压缩原始二进制格式保存在.npy格式的文件中
arr = np.arange(10)
np.save('some_test_array',arr)

# np.load ,读取
print(np.load('some_test_array.npy'))

# np.savez ，将多个数组保存在一个压缩文件中 .npz文件
np.savez('some_array_arc.npz',a=arr,b=arr)

arch = np.load('some_array_arc.npz')
print(arch['a'])


# 存取文本文件
fpath = os.path.abspath(os.path.join(os.getcwd(),'..'))
npath = (fpath+'/database/examples/array_ex.txt')
# np.loadtxt ,将数据加载到numpy
# np.genformtxt  , 将数据加载到numpy
arr1 = np.loadtxt(npath,delimiter=',')
print(arr1)
