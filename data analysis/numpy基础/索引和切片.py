#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''

import numpy as np

arr = np.arange(10)
print(arr)
print('---')
print(arr[5])
print(arr[0])
print(arr[5:8])


arr[5:8] = 12
print(arr)


arr_slice = arr[5:8]
arr_slice[1] = 12345
print(arr)

# 高维度数组；二维数组，各元素不是标量而是一维数组
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d)
print(arr2d[2])
print('----')
print(arr2d[0,2])
print(arr2d[0][2])
print('----')


arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr3d)
print('---')
print(arr3d[0])

# 标量和数组 均可赋值给 arr3d[0]
old_values = arr3d[0].copy()

arr3d[0] = 42
print(arr3d)
print('---')
arr3d [0] = old_values
print(arr3d)
print('---')
print(arr3d[1,0])

print('----')
# 切片索引

print(arr[1:6])
print('--')
print(arr2d)
print('---')
# 从0轴开始（第一个数据）
print(arr2d[:2])
print('---')

# 传入多个切片索引
print(arr2d[:2,1:])
print('---')
# 传入整数索引 和切片

print(arr2d[1,:2])
print(arr2d[2,:1])
#  : 冒号 表示选取整个轴


# 布尔型索引
names = np.array(['a','b','c','d','e','f','g'])
# 使用 random 的randn 生成，7行4列的 正态分布的随机数据
data = np.random.randn(7,4)

print(names)
print(data)