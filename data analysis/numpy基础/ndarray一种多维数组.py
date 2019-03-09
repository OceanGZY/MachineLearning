#!/usr/bin/python
#-*-coding:utf-8 -*-
'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''
'''
    array:      将输入数据(列表、元组、数组)转换为ndarray
    asarray:    将输入转换为ndarray 
    arange:     返回一个ndarry,不是列表
    ones、ones_like:     根据形状和dtype创建全1数组
    zeros、zeros_like:   根据形状和dtype创建全0数组
    empty、empty_like:   创建新数组，但不填充数值
    eye、identify:       创建单位矩阵
'''



import numpy as np
data =np.random.rand(2,3)
print(data)
print('--------------')
print(data *10)
print('--------------')
print(data+data)
print('--------------')
print(data.shape)
print('--------------')

# 创建ndarray
# 列表转为ndarray
# 一维数组
data1 = [6,7.5,8,0,1]
arr1 =np.array(data1)
print(arr1)
print('-----------')
print(arr1.shape)
print('-------------')

# 二维数组
data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
print(arr2)
print('----------')
print(arr2.shape)

# 三维数组
data3 = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
arr3 = np.array(data3)
print(arr3.shape)
print(arr3)
print('-----')

# 内置的arange函数
arr3 = np.arange(15)
print(arr3)
print(arr3.shape)
print(arr3.dtype)


