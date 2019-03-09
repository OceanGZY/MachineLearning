#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''

import numpy as np

# 1 byte 字节   8 位 bit
# 双精度浮点
arr1 = np.array([1,2,3] , dtype= np.float64)
print(arr1)
print(arr1.dtype)
print('----')

arr2 = np.array([4,5,6] , dtype= np.int32)
print(arr2)
print(arr2.dtype)
print('-----')


# 类型转换
arr = np.array([1,2,3,4,5])
print(arr.dtype)
print('---')
# 整型转为浮点型
float_arr = arr.astype(np.float64)
print(float_arr)
print(float_arr.dtype)

# 浮点型转为 整型 会把小数点后截断,不会四舍五入
arr3 = np.array([3.7,-1.2,-2.6,0.5,12.9,10.1])
print(arr3)
print(arr3.dtype)
print(arr3.astype(np.int32))

# dtype传递使用
int_arr = np.arange(10)
calibers = np.array([.22,.270,.357], dtype= np.float64)
print(int_arr)
print(int_arr.dtype)

print(int_arr.astype(calibers.dtype))
