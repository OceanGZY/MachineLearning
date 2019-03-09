#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''

import numpy as np

arr =np.array([[1.,2.,3.],[4.,5.,6.]])
print(arr)
print('---')
#大小相等的数组，之间的任何算术运算，运算会传播到各个元素

# 大小相等的数组相乘，同位置相乘
print(arr * arr)
print('---')
# 大小相等的数组相减 ，同位置相减
print(arr -arr)
print('---')
print(1/arr)
print('----')
print(arr ** 0.5)


# 不同大小的数组之间的运算，叫广播（broadcasting）