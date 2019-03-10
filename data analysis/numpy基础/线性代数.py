#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''
import numpy as np

x = np.array([[1.,2.,3.],[4.,5.,6.]])
print(x)
print('---')
y = np.array([[6.,23.],[-1,7],[8,9]])
print(y)
print('--')
# x与y 点积 ，
# dot函数，用于矩阵乘法
print(x.dot(y))
print('---')
print(np.dot(x,y))
print('---')

print(np.dot(x,np.ones(3)))


'''
    numpy.linalg 函数
    diag : 以一维数组的形式返回方阵的对角线（非对角线）元素； 将一维数组转换为方阵（非对角线元素为0）
    dot : 矩阵乘法
    trace ：计算对角线元素的和
    det : 计算矩阵的行列式
    eig : 计算方阵的本征值 和 本征向量
    inv ：计算方阵的逆
    pinv : 计算矩阵的Moore-Penrose伪逆
    qr : 计算QR分解
    svd : 计算奇异值分解（SVD）
    solve : 解线性方程组 Ax = b ,其中A的一个方阵
    lstsq : 计算Ax = b,最小二乘解
'''