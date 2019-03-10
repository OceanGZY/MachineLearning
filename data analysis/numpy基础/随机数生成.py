#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''

import numpy as np

# 使用normal生成一个标准正态分布的 4*4样本
samples = np.random.normal(size=(4,4))
print(samples)

'''
    numpy.random函数
    seed : 确定随机数生成器的种子
    permutation : 返回一个序列的随机排列 /  返回一个随机排列的范围 
    shuffle : 对一个序列 就地随机排列
    rand : 产生均匀分布的样本值
    randint : 从给定的上下限范围内随机选取整数
    randn : 产生正态分布（平均值0 ，标准差为1）的样本值，
    binomial : 产生二项分布的样本值
    normal : 产生正态（高斯）分布的样本值 
    beta : 产生Beta分布的样本值
    chisquare : 产生卡方分布的样本值
    gamma : 产生Gamma 分布的样本值
    uniform : 产生[0,1]中均匀分布的样本值
    
'''