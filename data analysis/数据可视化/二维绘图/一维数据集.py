#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(1000)
# 生成20个标准正态分布（伪）随机数，保存在一个 numpy的 ndarray中
y = np.random.standard_normal(20)
# pyplot 的plot函数
# x, 包含X 坐标（横坐标）的列表 或 数组
# y, 包含Y 坐标（纵坐标）的列表 活 数组
x = range(len(y))
plt.plot(x,y)
plt.show()

'''
    plt.axis函数选项
    Empty ：返回当前坐标轴限值
    off : 关闭坐标轴线和标签
    equal : 使用等刻度
    scaled : 通过尺寸变化平衡刻度
    tight : 使所有数据可见，（缩小限值）
    image : 使所有数据可见 （使用数据限值）
    [xmin , xmax , ymin ,ymax] : 将设置限制为 给定的一组值
    plt.xlim()
    plt.ylim()
'''

plt.plot(y.cumsum())
plt.grid(True)
plt.axis('tight')
plt.show()

