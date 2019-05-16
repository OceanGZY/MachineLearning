#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
    知识点：
        - 排列
        - 索引
        - 交叉
        - 透视表
        - 数据探索
'''
'''
    pandas主要方法：
        - 类似SQL的方式，对.csv,.tsv,.xlsx等格式数据进行处理
        - 主要数据结构：
            - Series ，类似一维数组对象，一组数据
            - DataFrame ， 二维数据结构，一张表格，每列数据的类型相同； 可视为Series 构成的字典
'''
import numpy as np
import pandas as pd
import warnings

# 读取数据
df = pd.read_csv('./telecom_churn.csv')
# 查看前5行
print(df.head())
# 查看数据的维度、特征名称、特征类型
print(df.shape)
# 查看数据列名
print(df.columns)

# 查看该DataFrame的总体信息
print(df.info())
'''
    数据类型：
        bool
        int64
        float64
        object
'''
# astype()可更改数据类型
df['Churn'] =df['Churn'].astype('int64')
# describe() 可以显示数值特征（int64 、float64），如未缺失值的数值，均值，标砖茶，范围，四分位数
print(df.describe())
# include 参数显示指定的数据类型
print(df.describe(include=['object','bool']))
# value_counts() 查看类别（类别为object）和布尔值(bool)的特征; 加上 normalize=True参数可以显示比例
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))

'''
    排序：
        DataFrame 根据某个变量的值（列），进行排序
'''
# sort_values(); ascending=False时倒序排列
print(df.sort_values(by='Total day charge',ascending=False).head())
# 多个排序参数
print(df.sort_values(by=['Churn','Total day charge'],ascending=[True,False]).head())

'''
    索引获取数据：
        
'''