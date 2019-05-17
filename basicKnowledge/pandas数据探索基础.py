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
# 分析电信运营商的客户离网率数据
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
df['Churn'] = df['Churn'].astype('int64')
# describe() 可以显示数值特征（int64 、float64），如未缺失值的数值，均值，标砖茶，范围，四分位数
print(df.describe())
# include 参数显示指定的数据类型
print(df.describe(include=['object', 'bool']))
# value_counts() 查看类别（类别为object）和布尔值(bool)的特征; 加上 normalize=True参数可以显示比例
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))

'''
    排序：
        DataFrame 根据某个变量的值（列），进行排序
'''
# sort_values(); ascending=False时倒序排列
print(df.sort_values(by='Total day charge', ascending=False).head())
# 多个排序参数
print(df.sort_values(by=['Churn', 'Total day charge'], ascending=[True, False]).head())

'''
    索引获取数据：
        通过列名，行名，行号进行索引
        loc ,通过名称索引
        iloc ,通过数字索引
        
'''
# 离网率的均值
print(df['Churn'].mean())

# 布尔值索引，df[P[df['Name]]] 检测Name列每个元素的满足条件P的，最后在dataframe中输出满足P条件的行
print(df[df['Churn'] == 1].mean())
# 离网用户白天电话总时长的均值
print(df[df['Churn'] == 1]['Total day minutes'].mean())

# 未使用国际套餐和忠实用户，拨打国际长途的最长时间
print(df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max())

# 输出数据0到5行，从State 到 Area code
print(df.loc[0:5, 'State':'Area code'])

# 输出前5行，前3列
print(df.iloc[0:5, 0:3])

# 输出首行
print(df[:1])
# 输出末行
print(df[-1:])

'''
    应用函数到单元格、行、列
        apply()
'''

# 应用函数max()至每列，输出每列最大值
print(df.apply(np.max))

'''
    分组（groupby）
        df.groupby(by=grouping_columns)[columns_to_show].function()
            根据 grouping_columns 的值进行分组，
            选中感兴趣的列columns_to_show，若未设置，则会选中所有非groupby列，
            应用一个或多个函数function
    
'''
columns_to_show = ['Total day minutes', 'Total eve minutes', 'Total night minutes']

print(df.groupby('Churn')[columns_to_show].describe(percentiles=[]))

print(df.groupby('Churn')[columns_to_show].agg([np.mean, np.std, np.min, np.max]))

'''
    汇总表
        pivot_table()建立透视表
        - values 表示需要统计数据的变量列表
        - index 表示分组数据的变量列表
        - aggfunc 表示需要计算哪些数据，总和，均值，最大值，最小值
        
        pd.crosstab()交叉表
        - 计算分组频率的特殊透视表
'''

# 查询不同区号下白天夜晚深夜电话量的均值
print(df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
                     ['Area code'],aggfunc='mean'))

#
print(pd.crosstab(df['Churn'],df['International plan']))

print(pd.crosstab(df['Churn'],df['International plan'],normalize=True))

'''
    增减DataFrame的行列
        insert() 添加列
        drop()  删除列
            - axis 参数，1，删除列，0删除行，默认值为0
            - inplace 参数,表示是否修改原始DataFrame；False表示不修改现有，返回一个新的 ； True表示修改当前
'''
# 插入一列total_calls
# 通过创建中间Series实例
total_calls=df['Total day calls']+df['Total eve calls']+df['Total night calls']+df['Total intl calls']
df.insert(loc=len(df.columns),column='Total calls',value=total_calls)
print(df.head())

# 直接添加
df['Total charge'] = df['Total day charge']+df['Total eve charge']+df['Total night charge']+df['Total intl charge']
print(df.head())

# 删除列
df.drop(['Total calls'],axis=1,inplace=True)
print(df.head())