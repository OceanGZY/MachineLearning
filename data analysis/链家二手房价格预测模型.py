#!/usr/bin/python
#-*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.font_manager import FontProperties

siyuanheiti = FontProperties(fname='../natural language processing/SourceHanSans-Normal.otf')

# 检测python版本
from sys import version_info
if version_info.major != 3 :
    raise Exception('请使用python3来完成项目')

# 导入链家二手房数据
lianjia_df = pd.read_csv('lianjia.csv')
# 添加新特征房屋均价
df = lianjia_df.copy()
#重新摆放列位置
columns = ['Region','District','Garden','Layout','Floor','Year','Size','Elevator','Direction','Renovation','PerPrice','Price']
df = pd.DataFrame(df,columns= columns)
'''
    构建特征工程
'''

# 移除结构类型异常值和房屋大小异常值
df = df[(df['Layout']!='叠拼别墅')&(df['Size']<1000)]

# 移除电梯缺失值
df['Elevator'] = df.loc[(df['Elevator'] == '有电梯')|(df['Elevator'] == '无电梯'),'Elevator']
# 填补电梯缺失值
df.loc[(df['Floor']>6) & (df['Elevator'].isnull()),'Elevator'] = '有电梯'
df.loc[(df['Floor']<=6) & (df['Elevator'].isnull()),'Elevator'] = '无电梯'


df['Year'] = pd.qcut(df['Year'],8).astype('object')
print(df['Year'].value_counts())


