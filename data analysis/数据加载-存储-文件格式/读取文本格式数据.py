#!/usr/bin/python
#-*-coding:utf-8 -*-
'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''

'''
    pandas中读取文件的函数
    pandas.read_csv :  从文件、URL、文件型对象中加载带分隔符的数据， 默认分隔符为 逗号 ，
    pandas.read_table : 从文件、URL、文件型对象中加载带分隔符的数据， 默认分隔符为 制表符   /t
    pandas.read_fwf : 读取定宽列格式的数据
    pandas.read_clipboard : 读取剪贴板中的数据，多用于讲网也转为 表格时候使用
'''

import pandas as pd
import numpy as np
import os

path = os.path.abspath(os.path.join(os.getcwd(),'..')) +'\database\examples\ex1.csv'
df1 = pd.read_csv(path)
print(df1)
print('---')
df2 = pd.read_table(path,sep =',')
print(df2)



