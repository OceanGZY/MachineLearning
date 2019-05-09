#!/usr/bin/python
# -*-coding:utf-8 -*-
'''
    auth:gzy
    date:20190505
    version:0.1.0
'''

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser

df_ferrara = pd.read_csv("WeatherData/ferrara_270615.csv")
df_milano = pd.read_csv("WeatherData/milano_270615.csv")
df_mantova = pd.read_csv("WeatherData/mantova_270615.csv")
df_ravenna = pd.read_csv("WeatherData/ravenna_270615.csv")
df_torino = pd.read_csv("WeatherData/torino_270615.csv")
df_asti = pd.read_csv("WeatherData/asti_270615.csv")
df_bologna = pd.read_csv("WeatherData/bologna_270615.csv")
df_piacenza = pd.read_csv("WeatherData/piacenza_270615.csv")
df_cesena = pd.read_csv("WeatherData/cesena_270615.csv")
df_faenza = pd.read_csv("WeatherData/faenza_270615.csv")

# print(df_ferrara)

# 温度数据分析
## 以Ferrara为例子

### 取出需要分析的温度和日期数据
y1_ferrara = df_ferrara['temp']
x1_ferrara = df_ferrara['day']

y2_milano = df_milano['temp']
x2_milano = df_milano['day']

y3_mantova = df_mantova['temp']
x3_mantova = df_mantova['day']

y4_ravenna = df_ravenna['temp']
x4_ravenna = df_ravenna['day']

y5_torino = df_torino['temp']
x5_torino = df_torino['day']

y6_asti = df_asti['temp']
x6_asti = df_asti['day']

y7_bologna = df_bologna['temp']
x7_bologna = df_bologna['day']

y8_piacenza = df_piacenza['temp']
x8_piacenza = df_piacenza['day']

y9_cesena = df_cesena['temp']
x9_cesena = df_cesena['day']

y10_faenza = df_faenza['temp']
x10_faenza = df_faenza['day']


### 把日期数据转换为datetime 格式
day_ferrara = [parser.parse(x) for x in x1_ferrara]
day_milano = [parser.parse(x) for x in x2_milano]  #数据存在缺失
day_mantova = [parser.parse(x) for x in x3_mantova]
day_ravenna = [parser.parse(x) for x in x4_ravenna]  # 数据存在缺失
day_torino = [parser.parse(x) for x in x5_torino]
day_asti = [parser.parse(x) for x in x6_asti]
day_bologna = [parser.parse(x) for x in x7_bologna]
day_piacenza = [parser.parse(x) for x in x8_piacenza]
day_cesena = [parser.parse(x) for x in x9_cesena]
day_faenza = [parser.parse(x) for x in x10_faenza]

# print(day_ferrara)
# print(day_milano)  #数据存在缺失
# print(day_mantova)
# print(day_ravenna)  # 数据存在缺失
# print(day_torino)
# print(day_asti)
# print(day_bologna)
# print(day_piacenza)
# print(day_cesena)
# print(day_faenza)

### 调用subplot函数，fig图像对象，ax坐标轴对象
fig,ax = plt.subplots()

### 设定时间的格式
hours = mdates.DateFormatter('%H:%M')

### 设定x轴显示的格式
ax.xaxis.set_major_formatter(hours)

### 画出图像， day_ferrara是X轴数据，y1_ferrara是Y轴数据， 'r'代表是red 红色
ax.plot(day_ravenna,y4_ravenna,'r',day_ferrara,y1_ferrara,'r',day_cesena,y9_cesena,'r')
ax.plot(day_asti,y6_asti,'g',day_torino,y5_torino,'g',day_piacenza,y8_piacenza,'g')

# plt.show()



# 收集10个城市的最高温和最低温，用线性图 表示气温最值点和离海远近之间的关系
## dist  城市距离海边距离的列表
dist = [
    df_ravenna['dist'][0],
    df_cesena['dist'][0],
    df_faenza['dist'][0],
    df_ferrara['dist'][0],
    df_bologna['dist'][0],
    df_mantova['dist'][0],
    df_piacenza['dist'][0],
    df_milano['dist'][0],
    df_asti['dist'][0],
    df_torino['dist'][0]
]

print(dist)

## temp_max 存放城市的最高温度的列表
temp_max = [
    df_ravenna['temp'].max(),
    df_cesena['temp'].max(),
    df_faenza['temp'].max(),
    df_ferrara['temp'].max(),
    df_bologna['temp'].max(),
    df_mantova['temp'].max(),
    df_piacenza['temp'].max(),
    df_milano['temp'].max(),
    df_asti['temp'].max(),
    df_torino['temp'].max()
]
# print(temp_max)


## temp_min 存放城市的最低温度的列表
temp_min = [
    df_ravenna['temp'].min(),
    df_cesena['temp'].min(),
    df_faenza['temp'].min(),
    df_ferrara['temp'].min(),
    df_bologna['temp'].min(),
    df_mantova['temp'].min(),
    df_piacenza['temp'].min(),
    df_milano['temp'].min(),
    df_asti['temp'].min(),
    df_torino['temp'].min()
]

# print(temp_min)

fig1,ax1 = plt.subplots()
ax1.plot(dist,temp_max,'ro')

# plt.show()


from sklearn.svm import SVR

# dist1 是靠近海的城市集合
# dist2 是远离海的集合