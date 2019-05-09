#!/usr/bin/python
#-*-coding:utf-8 -*-

'''
    auth:gzy
    date:20190509
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


# 读取湿度数据
y1 = df_ravenna['humidity']
x1 = df_ravenna['day']

y2 = df_faenza['humidity']
x2 = df_faenza['day']

y3 = df_cesena['humidity']
x3 = df_cesena['day']

y4 = df_milano['humidity']
x4 = df_milano['day']

y5 = df_asti['humidity']
x5 = df_asti['day']

y6= df_torino['humidity']
x6 = df_torino['day']

# 定义fig 和ax变量
fig,ax = plt.subplots()
plt.xticks(rotation = 70) ## 其实并没有看明白为啥要转70


# 将时间 从string 转化为标准的 datetime类型

day_ravenna = [parser.parse(x) for x in x1]
day_faenza = [parser.parse(x) for x in x2]
day_cesena = [parser.parse(x) for x in x3]
day_milano = [parser.parse(x) for x in x4]
day_asti = [parser.parse(x) for x in x5]
day_torino = [parser.parse(x) for x in x6]


# 规定时间的表示方式
hours = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(hours)

# 表示在图上
ax.plot(day_ravenna,y1,'r',day_faenza,y2,'r',day_cesena,y3,'r')
ax.plot(day_milano,y4,'g',day_asti,y5,'g',day_torino,y6,'g')
# plt.show()


## 城市与离海的距离
dist = [
    df_ferrara['dist'][0],
    df_milano['dist'][0],
    df_mantova['dist'][0],
    df_ravenna['dist'][0],
    df_torino['dist'][0],
    df_asti['dist'][0],
    df_bologna['dist'][0],
    df_piacenza['dist'][0],
    df_cesena['dist'][0],
    df_faenza['dist'][0]
]

## 获取最大湿度
hum_max = [
    df_ferrara['humidity'].max(),
    df_milano['humidity'].max(),
    df_mantova['humidity'].max(),
    df_ravenna['humidity'].max(),
    df_torino['humidity'].max(),
    df_asti['humidity'].max(),
    df_bologna['humidity'].max(),
    df_piacenza['humidity'].max(),
    df_cesena['humidity'].max(),
    df_faenza['humidity'].max()
]

## 获取最小湿度
hum_min = [
    df_ferrara['humidity'].min(),
    df_milano['humidity'].min(),
    df_mantova['humidity'].min(),
    df_ravenna['humidity'].min(),
    df_torino['humidity'].min(),
    df_asti['humidity'].min(),
    df_bologna['humidity'].min(),
    df_piacenza['humidity'].min(),
    df_cesena['humidity'].min(),
    df_faenza['humidity'].min()
]
print(dist)
print(hum_max)
print(hum_min)

fig1 ,ax1 = plt.subplots()
ax1.plot(dist,hum_max,'ro')

fig2,ax2 = plt.subplots()
ax2.plot(dist,hum_min,'bo')
plt.show()
