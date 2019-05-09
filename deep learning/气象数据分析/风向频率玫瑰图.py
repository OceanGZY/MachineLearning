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

## 风向频率涉及因素
### 风速，风向（力）

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

# print(df_ferrara[['wind_speed','wind_deg','day']])
## 散点图效果
# fig,ax = plt.subplots()
# ax.plot(df_ferrara['wind_speed'],df_ferrara['wind_deg'],'ro')
# plt.show()
## 散点图的效果并不理想，使用其它360度，改为使用极区图
hist,bins  = np.histogram(df_ferrara['wind_deg'],8,[0,360])


def showRoseWind(values,city_name,max_value):
    N = 8

    # theta = [pi*1/4, pi*2/4, pi*3/4, ..., pi*2]
    theta = np.arange(2 * np.pi / 16, 2 * np.pi, 2 * np.pi / 8)
    radii = np.array(values)
    fig1 = plt.subplots()
    # 绘制极区图的坐标系
    plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)

    # 列表中包含的是每一个扇区的 rgb 值，x越大，对应的color越接近蓝色
    colors = [(1-x/max_value, 1-x/max_value, 0.75) for x in radii]

    # 画出每个扇区
    plt.bar(theta, radii, width=(2*np.pi/N), bottom=0.0, color=colors)

    # 设置极区图的标题
    plt.title(city_name, x=0.2, fontsize=20)
    plt.show()


showRoseWind(hist,'Ferrara',15)
