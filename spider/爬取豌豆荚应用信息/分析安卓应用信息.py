#!/usr/bin/python
# -*-coding:utf-8 -*-
'''
    auth:gzy
    date:2019-04-27
    version:0.1.0
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

siyuanheiti = FontProperties(fname='../SourceHanSans-Normal.otf')
android_df = pd.read_csv('./安卓应用信息.csv',engine='python',encoding='utf-8',converters={'appsize':float})
df = android_df.copy()


