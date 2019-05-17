#!/usr/bin/python
# -*-coding:utf-8-*-
'''
    date:2019-05-05
    version:0.1.0
'''

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./adult.data.csv')
print(data.head())
'''
    问题一：数据集中有多少男性和女性
'''
print(data['sex'].value_counts())
