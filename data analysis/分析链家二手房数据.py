# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['simhei','Arial']})

#检测python版本
from sys import version_info
if version_info.major != 3 :
    raise Exception('请使用python3来完成项目')

#导入链家二手房数据
lianjia_df = pd.read_csv('lianjia.csv')
# display(lianjia_df.head(n=2))
# lianjia_df.info()
# lianjia_df.describe()

#添加新特征房屋均价
df = lianjia_df.copy()
df['PerPrice'] =lianjia_df['Price'] / lianjia_df['Size']

#重新摆放列位置
columns = ['Region','District','Garden','Layout','Floor','Year','Size','Elevator','Direction','Renovation','PerPrice','Price']
df = pd.DataFrame(df,columns= columns)

#重新审视数据
# display(df.head(n=2))


#数据可视化分析
#Region特征分析
#对二手房区域分组对比二手房数量和每平米房价

df_house_count = df.groupby('Region')['Price'].count().sort_values(ascending=False).to_frame().reset_index()
df_house_mean = df.groupby('Region')['PerPrice'].mean().sort_values(ascending=False).to_frame().reset_index()

f,[ax1,ax2,ax3] = plt.subplots(3,1,figsize=(20,15))
sns.barplot(x='Region', y='PerPrice', palette="Blues_d", data=df_house_mean, ax=ax1)
ax1.set_title('北京各大区二手房每平米单价对比',fontsize=15)
ax1.set_xlabel('区域')
ax1.set_ylabel('每平米单价')

sns.barplot(x='Region', y='Price', palette="Greens_d", data=df_house_count, ax=ax2)
ax2.set_title('北京各大区二手房数量对比',fontsize=15)
ax2.set_xlabel('区域')
ax2.set_ylabel('数量')

sns.boxplot(x='Region', y='Price', data=df, ax=ax3)
ax3.set_title('北京各大区二手房房屋总价',fontsize=15)
ax3.set_xlabel('区域')
ax3.set_ylabel('房屋总价')

plt.show()
