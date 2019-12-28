# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.font_manager import FontProperties

siyuanheiti = FontProperties(fname='../natural language processing/SourceHanSans-Normal.otf')

#检测python版本
from sys import version_info
if version_info.major != 3 :
    raise Exception('请使用python3来完成项目')

#导入链家二手房数据
lianjia_df = pd.read_csv('lianjia.csv')
# # 展示数据前几行
# display(lianjia_df.head(n=2))
# # 检查缺失值情况
# lianjia_df.info()
# lianjia_df.describe()

#添加新特征房屋均价
df = lianjia_df.copy()
df = df[(df['Layout']!='叠拼别墅')&(df['Size']<1000)]
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

# 不同地区的二手房房每平米单价
f1,ax1 = plt.subplots(1,1,figsize=(20,7))
sns.barplot(x='Region', y='PerPrice', palette="Blues_d", data=df_house_mean, ax=ax1)
plt.xticks(fontproperties=siyuanheiti)
ax1.set_title('北京各大区二手房每平米单价对比',fontsize=15,fontproperties = siyuanheiti)
ax1.set_xlabel('区域',fontproperties = siyuanheiti)
ax1.set_ylabel('每平米单价',fontproperties = siyuanheiti)

# 不同地区的二手房数量
f2,ax2 = plt.subplots(1,1,figsize=(20,7))
sns.barplot(x='Region', y='Price', palette="Greens_d", data=df_house_count, ax=ax2)
plt.xticks(fontproperties=siyuanheiti)
ax2.set_title('北京各大区二手房数量对比',fontsize=15,fontproperties = siyuanheiti)
ax2.set_xlabel('区域',fontproperties = siyuanheiti)
ax2.set_ylabel('数量',fontproperties = siyuanheiti)

# 不同地区的二手总价
f3,ax3 = plt.subplots(1,1,figsize=(20,7))
sns.boxplot(x='Region', y='Price', data=df, ax=ax3)
plt.xticks(fontproperties=siyuanheiti)
ax3.set_title('北京各大区二手房房屋总价',fontsize=15,fontproperties = siyuanheiti)
ax3.set_xlabel('区域',fontproperties = siyuanheiti)
ax3.set_ylabel('房屋总价',fontproperties = siyuanheiti)

# plt.show()

# size特征分析
# 房间大小
f4,[ax4,ax5] = plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['Size'],bins=20,ax=ax4,color='r')
sns.kdeplot(df['Size'],shade=True,ax=ax4)
ax4.set_title("北京各大区二手房大小分布",fontproperties=siyuanheiti)
sns.regplot(x='Size',y='Price',data=df,ax=ax5)
ax5.set_title("北京各大区二手房大小与价格分布",fontproperties=siyuanheiti)
plt.show()


# # 面积小于10的
# # print(df.loc[df['Size']<10])
# # 面积大于1000的
# print(df.loc[df['Size']> 1000])

# 户型特征分析
# Layout特征分析
f,ax6 = plt.subplots(1,1,figsize=(20,20))
sns.countplot(y='Layout',data=df,ax=ax6)
ax6.set_title("北京各大区户型特征",fontproperties=siyuanheiti)
ax6.set_xlabel('数量',fontproperties=siyuanheiti)
ax6.set_ylabel('户型',fontproperties=siyuanheiti)
plt.yticks(fontproperties=siyuanheiti)
# plt.show()


# 装修特征分析
# Renovation特征分析
# print(df['Renovation'].value_counts())
f,ax7=plt.subplots(1,1,figsize=(6,4))
plt.xticks(fontproperties=siyuanheiti)
sns.countplot(df['Renovation'],ax=ax7)

f,ax8=plt.subplots(1,1,figsize=(6,4))
plt.xticks(fontproperties=siyuanheiti)
sns.barplot(x='Renovation',y='Price',data=df,ax=ax8)

f,ax9=plt.subplots(1,1,figsize=(6,4))
plt.xticks(fontproperties=siyuanheiti)
sns.boxplot(x='Renovation',y='Price',data=df,ax=ax9)

# plt.show()

# 电梯数据分析
# Elevator数据分析
# # 空电梯数据量
# print(df.loc[df['Elevator'].isnull(),'Elevator'])
# 移除缺失值
df['Elevator'] = df.loc[(df['Elevator'] == '有电梯')|(df['Elevator'] == '无电梯'),'Elevator']
# 填补缺失值
df.loc[(df['Floor']>6) & (df['Elevator'].isnull()),'Elevator'] = '有电梯'
df.loc[(df['Floor']<=6) & (df['Elevator'].isnull()),'Elevator'] = '无电梯'
# # 填补数据后的数据情况
# print(df.loc[df['Elevator'].isnull(),'Elevator'])

f,ax10 = plt.subplots(1,1,figsize=(10,10))
sns.countplot(df['Elevator'],ax=ax10)
ax10.set_title("有无电梯的数量对比",fontproperties=siyuanheiti)
ax10.set_xlabel("是否有电梯",fontproperties=siyuanheiti)
ax10.set_ylabel("数量",fontproperties=siyuanheiti)
plt.yticks(fontproperties=siyuanheiti)
plt.xticks(fontproperties=siyuanheiti)

f,ax11=plt.subplots(1,1,figsize=(10,10))
sns.barplot(x='Elevator',y='Price',data=df,ax=ax11)
ax11.set_title("有无电梯的房价对比",fontproperties=siyuanheiti)
ax11.set_xlabel("是否有电梯",fontproperties=siyuanheiti)
ax11.set_ylabel("房价",fontproperties=siyuanheiti)
plt.yticks(fontproperties=siyuanheiti)
plt.xticks(fontproperties=siyuanheiti)

# 年限特征分析
# year特征分析
grid = sns.FacetGrid(df,row='Elevator',col='Renovation',palette='seismic',height=4)
grid.map(plt.scatter,'Year','Price')
grid.add_legend()


# floor特征
f,ax12=plt.subplots(1,1,figsize=(20,5))
sns.countplot(x='Floor',data=df,ax=ax12)
ax12.set_title("房屋户型",fontproperties=siyuanheiti)
ax12.set_xlabel("楼层数",fontproperties=siyuanheiti)
ax12.set_ylabel("数量",fontproperties=siyuanheiti)

plt.show()
