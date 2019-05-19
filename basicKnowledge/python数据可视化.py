#!/usr/bin/python
# -*-coding:utf-8-*-
'''
    数据集；
    单变量可视化；
    多变量可视化；
    全局数据集可视化；
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# 继续使用电信离网数据
df = pd.read_csv('./telecom_churn.csv')
print(df.head())

'''
    单变量可视化：一次只关注一个变量
        - 数量特征
            数量特征的值为有序数值：可能离散、可能连续
        - 直方图和密度图
            直方图：按照相等间隔，将值分成柱；（高斯分布，指数分布）；
            密度图：核密度图，可视为直方图的平滑版
'''
# 使用DataFrame的hist()方法绘制直方图
features = ['Total day minutes', 'Total intl calls']
df[features].hist(figsize=(10, 4))
plt.show()

#  创建密度图
df[features].plot(kind='density', subplots=True, layout=(1, 2),
                  sharex=False, figsize=(10, 4), legend=False, title=features)
plt.show()

# 使用seaborn 的distplot() ，同时显示直方图和密度图
sns.distplot(df['Total intl calls'])
plt.show()

'''
    箱线图：主要是箱子，须，和单独的数据点（离散值）    
'''
# 使用seaborn的 boxplot()绘制箱型图
sns.boxplot(x='Total intl calls', data=df)
plt.show()

'''
    提琴型图：聚焦于平滑后的整体分布
'''
# 使用seaborn的violinplot()
f, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.boxplot(data=df['Total intl calls'], ax=axes[0])
sns.violinplot(data=df['Total intl calls'], ax=axes[1])
plt.show()
'''
    数据描述
'''
# DataFrame的 describe()来获取分布的精确数值统计
print(df[features].describe())

'''
    类别特征和二元特征
        - 类别特征：反映样本的某个定性属性，具有固定数目的值，每个值将一个观测数据分配到相应的组，这些组成为类别
        - 二元特征：类别特征的特例，其可能值 有2个
'''

'''
    频率表
'''
# value_counts()获得一张频率表
print(df['Churn'].value_counts())

'''
    条形图 和直方图区别
        - 直方图适合查看数值变量的分布，而条形图用于查看类别特征。
        - 直方图的 X 轴是数值；条形图的 X 轴可能是任何类型，如数字、字符串、布尔值。
        - 直方图的 X 轴是一个笛卡尔坐标轴；条形图的顺序则没有事先定义。        
'''
# seabornd的countplot()，绘制条形图
f, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
sns.countplot(x='Churn', data=df, ax=axes1[0])
sns.countplot(x='Customer service calls', data=df, ax=axes1[1])
plt.show()

'''
    多变量可视化
            多变量图形可在单张图像中查看两个以上变量的联系
        - 相关矩阵
            揭示数据集中的数值变量的相关性
        - 散点图
            将两个数值变量的值显示为二维空间的笛卡尔坐标
'''

# 使用DataFrame的corr()计算每对特征键的相关性生成相关矩阵
# 使用seaborn的heatmap()把相关矩阵渲染
### 丢弃非数值变量
numerical = list(set(df.columns) - set(
    ['State', 'International plan', 'Voice mail plan', 'Area code', 'Churn', 'Customer service calls']))
### 计算和绘图
corr_matrix = df[numerical].corr()
sns.heatmap(corr_matrix)
plt.show()

# matplotlib库的scatter()方法绘制散点图
plt.scatter(df['Total day minutes'], df['Total night minutes'])
plt.show()
# seaborn的joinplot()方法在绘制散点图的同时绘制两张直方图
sns.jointplot(x='Total day minutes', y='Total night minutes', data=df, kind='scatter')
plt.show()

# seaborn的joinplot()方法在绘制平滑过的散点直方图
sns.jointplot('Total day minutes', 'Total night minutes', data=df, kind='kde', color='g')
plt.show()

# seaborn的pairplot()绘制散点矩阵图
sns.pairplot(df[numerical])
plt.show()

# seaborn的 lmplot()方法hue参数指定感兴趣的特征
sns.lmplot('Total day minutes','Total night minutes',data=df,hue='Churn',fit_reg=False)
plt.show()



'''
    全剧数据可视化
        - 降维（dimensionality reduction）
            大多数现实世界的数据集有很多特征，每一个特征都可以被看成数据空间的一个维度。
            为了从整体上查看一个数据集，需要在不损失很多数据信息的前提下，降低用于可视化的维度。
            降维是一个无监督学习（unsupervised learning）问题，因为它需要在不借助任何监督输入（如标签）的前提下，从数据自身得到新的低维特征。
        - tSNE 
            流形学习（Manifold Learning）
            为高维特征空间在二维平面（或三维平面）上寻找一个投影，使得在原本的 n 维空间中相距很远的数据点在二维平面上同样相距较远，而原本相近的点在平面上仍然相近。
        
'''