#!/usr/bin/python3
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
data_train = pd.read_csv('train.csv')
siyuanheiti = FontProperties(fname='../SourceHanSans-Normal.otf')

# 乘客属性分布
fig = plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid((2,3),(0,0))

data_train.Survived.value_counts().plot(kind='bar')
plt.title("获救情况",fontproperties = siyuanheiti)
plt.ylabel('人数',fontproperties=siyuanheiti)

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title("乘客等级分布",fontproperties=siyuanheiti)
plt.ylabel("人数",fontproperties=siyuanheiti)

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel("年龄",fontproperties=siyuanheiti)
plt.grid(b=True,which='major',axis='y')
plt.title("按年龄看获救分布",fontproperties=siyuanheiti)

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("年龄",fontproperties=siyuanheiti)
plt.ylabel("密度",fontproperties=siyuanheiti)
plt.title("各等级的乘客年龄分布",fontproperties=siyuanheiti)
plt.legend(("头等舱","2等舱","3等舱"),loc='best',prop=siyuanheiti)

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数",fontproperties=siyuanheiti)
plt.ylabel("人数",fontproperties=siyuanheiti)

# 查看不同等级乘客的获救情况
fig1 = plt.figure()
fig1.set(alpha=0.2)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
df.plot(kind="bar",stacked=True)
plt.legend(prop=siyuanheiti)
plt.title("各乘客等级的获救情况",fontproperties=siyuanheiti)
plt.xlabel("乘客等级",fontproperties=siyuanheiti)
plt.ylabel("人数",fontproperties=siyuanheiti)

# 查看不同性别的获救情况
fig2 = plt.figure()
fig2.set(alpha=0.2)
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'男性':Survived_m,'女性':Survived_f})
df.plot(kind='bar',stacked=True)
plt.legend(prop=siyuanheiti)
plt.title('按性别看获救情况',fontproperties=siyuanheiti)
plt.xlabel('性别',fontproperties=siyuanheiti)
plt.ylabel('人数',fontproperties=siyuanheiti)

# 各种舱级别情况下各性别的获救情况
fig3 = plt.figure()
fig3.set(alpha=0.2)
plt.title('根据舱等级和性别的获救情况',fontproperties=siyuanheiti)

ax1 = fig3.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',label="female highclass",color='#FA2479')
ax1.set_xticklabels(['获救数','未获救数'],rotation =0,fontproperties=siyuanheiti)
plt.legend(['女性/高级舱'],prop=siyuanheiti,loc='best')

ax2 = fig3.add_subplot(142,sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',label = "female lowclass",color='pink')
ax2.set_xticklabels(["未获救数","获救数"],rotation=0,fontproperties=siyuanheiti)
plt.legend(['女性/低级舱'],prop=siyuanheiti,loc='best')


plt.show()