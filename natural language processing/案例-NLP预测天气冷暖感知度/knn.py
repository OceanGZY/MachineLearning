#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
    author:gzy
    date:20190301
    version:0.1.0
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import math
class Temperature():
    #设置字字体，避免汉字乱码
    siyuanyahei = FontProperties(fname="../SourceHanSans-Normal.otf")

    #创建数据集，返回数据集和类标签
    def create_dataset(self):
        # 数据集
        datasets = np.array([[8,4,2],[7,1,1],[1,4,4],[3,0,5]])
        #类标签
        labels = ['非常热','非常热','一般热','一般热']
        return datasets,labels
    #可视化分析数据
    def analyze_data_plot(self,x,y):
        fig = plt.figure()
        # 将画布分隔为1行1列一块
        ax = fig.add_subplot(111)
        ax.scatter(x,y)

        #设置散点图的标题和横纵坐标
        plt.title('游客领热感知散点图',fontproperties=self.siyuanyahei)
        plt.xlabel('天热吃冰激凌数目',fontproperties=self.siyuanyahei)
        plt.ylabel('天热喝水的数目',fontproperties=self.siyuanyahei)
        plt.show()


    #构造KNN分类器
    def knn_Classifier(self,newV,datasets,labels,k):
        # 获取新样本的数据
        # 获取样本库的数据
        # 选择K值
        # 计算样本数据与样本库数据之间的距离
        # 根据距离进行排序
        # 针对k个点，统计各个类别的数量
        # 投票机制，少数服从多数原则

        pass


    # 欧氏距离计算
    #方式一 d^2 = （x1-x2）^2 +(y1-y2)^2"
    def ComputeEuclideanDistance(self,x1,x2,y1,y2):
        d = math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
        return d

    # 方式二
    def EuclideanDistance(self,instance1,instance2,length):
        d =0
        for x in range(length):
            d += pow((instance1[x]-instance2[x]),2)
        return math.sqrt(d)


if __name__ =='__main__':
    temperature = Temperature()
    # 创建数据集和类标签
    datasets,labels= temperature.create_dataset()
    print('数据集：','\n',datasets,'\n','类标签:','\n',labels)
    # 可视化数据
    temperature.analyze_data_plot(datasets[:,0],datasets[:,1])
    # 欧氏距离计算
    #方式一
    d1 = temperature.ComputeEuclideanDistance(2,4,8,2)
    print(d1)
    #方式二
    d2 = temperature.EuclideanDistance([2,4],[8,2],2)
    print(d2)

    # KNN分类器
    newV = [2,4,0]
    temperature.knn_Classifier(newV,datasets,labels,2)

