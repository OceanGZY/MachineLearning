# -*- coding:utf-8 -*-
'''
    date:2018/11/14
    version:0.1.0
    auth:YSilhouette
'''

import numpy
import scipy.special

class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #初始化神经网络，设置输入层，中间层，和输出层节点数
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #设置学习率
        self.lr = learningrate
        '''
            初始化权重矩阵，
            wih表示输入层和中间层节点间链路权重形成的矩阵，
            who表示中间层和输出层间 链路权重形成的矩阵
            
        '''
        self.wih = numpy.random.rand(self.hnodes,self.innodes) - 0.5
        self.who = numpy.random.rand(self.onodes,self.innodes) - 0.5

        self.actication_function = lambda x:scipy.special.expit(x)


        pass
    def train(self):
        #根据输入的训练数据更新节点链路权重
        pass
    def query(self,inputs):
        #根据输入数据计算并输出答案
        #计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih,inputs)

        #计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.actication_function(hidden_inputs)

        #计算最外层收到的信号量
        final_inputs = numpy.dot(self.who,hidden_outputs)

        #计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.actication_function(final_inputs)

        print(final_outputs)
        pass


if __name__ =='__main__':
    input_nodes= 3
    hidden_nodes =3
    output_nodes =3

    learning_rate = 0.3

    n = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)
    n.query([1.0,0.5,-1.5])