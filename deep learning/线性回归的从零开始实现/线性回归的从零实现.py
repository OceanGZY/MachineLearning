from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd,nd
import random

# 生成数据集
# 设训练数据集样本数为1000，输入个数（特征数）为2。给定随机生成的批量样本特征
# 使用线性回归模型真实权重w ,偏差b ,一个随机噪声e , 来生成标签y

num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels = labels + nd.random.normal(scale=0.01 , shape=labels.shape)

# features 的每一行是一个长度为2的向量
# lebels 的每一行是一个长度为 1 的向量
# print(features[0],labels[0])

def use_svg_display():
    # 矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize =(3.5,2,5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

# 读取数据
# 在训练模型的时候，需要遍历数据集并不断读取小批量数据样本
# 定义一个函数 每次返回batch_size(批量大小)个随机样本的特征和标签

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = nd.array(indices[i:min(i + batch_size ,num_examples)])
        yield features.take(j),labels.take(j)


batch_size = 10

for x,y in data_iter(batch_size,features,labels):
    print(x,y)
    break



# 初始化模型参数
# 将权重初始化成均值为0 标准差为0.01的正态随机数，偏差初始化为0
w = nd.random.normal(scale=0.01,shape=(num_inputs,1))
b = nd.zeros(shape=(1,))
# 之后模型训练，需要对这些参数求梯度来迭代参数的值， 创建他们的梯度
w.attach_grad()
b.attach_grad()

# 定义模型
# 线性回归的矢量计算表达式实现，使用dot函数做矩阵乘法
def linreg(X,w,b):
    return nd.dot(X,w) + b

# 定义损失函数
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 /2

# 定义优化算法
# sgd函数实现小批量随机梯度下降，通过不断迭代来优化损失函数
def sgd(params,lr,batch_size):
    for param in params:
        param[:] = param -lr * param.grad /batch_size


# 训练模型
# 训练中 多次迭代模型参数
# 每次迭代根据当前读取的小批量数据样本（特征X，标签y）, 通过调用反向函数backward计算小批量随机梯度
# 并调用优化算法sgd迭代模型参数

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        with autograd.record():
            l = loss()