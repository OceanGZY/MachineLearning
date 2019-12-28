#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
    date:20190505
    author:gzy
'''
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import os

# 基于多层感知训练模型(DNN) ，用于预测手写数字图片

# 数据准备
BUF_SIZE = 512
BATCH_SIZE = 128
# 下载训练数据集,每次缓存BUF_SIZE个数据项，每BATCH_SIZE个组成batch
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(),
        buf_size=BUF_SIZE,
    ),
    batch_size=BATCH_SIZE
)
# 下载测试数据集,每次缓存BUF_SIZE个数据项，每BATCH_SIZE个组成batch
test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.test(),
        buf_size=BUF_SIZE
    ),
    batch_size=BATCH_SIZE
)

train_data = paddle.dataset.mnist.train()
simple_data = next(train_data())
print(simple_data)


# 配置网络
# 定义多层感知器
def multilayer_perceptron(input):
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')

    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')

    prediction = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return prediction


# 定义数据层
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')

label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
predict = multilayer_perceptron(image)

# 定义损失函数和准确率
# 使用交叉熵函数，描述真实样本标签与预测概率之间的误差
cost = fluid.layers.cross_entropy(input=predict, label=label)

avg_cost = fluid.layers.mean(cost)
# 计算分类准确率
acc = fluid.layers.accuracy(input=predict, label=label)

# 定义优化函数
# 使用adam算法优化，学习率设定为0.001
optimizer = fluid.optimizer.AdamaxOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

# 模型训练
# 创建训练的Executor
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
test_program = fluid.default_main_program().clone(for_test=True)
exe = fluid.Executor(place)
exe.run(fluid.default_main_program())

# 告知网络传入数据分两部分，image，label
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 展示模型曲线
all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []


def draw_train_process(title, iters, costs, accs, label_cost, label_acc):
    plt.title(title, fontsize=24)
    plt.xlabel('iter', fontsize=20)
    plt.ylabel('cost/acc', fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=label_acc)
    plt.legend()
    plt.grid()
    plt.show()


exe.run(fluid.default_startup_program())
# 训练并保存模型
EPOCH_NUM = 2
model_save_dir = os.getcwd() + '/work/hand.inference.model'
for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        # 遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)

        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        # 每200个batch打印一次信息 误差 准确率
        if batch_id % 200 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    # 每训练一轮， 进行一次测试
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])

    # 测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

# 保存模型
# 如果路径不存在则创建
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
print('Save model to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,
                              ['image'],
                              [predict],
                              exe)
print("模型训练保存完成")
draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, 'training cost', 'training acc')

# 模型评估
