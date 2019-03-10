#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
    author:gzy
    date:2018-10-01
    version:v0.1.0
'''

## 期权定价、风险管理问题具有较强能力
## 纯Python实现
from time import time
from math import exp, sqrt ,log
from random import gauss, seed

seed(2000)
t0 =time()

# Parameters
s0 = 100.
K = 105.
T = 1.0
r = 0.05
sigma = 0.2
M = 50
dt = T/M
I = 2500
#
S = []
for i in range(I):
    path = []
    for t in range(M+1):
        if t==0:
            path.append(s0)
        else:
            z = gauss(0.0,1.0)
            St = path[t-1] * exp((r- 0.5*sigma **2) * dt + sigma * sqrt(dt) * z)
            path.append(St)
    S.append(path)

# 计算蒙特卡洛估值
C0 = exp(-r * T ) * sum([max(path[-1] - K ,0) for path in S]) / I

# 输出结果
tpy = time() -t0
print(C0)
print(tpy)


