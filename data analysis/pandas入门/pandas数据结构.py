# pandas 常见数据结构：Serires , DataFrame

import numpy as np
import pandas as pd

# Series : 带有名称和索引的一维数组
# 示例：存储一些用户信息，暂时包括年龄信息

# 存储 4 个年龄
user_age = pd.Series(data=[18,30,25,40])
print(user_age)

print('-------------------')

# 为年龄增加对应的姓名， 使用index索引
user_age.index= ['a','b','c','d']
print(user_age)
print('-------------------')

# 为索引字段增加 名称描述
user_age.index.name = 'name'
print(user_age)
print('-------------------')

# 为Series生成的数组设置名字
user_age.name = "user_age_info"
print(user_age)


# 快速构建Series数组
# 构建索引
name = pd.Index(["E","F","G","H"],name = "name")
# 构建Series
user_age = pd.Series(data=[18,33,26,40],index=name,name="user_age_info")
print(user_age)

print('--------------------')


# 快速构建Series数组
# 构建索引
name = pd.Index(["E","F","G","H"],name = "name")
# 构建Series，指定数据类型
user_age = pd.Series(data=[18,33,26,40],index=name,name="user_age_info",dtype=float)
print(user_age)

print('--------------------')

# Series的向量化操作
print(user_age +1)
print('--------------------')

# DataFrame 是一个带索引 的二维数据结构， 类似一个数据表
# 构建方式一
# 构建dict， key存储字段名, value是字段值信息,然后将dict传递给 data参数
index = pd.Index(data=["I","J","K","L"],name = "name")
data = {
    "age":[19,25,32,38],
    "city":["beijing","shanghai","guangzhou","shenzhen"],
}

user_info = pd.DataFrame(data=data,index=index)
print(user_info)
print('----------------------')

# 构建方式二
# 先构建一个二维数组，然后再生成一个列名称列表
index = pd.Index(data=["I","J","K","L"],name = "name")
data = [
    [18,"beijing"],
    [24,"shanghai"],
    [28,"guangzhou"],
    [26,"shenzhen"]
]
columns = ["age","city"]

user_info = pd.DataFrame(data=data,index=index,columns=columns)
print(user_info)
print('----------------')

# 访问行
print(user_info.loc["J"])
print('----------------')
# 访问列
print(user_info.age)
print('----------------')

# 新增列
user_info["sex"] = "male"
print(user_info)
print('----------------')

user_info["sex"] = ["male", "male", "female", "male"]
print(user_info)
print('----------------')


# 在不修改原DataFrame的条件下，新增一列
print(user_info.assign(age_add_one = user_info["age"] +1))
print('----------------')
print(user_info.assign(sex_code = np.where(user_info["sex"] == "male",1,0)))

# 删除列
user_info.pop("sex")
print(user_info)
print('----------------')