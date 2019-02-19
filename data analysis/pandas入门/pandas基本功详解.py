# 导入相关库
import numpy as np
import pandas as pd

# 构建DataFrame
# 构建dict， key存储字段名, value是字段值信息,然后将dict传递给 data参数
index = pd.Index(data=["I","J","K","L"],name = "name")
data = {
    "age":[19,25,32,38],
    "city":["beijing","shanghai","guangzhou","shenzhen"],
    "sex":["male","male","female","male"]
}

user_info = pd.DataFrame(data=data,index=index)
print(user_info)
print('----------------------')

# 1、查看数据的整体情况
print(user_info.info())
print('----------------------')
# 当数据量较大时， 只看头部N条数据，或者只看尾部N条数据
# 头部
print(user_info.head(2))
print('----------------------')
# 尾部
print(user_info.tail(2))
print('----------------------')

# 获取DataFrame 的原有数据
print(user_info.values)

# 2、描述与统计
# 最大值
print(user_info.age.max())
# 最小值
print(user_info.age.min())
# 平均值
print(user_info.age.mean())
# 中位数
print(user_info.age.quantile())
# 求和
print(user_info.age.sum())

print('----------------------')
# 一次性获取多个统计指标
print(user_info.describe())
print('----------------------')
# 调用value_counts方法快速获取 Series 中每个值出现的次数
print(user_info.sex.value_counts())
print('----------------------')
# 获取某列最大值或最小值对应的索引，可以使用idxmax或idxmin
print(user_info.age.idxmax())
print(user_info.age.idxmin())
print('----------------------')

# 3、离散化
# 将数据离散化（分桶），分成几个区间，使用cut方法
# 自动分区
print(pd.cut(user_info.age,3))
print('----------------------')
# 手动分区
print(pd.cut(user_info.age,[1,18,25,39]))
print('----------------------')
# 手动分区，并为每个区间设置名字 ,labels
print(pd.cut(user_info.age,[1,16,24,40],labels=["child","young","middle"]))
print('----------------------')

# cut 是根据每个值的大小来进行离散化的，qcut 是根据每个值出现的次数来进行离散化
print(pd.qcut(user_info.age,3))
print('----------------------')


# 4、排序功能
# pandas ,两种排序方式
# 按轴（索引／列）排序，  按实际值排序
# 按照索引排序，sort_index ,按索引正序排
print(user_info.sort_index())
print('----------------------')
# 按照索引，倒序排,设置参数axis=1, ascending=False
print(user_info.sort_index(axis=1,ascending=False))
print('----------------------')

# 按照某一列实际值排序
print(user_info.sort_values(by = "age"))
print('----------------------')

# 按照某多列实际值排序,可以设置参数 by 为一个 list ,list 中每个元素的顺序会影响排序优先级的
print(user_info.sort_values(by = ["age","city"]))
print('----------------------')

# 获取最大的n个值或最小值的n个值，使用 nlargest 和 nsmallest 方法
print(user_info.age.nlargest(2))
print('----------------------')


# 5、函数应用
# 常用到的函数有：map、apply、applymap
# map  对Series中的每个元素实现转换， Series中特有
# 判断是否大于等于24岁
print(user_info.age.map(lambda  x: "yes" if x>= 24 else "no"))
print('----------------------')

# 判断城市是北方还是南方
city_map = {
    "beijing":"north",
    "shanghai":"sourth",
    "guangzhou":"sourth",
    "shenzhen":"sourth",
}
print(user_info.city.map(city_map))
print('----------------------')

# apply ，在对 Series 操作时会作用到每个值上，在对 DataFrame 操作时会作用到所有行或所有列（通过 axis 参数控制）
# 对 Series 来说，apply 方法 与 map 方法区别不大。
print(user_info.age.apply(lambda x: "yes" if x >= 24 else "no"))
# 对 DataFrame 来说，apply 方法的作用对象是一行或一列数据（一个Series）
print(user_info.apply(lambda x: x.max(), axis=0))
# applymap 针对DataFrame
print(user_info.applymap(lambda x: str(x).lower()))
print('----------------------')

# 6、修改列／索引名称
# DataFrame  使用rename
# 修改列名字 columns
print(user_info.rename(columns={"age":"Age","city":"City","sex":"Sex"}))
print('----------------------')
# 修改索引名字 index
print(user_info.rename(index={"I":"i","J":"j"}))
print('----------------------')

# 7、类型操作
# 获取每种类型的列数 ,get_dtype_counts
print(user_info.get_dtype_counts())
print('----------------------')

# 转换数据类型，使用 astype
print(user_info["age"].astype(float))
print('----------------------')

# 将object类型转换为其它类型，
# 转为数字      to_numeric
# 转为日期      to_datetime
# 转为时间差     to_timedelta