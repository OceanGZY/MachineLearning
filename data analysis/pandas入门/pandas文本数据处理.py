import numpy as np
import pandas as pd

# str属性
## pandas 为Series提供了str属性
index = pd.Index(data=["A","B","C","D","E","F"],name="name")
data = {
    "age":[18,30,np.nan,40,np.nan,30],
    "city":["Bei Jing","Shang Hai","Guang Zhou","Shen Zhen",np.nan,""],
    "sex":[None,"male","female","male",np.nan,"unknown"],
    "birth":["2000-02-10","1988-10-17",None,"1978-08-08",np.nan,"1988-10-17"]
}
user_info = pd.DataFrame(data=data,index=index)
print(user_info)
print('---------------------------')
## 将出生日期转为时间戳
user_info["birth"] = pd.to_datetime(user_info.birth)
print(user_info)
print('---------------------------')
## 将城市信息转为小写
print(user_info.city.str.lower())
print('---------------------------')
## 统计城市信息的每个字符串的长度
print(user_info.city.str.len())
print('---------------------------')

# 替换和分割
## 使用.str属性，进行替换 replace 和分割 split操作
## 将城市信息里的空字符串替换成下划线
print(user_info.city.str.replace(" ","_"))
## 将城市信息的字段按照空字符串来分割
print(user_info.city.str.split(" "))
print('---------------------------')
## 分割后列表内的元素，可以使用get()  / [] 进行访问
print(user_info.city.str.split(" ").str.get(1))
print('---------------------------')
print(user_info.city.str.split(" ").str[0])

## 设置参数expand=True 可以轻松扩展此项以返回 DataFrame
print(user_info.city.str.split(" ",expand=True))