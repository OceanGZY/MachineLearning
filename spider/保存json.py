import json
import csv

str ='''
[
    {
        "auth":"gzy",
        "date":"2018-11-22",
        "time":"22:58",
        "version":"0.1.1"
    },
    {
        "auth":"gzy",
        "date":"2018-11-21",
        "time":"21:56",
        "version":"0.1.0"
    }
]
'''

# print(type(str))

data = json.loads(str)

# print(data)

# print(type(data))

# print(data[0].get('auth'))
with open('data.json','w')as  file:
    file.write(json.dumps(data))