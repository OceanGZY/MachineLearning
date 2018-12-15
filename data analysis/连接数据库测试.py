# -*- coding:utf-8 -*-

import pymysql

#打开数据库操作
db = pymysql.connect("10.211.55.5","root","gzy5211314","test")


#使用cursor()方法创建一个游标对象 cursor
#cursor = db.cursor()

db.close()
