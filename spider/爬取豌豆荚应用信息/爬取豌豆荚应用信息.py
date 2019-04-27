#!/usr/bin/python
# -*-coding:utf-8 -*-
'''
    auth:gzy
    date:2019-04-26
    version:0.1.0
'''

import requests
import json
from bs4 import BeautifulSoup
import pymysql
import time

class WDJAPPS(object):
    def __init__(self):
        self.BASEURL= 'https://www.wandoujia.com/'

    # url 生成器
    def get_page_url(self):
        api = self.BASEURL +'wdjweb/api/top/more'
        for n in range(42):
            data= {
                "resourceType":0,
                "ctoken":"cO30CXX_DSwVDC306ZSU8ddW",
                "page":n,
            }
            self.get_page_content(api,data)

    # page 下载器
    def get_page_content(self,api,data):
        try:
            res = requests.get(api,data)
            if res.status_code ==200:
                result = json.loads(res.text)
                content = result['data']['content']
                self.parse_page_content(content)
            else:
                print("请求数据异常",res.status_code)
        except Exception as e:
            print('PAGE下载器异常',e)
            return None


    # page 内容解析器
    def parse_page_content(self,content):
        if content:
            soup = BeautifulSoup(content,'lxml')
            app_names = soup.select('.app-title-h2')
            # print(app_names)
            app_comments = soup.select('div .comment')
            # print(app_comments)
            app_metas = soup.select('.meta')
            # print(app_metas)
            app_tags = soup.select('.tag-link')
            self.save_page_content(app_names,app_comments,app_metas,app_tags)

    # page 内容保存器
    def save_page_content(self,names,comments,metas,tags):
        for i,j,k,l in zip(names,comments,metas,tags):
            name = i.text
            # print(name)
            comment = j.text.strip()
            # print(comment)
            meta = k.text.strip().split('人下载 ・ ')
            install_count = meta[0]
            app_size = meta[1]
            # print(install_count)
            # print(app_size)
            app_tag = l.text
            self.save_to_db(name,comment,install_count,app_size,app_tag)
    # 创建数据表
    def create_tb(self):
        db = pymysql.connect('127.0.0.1','root','','mydata')
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS wdjapps")

        sql_create_table = """
                            CREATE TABLE wdjapps(
                            id INT NOT NULL AUTO_INCREMENT,
                            appName CHAR(250),
                            appSize CHAR(250),
                            appInstallCount CHAR(250),
                            appComment CHAR(250),
                            appTag CHAR(250),
                            PRIMARY KEY(id))
                            """
        try:
            cursor.execute(sql_create_table)
            db.commit()
            print('建表成功')
        except Exception as e:
            print('建表相关错误',e)
            db.rollback()
        cursor.close()
        db.close()

    #保存至数据库
    def save_to_db(self,name,comment,install_count,app_size,app_tag):
        db = pymysql.connect('127.0.0.1','root','','mydata')
        cursor = db.cursor()
        sql_insert_apps = """INSERT INTO wdjapps(
                                    appName,
                                    appSize,
                                    appInstallCount,
                                    appComment,
                                    appTag) VALUES (%s,%s,%s,%s,%s)"""
        try:
            # print(name,app_size,install_count,comment)
            cursor.execute(sql_insert_apps,(name,app_size,install_count,comment,app_tag))
            db.commit()
            print('数据插入成功')
        except Exception as e:
            print('插入数据相关错误',e)
            db.rollback()
        cursor.close()
        db.close()


if __name__ == '__main__':
    wdjapps = WDJAPPS()
    wdjapps.create_tb()
    wdjapps.get_page_url()