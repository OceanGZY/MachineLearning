#!/usr/bin/python3
'''
    author:gzy
    date:20191004
    version:0.1.0
'''

import requests
from bs4 import BeautifulSoup
import pymysql

class Db(object):
    #数据库连接
    def db_connect(self):
        db = pymysql.connect(
            host = "127.0.0.1",
            port = 3306,
            user = "root",
            password = "123456",
            db = "zhihu",
        )
        cursor = db.cursor()
        print("数据库连接成功")
        return db,cursor
    # 创建数据表
    def create_tb(self,db,cursor,dsql,csql):
        cursor.execute(dsql)
        try:
            cursor.execute(csql)
            db.commit()
        except Exception as e:
            db.rollback()
            print(e)
        cursor.close()
        db.close()

    # 插入数据
    def insert_tb(self,db,cursor,isql):
        try:
            cursor.execute(isql)
            db.commit()
        except Exception as e:
            db.rollback()
            print(e)
        cursor.close()
        db.close()


class Crawl(object):
    def __init__(self):
        self.BASE_URL = "http://www.shicimingju.com"
        self.db1 = Db()
        dsql = """DROP TABLE IF EXISTS nalanpoem"""
        csql = """CREATE TABLE nalanpoem (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        title VARCHAR (100),
                        poem LONGTEXT) DEFAULT CHARSET = utf8mb4 ;"""
        db, cursor = self.db1.db_connect()
        self.db1.create_tb(db, cursor, dsql, csql)

    def create_url(self):
        for i in range(1,6):
            url = self.BASE_URL + "/chaxun/zuozhe/29_" + str(i) +".html"
            self.get_page(url)

    def get_page(self,url):
        if url:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    html  = response.content
                    self.parse_link(html)
                else:
                    print(response.status_code,"get页面错误")
            except Exception as e:
                print(e)
        else:
            print("url是空的")

    def parse_link(self,html):
        if html:
            soup = BeautifulSoup(html,"html.parser")
            links = soup.select(".www-main-container h3 a")
            for j in links:
                link = j.get("href")
                link = self.BASE_URL + link
                self.get_poerm_page(link)


    def get_poerm_page(self,link):
        if link:
            try:
                res = requests.get(link)
                if res.status_code ==200:
                    html = res.content
                    self.parse_poem_page(html)
                else:
                    print(res.status_code)
            except Exception as e:
                print(e)
        else:
            print("link是空的")

    def parse_poem_page(self,html):
        if html:
            try:
                soup = BeautifulSoup(html,"html.parser")
                titles = soup.select(".shici-title")
                poems = soup.select(".shici-content")
                for m,n in zip(titles,poems):
                    title = m.text
                    title = "\""+ title.strip() + "\""
                    poem = n.text
                    poem = "\""+ poem.strip() + "\""
                    self.save_poem(title,poem)
            except Exception as e:
                print(e)
        else:
            print("html是空的")


    def save_poem(self,ctitle,cooem):
        isql = """
                INSERT INTO nalanpoem ( 
                title,
                poem) VALUES (%s,%s);""" %(ctitle,cooem)
        # print(isql)
        db,cursor = self.db1.db_connect()
        self.db1.insert_tb(db,cursor,isql)

if __name__ == "__main__":
    ccrawler = Crawl()
    ccrawler.create_url()


