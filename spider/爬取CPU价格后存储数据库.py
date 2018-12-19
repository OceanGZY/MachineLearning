# -*- coding:utf-8 -*-
'''
    auth:gzy
    date:2018/12/15
'''
import requests
from bs4 import BeautifulSoup
import pymysql

class CPUPrice(object):

    def __init__(self):
        self.BASE_URL ="http://www.zgcdiy.com"

    def get_urls(self):
        for i in range(1,9):
            url = self.BASE_URL +"/category-2-b0-min0-max0-attr0-"+str(i)+"-goods_id-DESC.html"
            self.get_pages(url)

    def get_pages(self,url):
        try:
            response = requests.get(url)
            if response.status_code ==200:
                self.parse_pages(response.content)
            else:
                print("请求失败",response.status_code)
        except Exception as e:
            print("请求异常",e)
            return None


    def parse_pages(self,html):
        if html:
            soup = BeautifulSoup(html,'lxml')
            # print(soup)
            cpu_names = soup.select('#globalGoodsList .itemGrid .item .name')
            cpu_prices = soup.select('#globalGoodsList .itemGrid .item .price .goodsPrice')
            self.parse_cpu(cpu_names,cpu_prices)


    def parse_cpu(self,cpuNames,cpuPrices):

        for cpu_name_item in cpuNames:
            cpuName = cpu_name_item.text.strip()
            print(cpuName)

        for cpu_price_item in cpuPrices:
            cpuPrice = cpu_price_item.text.strip()
            print(cpuPrice)



    def create_database_table(self):
        db = pymysql.connect('192.168.0.103','root','gzy5211314','test')
        cursor =db.cursor()
        cursor.execute("DROP TABLE IF EXISTS CPUPRICE")
        sql_create_database_table = """CREATE TABLE CPUPRICE(
                                        id INT NOT NULL AUTO_INCREMENT,
                                        cpuName CHAR(250),
                                        cpuPrice CHAR(250),
                                        PRIMARY KEY(id))"""

        try:
            cursor.execute(sql_create_database_table)
            db.commit()
            print("建表成功")
        except Exception as e:
            print(e)
            db.rollback()

        cursor.close()
        db.close()



    def save_cpudetial(self,cpuNames,cpuPrices):
        db = pymysql.connect('192.168.0.103', 'root', 'gzy5211314', 'test')
        cursor = db.cursor()

        sql_insert_data = """INSERT INTO CPUPRICE(
                             cpuName,
                             cpuPrice)
                             VALUES (%s,%s)"""
        try:
            cursor.execute(sql_insert_data,(cpuNames,cpuPrices))
            db.commit()
            print("数据插入成功")
        except Exception as e:
            print(e)
            db.rollback()

        cursor.close()
        db.close()


if __name__ =="__main__":
    CpuPrice = CPUPrice()
    # CpuPrice.create_database_table()
    CpuPrice.get_urls()
