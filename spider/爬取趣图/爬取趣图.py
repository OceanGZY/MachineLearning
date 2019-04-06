#!/usr/bin/python
#-*-coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
import time
import pymysql

class MZi():

    def __init__(self):
        self.BASE_URL = 'https://www.mzitu.com/'


    def get_pages_urls(self):
        for i in range(1,118):
            pages_url = self.BASE_URL + 'tag/youhuo/page/'+ str(i) +'/'
            # pages_url = self.BASE_URL + 'tag/youhuo/page/'+ str(i) +'/'
            print(pages_url)
            self.get_imgaes_preview_urls(pages_url)

    def headers(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36",
        }


    def get_imgaes_preview_urls(self,url):
        time.sleep(1)
        try:
            response = requests.get(url,headers =self.headers())
            if response.status_code ==200:
                # print(response.text)
                self.parse_imgaes_preview_urls(response.text)
            else:
                print("请求图像detail的urls异常",response.status_code)
        except Exception as e:
            print("获取图像detail的urls异常",e)
            return None


    def parse_imgaes_preview_urls(self,html):
        soup = BeautifulSoup(html,'lxml')
        imgaes_preview_srcs = soup.select("#pins .lazy")
        imgaes_preview = soup.select("#pins li span a")
        imgaes_preview_dates = soup.select("#pins .time")

        # print(imgaes_preview_urls)
        # print(imgaes_preview_srcs)

        for item,i,k in zip(imgaes_preview,imgaes_preview_srcs,imgaes_preview_dates):
            preview_url =item.get('href')
            previewTitle = item.text.strip()
            preview_src = i.get('data-original')
            previewDate = k.text.strip()

            # print(previewDate,previewTitle,preview_url,preview_src)

            # #存储文件
            # with open('imgaes_preview_urls.txt','a') as urlf:
            #     urlf.write(preview_url +','+'\n')

            self.save_images_preview(previewDate,previewTitle,preview_url,preview_src)


    def create_table(self):
        db = pymysql.connect('10.211.55.5','root','','test')
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS MeiZiTu")

        sql_create_table = """CREATE TABLE MeiZiTu (
                                    id INT NOT NULL AUTO_INCREMENT,
                                    date CHAR (100),
                                    title CHAR (250),
                                    nextPage CHAR (250),
                                    previewImg CHAR (250),
                                    PRIMARY KEY (id))"""
        try:
            cursor.execute(sql_create_table)
            db.commit()
            print('建表成功')
        except:
            db.rollback()

        cursor.close()
        db.close()



    def save_images_preview(self,previewDate,previewTitle,nextPage,previewSrc):
        db = pymysql.connect('10.211.55.5', 'root', '', 'test')
        cursor = db.cursor()
        sql_insert_data = """INSERT INTO MeiZiTu (
                                date,
                                title,
                                nextPage,
                                previewImg) VALUES (%s,%s,%s,%s)"""

        try:
            cursor.execute(sql_insert_data,(previewDate,previewTitle,nextPage,previewSrc))
            db.commit()
            print('数据插入成功')

        except Exception as e:
            print("数据插入异常",e)
            db.rollback()
        cursor.close()
        db.close()




if __name__ == '__main__':
    meizi = MZi()
    meizi.create_table()
    meizi.get_pages_urls()