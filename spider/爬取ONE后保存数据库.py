# coding:utf-8'''    auth:gzy    date:2018/12/13'''
import pymysql
import requests
from bs4 import BeautifulSoup


class GetONETEXT(object):
    def __init__(self):
        self.BASE_URL = "http://wufazhuce.com/one/"

    def get_all_urls(self):
        for n in range(14, 16):
            url = self.BASE_URL + str(n)
            self.get_all_pages(url)

    def get_all_pages(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:64.0) Gecko/20100101 Firefox/64.0"
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                self.parse_all_pages(response.content)
            else:
                print('请求页面状态码：', response.status_code)
        except Exception as e:
            print('请求页面异常', e)
            return None

    def parse_all_pages(self, html):
        if html:
            soup = BeautifulSoup(html, 'lxml')
            image_urls = soup.select('#main-container .one-imagen img')
            text_numbers = soup.select('#main-container .one-titulo')
            imgae_authers = soup.select('#main-container .one-imagen-leyenda')
            texts = soup.select('#main-container .one-cita-wrapper .one-cita')
            text_mons = soup.select('#main-container .one-cita-wrapper .one-pubdate .may')
            text_days = soup.select('#main-container .one-cita-wrapper .one-pubdate .dom')

            for image_url in image_urls:
                url= image_url.get('src')
                # print(url)

            for text_number in text_numbers:
                textNum = text_number.text.strip()
                # print(textNum)

            for image_auther in imgae_authers:
                imgAuth = image_auther.text.strip()
                # print(imgAuth)

            for text in texts:
                textCont = text.text.strip()
                # print(textCont)

            for text_mon in text_mons:
                mon = text_mon.text.strip()
                # print(mon)

            for text_day in text_days:
                day = text_day.text.strip()
                # print(day)

            # self.save_all(url,textNum,imgAuth,textCont,mon,day)
            # print(url,textNum,imgAuth,textCont,mon,day)

    def create_data_table(self):
        db = pymysql.connect('10.211.55.5', 'root', '数据库密码', 'test')
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS YOUONE")

        sql_create_table = """CREATE TABLE YOUONE (
                                      id INT NOT NULL AUTO_INCREMENT, 
                                      imgUrl CHAR(250),
                                      textNum CHAR(250),
                                      imgAuther CHAR(250),
                                      textContent CHAR(250),
                                      mon CHAR(250),
                                      day CHAR(250),
                                      PRIMARY KEY (id))"""
        try:
            cursor.execute(sql_create_table)
            db.commit()
            print("建表成功")
        except:
            db.rollback()

        cursor.close()
        db.close()


    def save_all(self,url,textNum,imgAuth,textCont,mon,day):
        # print(url,textNum,imgAuth,textCont,mon,day)
        db = pymysql.connect('10.211.55.5','root','数据库密码','test')
        print('连接成功')
        cursor =db.cursor()

        sql_insert_data = """INSERT INTO YOUONE (imgUrl,
                                      textNum,
                                      imgAuther,
                                      textContent,
                                      mon,
                                      day) VALUES (%s,%s,%s,%s,%s,%s)"""

        try:
            cursor.execute(sql_insert_data,(url,textNum,imgAuth,textCont,mon,day))
            db.commit()
            print("数据插入成功")
        except Exception as e:
            print(e)
            db.rollback()
        cursor.close()
        db.close()




if __name__ == "__main__":
    getOneText = GetONETEXT()
    # getOneText.create_data_table()
    getOneText.get_all_urls()
