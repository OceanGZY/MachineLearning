#!/usr/bin/python
# -*-coding:utf-8 -*-
'''
    author:gzy
    date:20190417
    version:V0.1.0
'''

import requests
from bs4 import BeautifulSoup

class XHJ():
    def __init__(self):
        self.BASEURL= 'https://37yzy.com/tag/detail/3600/'


    def get_page_url(self):
        for i in range(1,8):
            url = self.BASEURL +str(i)+'.html'
            print(url)
            self.get_page(url)

    def get_page(self,url):
        try:
            response = requests.get(url)
            if response.status_code ==200:
                # print(response.content.decode())
                self.parse_page(response.content)
            else:
                print('请求页面失败',response.status_code)
        except Exception as e:
            print(e)
            return None

    def parse_page(self,html):
        if html:
            soup = BeautifulSoup(html,'lxml')
            img_tag = soup.find_all("img",class_="ui image bqppsearch lazy")
            for j in img_tag:
                # print(j)
                img_url = j.get('src')
                img_name = j.get('title').strip('_斗图表情')
                print(img_url)
                print(img_name)
                self.get_img(img_url,img_name)


    def get_img(self,img_url,img_name):
        try:
            res = requests.get(img_url)
            if res.status_code ==200:
                img_con = res.content
                self.save_img(img_con,img_name)
            else:
                print("取图失败",res.status_code)
        except Exception as e:
            print(e)

    def save_img(self,img_con,img_name):
        file = img_name +'.gif'
        with open(file ,'wb') as f:
            f.write(img_con)
            print('存图成功')


if __name__ =='__main__':
    xhj = XHJ()
    xhj.get_page_url()
