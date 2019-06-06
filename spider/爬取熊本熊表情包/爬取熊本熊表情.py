#!/usr/bin/python
# -*-coding:utf-8-*-
'''
    date:2019-06-06
    version:0.1.0
    auth:gzy
'''
import requests
from bs4 import BeautifulSoup
import time

class XBX(object):
    def __init__(self):
        self.BASE_URL = 'https://fabiaoqing.com/'

    def make_page_url(self):
        for i in range(1,33):
            page_url = self.BASE_URL + 'tag/detail/id/57/page/' + str(i)+'.html'
            self.get_page(page_url)

    def get_page(self,url):
        time.sleep(1)
        try:
            res = requests.get(url)
            if res.status_code == 200:
                self.parse_page(res.content)
            else:
                print('读取页面异常',res.status_code)
        except Exception as e:
            print('try 页面链接异常',e)
            return None

    def parse_page(self,html):
        if html:
            soup = BeautifulSoup(html,'lxml')
            img_urls = soup.select(".tagbqppdiv img")
            for j in img_urls:
                img_url = j.get('data-original')
                img_name = j.get('title').strip()
                # print(img_url)
                # print(img_name)
                self.get_img(img_url,img_name)

    def get_img(self,imgurl,imgname):
        if imgurl:
            time.sleep(1)
            try:
                response = requests.get(imgurl)
                if response.status_code == 200:
                    self.save_img(response.content,imgname)
                else:
                    print('请求图片异常',response.status_code)
            except Exception as e:
                print('try 获取图片异常',e)
                return None

    def save_img(self,content,name):
        if content:
            file = name + '.gif'
            with open(file,'wb') as f:
                f.write(content)
                print('存图成功')



if __name__ == '__main__':
    xbx = XBX()
    xbx.make_page_url()
