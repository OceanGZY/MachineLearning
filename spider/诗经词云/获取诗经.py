#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
    auth:gzy
    version:0.1.0
    date:20190608
'''
import requests
from bs4 import BeautifulSoup
import time

class SJ(object):
    def __init__(self):
        self.BASE_URL = 'http://www.shicimingju.com'

    def create_page_url(self):
        for i in range(1,9):
            page_url = self.BASE_URL + '/chaxun/zuozhe/13046_'+ str(i)+'.html'
            self.get_page(page_url)
    def get_page(self,url):
        if url:
            time.sleep(1)
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    self.parse_page(response.content)
                else:
                    print('获取page异常',response.status_code)
            except Exception as e:
                print('try获取页面失败',e)
                return None
        else:
            print('page_url不能为空')

    def parse_page(self,html):
        if html:
            soup = BeautifulSoup(html,'lxml')
            poetry_urls = soup.select("h3 a")
            for i in poetry_urls:
                poetry_url = i.get('href')
                poetry_url = self.BASE_URL +poetry_url
                # print(poetry_url)
                self.get_poetry(poetry_url)
        else:
            print('page_content不能为空')

    def get_poetry(self,poetry_url):
        if poetry_url:
            time.sleep(1)
            try:
                res = requests.get(poetry_url)
                if res.status_code == 200:
                    self.parse_poetry(res.content)
                else:
                    print('获取poetry异常',res.status_code)
            except Exception as e:
                print('try poetry异常',e)
                return None
        else:
            print('poetry_url不能为空')

    def parse_poetry(self,poetry_content):
        if poetry_content:
            soup = BeautifulSoup(poetry_content,'lxml')
            poetry_titles = soup.select(".shici-title")
            poetry_infos = soup.select(".shici-info")
            poetries= soup.select(".shici-content")
            for j,k,l in zip(poetry_titles,poetry_infos,poetries):
                poetry_title = j.text.strip()
                poetry_info = k.text.strip()
                poetry = l.text.strip()
                # print(poetry_title,'---',poetry_info,'---',poetry)
                poetry_detial=  poetry_title +'-'+poetry_info +'-' + poetry +'\n'
                self.save_poetry(poetry_detial)

        else:
            print('poetry_content不能为空')

    def save_poetry(self,poetry_detial):
        file = '诗经'+'.text'
        with open(file,'a+',encoding="utf-8") as f:
            f.write(poetry_detial)
            print("已写入一首")

if __name__ == '__main__':
    sj = SJ()
    sj.create_page_url()