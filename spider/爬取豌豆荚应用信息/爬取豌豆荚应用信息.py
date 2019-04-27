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
import time

class WDJAPPS(object):
    def __init__(self):
        self.BASEURL= 'https://www.wandoujia.com/'

    # url 生成器
    def get_page_url(self):
        api = self.BASEURL +'wdjweb/api/top/more'
        data= {
            "resourceType":0,
            "ctoken":"cO30CXX_DSwVDC306ZSU8ddW",
            "page":1,
        }
        return api,data

    # page 下载器
    def get_page_content(self,api,data):
        try:
            res = requests.get(api,data)
            if res.status_code ==200:
                result = json.loads(res.text)
                content = result['data']['content']
                return content
            else:
                print("请求数据异常",res.status_code)
        except Exception as e:
            print(e)
            return None


    # page 内容解析器
    def parse_page_content(self,content):
        if content:
            soup = BeautifulSoup(content,'lxml')
            app_names = soup.select('.app-title-h2')
            print(app_names)

            app_comments = soup.select('div .comment')
            print(app_comments)

            app_metas = soup.select('.meta')
            print(app_metas)
            





            # print(content)


    # page 内容保存器



if __name__ == '__main__':
    wdjapps = WDJAPPS()
    api ,data = wdjapps.get_page_url()
    c = wdjapps.get_page_content(api,data)
    wdjapps.parse_page_content(c)