#!/usr/bin/python
#-*-coding:utf-8-*-
'''
    author:gzy
    date:20190419
    version:V0.1.0
'''
import requests
import re
import time
from bs4 import BeautifulSoup

class ZuFang():
    def __init__(self):
        self.BASEURL='https://nj.58.com/chuzu/pn'

    def get_page_url(self):
        for i in range(1,2):
            url = self.BASEURL +str(i)+'/?utm_source=sem-sales-baidu-pc&spm=57673705953.14911910706&utm_campaign=sell&utm_medium=cpc&showpjs=pc_fg'
            print(url)
            self.get_page(url)

    def get_page(self,url):
        try:
            time.sleep(2)
            res = requests.get(url)
            if res.status_code ==200:
                html = res.content.decode()
                # print(html)
                self.get_page(html)
            else:
                print('请求页面异常',res.status_code)
        except Exception as e:
            print(e)
            return None

    def parse_page(self,html):
        if html:
            soup = BeautifulSoup(html,'lxml')
            fzs = soup.find_all("div",class_="des")
            print(fzs)








if __name__ =='__main__':
    zufang = ZuFang()
    zufang.get_page_url()