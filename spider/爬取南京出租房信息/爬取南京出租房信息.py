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
        for i in range(2,3):
            url = self.BASEURL +str(i)+'/?utm_source=sem-sales-baidu-pc&spm=57673705953.14911910706&utm_campaign=sell&utm_medium=cpc&showpjs=pc_fg'
            print(url)
            self.get_page(url)

    def get_page(self,url):
        try:
            time.sleep(2)
            headers = {
                'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko)Chrome/74.0.3729.131 Safari/537.36'
            }
            res = requests.get(url,headers=headers)
            if res.status_code ==200:
                html = res.content.decode()
                # print(html)
                self.parse_page(html)
            else:
                print('请求页面异常',res.status_code)
        except Exception as e:
            print(e)
            return None

    def parse_page(self,html):
        if html:
            # print(html)
            soup = BeautifulSoup(html,'lxml')

            imgs  = soup.select('.listUl .img_list')
            titles = soup.select('.listUl .des h2')
            sizes = soup.find_all("p",class_ = "room strongbox")
            locations = soup.find_all("p",class_ = "add")
            moneys = soup.select('.listUl .listliright .money')
            for i in moneys:
                money = i.text
                print(money)
            # print(titles)
            # print(sizes)
            # print(locations)
            # print(moneys)



if __name__ =='__main__':
    zufang = ZuFang()
    zufang.get_page_url()