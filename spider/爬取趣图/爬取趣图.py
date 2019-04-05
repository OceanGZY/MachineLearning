#!/usr/bin/python
#-*-coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
import time


class MZi():

    def __init__(self):
        self.BASE_URL = 'https://www.mzitu.com/'


    def get_pages_urls(self):
        for i in range(4,118):
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
        imgaes_preview_urls = soup.select("#pins li a")
        imgaes_preview_srcs = soup.select("#pins .lazy")
        # print(imgaes_preview_urls)
        # print(imgaes_preview_srcs)

        for item in imgaes_preview_urls:
            preview_url =item.get('href')
            print(preview_url)
            with open('imgaes_preview_urls.txt','a') as urlf:
                urlf.write(preview_url +','+'\n')
        for i in imgaes_preview_srcs:
            preview_src =i.get('data-original')
            print(preview_src)
            with open('imgaes_preview_srcs.txt','a') as srcf:
                srcf.write(preview_src +','+'\n')






if __name__ == '__main__':
    meizi = MZi()
    meizi.get_pages_urls()