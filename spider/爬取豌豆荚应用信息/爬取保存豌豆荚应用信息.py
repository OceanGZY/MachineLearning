#!/usr/bin/python
# -*-coding:utf-8 -*-
'''
    auth:gzy
    date:2019-04-27
    version:0.1.0
'''
import requests
from bs4 import BeautifulSoup
import json
import time
import csv
import os

class WDJAPP(object):

    def __init__(self):
        self.BASR_URL = 'https://www.wandoujia.com/'
        self.applists = []

    # 构造页面数据URL生成器
    def create_page_url(self):
        api = self.BASR_URL + 'wdjweb/api/top/more'
        for n in range(42):
            data = {
                "ctoken":"gquLOOZyoog2cMp1PA5AWOE9",
                "page":n,
                "resourceType":0,
            }
            self.get_page_content(api,data)



    # 抓取页面内容信息
    def get_page_content(self,url,data):
        try:
            time.sleep(1)
            response = requests.get(url,data)
            if response.status_code == 200:
                resjson = json.loads(response.text)
                html = resjson['data']['content']
                # print(html)
                self.parse_page_content(html)
            else:
                print("页面请求失败",response.status_code)
        except Exception as e:
            print("尝试获取页面异常",e)
            return None


    # 解析页面信息
    def parse_page_content(self,html):
        if html:
            soup = BeautifulSoup(html,'lxml')
            app_names = soup.select('.app-title-h2 a')
            app_metas= soup.select('.app-desc .meta')
            app_comments = soup.select('.app-desc .comment')

            # print(app_names)
            # print('---')
            # print(app_metas)
            # print('---')
            # print(app_comments)

            for i,j,k in zip(app_names,app_metas,app_comments):
                appname = i.text
                # print(appname)
                # print('---')

                applink = i.get('href')
                # print(applink)
                # print('---')

                try:
                    app_meta = j.text.strip().split('人下载 ・ ')
                    if len(app_meta) ==2:
                        appinstallcount=app_meta[0]
                        # print(appinstallcount)
                        # print('---')

                        appsize = app_meta[1].strip('MB')
                        # print(appsize)
                        # print('---')
                    else:
                        appinstallcount = ''
                        # print(appinstallcount)
                        # print('---')
                        appsize = ''
                        # print(appsize)
                        # print('---')

                except Exception as e:
                    print(e)

                appcomment = k.text.strip()
                # print(appcomment)
                # print('---')
                applist = (appname,applink,appinstallcount,appsize,appcomment)
                self.applists.append(applist)

            self.save_page_data(self.applists)


        else:
            print("没有拿到html数据")


    # 保存数据表
    def save_page_data(self,applists):
        # print(applists)
        csvFile = '安卓应用信息.csv'
        with open(csvFile,"w") as f:
            fwriter =csv.writer(f)
            fwriter.writerow(["appname","applink","appinstallcount","appsize","appcomment"])
            for item in applists:
                fwriter.writerow(item)


if __name__ == "__main__":
    wdjapps = WDJAPP()
    wdjapps.create_page_url()




