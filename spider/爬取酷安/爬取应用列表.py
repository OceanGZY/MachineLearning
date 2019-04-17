#!/usr/bin/python
#-*-coding:utf-8-*-
'''
    date:20190415
    version:v0.1.0
    author:gzy
'''

import requests
from bs4 import BeautifulSoup

class KuAnAPPS():
    def __init__(self):
        BASE_URL = 'https://www.coolapk.com/'
        self.app_url = BASE_URL +'apk?p='
        self.game_url = BASE_URL +'game?p='


    def get_apps(self):
        try:
            app_url = self.app_url +'1'
            response = requests.get(app_url)
            if response.status_code == 200:
                # print(response.content.decode())
                self.parse_apps(response.content)
            else:
                print(response.status_code)
        except Exception as e:
            print(e)
            return None

    def parse_apps(self,html):
        soup = BeautifulSoup(html,'lxml')
        app = soup.select(".app_left_list a")
        list_app_title =soup.select(".alllist_app .alllist_app_side .list_app_title")
        alllist_img =soup.select(".alllist_app .alllist_app_side .alllist_img")
        list_app_info =soup.select(".alllist_app .alllist_app_side .list_app_info")
        list_app_description = soup.select(".alllist_app .alllist_app_side .list_app_description")


        for u,i,j,k,l in zip(range(0,10),list_app_title,alllist_img,list_app_info,list_app_description):
            app_package= app[u].get('href')
            app_packagename = app_package.strip('/apk/')
            app_name = i.text.title()
            app_icon = j.get('src')
            a= k.text.split()
            app_grade = a[0]
            app_size = a[1]
            app_downloads = a[2]
            app_updatetime = a[3]
            app_description = l.text.title()
            print(type(app_grade))
            print(type(app_size))
            # print(app_packagename,app_name,app_icon,app_grade,app_size,app_description,app_downloads,app_updatetime)




if __name__ =='__main__':
    kuanapps = KuAnAPPS()
    kuanapps.get_apps()

