#!/usr/bin/python3
#-*- coding:utf-8 -*-
import requests
from bs4 import BeautifulSoup
import time
import os
from selenium import webdriver

class HeroSkins():
    def __init__(self):
        self.BASE_URL = "https://pvp.qq.com/web201605/"

    def get_hero_url(self):
        source_url = self.BASE_URL + "herolist.shtml"
        res = requests.get(source_url)
        if res.status_code == 200 :
            self.parse_hero_url(res.content)
        else:
            print("请求失败")

    def parse_hero_url(self,html):
        if html:
            try:
                soup = BeautifulSoup(html,'html.parser')
                hero_lists = soup.select('.herolist-content ul li a')
                for i in hero_lists:
                    hero = i.get("href")
                    hero_name  = i.text.strip()
                    hero_url = self.BASE_URL + hero
                    FILE_PATH = os.getcwd()
                    skin_path = FILE_PATH + "/" + str(hero_name)
                    try:
                        os.mkdir(skin_path)
                    except Exception as e:
                        print(e)
                    self.get_skin_url(hero_url,skin_path)
            except Exception as e:
                print("解析失败",e)
        else:
            print("没有hero_list_source")

    def get_skin_url(self,url,path):
        if url:
            time.sleep(1)
            opener = webdriver.Chrome()
            try:
                opener.get(url)
                self.parse_skin_url(opener.page_source,path)
            except Exception as e:
                print("解析英雄页失败",e)
        else:
            print("英雄详情页不存在")

    def parse_skin_url(self,html,path):
        if html:
            try:
                soup = BeautifulSoup(html,'html.parser')
                hero_skin_lists = soup.select('.pic-pf ul li i img')
                for i in hero_skin_lists:
                    hero_skin_name = i.get('data-title')
                    hero_skin_url = 'http:' +i.get('data-imgname')
                    print(hero_skin_name,hero_skin_url)
                    self.save_skin(hero_skin_name,hero_skin_url,path)
            except Exception as e:
                print("解析皮肤地址失败",e)
        else:
            print("找不到英雄详情页")

    def save_skin(self,skin_name,skin_url,path):
        if (skin_name,skin_url,path):
            skin_file = path + '/' + str(skin_name) + '.jpg'
            try:
                res = requests.get(skin_url)
                if res.status_code ==200 :
                    with open(skin_file,'wb') as f:
                        f.write(res.content)
                        print("保存皮肤成功")
                else:
                    print("请求皮肤失败")
            except Exception as e:
                print(e)

if __name__ == "__main__":
    heroskins = HeroSkins()
    heroskins.get_hero_url()