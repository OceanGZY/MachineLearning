#!/usr/bin/python
# -*-coding:utf-8-*-
'''
    date:2019-04-13
    author:gzy
    version:0.1.0
'''

import requests
import re
import json
import os

class LOLSkins():
    def __init__(self):
        self.heros_info_jsurl = 'https://lol.qq.com/biz/hero/champion.js'
    #获取英雄信息列表
    def get_heros_info(self):
        try:
            response = requests.get(self.heros_info_jsurl)
            if response.status_code == 200:
                response_js = response.content.decode()
                pattern= re.findall("LOLherojs.champion=(.+?);",response_js)
                # h = json.loads(pattern[0])
                # heros_dict = json.loads(pattern[0])['keys']
                # print(heros_dict)
                heros_info_dicts = json.loads(pattern[0])['keys']
                # print(heros_info_dicts)
                self.heros_info_dicts = heros_info_dicts
                self.parse_hero_info(heros_info_dicts)
            else:
                print('请求异常',response.status_code)
        except Exception as e:
            print('获取异常',e)
            return None

    #解析英雄信息
    def parse_hero_info(self,dicts):
        if dicts:
            for i,value in dicts.items():
                # print(i,value)
                self.get_hero_skin_js(value)
        else:
            print('英雄信息字典不存在')

    #获取英雄皮肤js
    def get_hero_skin_js(self,value):
        skin_js = 'https://lol.qq.com/biz/hero/'+value+'.js'
        self.get_hero_skin(skin_js)

    #获取英雄皮肤信息
    def get_hero_skin(self,skin_js):
        try:
            res = requests.get(skin_js)
            if res.status_code ==200:
                res_js = res.content.decode()
                res_pat = re.findall('{"data":(.+?);',res_js)
                res_pattern = '{"data":'+res_pat[0]
                res_json = json.loads(res_pattern)
                # print(res_json)
                skins =res_json['data']['skins']
                ename = res_json['data']['id']
                cname = res_json['data']['name']
                nname = res_json['data']['title']
                # print(ename,cname,nname,skin)
                self.save_hero_skin(skins,ename,cname,nname)

            else:
                print('解析皮肤js获取皮肤信息异常')
        except Exception as e:
            print(e)
            return None


    #下载并保存英雄皮肤
    def save_hero_skin(self,skins,ename,cname,nname):
        skin_base_url = 'http://ossweb-img.qq.com/images/lol/web201310/skin/big'
        for skin in skins:
            # print(skin['id'],skin['name'])
            skin_url = skin_base_url + skin['id']+'.jpg'
            FILEPATH = os.getcwd()
            skinpath = FILEPATH + '/' + str(ename) + '_'+ str(cname) +'_'+ str(nname)
            try:
                os.mkdir(skinpath)
            except Exception as e:
                print(e)

            skin_file =  skinpath+'/'+str(skin['name'])+'.jpg'
            print(skin_file)
            try:
                r = requests.get(skin_url)
                if r.status_code ==200:
                    with open(skin_file ,'wb') as f:
                        f.write(r.content)
                        print('下载成功')
                else:
                    print("下载图片异常",r.status_code)
            except Exception as e:
                print(e)


if __name__ =='__main__':
    lolskin = LOLSkins()
    lolskin.get_heros_info()