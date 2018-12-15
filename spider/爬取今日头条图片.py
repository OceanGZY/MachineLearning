# -*- coding:utf-8 -*-
'''
    date:2018/11/23
    version:0.1.0
    auth:gzy
'''

import requests
from urllib.parse import urlencode
import json

class PPIC(object):

    def __init__(self):
        return None

    def get_pages(self,offset):
        params ={
            'offset':offset,
            'format':'json',
            'keyword':'街拍',
            'count':'20',
            'cur_tab':'1',
        }
        url = 'https://www.toutiao.com/search/?'+ urlencode(params)

        try:
            response = requests.get(url)
            if response.status_code ==200:
                self.get_images(response.json())
        except requests.ConnectionError :
            return None


    def get_images(self,json):
        if json.get('data'):
            for item in json.get('data'):
                title = item.get('title')
                # images = item.get('imgae_url')
                print(title)



if __name__ == '__main__':
    ppic =PPIC()
    ppic.get_pages(0)









