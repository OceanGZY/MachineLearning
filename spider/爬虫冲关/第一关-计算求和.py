#!/usr/bin/python3
'''
    date:20190904
    author:GZY
'''

import requests
from bs4 import BeautifulSoup

class CSUM(object):

    def __init__(self):
        self.vsum = 0
        self.BASE_URL = "http://glidedsky.com/"


    def create_cpage_url(self):
        cpage_url = self.BASE_URL + "level/web/crawler-basic-1"
        return cpage_url

    def get_cpage_url(self,cpage_url):
        headers = {
            "Cookie":"_ga=GA1.2.537209483.1567569663; _gid=GA1.2.226938426.1567569663; Hm_lvt_020fbaad6104bcddd1db12d6b78812f6=1567569663; XSRF-TOKEN=eyJpdiI6InVRU2xsSmhLUHVVRmgrcUxTNVQ1bXc9PSIsInZhbHVlIjoiWDRnbnhpOFVadG9JXC9rbGhMVnczMnBxZmRhK0hBNWJWOFVjTUhqdmNqenl6K01HTEE4VXB3eFBNZ3Z1RWRXbG0iLCJtYWMiOiJmOWY1N2IwZjQwMmU0MmI2ZTNjYjU5OTE3NzA2NDk5MWNhOGFlODQ4MjBhYjQ2NzZlZmY2ZjQ5N2MyNTk5ODYxIn0%3D; glidedsky_session=eyJpdiI6IjhoUytNdmVJeWt5SDVtSjNybVwvTGtRPT0iLCJ2YWx1ZSI6IlhKRHBvdzNxdlR2MHV1Z1M5S01lV0VEZDFpQWx1SkVmazk0SjkyNlhlYktvRmxsaENRMVlSXC9EUDM5ZlE2MzBqIiwibWFjIjoiOWNhMzY4ZDZhZTRkNmVkZDZiYzg0OTlkYjJkODMzZDUwZTE1NjczZjU0ZTQwOTVlMTk1MTg1YjkxYjRlYTk5YiJ9; _gat_gtag_UA_75859356_3=1; Hm_lpvt_020fbaad6104bcddd1db12d6b78812f6=1567583437",
            "User-Agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Mobile Safari/537.36"
        }
        if cpage_url:
            try:
                csession = requests.session()
                res = csession.get(cpage_url,headers=headers)
                html = res.content
            except Exception as e:
                print(e,"请求页面失败")
        else:
            print("页面链接是空")
        return html

    def parse_cpage_url(self,html):
        if html:
            try:
                soup = BeautifulSoup(html,'lxml')
                values = soup.select(".col-md-1")
                for value in values:
                    v = value.text.strip()
                    self.vsum = self.vsum +int(v)
            except Exception as e:
                print(e)
        else:
            print("页面内容不存在")
        return self.vsum



if __name__ == "__main__":
    csum = CSUM()
    cpage_url = csum.create_cpage_url()
    html = csum.get_cpage_url(cpage_url)
    fsum = csum.parse_cpage_url(html)
    print(fsum)





