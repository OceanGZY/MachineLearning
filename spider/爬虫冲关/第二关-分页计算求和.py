#!/usr/bin/python3
'''
    date:20190904
    author:GZY
'''
import requests
import time
from bs4 import BeautifulSoup

class TSUM(object):

    def __init__(self):
        self.BASE_URL = "http://glidedsky.com/"
        self.tsum = 0

    def create_tpage_url(self):
        base_url = self.BASE_URL + "level/web/crawler-basic-2"
        tpage_urls = []
        for i in range(1,1001):
            tpage_url = base_url + "?" + "page=" + str(i)
            tpage_urls.append(tpage_url)
        return tpage_urls

    def get_tpage_url(self,url):
        headers = {
            "Cookie": "_ga=GA1.2.537209483.1567569663; _gid=GA1.2.226938426.1567569663; Hm_lvt_020fbaad6104bcddd1db12d6b78812f6=1567569663; XSRF-TOKEN=eyJpdiI6InVRU2xsSmhLUHVVRmgrcUxTNVQ1bXc9PSIsInZhbHVlIjoiWDRnbnhpOFVadG9JXC9rbGhMVnczMnBxZmRhK0hBNWJWOFVjTUhqdmNqenl6K01HTEE4VXB3eFBNZ3Z1RWRXbG0iLCJtYWMiOiJmOWY1N2IwZjQwMmU0MmI2ZTNjYjU5OTE3NzA2NDk5MWNhOGFlODQ4MjBhYjQ2NzZlZmY2ZjQ5N2MyNTk5ODYxIn0%3D; glidedsky_session=eyJpdiI6IjhoUytNdmVJeWt5SDVtSjNybVwvTGtRPT0iLCJ2YWx1ZSI6IlhKRHBvdzNxdlR2MHV1Z1M5S01lV0VEZDFpQWx1SkVmazk0SjkyNlhlYktvRmxsaENRMVlSXC9EUDM5ZlE2MzBqIiwibWFjIjoiOWNhMzY4ZDZhZTRkNmVkZDZiYzg0OTlkYjJkODMzZDUwZTE1NjczZjU0ZTQwOTVlMTk1MTg1YjkxYjRlYTk5YiJ9; _gat_gtag_UA_75859356_3=1; Hm_lpvt_020fbaad6104bcddd1db12d6b78812f6=1567583437",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Mobile Safari/537.36"
        }
        if url:
            try:
                res = requests.get(url,headers=headers)
                html = res.content
            except Exception as e:
                print(e,"请求函数异常")
        else:
            print("页面URL是不存在")
        return html

    def parse_tpage_url(self,html):
        if html:
            try:
                soup = BeautifulSoup(html,'lxml')
                values=soup.select(".col-md-1")
                for value in values:
                    num = value.text.strip()
                    self.tsum = self.tsum + int(num)
            except Exception as e:
                print(e,"解析函数异常")
        else:
            print("页面内容是不存在的")
        return self.tsum

if __name__ == "__main__":
    tsum = TSUM()
    page_urls = tsum.create_tpage_url()
    for page_url in page_urls:
        html = tsum.get_tpage_url(page_url)
        fsum = tsum.parse_tpage_url(html)
    print(fsum)

