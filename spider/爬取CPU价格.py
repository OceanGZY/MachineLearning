# _*_ coding:utf-8 _*_
'''
    auth:GZY
    date:2018/11/20
    version:0.1.0
'''

import urllib
import requests
from bs4 import BeautifulSoup
from lxml import etree
class CPUPrice(object):
    BASE_URL = 'http://detail.zol.com.cn/cpu/'
    keywords = ['index1204791','index1177026','index1233484']
    cpu =['2700x','8700k','9900k']


    def __init__(self):
        return  None


    # 获取需要爬取的URL地址
    def get_url(self):
        for keyword in self.keywords:
            urls = self.BASE_URL + keyword + '.shtml'
            self.get_pages(urls)

    #尝试获取URL地址的页面内容
    def get_pages(self,url):

        try:
            response = requests.get(url)
            if response.status_code ==200:
                self.get_prices(response.content)
            else:
                print('页面请求异常',response.status_code)
        except Exception as e:
            print('请求页面异常',e)

    # 解析页面内容，存储cpu价格
    def get_prices(self, html):
        # soup = BeautifulSoup(html, 'html.parser')
        selector = etree.HTML(html)
        ckprices = selector.xpath('/html/body/div[13]/div[2]/div[2]/div/span/b[2]')
        jdprices = selector.xpath('/html/body/div[13]/div[2]/div[3]/dl[2]/dd/ul/li[1]/a[2]')
        tmprices = selector.xpath('/html/body/div[13]/div[2]/div[3]/dl[2]/dd/ul/li[2]/a[2]')
        for i in range(len(ckprices)):
            print('参考价格',ckprices[i].text)

        for index in range(len(jdprices)):
            print('京东',jdprices[index].text)

        for n in range(len(tmprices)):
            print('天猫',tmprices[n].text)






if __name__ =="__main__":
    cpuPrice = CPUPrice()
    cpuPrice.get_url()