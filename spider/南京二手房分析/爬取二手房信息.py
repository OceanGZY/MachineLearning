import requests
from bs4 import BeautifulSoup


class AJK(object):
    def __init__(self):
        self.BASE_URL = 'https://nanjing.anjuke.com/'
        self.second_hand_house_url = self.BASE_URL + 'sale/p1/#filtersort'

    def get_second_hand_house_page(self):
        try:
            res = requests.get(self.second_hand_house_url)
            if res.status_code == 200:
                print(res.text)
            else:
                print('请求失败了',res.status_code)
        except Exception as e:
            print('try 失败了',e)

    def parse_sencond_hand_house_page(self,html):
        soup = BeautifulSoup(html,'lxml')






if __name__ == '__main__':
    ajk = AJK()
    ajk.get_second_hand_house_page()