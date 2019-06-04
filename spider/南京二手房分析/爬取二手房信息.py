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
                self.parse_sencond_hand_house_page(res.content)
            else:
                print('请求失败了',res.status_code)
        except Exception as e:
            print('try 失败了',e)

    def parse_sencond_hand_house_page(self,html):
        soup = BeautifulSoup(html,'lxml')
        house_img_urls = soup.select('.list-item .item-img img')
        house_titles = soup.select('.house-details .house-title a')
        house_types = soup.select('.house-details .details-item')
        house_prices = soup.select('.pro-price .price-det')
        house_unit_prices = soup.select('.pro-price .unit-price')

        for k,(i,j,n,l,m) in enumerate(zip(house_img_urls,house_titles,house_types,house_prices,house_unit_prices)):
            print(i)
            house_img_url = i.get('src')
            print(house_img_url)

            print(j)
            house_title = j.text.strip()
            print(house_title)

            if (k % 2 == 0):
                print(k)
                house_type = n.text.strip().split('|')
                if (len(house_type) > 3):
                    style = house_type[0]
                    area = house_type[1].strip('m²')
                    floor = house_type[2]
                    year = house_type[3]
                else:
                    print(k)
                    style = house_type[0]
                    area = house_type[1].strip('m²')
                    floor = house_type[2]
                    year = ''

                print(style)
                print(area)
                print(floor)
                print(year)
                print('-----------------------------------')
            else:
                house_adress = n.text.strip().split('\xa0\xa0\n                    ')
                print('地址', house_adress)
                print('-----------------------------------')

            house_price = l.text.strip('万')
            house_price = int(house_price)
            print(house_price)

            house_unit_price = m.text.strip("元/m²")
            house_unit_price = int(house_unit_price)
            print(house_unit_price)







if __name__ == '__main__':
    ajk = AJK()
    ajk.get_second_hand_house_page()