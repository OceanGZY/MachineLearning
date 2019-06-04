import requests
from bs4 import BeautifulSoup


class AJK(object):
    def __init__(self):
        self.BASE_URL = 'https://nanjing.anjuke.com/'
        self.second_hand_house_url = self.BASE_URL + 'sale/p1/#filtersort'

    def get_second_hand_house_page(self):
        headers = {
            

        }
        try:
            res = requests.get(self.second_hand_house_url,headers = headers)
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
        count = len(house_titles)
        # print(count)
        house_models = []
        house_adresses = []
        for x in range(2 * count):
            if (x % 2 == 0):
                # print(x)
                house_models.append(house_types[x])
            else:
                # print(x)
                house_adresses.append(house_types[x])

        # print(house_models)
        # print(house_adresses)

        for (i, j, k, l, m, n) in zip(house_img_urls, house_titles, house_models, house_prices, house_unit_prices,
                                      house_adresses):

            img_url = i.get('src')
            print('预览图', img_url)

            title = j.text.strip()
            print('名字', title)

            house_price = l.text.strip('万')
            price = float(house_price)
            print('总价', price)

            house_unit_price = m.text.strip("元/m²")
            unit_price = float(house_unit_price)
            print('每平米价格', unit_price)

            house_adress = n.text.strip().split('\xa0\xa0\n                    ')
            street = house_adress[0]
            locations = house_adress[1].split('-')
            offset = locations[0]
            position = locations[1]
            roadnumber = locations[2]

            print('街道', street)
            print('区名', offset)
            print('位置', position)
            print('路号', roadnumber)

            house_model = k.text.strip().split('|')
            if (len(house_model) > 3):
                style = house_model[0]
                area = house_model[1].strip('m²')
                floor = house_model[2]
                year = house_model[3]
            else:
                style = house_model[0]
                area = house_model[1].strip('m²')
                floor = house_model[2]
                year = ""
            print('户型', style)
            print('面积', area)
            print('楼层', floor)
            print('建造时间', year)


if __name__ == '__main__':
    ajk = AJK()
    ajk.get_second_hand_house_page()