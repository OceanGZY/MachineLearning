import requests
from bs4 import BeautifulSoup
import time
import os


class GetPic():
    def __init__(self):
        self.BASE_URL = "https://www.meitulu.com"

    def create_person_url(self):
        person_name = "wangyuchun_2"
        person_url = self.BASE_URL + "/t/wangyuchun/2.html"
        return person_url, person_name

    def get_person_url(self, url, person_name):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # print(response.content)
                self.parse_person_url(response.content, person_name)
            else:
                print("获取person_url失败")
        except Exception as e:
            print("get_person_url错误", e)

    def parse_person_url(self, html, person_name):
        person_img_set_list = []
        if html:
            try:
                soup = BeautifulSoup(html, "html.parser")
                person_img_sets = soup.select(".boxs .img li .p_title a")
                for i, item in enumerate(person_img_sets):
                    person_img_set_url = item.get("href")
                    person_img_set_name = person_name + str(i + 1)
                    img_dir = os.getcwd() + "/" + person_img_set_name
                    # print(img_dir)
                    if not os.path.exists(img_dir):
                        os.mkdir(img_dir)
                    # person_img_set_list.append(
                    #     {
                    #         "person_img_set": {"person_img_set_name": person_img_set_name,
                    #                            "person_img_set_url": person_img_set_url, }
                    #     }
                    # )
                    self.get_person_img_set_url(person_img_set_url, img_dir)
            except Exception as e:
                print("parse_person_url: 错误", e)
        else:
            print("person_url_html不存在")

    def get_person_img_set_url(self, url, img_dir):
        if url:
            try:
                time.sleep(1)
                response = requests.get(url)
                if response.status_code == 200:
                    self.parse_person_img_set_url(response.content, img_dir)
                else:
                    print("抓取get_person_img_set_url失败")
            except Exception as e:
                print("get_person_img_set_url 错误:", e)
        else:
            print("get_person_img_set_url是空的")

    def parse_person_img_set_url(self, html, img_dir):
        if html:
            try:
                soup = BeautifulSoup(html, "html.parser")
                person_img_sets = soup.select("center #pages a")[:-1]
                for item in person_img_sets:
                    url = item.get("href")
                    person_img_sets_item_url = self.BASE_URL + url
                    self.get_person_img_sets_item_url(person_img_sets_item_url, img_dir)
            except Exception as e:
                print("parse_person_img_set_url 错误:", e)
        else:
            print("get_person_img_set_url的内容是空的")

    def get_person_img_sets_item_url(self, url, img_dir):
        if url:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    self.parse_person_img_sets_item_url(response.content, img_dir)
                else:
                    print("抓取person_img_sets_item_url 失败")
            except Exception as e:
                print("get_person_img_sets_item_url 错误:", e)
        else:
            print("person_img_sets_item_url是空的")

    def parse_person_img_sets_item_url(self, html, img_dir):
        if html:
            try:
                soup = BeautifulSoup(html, "html.parser")
                person_img_sets_item_imgs = soup.select(".content center img")
                for i, item in enumerate(person_img_sets_item_imgs):
                    person_img_sets_item_img_url = item.get("src")
                    name = person_img_sets_item_img_url.split("/")[-1:][0]
                    self.getpic(person_img_sets_item_img_url, name, img_dir)
            except Exception as e:
                print("parse_person_img_sets_item_url 错误", e)
        else:
            print("get_person_img_sets_item_url的内容是空的")

    def getpic(self, url, name, img_dir):
        time.sleep(1)
        response = requests.get(url)
        if response.status_code == 200:
            file = img_dir + "/" + name
            with open(file, "wb") as f:
                f.write(response.content)
                print(file,"：保存成功")


getpic = GetPic()
url, person_name = getpic.create_person_url()
getpic.get_person_url(url, person_name)
