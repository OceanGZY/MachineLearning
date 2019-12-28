import requests
from bs4 import BeautifulSoup
import time

class QueryText(object):
    def __init__(self):
        self.BASE_URL = "https://7.n15.info/"

    def make_pageid(self):
        for pageid in range(1,125):
            self.get_story_list(pageid)

    def get_story_list(self,pageid):
        time.sleep(1)
        url = self.BASE_URL + '/stories/all/free?page=' + str(pageid)
        response = requests.get(url)
        if response.status_code == 200:
            self.parser_story_list(response.content)
        else:
            print(response.status_code)

    def parser_story_list(self,html):
        ht = BeautifulSoup(html, 'html.parser')
        story_urls = ht.select('.row .colNovelList .novelElem')
        story_titles = ht.select('.row .colNovelList .novelElem .container-title')
        for i, j in zip(story_urls, story_titles):
            story_url = self.BASE_URL + i.get('href')
            story_title = j.text.strip()
            self.get_story(story_url,story_title)

    def get_story(self,url,title):
        resp = requests.get(url)
        if resp.status_code == 200:
            self.save_story(resp.content,title)
        else:
            print(resp.status_code)

    def save_story(self,html,title):
        htm = BeautifulSoup(html, 'html.parser')
        stories = htm.select('.row .novelContent ')
        for j in stories:
            story = j.text.strip()
            story_file_path =  title +'.txt'
            with open (story_file_path,'w+',encoding='utf-8') as f:
                f.write(story)
                print("保存"+ title+"小说成功")


if __name__ == "__main__":
    querytext = QueryText()
    querytext.make_pageid()
    print("保存结束")



