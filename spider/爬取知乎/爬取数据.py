import json
import datetime
from bs4 import BeautifulSoup
import time
from selenium import webdriver
import pymysql

class CrawlZHIHU(object):
    def __init__(self):
        self.BASE_URL = "https://www.zhihu.com/"

    def make_api_url(self):
        for i in range(0,6842,5):
            api_url = self.BASE_URL +"api/v4/questions/22748892/answers?include=data[*].is_normal,admin_closed_comment,reward_info,is_collapsed,annotation_action,annotation_detail,collapse_reason,is_sticky,collapsed_by,suggest_edit,comment_count,can_comment,content,editable_content,voteup_count,reshipment_settings,comment_permission,created_time,updated_time,review_info,relevant_info,question,excerpt,relationship.is_authorized,is_author,voting,is_thanked,is_nothelp,is_labeled,is_recognized,paid_info,paid_info_content;data[*].mark_infos[*].url;data[*].author.follower_count,badge[*].topics&limit=5&offset=" + str(i) + "&platform=desktop&sort_by=default"
            # api_url = self.BASE_URL +"api/v4/questions/22748892/answers?include=data[*].is_normal,admin_closed_comment,reward_info,is_collapsed,annotation_action,annotation_detail,collapse_reason,is_sticky,collapsed_by,suggest_edit,comment_count,can_comment,content,editable_content,voteup_count,reshipment_settings,comment_permission,created_time,updated_time,review_info,relevant_info,question,excerpt,relationship.is_authorized,is_author,voting,is_thanked,is_nothelp,is_labeled,is_recognized,paid_info,paid_info_content;data[*].mark_infos[*].url;data[*].author.follower_count,badge[*].topics&limit=5&offset=0&platform=desktop&sort_by=default"
            self.get_api_url(api_url)

    def get_api_url(self,url):
        opener = webdriver.Chrome()
        if url:
            print(url)
            time.sleep(3)
            opener.get(url)
            html = BeautifulSoup(opener.page_source,"html.parser")
            predata = html.select("pre")
            for i in predata:
                data = i.text
                djson = json.loads(data,encoding="utf-8")
                ddata = djson["data"]
                for j in ddata:
                    username = j["author"]["name"]
                    userimage = j["author"]["avatar_url"]
                    ccontent = j["content"]
                    favour = j["voteup_count"]
                    creat_time = j["created_time"]
                    dateArray = datetime.datetime.utcfromtimestamp(creat_time)
                    ctime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
                    self.save_comit(username,userimage,ccontent,favour,ctime)

    def save_comit(self,name,image,ccontent,favour,ctime):

        print(name,image,ccontent,favour,ctime)


if __name__ == "__main__":
    crawlzhihu = CrawlZHIHU()
    crawlzhihu.make_api_url()