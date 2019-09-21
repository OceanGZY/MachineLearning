#!/usr/bin/python3
'''
    date:20190916
    author:gzy
    version:V0.1.0
'''
import time
import json
import pymysql
from selenium import webdriver
from bs4 import BeautifulSoup
import datetime



class CrawlZHIHU(object):
    def __init__(self):
        self.BASE_URL = "https://www.zhihu.com/"

    #连接数据库
    def conn_db(self):
        db = pymysql.connect(
            host="127.0.0.1",
            port = 3306,
            user = "root",
            password = "123456",
            db ="zhihu"
        )

        cursor = db.cursor()
        print("数据库连接成功")
        return db,cursor

    # 创建数据表
    def create_table(self,db,cursor,dsql,csql):
        cursor.execute(dsql)
        try:
            cursor.execute(csql)
            db.commit()
            print("创建表成功")
        except Exception as e:
            db.rollback()
            print(e)

        cursor.close()
        db.close()
        print("数据库已关闭")


    # 插入数据
    def insert_table(self,db,cursor,isql):
        try:
            cursor.execute(isql)
            db.commit()
            print("插入数据成功")
        except Exception as e:
            db.rollback()
            print(e)

        cursor.close()
        db.close()
        print("数据库已关闭")



    def make_api_url(self):
        for i in range(0,1328,10):
            api_url = self.BASE_URL +"api/v4/questions/333264576/answers?include=data%5B%2A%5D.is_normal%2Cadmin_closed_comment%2Creward_info%2Cis_collapsed%2Cannotation_action%2Cannotation_detail%2Ccollapse_reason%2Cis_sticky%2Ccollapsed_by%2Csuggest_edit%2Ccomment_count%2Ccan_comment%2Ccontent%2Ceditable_content%2Cvoteup_count%2Creshipment_settings%2Ccomment_permission%2Ccreated_time%2Cupdated_time%2Creview_info%2Crelevant_info%2Cquestion%2Cexcerpt%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%2Cis_labeled%2Cis_recognized%2Cpaid_info%2Cpaid_info_content%3Bdata%5B%2A%5D.mark_infos%5B%2A%5D.url%3Bdata%5B%2A%5D.author.follower_count%2Cbadge%5B%2A%5D.topics&limit=10&offset="+ str(i)+"&platform=desktop&sort_by=default"
            self.get_api_url(api_url)
        # url = "view-source:https://www.zhihu.com/api/v4/questions/333264576/answers?include=data%5B%2A%5D.is_normal%2Cadmin_closed_comment%2Creward_info%2Cis_collapsed%2Cannotation_action%2Cannotation_detail%2Ccollapse_reason%2Cis_sticky%2Ccollapsed_by%2Csuggest_edit%2Ccomment_count%2Ccan_comment%2Ccontent%2Ceditable_content%2Cvoteup_count%2Creshipment_settings%2Ccomment_permission%2Ccreated_time%2Cupdated_time%2Creview_info%2Crelevant_info%2Cquestion%2Cexcerpt%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%2Cis_labeled%2Cis_recognized%2Cpaid_info%2Cpaid_info_content%3Bdata%5B%2A%5D.mark_infos%5B%2A%5D.url%3Bdata%5B%2A%5D.author.follower_count%2Cbadge%5B%2A%5D.topics&limit=5&offset=15&platform=desktop&sort_by=default"
        # self.get_api_url(url)

    def get_api_url(self,url):
        opener = webdriver.Chrome()
        if url:
            print(url)
            time.sleep(3)
            opener.get(url)
            soup = BeautifulSoup(opener.page_source,"html.parser")
            html = soup.select("pre")
            for hdata in html:
                predata = hdata.text
                prejson = json.loads(predata,encoding="utf-8")
                pdata = prejson["data"]

                for i in pdata:
                    question_id = i["question"]["id"]
                    question_title = i["question"]["title"]
                    created_time = i["created_time"]
                    dateArray1 = datetime.datetime.utcfromtimestamp(created_time)
                    created_time = dateArray1.strftime("%Y-%m-%d")

                    updated_time = i["updated_time"]
                    dateArray2 = datetime.datetime.utcfromtimestamp(updated_time)
                    updated_time = dateArray2.strftime("%Y-%m-%d")

                    author_name = i["author"]["name"]
                    author_avatar = i["author"]["avatar_url"]
                    author_gender = i["author"]["gender"]

                    answer = i["content"]
                    favour = i["voteup_count"]

                    # print(question_id,question_title,created_time,updated_time,author_name,author_avatar,author_gender,answer,favour)
                    isql = """
                            INSERT INTO ZHIHU2(
                            question_id,
                            question_title,
                            created_time,
                            updated_time,
                            author_name,
                            author_avatar,
                            author_gender,
                            answer,
                            favour
                            ) VALUES('%s','%s','%s','%s','%s','%s','%s','%s','%s')""" %(question_id,question_title,created_time,updated_time,author_name,author_avatar,author_gender,answer,favour)
                    db,cursor = self.conn_db()
                    self.insert_table(db,cursor,isql)





if __name__ == '__main__':
    crawlzhihu = CrawlZHIHU()
    db,cursor = crawlzhihu.conn_db()
    dsql = """
            DROP TABLE IF EXISTS zhihu2;
            """
    csql = """
            CREATE TABLE ZHIHU2(
            question_id VARCHAR (20),
            question_title VARCHAR (256),
            created_time DATE,
            updated_time DATE ,
            author_name VARCHAR (256),
            author_avatar VARCHAR (512),
            author_gender INT,
            answer LONGTEXT,
            favour INT) DEFAULT CHARSET=utf8mb4;
            """
    crawlzhihu.create_table(db,cursor,dsql,csql)
    crawlzhihu.make_api_url()

