import pymysql
import os
import csv

'''
    author:gzy
    date:20191004
    version:0.1.0
'''

class Db(object):
    def db_conn(self):
        db = pymysql.connect(
            host ="127.0.0.1",
            port = 3306,
            db = "zhihu",
            user ="root",
            password = "123456",
        )

        cursor = db.cursor()
        print("数据库已连接")
        return db,cursor

    def query_tb(self,db,cursor,qsql):
        file = os.getcwd() + "/nalanci.csv"
        headers = ["id","title","poem"]
        with open(file,"w+") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)

        try:
            cursor.execute(qsql)
            results = cursor.fetchall()
            for j in results:
                with open(file,"a+") as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(j)
        except Exception as e:
            print(e)
        cursor.close()
        db.close()
        print("数据库已关闭")

if __name__ == "__main__":
    squry = Db()
    db,cursor = squry.db_conn()
    qsql = """
        SELECT * FROM nalanpoem;
    """
    squry.query_tb(db,cursor,qsql)



