import pymysql

# 数据库连接
def conn_db():
    db = pymysql.connect(
        host ='127.0.0.1',
        port=3306,
        user='root',
        db='test',
        password='123456',
    )
    cursor =db.cursor()
    print("连接数据库成功")
    return db,cursor

#创建表
def create_table(db,cursor,dsql,csql):
    cursor.execute(dsql)
    try:
        cursor.execute(csql)
        db.commit()
    except Exception as e:
        print(e)
        db.rollback()
    db.close()
    print("数据库已关闭")

#更新、插入数据
def exe_update(db,cursor,sql):
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print(e)
        db.rollback()
    db.close()
    print("数据库已关闭")

# 查询数据
def query_data(db,cursor,sql):


db,cursor = conn_db()
dsql = "DROP TABLE IF EXISTS ZHIHU1"
csql = """
        CREATE TABLE ZHIHU1 (
        id INT ,
        question VARCHAR (256),
        username VARCHAR (256),
        userimage VARCHAR (512),
        content LONGTEXT,
        favour INT ,
        ctime DATE ) DEFAULT CHARSET=utf8;"""
create_table(db,cursor,dsql,csql)
