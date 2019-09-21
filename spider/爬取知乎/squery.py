import pymysql
# 数据库连接
def conn_db():
    db = pymysql.connect(
        "127.0.0.1",
        "root",
        "",
        "zhihu",
    )
    cursor = db.cursor()
    print("数据库连接成功")
    return db,cursor


# 查询数据
def squery_db(db,cursor,qsql):
    try:
        cursor.execute(qsql)
        # result = cursor.fetchone()
        results = cursor.fetchmany(5)
        for i in results:
            print(i)

    except Exception as e:
        print(e)
    cursor.close()
    db.close()
    print("数据库已关闭")



db,cursor = conn_db()
qsql = """
        SELECT * FROM zhihu1;
    """
squery_db(db,cursor,qsql)

