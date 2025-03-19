import mysql.connector
from mysql.connector import Error

def connect_to_database(host, database, user, password):
    """
    连接到MySQL数据库
    """
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        if connection.is_connected():
            print("成功连接到数据库")
            return connection
    except Error as e:
        print(f"错误: {e}")
        return None

def execute_query(connection, query):
    """
    执行SQL查询并返回结果
    """
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        columns = cursor.column_names
        return columns, result
    except Error as e:
        print(f"查询错误: {e}")
        return None, None
    finally:
        cursor.close()
