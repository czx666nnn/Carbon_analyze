# schema.py
import mysql.connector
from mysql.connector import Error

def get_table_names(connection):
    """
    获取数据库中所有表的名称
    """
    cursor = connection.cursor()
    try:
        # 查询当前数据库的所有表名
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = DATABASE()")
        rows = cursor.fetchall()
        # 提取表名并返回为列表
        table_names = [row[0] for row in rows]
        return table_names
    except Error as e:
        print(f"获取表名错误: {e}")
        return []
    finally:
        cursor.close()

def get_db_schema(connection):
    """
    获取数据库的模式，包括所有表和列
    """
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE()")
        rows = cursor.fetchall()
        schema = {}
        for table, column, data_type in rows:
            if table not in schema:
                schema[table] = []
            schema[table].append(f"{column} {data_type}")
        schema_str = ""
        for table, columns in schema.items():
            schema_str += f"表 `{table}` ("
            schema_str += ", ".join(columns)
            schema_str += ")\
"
        return schema_str
    except Error as e:
        print(f"获取模式错误: {e}")
        return ""
    finally:
        cursor.close()
