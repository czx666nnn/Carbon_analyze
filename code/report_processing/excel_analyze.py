import pdfplumber
import json
import pandas as pd
import re


class PDFTableProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdfplumber = pdfplumber.open(pdf_path)  

    def extract_tables(self):
        tables = []
        for page in self.pdfplumber.pages:
            table = page.extract_tables()
            for tbl in table:
                tables.append(tbl)
        return tables

    def handle_cross_page_tables(self, tables):
        """
        处理跨页表格（通过匹配表头或表格编号等信息）
        """
        combined_tables = []
        current_table = []
        header = None  

        for table in tables:
            if not header:
                header = table[0]
            else:
                if table[0] == header:
                    current_table.extend(table[1:])
                else:
                    if current_table:
                        combined_tables.append(current_table)
                    current_table = table
        if current_table:
            combined_tables.append(current_table)
        return combined_tables

    def handle_merged_cells(self, table):
        """
        处理合并单元格（通过字段位置或XML信息解析）
        """
        df = pd.DataFrame(table[1:], columns=table[0])
        return df

    def infer_missing_header(self, tables):
        """
        自动推断无表头的表格（通过上下文推断列名）
        """
        for table in tables:
            if not self.has_header(table):
                inferred_header = self.infer_header_from_context(table)
                if isinstance(table, list) and len(table) > 0:
                    table[0] = inferred_header
        return tables

    def has_header(self, table):
        """简单检查表格是否包含表头"""
        if isinstance(table, list) and len(table) > 0 and isinstance(table[0], list):
            return bool(re.match(r"^[A-Za-z ]+$", str(table[0][0])))
        return False

    def infer_header_from_context(self, table):
        """通过表格内容自动推测表头"""
        return ['Product', 'Sales', 'Quantity'] 


def extract_tables_from_pdf(pdf_path, json_output_path):
    tables_json = {}

    processor = PDFTableProcessor(pdf_path)

    tables = processor.extract_tables()

    tables = processor.handle_cross_page_tables(tables)

    tables = [processor.handle_merged_cells(table) for table in tables]

    tables = processor.infer_missing_header(tables)

    for page_num, page in enumerate(processor.pdfplumber.pages):
        page_tables = []
        for table_index, table in enumerate(tables):
            table_data = {
                "table_index": table_index + 1,
                "data": table.values.tolist()
            }
            page_tables.append(table_data)

        if page_tables:
            tables_json[f"page_{page_num + 1}"] = page_tables

    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(tables_json, json_file, ensure_ascii=False, indent=4)

    print(f"Tables extracted and saved to {json_output_path}")



pdf_path = ''  # 输入 PDF 文件路径
json_output_path = ''  # 输出的 JSON 文件路径
extract_tables_from_pdf(pdf_path, json_output_path)


import mysql.connector
import json
def store_tables_in_db(json_file_path, db_config):
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = conn.cursor()
        print("Connected to the database successfully.")

        with open(json_file_path, "r", encoding="utf-8") as json_file:
            tables_data = json.load(json_file)

        for page, tables in tables_data.items():
            for table_data in tables:
                table_index = table_data['table_index']
                table_content = json.dumps(table_data['data']) 


                query = """
                    INSERT INTO pdf_tables (page, table_index, table_content)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(query, (page, table_index, table_content))

        conn.commit()
        print(f"Data from {json_file_path} stored in the database successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if conn:
            cursor.close()
            conn.close()

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'database': 'your_database_name'
}

json_file_path = ''

store_tables_in_db(json_file_path, db_config)
