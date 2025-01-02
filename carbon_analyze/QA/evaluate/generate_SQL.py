# -*- encoding: utf8 -*-

import pandas as pd
import requests
import json


def find_matching_row(value, df2):
    # 移除table列值中的方括号
    value_stripped = value.strip("[]\"")

    # 在df2中查找匹配的行
    matching_row = df2.loc[df2['tab_name'] == value_stripped, 'tab_schema']

    # 如果找到匹配的行，返回tab_schema值
    if not matching_row.empty:
        return matching_row.iloc[0]

    # 如果没有找到匹配的行，返回None
    return None


def convert_time_string_to_list(time_string):
    # 使用 json.loads 将字符串转换为列表
    time_list = json.loads(time_string)
    return time_list


def format_date(date):
    year, month, day = date.split('-')
    return f"{year}年{month}月{day}日"


def extract_time_info(time):
    # time_descriptions = json.loads(time)
    sentence_parts = []

    if time:
        for description in time:
            if description:
                # 将时间范围从 "YYYY-MM-DD～YYYY-MM-DD" 转换为 "YYYY年MM月DD日到YYYY年MM月DD日"
                start, end = description['time'].split('～')
                if start == end:
                    formatted_start_date = format_date(start)
                    formatted_time = formatted_start_date
                # 拼接时间描述和时间范围
                else:
                    formatted_start_date = format_date(start)
                    formatted_end_date = format_date(end)
                    formatted_time = f"{formatted_start_date}到{formatted_end_date}"
                sentence_parts.append(f"{description['time_des']}指{formatted_time}")
            else:
                return ""

        full_sentence = '，'.join(sentence_parts) + '。'

        # print(full_sentence)
        return full_sentence
    else:
        return ""

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage


#星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = ''
#星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = ''
SPARKAI_API_SECRET = ''
SPARKAI_API_KEY = ''
#星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = ''
import re
def response(prompt):
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False
    )
    # 创建ChatMessage对象
    messages = [ChatMessage(
        role="user",
        content=prompt
    )]

    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    response_json = a.json()
    print(response_json)
    response_json = json.loads(response_json)
    generations = response_json["generations"]
    generated_text = generations[0][0]["text"]

    print(generated_text)

    return generated_text



replacements = {
    "全省": "河北省",
    "全市": "石家庄市",
    "本省": "河北省",
    "本市": "石家庄市"
}


def replace_query(query, replacements):
    for old, new in replacements.items():
        query = query.replace(old, new)
    return query


def generate_sql_with_time(query, tab_schema, table, time, remarks):
    sys_prompt = "你是MySQL专家,对于给定问题和schema等信息，你的任务是创建一个语法准确、准确回答用户问题的高效MySQL查询"
    query = replace_query(query, replacements)
    if remarks != '':
        # if sample != '':
        # SQL_Prompt = f'''根据给定问题和已知信息生成满足要求的SQL。\n 问题：{query}\n 已知信息: ``` \n 数据表信息：{table}. \n数据表的建表信息：{tab_schema}. \n 已知{time},直接使用此日期，无需推算。\n 生成过程中参考如下信息：\n{remarks}. \n ``` \n  要求：\n 在给定数据表范围内查询回答问题所必须的列，注意使用列的单位，SQL必须正确使用表中含有的列名，根据schema和问题为查询的列名或计算结果设置中文别名。不明确查询日期但是表有含有日期的列时，默认是查询表中最新日期的数据。\n 你可以一步步思考如何得到正确的SQL，但最终只给出MySQL语句。除了生成的SQL不要包含任何其他。'''

        SQL_Prompt = f'''根据给定问题和已知信息生成满足要求的SQL。\n 问题：{query}\n 已知信息: ``` \n 数据表信息：{table}. \n数据表的建表信息：{tab_schema}. \n 已知{time},直接使用此日期，无需推算。\n 生成过程中参考如下信息：\n{remarks}. \n ```  \n 要求：\n 在给定数据表范围内查询回答问题所必须的列，注意使用列的单位，SQL必须正确使用表中含有的列名，根据schema和问题为查询的列名或计算结果设置中文别名。不明确查询日期但是表有含有日期的列时，默认是查询表中最新日期的数据。\n 你可以一步步思考如何得到正确的SQL，但最终只给出MySQL语句。除了生成的SQL不要包含任何其他。'''

    else:
        #   if sample != '':
        #      SQL_Prompt = f'''根据给定问题和已知信息生成满足要求的SQL。\n 问题：{query}\n 已知信息: ``` \n 数据表信息：{tab_schema}. \n 已知{time},直接使用此日期，无需推算。 \n ``` \n  要求：\n 在给定数据表范围内查询回答问题所必须的列，注意使用列的单位，SQL必须正确使用表中含有的列名，根据schema和问题为查询的列名或计算结果设置中文别名。不明确查询日期但是表有含有日期的列时，默认是查询表中最新日期的数据。\n 你可以一步步思考如何得到正确的SQL，但最终只给出MySQL语句。除了生成的SQL不要包含任何其他。'''

        SQL_Prompt = f'''根据给定问题和已知信息生成满足要求的SQL。\n 问题：{query}\n 已知信息: ``` \n 数据表信息：{table}. \n数据表的建表信息：{tab_schema}. \n 已知{time},直接使用此日期，无需推算。 \n ```  \n 要求：\n 在给定数据表范围内查询回答问题所必须的列，注意使用列的单位，SQL必须正确使用表中含有的列名，根据schema和问题为查询的列名或计算结果设置中文别名。不明确查询日期但是表有含有日期的列时，默认是查询表中最新日期的数据。\n 你可以一步步思考如何得到正确的SQL，但最终只给出MySQL语句。除了生成的SQL不要包含任何其他。'''

    print(SQL_Prompt)

    return response(SQL_Prompt)


def extract_sql(response_txt):
    import re
    # 正则表达式匹配 SELECT 语句，考虑大小写，并允许多行
    # SQL 语句可能不以分号结束，因此这里不将分号作为必须的结束符
    # 这个正则表达式仍然可能需要根据实际返回的文本格式进行调整
    pattern = re.compile(r"select.*?(?:;|$)", re.S | re.I)
    match = pattern.search(response_txt)
    if match:
        # 清理多行语句中的换行符
        sql_statement = ' '.join(match.group(0).splitlines())
        return sql_statement.strip()  # 返回匹配到的 SQL 语句，去除首尾空白
    return ""  # 如果没有找到匹配项，返回空字符串


def get_table_names1(query):
    url = ""
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"query": query})
    response = requests.post(url, headers=headers, data=data)
    extracted_data = json.loads(response.text)["table_names"]
    list_str = str(extracted_data).strip("[]").replace("'", '"')
    extracted_data = f'[{list_str}]'
    return extracted_data


def choose_table2(query):
    url = ""
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"query": query})
    response = requests.post(url, headers=headers, data=data)
    print(response.text)
    extracted_data = json.loads(response.text)["table_names"]
    list_str = str(extracted_data).strip("[]").replace("'", '"')
    extracted_data = f'[{list_str}]'
    return extracted_data


def run_sql(sql):
    url = ""
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"sql": sql})
    response = requests.post(url, headers=headers, data=data)
    print(response.text)
    extracted_data = json.loads(response.text)["data"]
    error = json.loads(response.text)["error"]
    return extracted_data, error


def calculate_recall_precision(useTable, queryTable, chooseTable):
    """
    对于给定的查询和选择集，计算召回率和精确率。
    """
    useTable_set = set(useTable)
    queryTable_set = set(queryTable)
    chooseTable_set = set(chooseTable)

    recall = 0 if not useTable_set else 1 if useTable_set.issubset(queryTable_set) else 0

    precision = 0 if not set(useTable) else 1 if useTable_set == chooseTable_set else 0
    '''# 计算召回率
    recall = 1 if set(useTable).issubset(set(queryTable)) else 0

    # 计算精确率
    precision = 1 if useTable == chooseTable else 0'''

    return recall, precision


# 存储结果的字典
data = {
    'query': [],
    'remarks': [],
    'time': [],
    'table': [],
    'tab_schema': [],
    'oriSql': [],
    'newSql': [],
    'get_table_names': [],
    'choose_table': [],
    'sql_result': [],
    'error': [],
    'recall': [],
    'precision': []

}
from langchain_chroma import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
qianfan_ak = ''
qianfan_sk = ''
embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)
persist_directory = ""
Cot_directory = ""

def RAG_cot_prompt(query):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    Example = retriever.invoke(query)
    return Example
def extract_info_from_excel(file_path, file_path2, out_path):
    # 读取Excel文件
    excel_file = pd.ExcelFile(file_path)
    df2 = pd.read_excel(file_path2)
    # 遍历每个工作表
    for sheet in excel_file.sheet_names:
        # 读取工作表数据
        df = excel_file.parse(sheet)
        # 提取信息
        for index, row in df.iterrows():
            query = row['query']
            oriSql = row['oriSql']
            time = row['time']
            table = row['table']
            tab_schema = find_matching_row(table, df2)
            if time is not []:
                time = convert_time_string_to_list(time)
                time = extract_time_info(time)
            else:
                time = time

            if query and tab_schema and table:
                # 调用generate_sql_with_time函数获取生成的SQL语句
                remarks = RAG_cot_prompt(query)
                response_txt = generate_sql_with_time(query, tab_schema, table, time, remarks)
                # print('response_txt::::',response_txt)
                newSql = extract_sql(response_txt)

                sql_result, error = run_sql(newSql)
                get_table_names = get_table_names1(query)
                choose_table = choose_table2(query)
                recall, precision = calculate_recall_precision(table, get_table_names, choose_table)

                data['query'].append(query)
                data['time'].append(time)
                data['remarks'].append(remarks)
                data['table'].append(table)
                data['oriSql'].append(oriSql)
                data['tab_schema'].append(tab_schema)
                data['newSql'].append(newSql)
                data['sql_result'].append(sql_result)
                data['error'].append(error)
                data['get_table_names'].append(get_table_names)
                data['choose_table'].append(choose_table)
                data['recall'].append(recall)
                data['precision'].append(precision)

    out_df = pd.DataFrame(data)
    print(out_df)
    out_df.to_csv(out_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    file_path = ''
    file_path2 = ''
    out_file = ''
    extract_info_from_excel(file_path, file_path2, out_file)

























