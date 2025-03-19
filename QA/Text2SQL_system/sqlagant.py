# agent.py
from langchain.agents import Tool, create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentOutputParser
import re
from pydantic import BaseModel
from typing import Any, Dict
from sql_tool import SQLGenerator
from db import execute_query
from schema import get_db_schema,get_table_names
# from security import is_safe_sql
from tabulate import tabulate
import json
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
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
    # print(a)
    response_json = a.json()
    # print(response_json)
    response_json = json.loads(response_json)
    generations = response_json["generations"]
    generated_text = generations[0][0]["text"]
    return generated_text
##################################################

class ExecuteSQLArgs(BaseModel):
    query: str

def is_safe_sql(sql):
    """
    简单的SQL安全检查函数
    """
    # 禁止删除、更新等危险操作
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    for word in forbidden:
        if re.search(rf"\\b{word}\\b", sql, re.IGNORECASE):
            return False
    return True

def execute_sql_tool(args: ExecuteSQLArgs, sql_generator: SQLGenerator, db_connection) -> str:
    query = args.query
    sql = sql_generator.generate_sql(query)
    print(f"生成的 SQL 语句:\n{sql}")  # 调试输出
    if not sql or not is_safe_sql(sql):
        return "生成的SQL语句为空或包含不允许的操作。"

    # 执行 SQL 查询
    columns, result = execute_query(db_connection, sql)
    if columns and result:
        # 将结果格式化为表格字符串
        table = tabulate(result, headers=columns, tablefmt="grid")
        return f"查询结果:\n{table}"
    return "查询失败或无结果。"

def choose_table_api(query,table_names, table_des,supple_info=''):#星火api
    sys_prompt = "你精通于分析和解读问题，并能够高效地从多个数据表中识别和定位出查询所需的关键数据表"
    try:
        supple_info= supple_info[:100]
    except:
        supple_info = ''

    chose_tabe_prompt = f"你精通于分析和解读问题，并能够高效地从多个数据表中识别和定位出查询所需的关键数据表,针对问题：\n {query}\n,初步筛选了以下候选表用来生成sql以回答该问题：\n {table_names}：\n 这些表的schema信息如下：\n {table_des}.根据问题和表的schema表描述，从候选表定位出你回答问题需要查询的所有数据表，回答返回一个list，字母要大写，格式：['TAB1','TAB2'],如果候选表均与问题无关，返回[]即可。"
    print('chose_tabe_prompt',chose_tabe_prompt)
    answer = response(chose_tabe_prompt)
    # import json
    #
    # # 将列表转换为 JSON 格式的字符串
    # output_str = json.dumps(answer)
    # intermediate_result = answer[0]
    list_str = str(answer).strip("[]").replace("'", '"')
    extracted_data = f'[{list_str}]'
    print('api_result:',extracted_data)
    return extracted_data


import mysql.connector
from mysql.connector import Error


def get_create_table_statements(connection, table_names):
    """
    查询指定表名的建表语句。

    :param connection: MySQL 数据库连接对象
    :param table_names: 包含表名的列表
    :return: 字典，键为表名，值为对应的建表语句
    """
    create_statements = {}
    try:
        cursor = connection.cursor()
        for table_name in table_names:
            try:
                # 使用 SHOW CREATE TABLE 查询建表语句
                cursor.execute(f"SHOW CREATE TABLE {table_name}")
                result = cursor.fetchone()
                if result:
                    create_statements[table_name] = result[1]  # 建表语句通常是第二列
                else:
                    create_statements[table_name] = "建表语句未找到"
            except Error as e:
                create_statements[table_name] = f"查询失败: {e}"
    except Error as e:
        print(f"数据库错误: {e}")
    finally:
        cursor.close()
    return create_statements

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

    return response(SQL_Prompt)


def create_agent(openai_api_key, db_connection, user_input):
    # 获取数据库模式
    db_schema = get_db_schema(db_connection)
    db_name = get_table_names(db_connection)
    choose_table_xinghuo = choose_table_api(user_input, db_name, db_schema)
    tab_schema = get_create_table_statements(choose_table_xinghuo)

    # 初始化 SQL 生成工具
    sql_generator = SQLGenerator(openai_api_key, db_schema)

    # 定义执行 SQL 的工具
    def execute_sql_function(user_input):
        sql = sql_generator.generate_sql(user_input)
        columns, result = execute_query(db_connection, sql)
        if columns and result:
            # 将结果格式化为表格字符串
            from tabulate import tabulate
            table = tabulate(result, headers=columns, tablefmt="grid")
            return f"查询结果:\n{table}"
        return "查询失败或无结果。"

    execute_sql_tool_instance = Tool(
        name="execute_sql",
        func=execute_sql_function,
        description="执行给定的SQL查询，并返回结果。用户提供的查询将被转化为SQL语句。",
        args_schema=ExecuteSQLArgs
    )

    # 创建 OpenAI LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)

    tools = [execute_sql_tool_instance]

    prompt = '''从问题中提取时间段，记录开始和结束的的时间。 问题：{user_input}。
     已知现在{now_date}。时间结果返回一个json，格式为：{{ 问题中包含时间相关的描述，精确到日：yyyy-MM-dd～yyyy-MM-dd}}。
     注意：当问题中不明确时间时，如最近、目前、当前时，返回{{}}.
     示例：昨天下午4点怎么样？时间结果：{{昨天：2024-03-01～2024-03-01}}
     示例：上个月初地铁运行情况怎样？时间结果：{{上个月初：2024-03-01～2024-03-10}}。
    问题：今年的销售额较去年如何？ 时间结果：{{"今年":2024-01-01～2024-12-31,"去年":2023-01-01～2023年-12-31}}.对于涉及到推算的，
    除json外，另起一行给出准确详细的推算过程，除此不要输出其他。示例：上上周二的情况怎么样？已知当前是2024年5月6号星期一。
    时间结果：{{"上上周二":2024-04-23～2024-04-23}}\n 推算过程：- 首先，确定当前日期为2024年5月6日，星期一。
    - 要计算上上周二，即需要回溯两周前的星期二。
    - 2024年5月6日是星期一，回溯一周是4月29日（星期一），再回溯一周即到达4月22日（星期一）。
    - 由于我们需要找到星期二，所以再往后推移一天，即可得到上上周二，也就是4月23日。'''

    from datetime import date

    # 获取今日日期
    today = date.today()

    # 将日期格式化为YYYY-MM-DD
    formatted_today = today.strftime('%Y-%m-%d')

    prompt_with_time = prompt.format(user_input=user_input, now_date=formatted_today)
    time_info = llm.invoke(prompt_with_time)

    remarks = RAG_cot_prompt(user_input)
    # 定义 PromptTemplate
    prompt_template = generate_sql_with_time(user_input, tab_schema, choose_table_xinghuo, time_info, remarks)


    # 创建代理
    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template,
        # verbose=True
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"input": user_input,"agent_scratchpad":[],"tools":tools})

    return response["output"]
