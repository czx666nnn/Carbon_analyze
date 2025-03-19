
import logging
import os

from flask import Flask, request, jsonify, render_template
import qianfan
import requests
from langchain_community.embeddings import QianfanEmbeddingsEndpoint,XinferenceEmbeddings
from langchain_community.llms import QianfanLLMEndpoint,Tongyi
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_transformers import (
    LongContextReorder,
)
reordering = LongContextReorder()
#########################
# from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
# from sparkai.core.messages import ChatMessage
# import json
#
# #星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
# SPARKAI_URL = ''
# #星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
# SPARKAI_APP_ID = ''
# SPARKAI_API_SECRET = ''
# SPARKAI_API_KEY = ''
# #星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
# SPARKAI_DOMAIN = ''
# import re
# def response(prompt):
#     spark = ChatSparkLLM(
#         spark_api_url=SPARKAI_URL,
#         spark_app_id=SPARKAI_APP_ID,
#         spark_api_key=SPARKAI_API_KEY,
#         spark_api_secret=SPARKAI_API_SECRET,
#         spark_llm_domain=SPARKAI_DOMAIN,
#         streaming=False
#     )
#     # 创建ChatMessage对象
#     messages = [ChatMessage(
#         role="user",
#         content=prompt
#     )]
#
#     handler = ChunkPrintHandler()
#     a = spark.generate([messages], callbacks=[handler])
#     response_json = a.json()
#     print(response_json)
#     response_json = json.loads(response_json)
#     generations = response_json["generations"]
#     generated_text = generations[0][0]["text"]
#
#     print(generated_text)
#
#     return generated_text
#
#########################
def change_llm(query):
    intent_prompt = f'''
    输入查询： {query}
    请你按照以下输出要求，逐步完成。
    输出要求：
    1.将查询分解成多个简单、清晰的子查询。
    2.识别并消除查询中的歧义，提供最可能的解释。
    3.提炼出查询的核心意图和概念元素。
    4.生成一个高层次的简化表示，保留查询的本质含义。
    '''
    intent_result = model.invoke(intent_prompt)
    return intent_result

def Multiple_expressions_llm(query):
    intent_prompt = f'''
    根据下列问题，生成多个不同的表述或回答方式，确保每个回答都涵盖问题的核心内容，但在表达方式上有所不同。请注意，确保每种回答在语气、结构或用词上有明显区别，但始终保持信息的准确性和一致性。以下是问题：{query}
    \n注意：只返回多样表达后的结果，不要有其他任何说明。每个结果之间使用"/"隔开
    '''
    intent_result = model.invoke(intent_prompt)
    result = intent_result.split('/')
    return result
def convert_to_documents(data):
    documents = []
    for result in data['results']:
        content = result['document']
        metadata = {
            'id': data['id'],
            'index': result['index'],
            'relevance_score': result['relevance_score']
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents

def Key_sentences_with_llm(query):
    intent_prompt = f'''
    从以下文本中提取关键句：{query};(关键句应该是能够概括主要内容的单词或短语，选择反映核心主题或概念的语句，避免提取常见无关词汇（如：连词、副词等）。关键句数量不超过8个。只返回关键句，不要有其他任何说明。关键句之间使用"/"隔开)
    '''
    intent_result = model.invoke(intent_prompt)
    result = intent_result.split('/')
    return result

def RAG_cot_prompt(query):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    Example = retriever.invoke(query)
    return Example

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置环境变量
model = Tongyi(model="qwen-max",api_key="",)
# 初始化 LLM 和嵌入模型
# model = QianfanLLMEndpoint(model=llm_model_name, streaming=True, init_kwargs=kwargs)
qianfan_ak = ''
qianfan_sk = ''
embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)
persist_directory = ""
Cot_directory = ""
qianfan.enable_log()

from qianfan.resources import Reranker
model2 = Reranker(model="")
# reordered_docs = reordering.transform_documents(ensemble_results)

#关键词提取
# import xmnlp
# xmnlp.set_model('')

class ChatBot:
    conversation_history = []
    mode = "qa"  # 默认模式为聊天模式
    @staticmethod
    def retrieve_all_documents():
        """
        静态方法：检索向量数据库中的所有文档并返回。
        """
        persist_directory = ""
        qianfan_ak = ''
        qianfan_sk = ''
        embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        all_docs = vectorstore._collection.get(include=["documents"])
        documents = []
        for doc in all_docs["documents"]:
            document = Document(page_content=doc)
            documents.append(document)
        for i, doc in enumerate(documents):
            logging.info(f"文档 {i + 1}: {doc.page_content}")

        return documents

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    all_documents = retrieve_all_documents()
    bm25_retriever = BM25Retriever.from_documents(all_documents)

    import pandas as pd
    # 输入和输出文件名
    file_name ="数据集.xlsx"

    try:
        df = pd.read_excel(file_name)  # 用 df 替代 data 避免覆盖
    except Exception as e:
        raise ValueError(f"无法加载文件 '{file_name}'，错误信息：{str(e)}")

    # 检查是否有“问题”列
    if "问题" not in df.columns:
        raise ValueError("Excel 文件中没有名为 '问题' 的列，请检查文件格式。")

    # 确保有“输出”和“候选列表”列
    if "输出" not in df.columns:
        df["输出"] = ""
    if "候选列表" not in df.columns:
        df["候选列表"] = ""

    # 遍历每一行的“问题”列
    for index, question in enumerate(df["问题"]):
        if pd.notna(question):  # 检查问题是否为空
            print(f"处理问题：{question}")

            try:
                query_content = question.strip()
                prompts = '''你被赋予了碳排放及气候科学家的角色，负责分析一家公司的可持续发展报告。。你的工作是帮助企业人员更好的理解问题，不要回答问题。
                简要分析，字数控制在100字以内，尽量囊括足够多的背景信息。问题：{query},示例：{Example}'''
                system_prompt = '''
                根据背景信息，筛选出最适合回答问题的规则或内容，保持其完整性并直接引用原文作答，没有适合回答的就不要生成。                                                   
                以下是背景信息：{context}\n
                这是问题：{question}
                '''
                conversation_history.append({'user': query_content})
                print(query_content)
                Example = RAG_cot_prompt(query_content)
                prompt = prompts.format(query=query_content, Example=Example)
                outputs = model.invoke(prompt)
                outputs = f'{query_content}+{outputs}'
                print(query_content)

                ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever] , search_kwargs={"k": 3})
                ensemble_results = ensemble_retriever.get_relevant_documents(query_content)
                # ensemble_results = retriever.invoke(query_content)
                query_content2 = change_llm(query_content)

                Multiple_expressions = Multiple_expressions_llm(query_content2)
                Multiple_expressions_list = []
                for item in Multiple_expressions:
                    print(item)
                    ##########################
                    expressions_results = ensemble_retriever.get_relevant_documents(item)
                    expressions_contents = [doc.page_content for doc in expressions_results]
                    Multiple_expressions_list.extend(expressions_contents)
                a = model2.do(query_content, Multiple_expressions_list)
                b = convert_to_documents(a)
                c = [doc.page_content for doc in b]

                page_contents = [doc.page_content for doc in ensemble_results]
                context = "\n".join([doc.page_content for doc in ensemble_results])
                inputs = {"context": context, "query": query_content}

                data = Key_sentences_with_llm(outputs)
                combined_list = []
                for item in data:
                    print(item)
                    ##########################
                    key_results = ensemble_retriever.get_relevant_documents(item)
                    key_contents = [doc.page_content for doc in key_results]
                    combined_list.extend(key_contents)
                d = model2.do(query_content, combined_list)
                e = convert_to_documents(d)
                f = [doc.page_content for doc in e]

                from collections import OrderedDict
                combined_list = list(OrderedDict.fromkeys(f + page_contents + c))
                c = model2.do(query_content, combined_list)

                d = convert_to_documents(c)
                context = "\n".join([doc.page_content for doc in d[:4]])
                inputs = {"context": context, "query": query_content}
                print(context)

                system_prompt = system_prompt.format(context=context, question=query_content)
                response_data = model.invoke(system_prompt)
                print(response_data)
                a = response_data

                if isinstance(a, str):
                    response_text = a.strip()
                else:
                    response_text = str(a).strip()
                df.at[index, "输出"] = response_text
                df.at[index, "候选列表"] = page_contents
            except Exception as e:
                print(f"处理问题时出错：{question}，错误信息：{str(e)}")
                df.at[index, "输出"] = f"Error: {str(e)}"
                df.at[index, "候选列表"] = "Error"




