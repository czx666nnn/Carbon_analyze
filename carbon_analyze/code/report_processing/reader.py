import os
import re
import markdown
import asyncio
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
import cfg
import json
import tiktoken
from qianfan.resources.tools import tokenizer
import json
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.llms import QianfanLLMEndpoint,Tongyi
from langchain_core.messages import HumanMessage
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
os.environ["QIANFAN_AK"] = ""
os.environ["QIANFAN_SK"] = ""
from qianfan.resources import Reranker
model2 = Reranker(model="")
model = Tongyi(model="qwen-max",api_key="",)
import logging
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

def Key_sentences_with_llm(query):
    intent_prompt = f'''
    从以下文本中提取关键提问及关键句：{query};(关键句应该是能够概括主要内容的句子，选择反映核心主题或概念的语句，避免提取常见无关词汇（如：连词、副词等）。关键句数量不超过5个。只返回对应语句，不要有其他任何说明。)
    \n注意：1、不要生成任何说明！！！！！\n2、每个句子之间使用"/"隔开\n3、只提取完整的句子，不要有""
    '''
    llm = QianfanLLMEndpoint(streaming=False,temperature=0.1)
    intent_result = llm.invoke(intent_prompt)
    result = intent_result.split('/')
    return result
def Multiple_expressions_llm(query):

    intent_prompt = f'''
    根据下列问题，生成多个不同的表述或回答方式，确保每个回答都涵盖问题的核心内容，但在表达方式上有所不同。请注意，确保每种回答在语气、结构或用词上有明显区别，但始终保持信息的准确性和一致性。以下是问题：{query}
    \n注意：只返回多样表达后的结果，不要有其他任何说明。每个结果之间使用"/"隔开
    '''
    llm = QianfanLLMEndpoint(streaming=False, temperature=0.1)
    intent_result = llm.invoke(intent_prompt)
    result = intent_result.split('/')
    return result
def identify_intent_with_llm(query):
    intent_prompt = f'''
    你是一个双碳领域的专家，请判断以下提问是否与国家碳排放政策有关。如果相关，只输出“需要查询”；如果不相关，只输出“不需要查询”。不能输出其他任何字符与多余的解释。
    问题：{query}
    '''
    llm = QianfanLLMEndpoint(streaming=False,temperature=0.1)
    intent_result = llm.invoke(intent_prompt)
    intent_result = intent_result.strip()
    return intent_result

def H_llm(generated_answer,report_content,SQL_result=''):
    intent_prompt = f'''
        你是一个专业的文档核查助手，负责验证企业相关回答的准确性和可靠性。请确保所有回答内容均可追溯至原始来源，避免包含任何未经验证的推测。
        以下是需要核查的内容：
        报告内容：{report_content}
        SQL执行结果：{SQL_result}（如无内容表示未调用企业数据库）
        系统回答：{generated_answer}
        \n核查任务：
        1.对比 报告内容 和 系统回答，标记任何不符或超出的信息为“幻觉”。
        2.如果发现幻觉，提供修正后的回答，确保其与报告内容完全一致，且语义准确。
        输出格式（如果没有幻觉）：
        直接返回系统回答。不需要有任何多余的描述。
        \n如果回答中产生幻觉，输出格式（如果有幻觉）：
        【幻觉标记】：列出不符合报告内容的部分，并说明原因(必须给出幻觉标记）。
        【修正后的回答】：提供准确且符合事实的回答。
        【原因说明】：简要说明幻觉部分的原因及修正依据。
        \n注意事项：
        请严格依据报告内容和SQL结果完成核查。
        修正后的回答必须清晰、准确，无任何推测或不符之处。
    '''
    llm = QianfanLLMEndpoint(streaming=False,temperature=0.1)
    intent_result = llm.invoke(intent_prompt)
    intent_result = intent_result.strip()

    return intent_result


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
# #text2sql
# import os
# from db import connect_to_database
# from sqlagant import create_agent
# from langchain_core.pydantic_v1 import BaseModel
#
# # 初始化数据库连接
# db_connection = connect_to_database(
#     host='0.0.0.0',
#     database='db',
#     user='user',
#     password='password'
# )
#
# if not db_connection:
#     raise Exception("无法连接到数据库")
#
# # 创建Agent
# # openai_api_key = os.getenv('OPENAI_API_KEY')
# # if not openai_api_key:
# #     raise Exception("请设置 OPENAI_API_KEY 环境变量")
#
# openai_api_key="请输入自己的key"
#
# def query_database(user_query: str) -> str:
#     """
#     根据用户的查询执行数据库和模型操作，返回结果
#     """
#     print(f"用户查询：{user_query}")
#     try:
#         result = create_agent(openai_api_key, db_connection, user_query)
#         print(f"查询结果：{result}")
#         return result
#     except Exception as e:
#         print(f"发生错误：{e}")
#         return f"错误：{e}"

 # response = query_database(example_query)


TOP_K = 5
PROMPTS = {
    'general':
        """你被赋予了碳排放及气候科学家的角色，负责分析一家公司的可持续发展报告。根据以下从可持续发展报告中提取的部分内容，回答给定的问题，并给出可持续发展报告中的内容依据。
如果你不知道答案，就直接说你不知道。不要试图编造答案。
请按照以下键值将答案格式化为 JSON 格式： COMPANY_NAME, COMPANY_SECTOR, and COMPANY_LOCATION.

QUESTIONS: 
1. 该报告中涉及的企业名称是什么？
2. 企业所属的行业类别是什么？
3. 企业的地理位置在哪里？

=========
{context}
=========
你的 FINAL_ANSWER 应为 JSON 格式（确保没有格式错误）：
""",
    'GHG_qa_source': """作为一名具备气候科学专业知识的专家，你正在评估一家企业的可持续发展报告，并重点关注其与国家双碳政策相关的合规情况及碳排放管理，并给出可持续发展报告中的内容依据。以下是提供给你的背景信息：

{basic_info}

根据上述信息和以下从可持续发展报告中提取的部分内容（这些内容的开头和结尾可能是不完整的），请回答所提出的问题，并确保引用相关部分（“SOURCES”）。请将你的回答格式化为 JSON 格式，包含以下两个键值：
1. ANSWER：此处应包含不带来源引用的答案字符串。
2. SOURCES：此处应为答案中引用的来源编号列表。

QUESTION: {question}
=========
{summaries}
=========

请在回答中遵循以下指导原则：
1. 你的回答必须准确、全面，并基于报告中的具体摘录以验证其真实性，并给出可持续发展报告中的内容依据。
2. 如果不确定答案，简单地承认知识的不足，而不是编造答案。
3. 将你的 ANSWER 保持在 {answer_length} 字以内。
4. 对报告中披露的信息保持怀疑态度，因为可能存在绿washing（夸大公司的环境责任）。始终以批判的语气回答。
5. “空洞言论”是指那些成本低廉、可能不反映公司真实意图或未来行动的陈述。对报告中发现的所有空洞言论保持批判。
6. 始终承认提供的信息是基于公司报告的观点。
7. 仔细审查报告是否基于可量化、具体的数据，还是模糊、无法验证的陈述，并传达你的发现。

{guidelines}

你的 FINAL_ANSWER 应为 JSON 格式（确保没有格式错误）：
""",
    'user_qa_source': """作为一名具备气候科学专业知识的专家，你正在评估一家企业的可持续发展报告，并重点关注其与国家双碳政策相关的合规情况及碳排放管理，并给出可持续发展报告中的内容依据。以下是提供给你的背景信息：

{basic_info}

根据上述信息和以下从可持续发展报告中提取的部分内容（这些内容的开头和结尾可能是不完整的），请回答所提出的问题，并确保引用相关部分（“SOURCES”）。
请将你的回答格式化为 JSON 格式，包含以下两个键值：
1. ANSWER：此处应包含不带来源引用的答案字符串。
2. SOURCES：此处应为答案中引用的来源编号列表。

QUESTION: {question}
=========
{summaries}
=========

请在回答中遵循以下指导原则：
1. 你的回答必须准确、全面，并基于报告中的具体摘录以验证其真实性，并给出可持续发展报告中的内容依据。
2. 如果某些信息不清楚或不可用，承认知识的不足，而不是编造答案。
3. 仅根据提供的摘录回答问题。如果可用的信息不足，明确说明无法基于给定报告回答该问题。
4. 将你的 ANSWER 保持在 {answer_length} 字以内。
5. 对报告中披露的信息保持怀疑态度，因为可能存在绿washing（夸大公司的环境责任）。始终以批判的语气回答。
6. “空洞言论”是指那些成本低廉、可能不反映公司真实意图或未来行动的陈述。对报告中发现的所有空洞言论保持批判。
7. 始终承认提供的信息是基于公司报告的观点。
8. 仔细审查报告是否基于可量化、具体的数据，还是模糊、无法验证的陈述，并传达你的发现。

你的 FINAL_ANSWER 应为 JSON 格式（确保没有格式错误）：
""",
    'GHG_summary_source': """你的任务是分析和总结公司可持续发展报告中与以下 <CRITICAL_ELEMENT> 相关的任何披露：

<CRITICAL_ELEMENT>: {question}

以下是关于正在评估的公司的基本信息：

{basic_info}

除了上述信息外，以下从可持续发展报告中提取的部分也已提供给你进行审查：

{summaries}

你的任务是根据这些摘录中的信息总结公司对上述 <CRITICAL_ELEMENT> 的披露。请在总结中遵循以下指导原则：

1. 如果报告中披露了 <CRITICAL_ELEMENT>，请尝试通过直接摘录的方式进行总结，并引用提供的摘录中的来源以确认其可信度。
2. 如果报告中未涉及 <CRITICAL_ELEMENT>，请明确说明这一点，避免尝试推测或编造信息。
3. 将你的 SUMMARY 保持在 {answer_length} 字以内。
4. 对报告中披露的信息保持怀疑态度，因为可能存在绿washing（夸大公司的环境责任）。始终以批判的语气回答。
5. “空洞言论”是指那些成本低廉、可能不反映公司真实意图或未来行动的陈述。对报告中发现的所有空洞言论保持批判。
6. 始终承认提供的信息是基于公司报告的观点。
7. 仔细审查报告是否基于可量化、具体的数据，还是模糊、无法验证的陈述，并传达你的发现，并给出可持续发展报告中的内容依据。
{guidelines}
你的总结应以 JSON 格式呈现，包含以下两个键值：
1. SUMMARY：此处应包含不带来源引用的总结。
2. SOURCES：此处应为总结中引用的来源编号列表。

你的 FINAL_ANSWER 应为 JSON 格式（确保没有格式错误）：
""",
    'GHG_qa': """作为一名具备气候科学专业知识的专家，你正在评估一家公司的可持续发展报告，以下是有关报告的重要信息：

{basic_info}

根据上述信息和以下从可持续发展报告中提取的部分内容（这些内容的开头和结尾可能是不完整的），请回答所提出的问题。 
你的回答应准确、全面，并基于报告中的直接摘录以建立其可信度。
如果你不知道答案，就直接说你不知道。不要试图编造答案。

问题：{question}
=========
{summaries}
=========
""",
    'GHG_assessment': """你的任务是对可持续发展报告中对以下 <CRITICAL_ELEMENT> 的披露质量进行评分：

<CRITICAL_ELEMENT>: {question}

以下是高质量披露所需的必要组成部分的 <REQUIREMENTS>：

<REQUIREMENTS>:
====
{requirements}
====

以下是与 <CRITICAL_ELEMENT> 相关的可持续发展报告摘录：

<DISCLOSURE>:
====
{disclosure}
====

请分析给定的 <DISCLOSURE> 满足上述 <REQUIREMENTS> 的程度。你的分析应具体说明哪些 <REQUIREMENTS> 已满足，哪些未满足。
你的回应应以 JSON 格式呈现，包含以下两个键值：
1. ANALYSIS：一段分析（以字符串格式呈现）。不超过 150 字。
2. SCORE：一个 0 到 100 的整数分数。分数 0 表示大多数 <REQUIREMENTS> 未满足或细节不足。分数 100 表示大多数 <REQUIREMENTS> 已满足，并附有具体细节。

你的 FINAL_ANSWER 应为 JSON 格式（确保没有格式错误）：
""",
    'scoring': """你的任务是对可持续发展报告的披露质量进行评分。你将获得一个 <REPORT SUMMARY>，其中包含 {question_number} 对（DISCLOSURE_REQUIREMENT，DISCLOSURE_CONTENT）。DISCLOSURE_REQUIREMENT 对应于报告应披露的关键信息。DISCLOSURE_CONTENT 总结了报告中对该主题的披露信息。 
对于每对内容，你应分配一个分数，反映披露信息的深度和全面性。分数 1 表示详细和全面的披露。分数 0.5 表示披露的信息缺乏细节。分数 0 表示所请求的信息未披露或披露的信息没有任何细节。
请以 JSON 结构格式化你的回答，包含 'COMMENT'（提供对报告质量的总体评估）和 'SCORES'（一个包含 {question_number} 个分数的列表，分别对应每对问题和答案）。

<REPORT SUMMARY>:
====
{summaries}
====
你的 FINAL_ANSWER 应为 JSON 格式（确保没有格式错误）：
""",
  'to_question': """检查以下陈述，并将其转换为适合 ChatGPT 提示的问题，如果它尚未以问题形式表达。如果该陈述已经是一个问题，请按原样返回。
陈述：{statement}"""
}

QUERIES = {
	'general': ["该报告中涉及的企业名称是什么？", "企业所属的行业类别是什么？", "企业的地理位置在哪里？"
          ],
'GHG_1': "企业在环境报告中如何定义排放核算的边界？是否详细列出了全资、控股和合资公司的排放源？排放核算范围的划定依据是什么（如运营控制、财务控制或权益比例原则），是否能在报告中明确找到相关说明？核查组织架构及运营控制信息时，有无证据表明所有相关活动已包含在核算范围内？",
'GHG_2': "报告中是否列明了直接排放源（Scope 1）的分类和范围，例如固定排放源、移动排放源及其他潜在排放活动？企业是否提供了燃料使用量、行驶里程、制冷剂泄漏等关键数据，并通过标准排放因子计算排放量？相关计算依据在哪里体现？在直接排放的核算和量化过程中，报告是否识别出潜在的不确定性或数据空白点？",
'GHG_3': "报告中对外购能源（Scope 2）的使用情况有无完整统计，包括年度电力、热力和冷力的使用量？企业是否披露了区域电网的碳排放因子及其使用情况？能源消耗的总排放量如何体现？能源结构优化和可再生能源应用的进展是否有明确的指标或数据支撑？是否能在报告中找到相关成果展示？",
'GHG_4': "报告是否涵盖了上下游供应链和产品生命周期的所有排放环节（Scope 3），包括采购、生产、运输和废弃处理等？上下游供应链中有哪些环节被认定为高排放来源？企业在管理供应商或优化产品设计上采取了哪些举措？",
'GHG_5': "环境报告中是否包含员工和访客通勤的调查数据，例如通勤方式分布、通勤距离及频率？交通工具的碳排放因子在核算通勤排放量时如何使用？相关数据是否被完整记录？企业是否提出或实施了促进低碳通勤的具体措施，这些措施在报告中的效果如何体现？",
'GHG_6': "报告是否全面说明了核算方法、排放因子及数据来源？相关内容是否易于查阅和验证？企业的环境报告在与GHG Protocol等国际标准的符合性方面有无评估？是否列出了对标依据和改进方向？参考文献或数据附录的质量如何？是否足以确保报告内容的可追溯性和专业性？",
'GHG_7': "报告是否清晰记录了废弃物的分类及处理方式（如焚烧、填埋、回收），以及相关的排放量核算？企业的废弃物减量化或循环利用实践是否有量化的指标或案例支撑？在报告中是否展示了这些努力的成果？在供水与污水处理环节，企业是否披露了能耗与碳排放数据？是否明确了未来的节水和减排目标？",
'GHG_8': "环境报告的时间跨度和数据更新频率是否清晰标注？是否有跨年度数据对比的内容？多版本报告之间的数据差异如何解释？企业是否提供了版本对比的分析方法或结果？在历史数据追踪和版本管理中，有哪些体现优化或改进的具体案例？",
'GHG_9': "该企业是如何跟踪长期排放量？该企业是在跟踪长期排放量时应选择什么样的基准年？该企业是应制定什么政策以管理基准年排放量的重算？当企业发生变化影响排放信息时，应如何处理基准年排放量？",
'GHG_10': "该企业是如何确定信息的披露程度？该企业是在选择报告信息的详细程度时应考虑哪些因素？该企业是在保证商业机密的情况下，如何处理温室气体排放数据？",
'GHG_11': "该企业是如何核算温室气体减排量？该企业是如何核算温室气体减排量以满足政府要求或履行自愿减排承诺？该企业是在评估间接减排量时考虑了哪些关键因素？",
'GHG_12': "该企业在根据情况进行预测，设定未来温室气体目标时，是如何保证高级管理层保持对某个问题的监控的？该企业是确定目标类型以及边界？该企业是如何选择目标基准年？",
'GHG_13': "该企业是在进行预测并设定未来温室气体目标时，设定的怎样的目标承诺期长度，如何确定温室气体抵消量或信用额度的使用？如何制定目标重复计算政策？如何确定目标水平以及进行跟踪与报告进度？",
'GHG_14': "该企业是在短期、中期和长期内识别到的与国家双碳政策相关的主要碳排放风险和机遇是什么？这些风险是否明确与某一时间范围相关联？",
}
GHG_ASSESSMENT = {
'GHG_1': """在描述如何确保清单全面服务于决策需要时，重点在于选择适当的排放清单边界，该企业应考虑以下内容：
1. 组织结构：公司在设定组织边界时，选择何种合并温室气体排放量的方法，进而采用选定方法界定这家公司的业务活动和运营，从而对温室气体排放量进行核算和报告。
2. 运营边界：公司如何识别与其运营相关的排放，其如何分为直接与间接排放，及选定的间接排放的核算与报告范围。
3. 业务范畴 ：公司选择排放清单边界时，是否考虑活动性质、地理位置、行业部门、信息用途和用户相关因素。
""",
'GHG_2': """在报告其选定排放清单边界内所有温室气体排放源和活动时，该企业必须披露：
1. 范围一和范围二的总排放量，并与出售、购买、转让或储蓄排放配额等温室气体交易区分开来。在可以取得可靠排放数据时，相关的范围三活动排放。
2. 分别报告每个范围的排放信息。
3. 针对六种温室气体分别报告其排放数据。选定作为基准年的年份，阐明重算基准年排放量的政策，并按照该政策所算出来的一段时间内的排放变化。
4. 阐述引起任何基准年排放量重算的重大变化。
5. 从生物源产生的直接二氧化碳排放数据，在各范围外单独报告。
6. 用于计算和测量排放量的方法学，为所用计算工具提供来源参考或链接。
""",
    'GHG_3': """该企业在确定排放清单边界后，一般采取下列步骤计算温室气体排放量：
1. 识别温室气体排放源 ：
首先，该企业应当识别排放源中的直接排放源，即识别范围一的排放。接下来的步骤是识别由于消耗外购的电力、热力或蒸汽所产生的间接排放源，即识别范围二的排放。之后识别尚未包含于范围一或范围二中的该企业上游和下游活动产生的其他间接排放，以及与外包／合同制造、租赁或特许经营有关的排放，即识别范围三的排放，通过识别范围三的排放，该企业能够根据价值链扩展其排放清单的边界，并识别全部相关的温室气体排放。
2. 选择温室气体排放量计算方法：该企业应当采用适合其报告情况且可行的最精确的计算方法。
3. 收集活动数据和选择排放因子：多数情况下，如果有具体排放源或设施的排放因子，应该优先使用这些因子而非通用的排放因子。
4. 应用计算工具：多数该企业需要采用一种以上的计算工具计算全部温室气体排放源的排放量。 
5. 将温室气体排放数据汇总到该企业一级：该企业把温室气体报告和已有的报告工具与流程进行整合，从而利用各种已经收集并报告给该企业部门或办公室、监管机构或其他利益相关方的数据。用来报告数据的工具与流程必须基于已有的信息和沟通机制（即必须考虑把新数据类别纳入该企业现有数据库的难易程度）。另外取决于该企业总部对各设施报告的详细程度的具体要求。
""",
'GHG_4': """在披露没有计入的排放源及其活动时，该企业应该披露排除在外的具体排放源、设施或业务，并说明被认为无关紧要或不适用于计入的理由，同时应考虑其余公司运营的相关性，是否对该企业的整体温室气体排放有显著影响。
""",
'GHG_5': """该企业可以按照业务单元／设施、国家、排放源类型和活动类型进一步细分的排放数据，相关的绩效比率指标，温室气体管理减排计划或战略，造成排放量变化，但没有引起基准年排放重算的有关原因，关于温室气体捕获的信息，排放清单包含的设施列表以及购买的或开发的在排放清单边界以外的碳抵消额度的信息，在排放清单边界以内的排放源所产出的，并已作为碳抵消额度出售或转移给第三方的减排量信息等。
""",
'GHG_6': """
1.该企业进行核查温室气体排放前，应当清楚地确定进行核查的目标和核查的方式（如外部或内部）及核查人员，并确定是否是实现这些目标的最佳方式。
2.为了表明对数据或信息的看法，核查人员应当明确所有识别出来的误差或不确定性是否具有实质性。
3.核查人员应当评估温室气体信息收集与报告过程的各个组成部分出现实质性偏差的
风险
4.根据核查要求的不同保证水平，核查人员需要考察多处现场以获取足够、适当的证据
5.核查人员可以在温室气体排放清单准备与报告过程的各个时间点介入。
""",
'GHG_7': """该企业跟踪长期排放量，需要考虑：选择和报告有可供核查的排放数据的基准年，并具体说明选择这一特定年份的理由，公司须制定基准年排放量重算政策，明确规定重算的依据和相关因素，公司有责任确定引起基准年排放量重算的“重要限度”并予以披露。当该企业发生的变化影响该企业报告的温室气体排放信息的一致性和相关性时，须溯及既往重新计算基准年排放量。公司确定如何重新计算基准年排放量的政策后，须一致地执行这项政策。
""",
'GHG_8': """针对于必须要披露的信息，该企业应当确定一个全面的标准，公开报告所需的细节；选报信息的详细程度，需要根据报告目的和目标读者决定。针对于特定温室气体、设施或业务单元的排放数据设计商业机密的情况下，该企业不必公开报告这些数据，但是在保证商业机密的情况下，可以向温室气体排放数据审核人员提供数据。
""",
'GHG_9': """该企业的温室气体排放清单计划包括全部的制度、管理及技术的安排，从而确保收集数据、 编制清单和具体执行的质量控制，该企业需要建立和执行排放清单的质量管理体系，具体考虑的内容包括：
1. 排放清单计划的框架，该企业应当努力确保每个层级的排放清单设计中方法、数据、流程与体系、文件记录的质量。
2. 实施排放清单质量管理体系：成立排放清单质量管理小组，建立质量管理方案，方案应当包括所有组织层级的规程和排放清单编制流程。进行一般性质量检查，检查具体排放源的质量，审查最终排放清单估算数据和报告，建立正式反馈制度，记录保管规程，具体规定出于该企业内部要求，应当记录哪些信息，这些信息如何归档，以及向外部利益相关方报告哪些信息。
3. 实际执行措施：该企业应当在内部的多个层级采取措施，从原始数据采集到最后的该企业对排放清单审批流程。
4. 针对于特定排放源，要依据排放因子和其他参数（例如设备利用率、氧化率和）计算排放量，这些因子和参数应当基于特定的该企业、特定现场或基于直接排放量和其他测量值，质量调查应当评价排放因子和其他参数能否代表该企业的具体特征，并对实测值和默认值之间的差值作出定性解释，然后基于该企业运营特点说明原因。该企业应当建立严格的数据收集规程，保证收集高质量活动水平数据。特定排放类别的排放量估算值与历史数据或其他估算值进行比较，从而确保其处于合理范围之内。该企业应当考虑估算的不确定性分类，是否有针对模型不确定性的解决方法，是否有针对参数不确定性的解决方法。
""",
'GHG_10': """核算温室气体减排量，该企业可以通过比较不同时间相关层级的实际排放量，达到政府的要求或履行自愿减排承诺。需要考虑该企业是否采用合理的方法评估间接减排量，该企业应当采用项目量化方法来确定日后用作抵消额度的项目减排量，比如选择基准情景和排放量，额外性论证，识别并量化相关的次要影响，考虑可逆性，避免重复计算等。该企业需要报告项目减排量。
""",
'GHG_11': """需要考虑的方面：该企业需要高级管理层特别是董事会 / 首席执行官层面的支持和承诺；温室气体排放目标主要可分为两种：绝对目标和强度目标，该企业应当根据自身情况谨慎选择，也可以全部设定，采用强度目标的该企业还应当报告目标所涉及的排放源的绝对排放量；目标边界规定了目标所涵盖的温室气体、运营地区、排放源和活动。目标边界可以与排放清单边界相同，也可以只规定排放清单边界以内的特定排放源子集，该企业确定的目标边界，具体确定包括的温室气体，运营地区，直接与间接排放源，不用业务类型设定的目标；该企业需要选择固定的目标基准年或滚动的目标基准年，实现对照以往的排放量界定目标排放量。
""",
'GHG_12': """需要考虑的方面：该企业根据自身情况确定选择单年承诺期或多年承诺期；该企业应当具体说明是否使用抵消量以及因此实现多少目标减排量；该企业应当制定自己的“目标重复计算政策”。这种政策应当规定如何在减排、与其他目标和计划有关的交易与该企业目标之间进行调节对账，并相应规定哪些重复计算情形是相关的；该企业在确定各目标水平时，需要考虑商业度量之间的关系，对该企业发展的影响，是否会影响其他目标等因素；该企业应定期进行绩效检查，比照预测目标报告信息。
""",
'GHG_13': """在描述该企业识别的短期、中期和长期碳排放风险和机会时，该企业应提供以下信息：
1. 他们认为相关的短期、中期和长期时间范围的描述，考虑到该企业资产或基础设施的使用寿命以及碳排放问题通常在中长期内显现；
2. 在每个时间范围（短期、中期和长期）可能出现的具体碳排放问题的描述，这些问题可能对该企业产生实质性的财务影响；
3. 确定哪些碳排放风险和机会可能对该企业产生实质性财务影响的过程的描述。该企业应考虑按行业和/或地理位置提供风险和机会的描述。
""",
'GHG_14': """在描述该企业管理碳排放风险的过程时，该企业应描述其管理碳排放风险的过程，包括如何做出减轻、转移、接受或控制这些风险的决定。此外，该企业应描述其优先排序碳排放风险的过程，包括如何在该企业内部进行重要性判断。
""",
}
GHG_GUIDELINES = {
'GHG_1': """8. 请在选择边界时反映该该企业业务关系的本质和经济状况，而不只是它的法律形式。""",
'GHG_2': """8. 请避免讨论不在选定排放清单边界内的排放源和活动，同时避免讨论非温室气体排放，如常规污染物，除非它们与温室气体排放直接相关。""",
'GHG_3': """8. 请注意遵循标准化指导，避免推广或使用非官方认可的计算方法。""",
'GHG_4': """8. 请不要讨论与排放源计入无关的信息，注意避免详细讨论与当前排放源计入无关的未来计划。""",
'GHG_5': """8. 注意避免对信息过度简化，以所能取得的最优数据为基础。""",
'GHG_6': """8. 确保讨论焦点集中在温室气体排放量的核查上，避免讨论与核查无关的话题，同时避免出现实质性差异。""",
'GHG_7': """8. 可以采用指导方法，也可以自行制定方法，请注意避免选择不一致的方法。""",
'GHG_8': """8. 请该企业和集团该企业内其他设施、业务单元注意避免将信息重复计入或同一计入不同范围。""",
'GHG_9': """8. 请注意避免深入讨论特定的技术细节，除非这些细节直接影响到排放清单的质量管理。请注意避免讨论不明确或不一致的数据收集边界和范围，这可能导致数据质量问题的忽视。""",
'GHG_10': """8. 请注意避免过分关注技术解决方案，避免使用不适当的或过时的排放因子。""",
'GHG_11': """8. 请注意避免讨论与气候变化相关的政治立场或争议，专注于该企业的实际行动和目标设定。""",
'GHG_12': """8. 请注意避免深入讨论技术细节，保持讨论的高层次和战略性。""",
'GHG_13': """8. 请避免讨论公司范围的风险管理系统或这些碳排放风险和机会的识别和管理方式。""",
'GHG_14': """8. 请专注于管理碳排放风险的具体行动和策略，排除风险识别或评估过程。""",
}
SYSTEM_PROMPT = "你是一个具有气候科学专业知识的专家，分析公司的可持续性报告。"


def remove_brackets(string):
    return re.sub(r'\([^)]*\)', '', string).strip()


def _docs_to_string(docs, num_docs=TOP_K, with_source=True):
    output = ""
    docs = docs[:num_docs]
    for doc in docs:
        output += "Content: {}\n".format(doc.page_content)
        if with_source:
            output += "Source: {}\n".format(doc.metadata['source'])
        output += "\n---\n"
    return output

def _find_answer0(string,report_content, name="ANSWER"):
    for l in string.split('\n'):
        if name in l:
            start = l.find(":") + 3
            end = len(l) - 1
            print("ANSWER:",l[start:end])
            a = H_llm(l[start:end],report_content)
            return a
    return string
def _find_answer(string, name="ANSWER"):
    for l in string.split('\n'):
        if name in l:
            start = l.find(":") + 3
            end = len(l) - 1
            return l[start:end]
    return string

def _find_sources(string):
    pattern = r'\d+'
    numbers = [int(n) for n in re.findall(pattern, string)]
    return numbers

def _find_float_numbers(string):
    pattern = r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
    float_numbers = [float(n) for n in re.findall(pattern, string)]
    return float_numbers

def _find_score(string):
    for l in string.split('\n'):
        if "SCORE" in l:
            d = re.search(r'[-+]?\d*\.?\d+', l)
            break
    return d[0]




class Reader:
    def __init__(self, llm_name='', answer_key_name='ANSWER', max_token=1024, q_name='Q', a_name='A',
                 queries= QUERIES, qa_prompt='GHG_qa_source', guidelines = GHG_GUIDELINES,
                 assessments = GHG_ASSESSMENT,
                 answer_length='300',
                 root_path='./',
                 gitee_key='',
                 user_name='defualt', language='zh'):
        self.user_name = user_name  # user name
        self.language = language
        self.root_path = root_path
        self.max_token = max_token
        self.llm_name = llm_name
        #
        # self.tiktoken_encoder = tiktoken.encoding_for_model(self.llm_name)


        self.cur_api = 0
        self.file_format = 'md'  # or 'txt'
        self.prompts = PROMPTS
        self.assessments = assessments
        self.queries = queries
        self.guidelines = guidelines
        self.qa_prompt = qa_prompt
        self.answer_key_name = answer_key_name
        self.q_name = q_name
        self.a_name = a_name
        self.answer_length = answer_length
        self.basic_info_answers = []
        self.answers = []
        self.assessment_results = []
        self.user_questions = []
        self.user_answers = []
        # self.save_image = False
        # if self.save_image:
        #    self.gitee_key = self.config.get('Gitee', 'api')
        # else:
        #    self.gitee_key = ''

    @staticmethod
    def retrieve_all_documents():
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

    async def qa_with_chat(self, report_list):
        htmls = []
        for report_index, report in enumerate(report_list):
            basic_info_prompt = PromptTemplate(template=self.prompts['general'], input_variables=["context"])

            message = basic_info_prompt.format(
                context=_docs_to_string(report.section_text_dict['general'], with_source=False))
            llm = QianfanChatEndpoint(temperature=0.5, max_tokens=512)
            output_text = llm.invoke(message)

            output_text2 = output_text.content
            basic_info_dict = output_text2.replace('"', "'")
            pattern = r"'(?P<key>.*?)\s*:\s*'(?P<value>.*?)'"


            matches = re.findall(pattern, basic_info_dict)

            company_name = matches[0][1] if matches else None
            company_sector = matches[1][1] if matches else None
            company_location = matches[2][1] if matches else None

            basic_info_string = """Company name: {name}\nCompany sector: {sector}\nCompany Location: {location}"""

            basic_info_string = basic_info_string.format(name=company_name, sector=company_sector, location=company_location)

            print('basic_info_string::',basic_info_string)

            self.basic_info_answers.append(basic_info_string)

            GHG_questions = {k: v for k, v in self.queries.items() if 'GHG' in k}

            GHG_prompt = PromptTemplate(template=self.prompts[self.qa_prompt],
                                         input_variables=["basic_info", "summaries", "question", "guidelines",
                                                          "answer_length"])
            answers = {}
            messages = []
            keys = []
            for k, q in GHG_questions.items():
                intent_result = identify_intent_with_llm(q)
                if intent_result == "需要查询":
                    prompts = '''你被赋予了碳排放及气候科学家的角色，负责分析一家公司的可持续发展报告。。你的工作是帮助企业人员更好的理解问题，不要回答问题。
                    简要分析，字数控制在100字以内，尽量囊括足够多的背景信息。问题：{query},示例：{Example}'''

                    print(q)
                    Example = RAG_cot_prompt(q)
                    prompt = prompts.format(query=q, Example=Example)
                    outputs = model.invoke(prompt)
                    outputs = f'{q}+{outputs}'
                    print(q)

                    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],
                                                           search_kwargs={"k": 3})
                    ensemble_results = ensemble_retriever.get_relevant_documents(q)
                    # ensemble_results = retriever.invoke(q)
                    q2 = change_llm(q)


                    Multiple_expressions = Multiple_expressions_llm(q2)
                    Multiple_expressions_list = []
                    for item in Multiple_expressions:
                        print(item)
                        ##########################
                        expressions_results = ensemble_retriever.get_relevant_documents(item)
                        expressions_contents = [doc.page_content for doc in expressions_results]
                        Multiple_expressions_list.extend(expressions_contents)
                    a = model2.do(q, Multiple_expressions_list)
                    b = convert_to_documents(a)
                    c = [doc.page_content for doc in b]

                    page_contents = [doc.page_content for doc in ensemble_results]
                    context = "\n".join([doc.page_content for doc in ensemble_results])
                    inputs = {"context": context, "query": q}

                    data = Key_sentences_with_llm(outputs)
                    combined_list = []
                    for item in data:
                        print(item)
                        ##########################
                        key_results = ensemble_retriever.get_relevant_documents(item)
                        key_contents = [doc.page_content for doc in key_results]
                        combined_list.extend(key_contents)
                    d = model2.do(q, combined_list)
                    e = convert_to_documents(d)
                    f = [doc.page_content for doc in e]

                    from collections import OrderedDict
                    combined_list = list(OrderedDict.fromkeys(f + page_contents + c))
                    c = model2.do(q, combined_list)

                    d = convert_to_documents(c)
                    basic_info_string3 = "\n".join([doc.page_content for doc in d[:10]])
                    ######################################
                else:
                    print("问题不需要查询向量数据库。")

                    basic_info_string3 = "无"
                current_prompt = GHG_prompt.format(basic_info=f'{basic_info_string}；温室气体核算体系(企业核算与报告标准)：{basic_info_string3}',
                                                    summaries=_docs_to_string(report.section_text_dict[k]),
                                                    question=q, guidelines=self.guidelines[k],
                                                    answer_length=self.answer_length)
                report_content = _docs_to_string(report.section_text_dict[k])
                # else:
                    # import os
                    # from db import connect_to_database
                    # from sqlagant import create_agent
                    # from langchain_core.pydantic_v1 import BaseModel
                    #
                    # # 初始化数据库连接
                    # db_connection = connect_to_database(
                    #     host='0.0.0.0',
                    #     database='db',
                    #     user='user',
                    #     password='password'
                    # )
                    #
                    # if not db_connection:
                    #     raise Exception("无法连接到数据库")
                    #
                    # # 创建Agent
                    # # openai_api_key = os.getenv('OPENAI_API_KEY')
                    # # if not openai_api_key:
                    # #     raise Exception("请设置 OPENAI_API_KEY 环境变量")
                    #
                    # openai_api_key="请输入自己的key"
                    #
                    # def query_database(user_query: str) -> str:
                    #     """
                    #     根据用户的查询执行数据库和模型操作，返回结果
                    #     """
                    #     print(f"用户查询：{user_query}")
                    #     try:
                    #         result = create_agent(openai_api_key, db_connection, user_query)
                    #         print(f"查询结果：{result}")
                    #         return result
                    #     except Exception as e:
                    #         print(f"发生错误：{e}")
                    #         return f"错误：{e}"

                    # response = query_database(example_query)

                # message = current_prompt

                # token_cnt = tokenizer.Tokenizer().count_tokens(text=message,mode='remote',model="ernie-bot-4")
                # #print('token_cnt', token_cnt)
                # while token_cnt > 3500 and num_docs > 10:
                #     current_prompt = GHG_prompt.format(basic_info=basic_info_string,
                #                                         summaries=_docs_to_string(report.section_text_dict[k],
                #                                                                   num_docs=num_docs), question=q,
                #                                         guidelines=self.guidelines[k], answer_length=self.answer_length)
                #     #print("current_prompt:",current_prompt)

                # if "turbo" in self.llm_name:
                #     message = [
                #         SystemMessage(content=SYSTEM_PROMPT),
                #         HumanMessage(content=current_prompt)
                #     ]
                # else:
                #     message = current_prompt
                # #print("current_prompt",current_prompt)

                message = current_prompt
                #print("message:::",message)

                keys.append(k)
                messages.append(message)
            import time
            time.sleep(0.3)  # 暂停5秒
            llm = QianfanLLMEndpoint(streaming=True)
            # #print("messages:",messages)
            outputs = llm.generate(messages)
            print("outputs:", outputs)
            # output_texts = {k: g[0].text for k, g in zip(keys, outputs.generations)}
            # 初始化一个空字典
            output_texts = {}

            # 遍历 keys 和 outputs.generations 的内容
            for k, g in zip(keys, outputs.generations):
                # 假设 g 是一个列表，取其第一个元素，并获取该元素的 text 属性
                first_generation_text = g[0].text
                # 将 key 和提取的 text 添加到字典中
                print(first_generation_text)

                output_texts[k] = first_generation_text
            for k, text in output_texts.items():
                try:
                    answers[k] = json.loads(text)
                    if 'SOURCES' not in answers[k].keys() or self.answer_key_name not in answers[k].keys():
                        raise ValueError("Key name(s) not defined!")
                except ValueError as e:
                    answers[k] = {self.answer_key_name: _find_answer0(text,report_content, name=self.answer_key_name),
                                  'SOURCES': _find_sources(text)}
                page_source = []
                for s in answers[k]['SOURCES']:
                    try:
                        page_source.append(report.page_idx[s])
                    except Exception as e:
                        pass
                answers[k]['PAGE'] = list(set(page_source))
                answers[k][self.answer_key_name] = remove_brackets(answers[k][self.answer_key_name])
                #print(answers[k])
            self.answers.append(answers)

            questionnaire_governance = ""
            questionnaire_strategy = ""
            questionnaire_risk = ""
            questionnaire_metrics = ""
            questionnaire_metrics2 = ""
            for idx, (k, q) in enumerate(GHG_questions.items()):

                if 2 > idx >= 0:
                    if idx == 0:
                        questionnaire_governance += "\n\n"
                    questionnaire_governance += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_governance += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                                  answers[k][self.answer_key_name])
                    questionnaire_governance += "\n"
                elif 5 > idx >= 2:
                    if idx == 2:
                        questionnaire_strategy += "\n\n"
                    questionnaire_strategy += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_strategy += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                                answers[k][self.answer_key_name])
                    questionnaire_strategy += "\n"
                elif 8 > idx >= 5:
                    if idx == 5:
                        questionnaire_risk += "\n\n"
                    questionnaire_risk += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_risk += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                            answers[k][self.answer_key_name])
                    questionnaire_risk += "\n"
                elif 12 > idx >= 8:
                    if idx == 8:
                        questionnaire_metrics += "\n\n"
                    questionnaire_metrics += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_metrics += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                               answers[k][self.answer_key_name])
                    questionnaire_metrics += "\n"
                elif 14 > idx >= 12:
                    if idx == 12:
                        questionnaire_metrics2 += "\n\n"
                    questionnaire_metrics2 += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_metrics2 += self.a_name + "{}: {}\n\n".format(int(idx + 1),
                                                                               answers[k][self.answer_key_name])
                    questionnaire_metrics2 += "\n"
            questionnaire = questionnaire_governance + questionnaire_strategy + questionnaire_risk + questionnaire_metrics + questionnaire_metrics2

            htmls.append(markdown.markdown(questionnaire))
        return htmls

    async def analyze_with_chat(self, report_list):
        htmls = []
        for report_index, report in enumerate(report_list):
            # import pdb
            # pdb.set_trace()
            GHG_assessment_prompt = PromptTemplate(template=self.prompts['GHG_assessment'],
                                                    input_variables=["question", "requirements", "disclosure"])
            GHG_questions = {k: v for k, v in self.queries.items() if 'GHG' in k}
            # #print('GHG_questions:::',GHG_questions)

            assessments = {}
            messages = []
            keys = []
            for idx, k in enumerate(self.assessments.keys()):
                num_docs = 15
                current_prompt = GHG_assessment_prompt.format(question=self.queries[k],
                                                               requirements=self.assessments[k],
                                                               disclosure=_docs_to_string(
                                                                   report.section_text_dict[k], with_source=False))

                token_cnt = tokenizer.Tokenizer().count_tokens(text=current_prompt, mode='remote',
                                                                    model="")

                #     # while len(self.tiktoken_encoder.encode(current_prompt)) > 3200 and num_docs > 10:
                while token_cnt > 3200 and num_docs > 10:
                    num_docs -= 1
                    current_prompt = GHG_assessment_prompt.format(question=self.queries[k],
                                                                   requirements=self.assessments[k],
                                                                   disclosure=_docs_to_string(
                                                                       report.section_text_dict[k],
                                                                       num_docs=num_docs,
                                                                       with_source=False))

                message = current_prompt
                print('message:::',message)
                keys.append(k)
                messages.append(message)
            import time
            time.sleep(0.3)
            llm = QianfanLLMEndpoint(streaming=True)
            outputs = await llm.agenerate(messages)
            output_texts = {k: g[0].text for k, g in zip(keys, outputs.generations)}

            for k, text in output_texts.items():
                try:
                    assessments[k] = json.loads(text)
                    if 'SCORE' not in assessments[k].keys() or 'ANALYSIS' not in assessments[k].keys():
                        raise ValueError("Key name(s) not defined!")
                except ValueError as e:
                    assessments[k] = {'ANALYSIS': _find_answer(text, name='ANALYSIS'),
                                      'SCORE': _find_score(text)}
                analysis_text = remove_brackets(assessments[k]['ANALYSIS'])
                if "<CRITICAL_ELEMENT>" in analysis_text:
                    analysis_text = analysis_text.replace("<CRITICAL_ELEMENT>", "GHG recommendation point")
                if "<DISCLOSURE>" in analysis_text:
                    analysis_text = analysis_text.replace("<DISCLOSURE>", "report's disclosure")
                if "<REQUIREMENTS>" in analysis_text:
                    analysis_text = analysis_text.replace("<REQUIREMENTS>", "GHG guidelines")
                assessments[k]['ANALYSIS'] = analysis_text
            self.assessment_results.append(assessments)

            questionnaire_governance = ""
            questionnaire_strategy = ""
            questionnaire_risk = ""
            questionnaire_metrics = ""
            questionnaire_metrics2 = ""
            for idx, (k, q) in enumerate(GHG_questions.items()):

                if 2 > idx >= 0:
                    if idx == 0:
                        questionnaire_governance += ":\n\n"
                    questionnaire_governance += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_governance += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_governance += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_governance += "\n"
                elif 5 > idx >= 2:
                    if idx == 2:
                        questionnaire_strategy += ":\n\n"
                    questionnaire_strategy += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_strategy += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_strategy += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_strategy += "\n"
                elif 8 > idx >= 5:
                    if idx == 5:
                        questionnaire_risk += ":\n\n"
                    questionnaire_risk += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_risk += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_risk += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_risk += "\n"
                elif 12 > idx >= 8:
                    if idx == 8:
                        questionnaire_metrics += ":\n\n"
                    questionnaire_metrics += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_metrics += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_metrics += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_metrics += "\n"
                elif 15 > idx >= 12:
                    if idx == 8:
                        questionnaire_metrics2 += ":\n\n"
                    questionnaire_metrics2 += self.q_name + "{}: {}\n\n".format(int(idx + 1), q)
                    questionnaire_metrics2 += "Analysis{}: {}\n\n".format(int(idx + 1), assessments[k]['ANALYSIS'])
                    questionnaire_metrics2 += "Score{}: {}\n\n".format(int(idx + 1), assessments[k]['SCORE'])
                    questionnaire_metrics2 += "\n"
            questionnaire = questionnaire_governance + questionnaire_strategy + questionnaire_risk + questionnaire_metrics + questionnaire_metrics2
            all_scores = [float(s['SCORE']) for s in assessments.values()]

            htmls.append(markdown.markdown(questionnaire + '\n\n' + "Average score: {}".format(sum(all_scores) / 14)))
        return htmls