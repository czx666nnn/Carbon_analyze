import os
import configparser

# from langchain.llms import openai
# from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
from reader import _find_answer, _find_sources, _docs_to_string, remove_brackets
from qianfan.resources.tools import tokenizer
import cfg
import json
import tiktoken
import os
from qianfan.resources import Reranker
from langchain_community.llms import QianfanLLMEndpoint,Tongyi
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage
from langchain_community.document_transformers import (
    LongContextReorder,
)

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
os.environ["QIANFAN_AK"] = ""
os.environ["QIANFAN_SK"] = ""

model2 = Reranker(model="")
model = Tongyi(model="qwen-max",api_key="",)
import logging

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
#
# config = configparser.ConfigParser()
# config.read('apikey.ini')
# chat_api_list = config.get('OpenAI', 'OPENAI_API_KEYS')[1:-1].replace('\'', '').split(',')
# os.environ["OPENAI_API_KEY"] = chat_api_list[0]





TOP_K = cfg.retriever_top_k
PROMPTS = cfg.prompts
SYSTEM_PROMPT = cfg.system_prompt

import logging

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
class UserQA:
    def __init__(self, llm_name='', answer_key_name='ANSWER', max_token=512,
                 root_path='./',
                 gitee_key='',
                 user_name='defualt', language='en'):
        self.user_name = user_name  # user name
        self.language = language
        self.root_path = root_path
        self.max_token = max_token
        self.llm_name = llm_name
        #
        self.cur_api = 0
        self.prompts = PROMPTS
        self.answer_key_name = answer_key_name
        self.basic_info_answers = []
        self.user_questions = []
        self.user_answers = []
    @staticmethod
    def retrieve_all_documents():

        persist_directory = ""
        qianfan_ak = ''
        qianfan_sk = ''
        embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)
        # 初始化向量数据库
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # 提取所有文档
        all_docs = vectorstore._collection.get(include=["documents"])

        documents = []
        for doc in all_docs["documents"]:
            # 转换为 Document 对象（可以根据需要格式化）
            document = Document(page_content=doc)
            documents.append(document)

        # 打印或返回文档内容
        for i, doc in enumerate(documents):
            logging.info(f"文档 {i + 1}: {doc.page_content}")

        return documents
    def user_qa(self, question, report, basic_info_path, answer_length=60, prompt_template=None,
                top_k=20):
        if prompt_template is None:
            prompt_template = self.prompts['user_qa_source']
        # to_question_prompt = PromptTemplate(template=self.prompts['to_question'], input_variables=["statement"])
        # to_question_message = [
        #     SystemMessage(content="You are a helpful AI assistant."),
        #     HumanMessage(content=to_question_prompt.format(statement=question))
        # ]
        # llm = ChatOpenAI(temperature=0)
        # question = llm(to_question_message).content
        if os.path.exists(basic_info_path):
            with open(basic_info_path, 'r') as f:
                basic_info_dict = json.load(f)
            basic_info_string = str(basic_info_dict)
        else:
            basic_info_prompt = PromptTemplate(template=self.prompts['general'], input_variables=["context"])

            message = basic_info_prompt.format(
                context=_docs_to_string(report.section_text_dict['general'], with_source=False))
            llm = QianfanLLMEndpoint(streaming=True)
            output_text = llm(message)
            try:
                basic_info_dict = json.loads(output_text)
            except ValueError as e:
                basic_info_dict = {'COMPANY_NAME': _find_answer(output_text, name='COMPANY_NAME'),
                                   'COMPANY_SECTOR': _find_answer(output_text, name='COMPANY_SECTOR'),
                                   'COMPANY_LOCATION': _find_answer(output_text, name='COMPANY_LOCATION')}
            basic_info_string = str(basic_info_dict)
            with open(basic_info_path, 'w') as f:
                json.dump(basic_info_dict, f)
            self.basic_info_answers.append(basic_info_dict)

        self.user_questions.append(question)
        # get the retriever, where the vector database is loaded from report.db_path
        # import pdb
        # pdb.set_trace()
        retriever, _ = report._get_retriever(report.db_path)
        docs = retriever.invoke(question)
        GHG_prompt = PromptTemplate(template=prompt_template,
                                     input_variables=["basic_info", "summaries", "question", "answer_length"])
        num_docs = top_k
        ######################################################
        prompts = '''你被赋予了碳排放及气候科学家的角色，负责分析一家公司的可持续发展报告。。你的工作是帮助企业人员更好的理解问题，不要回答问题。
                        简要分析，字数控制在100字以内，尽量囊括足够多的背景信息。问题：{query},示例：{Example}'''
        system_prompt = '''
                        根据背景信息，筛选出最适合回答问题的规则或内容，保持其完整性并直接引用原文作答，没有适合回答的就不要生成。                                                   
                        以下是背景信息：{context}\n
                        这是问题：{question}
                        '''
        print(question)
        Example = RAG_cot_prompt(question)
        prompt = prompts.format(query=question, Example=Example)
        outputs = model.invoke(prompt)
        outputs = f'{question}+{outputs}'
        print(question)

        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever], search_kwargs={"k": 3})
        ensemble_results = ensemble_retriever.get_relevant_documents(question)
        # ensemble_results = retriever.invoke(question)

        Multiple_expressions = Multiple_expressions_llm(question)
        Multiple_expressions_list = []
        for item in Multiple_expressions:
            print(item)
            ##########################
            expressions_results = ensemble_retriever.get_relevant_documents(item)
            expressions_contents = [doc.page_content for doc in expressions_results]
            Multiple_expressions_list.extend(expressions_contents)
        a = model2.do(question, Multiple_expressions_list)
        b = convert_to_documents(a)
        c = [doc.page_content for doc in b]

        page_contents = [doc.page_content for doc in ensemble_results]
        context = "\n".join([doc.page_content for doc in ensemble_results])
        inputs = {"context": context, "query": question}

        data = Key_sentences_with_llm(outputs)
        combined_list = []
        for item in data:
            print(item)
            ##########################
            key_results = ensemble_retriever.get_relevant_documents(item)
            key_contents = [doc.page_content for doc in key_results]
            combined_list.extend(key_contents)
        d = model2.do(question, combined_list)
        e = convert_to_documents(d)
        f = [doc.page_content for doc in e]

        from collections import OrderedDict
        combined_list = list(OrderedDict.fromkeys(f + page_contents + c))
        c = model2.do(question, combined_list)

        d = convert_to_documents(c)
        basic_info_string3 = "\n".join([doc.page_content for doc in d[:10]])

        current_prompt = GHG_prompt.format(basic_info=f'{basic_info_string}；温室气体核算体系(企业核算与报告标准)：{basic_info_string3}',
                                            summaries=_docs_to_string(docs),
                                            question=question,
                                            answer_length=str(answer_length))
        # token_cnt = tokenizer.Tokenizer().count_tokens(text=current_prompt,mode='remote',model="ernie-bot-4")
        # while token_cnt > 3500 and num_docs > 10:
        #     num_docs -= 1
        #     current_prompt = GHG_prompt.format(basic_info=self.basic_info_answers[0],
        #                                         summaries=_docs_to_string(docs, num_docs=num_docs),
        #                                         question=question,
        #                                         answer_length=str(answer_length))
        message = current_prompt
        print("qa_message:",message)
        output_text = llm.invoke(message)
        try:
            answer_dict = json.loads(output_text)
        except ValueError as e:
            answer_dict = {self.answer_key_name: _find_answer(output_text, name=self.answer_key_name),
                           'SOURCES': _find_sources(output_text)}
        page_source = []
        for s in answer_dict['SOURCES']:
            try:
                page_source.append(report.page_idx[s])
            except Exception as e:
                pass
        used_chunks = []
        for doc in docs:
            if int(doc.metadata['source']) in answer_dict['SOURCES']:
                used_chunks.append(doc.page_content)
        answer_dict[self.answer_key_name] = remove_brackets(answer_dict[self.answer_key_name])
        answer_dict['PAGE'] = list(set(page_source))
        answer_dict['QUESTION'] = question
        answer_dict['ANSWER_LENGTH'] = answer_length
        answer_dict['USED_CHUNKS'] = used_chunks
        self.user_answers.append(answer_dict)

        return answer_dict, docs
