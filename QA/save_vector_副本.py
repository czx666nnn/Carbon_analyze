# coding=utf-8
import hashlib
import json
import logging
import math
import os
import time
import pandas as pd
import qianfan
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint,XinferenceEmbeddings
from langchain_community.llms import QianfanLLMEndpoint
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf1 import split_pdf_by_headings
from langchain.schema import Document
from pdf2 import pdf_to_text_page
from pdf3 import split_title

# 设置日志级别
logging.root.setLevel('INFO')

# 设置环境变量
os.environ["QIANFAN_AK"] = "ZJjlM0YRkktVVLe3fGHuA9gZ"
os.environ["QIANFAN_SK"] = "vlQJlrywCUL4SYVQkPenscB7m1C8XxG6"

# 定义 LLM 和嵌入模型名称
llm_model_name = "ERNIE-Bot-4"
embeddings_model_name = "bge-m3"
kwargs = {'max_output_tokens': 2000}

qianfan_ak = 'ZJjlM0YRkktVVLe3fGHuA9gZ'
qianfan_sk = 'vlQJlrywCUL4SYVQkPenscB7m1C8XxG6'
embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)
# 初始化 LLM 和嵌入模型
model = QianfanLLMEndpoint(model=llm_model_name, streaming=True, init_kwargs=kwargs)
# embeddings = XinferenceEmbeddings(server_url="http://192.168.211.107:9997/",model_uid="bge-m3-czx")
persist_directory = "/Users/caozhixuan/Desktop/chatreport-main/Demo-BaiduQianfan-Chatbot-master/.venv/chromadb4"
qianfan.enable_log()

class ChatBot:

    @staticmethod
    def clear_vector_database():
        """
        清空向量数据库及相关的哈希文件。
        """
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectorstore.delete_collection()  # 删除数据库中的所有集合
        hashmap_json = {}
        with open(os.path.join("../code/.venv/hash.json"), "w") as f:
            json.dump(hashmap_json, f)  # 清空哈希记录文件
        logging.info("向量数据库内容已清除")
        return json.dumps({"output": "cleared"})

    @staticmethod
    def get_hash(file):
        """
        计算文档的哈希值，用于避免重复上传。
        """
        return hashlib.sha256(file.page_content.encode()).hexdigest()

    @staticmethod
    def process_files():
        """
        处理并上传文档到向量数据库。
        """
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        hashmap_json = {}
        # 读取现有的哈希记录

        if os.path.exists("../code/.venv/hash.json"):
            hashmap_json = json.load(open("../code/.venv/hash.json"))
        else:
            with open(os.path.join("../code/.venv/hash.json"), "w") as f:
                json.dump(hashmap_json, f)
        # 加载指定目录下的 CSV 文件
        loader = DirectoryLoader("../code/.venv/data", glob="*.csv", loader_cls=CSVLoader)

        docs = loader.load()
        docs_to_upload = []
        for doc in docs:
            hash_value = ChatBot.get_hash(doc)
            if hash_value in hashmap_json:
                continue
            else:
                docs_to_upload.append(doc)
                hashmap_json[hash_value] = "1"

        if len(docs_to_upload) == 0:
            logging.info(f"没有需要上传的文档")
            return json.dumps({"output": "finished - no changes made", "status": "finished"})

        logging.info(f"文档长度：{len(docs)}")
        logging.info(f"需要上传的文档长度：{len(docs_to_upload)}")
        # 根据嵌入模型名称选择切分参数
        if embeddings_model_name == "tao-8k":
            split_unit = 384
            overlap_size = 80
        elif embeddings_model_name == "bge-large-zh":
            split_unit = 384
            overlap_size = 80
        else:
            split_unit = 384
            overlap_size = 80
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=split_unit, chunk_overlap=overlap_size)
        splits = text_splitter.split_documents(docs_to_upload)
        if embeddings_model_name == "tao-8k":
            logging.warning("tao-8k Embeddings接口不支持文件批量上传转换为索引值且有QPS(Query Per Second)限制，将尝试依次上传。"
                            f"共{len(splits)}个文件切片，上传过程将持续约{math.ceil(len(splits)/60)}分钟。")
            i = 0
            time.sleep(1)
            for document in splits:
                vectorstore.add_documents(documents=[document],
                                          embedding=embeddings,
                                          persist_directory=persist_directory)
                time.sleep(0.6)
                logging.info(f"当前进度：{i}/{len(splits)}")
                yield json.dumps({"output": f"进度{i/len(splits)*100:.2f}%", "status": "active"})
                i += 1
            logging.info("文本切片已全部上传")
        else:
            vectorstore.from_documents(documents=splits,
                                       embedding=embeddings,
                                       persist_directory=persist_directory)
        with open(os.path.join("../code/.venv/hash.json"), "w") as f:
            json.dump(hashmap_json, f)
        logging.info(f"库中文本数量：{vectorstore._collection.count()}")
        test_text = "这是一条测试文本"
        test_embedding = embeddings.embed_query(test_text)
        logging.info(f"嵌入检查：{len(test_embedding)}")
        return json.dumps({"output": "uploaded", "status": "finished"})

    @staticmethod
    def process_pdf():
        """
               处理并上传pdf到向量数据库。
               """
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        hashmap_json = {}
        data = {
            'text_splitter': [],
        }
        # # 读取现有的哈希记录
        # if os.path.exists(".venv/hash.json"):
        #     hashmap_json = json.load(open(".venv/hash.json"))
        # else:
        #     with open(os.path.join(".venv/hash.json"), "w") as f:
        #         json.dump(hashmap_json, f)
        # 加载指定目录下的 pdf 文件####################
        # from langchain.document_loaders import PyMuPDFLoader
        # loader = PyMuPDFLoader("/Users/caozhixuan/Desktop/chatreport-main/企业碳排放报告/惠普.pdf")
        # docs = loader.load()
        # print(docs)
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=80, separators=["\n\n", "\n", " ", "", "。", "，"])
        # print("text_splitter:", text_splitter)
        # splits = text_splitter.split_documents(docs)
        # print(f"{len(splits)} documents block loaded")
        # print("splits:", splits)
        # ####################################
        #自定义切分（按第几条切分）
        # splits = split_pdf_by_headings("Demo-BaiduQianfan-Chatbot-master/《供电营业规则》.pdf")
        # documents = [Document(page_content=section) for section in splits]
        #按页切分
        # splits = pdf_to_text_page("Demo-BaiduQianfan-Chatbot-master/营销2.0系统/营销2.0系统_601-1090.pdf")
        # documents = [Document(page_content=section) for section in splits]
        #按标题切分
        file_path = "/Users/caozhixuan/Desktop/chatreport-main/国家及省市级政策文件/2019-2020年全国碳排放权交易配额总设定分配实施方案（发电行业）.pdf"  # 待解析的文件路径(你自己的目录和文件)
        documents,res_paras = split_title(file_path)
        splits = res_paras.content['paragraphs']
        print(splits)
        # import pdb
        # pdb.set_trace()
        if embeddings_model_name == "bge-m3":
            logging.warning(" Embeddings接口不支持文件批量上传转换为索引值且有QPS(Query Per Second)限制，将尝试依次上传。"
                            f"共{len(splits)}个文件切片，上传过程将持续约{math.ceil(len(splits) / 60)}分钟。")
            i = 0
            time.sleep(0.3)
            for doc in splits:
                print('document::',doc)
                data['text_splitter'].append(doc)

                vectorstore.add_documents(
                                          documents=[doc],
                                          embedding=embeddings,
                                          persist_directory=persist_directory)
                time.sleep(0.3)
                logging.info(f"当前进度：{i}/{len(splits)}")
                yield json.dumps({"output": f"进度{i / len(splits) * 100:.2f}%", "status": "active"})
                i += 1
            logging.info("文本切片已全部上传")
            out_df = pd.DataFrame(data)
            out_df.to_csv("/Users/caozhixuan/Desktop/chatreport-main/Demo-BaiduQianfan-Chatbot-master/3.csv", index=False)
        else:
            vectorstore.from_documents(
                                       documents=splits,
                                       embedding=embeddings,
                                       persist_directory=persist_directory
                                       )
        # with open(os.path.join(".venv/hash.json"), "w") as f:
        #     json.dump(hashmap_json, f)
        logging.info(f"库中文本数量：{vectorstore._collection.count()}")
        test_text = "这是一条测试文本"
        test_embedding = embeddings.embed_query(test_text)
        logging.info(f"嵌入检查：{len(test_embedding)}")
        return json.dumps({"output": "uploaded", "status": "finished"})

    @staticmethod
    def process_pdf2():
        """
        遍历整个文件夹的 PDF 文件，处理并上传到向量数据库，切分快内将文件名称和页码拼接到内容中。
        """
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        hashmap_json = {}
        data = {
            'text_splitter': [],
        }

        from langchain_community.document_loaders import PyMuPDFLoader
        import os

        folder_path = "/Users/caozhixuan/Desktop/chatreport-main/国家及省市级政策文件"
        if not os.path.exists(folder_path):
            raise ValueError(f"文件夹路径不存在: {folder_path}")

        # 遍历文件夹，找到所有 PDF 文件
        pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
        if not pdf_files:
            raise ValueError(f"指定文件夹中没有找到 PDF 文件: {folder_path}")

        all_splits = []

        for pdf_file in pdf_files:
            print(f"正在处理 PDF 文件: {pdf_file}")
            loader = PyMuPDFLoader(pdf_file)
            docs = loader.load()

            # 切分文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=384, chunk_overlap=80, separators=["\n\n", "\n", " ", "", "。", "，"]
            )
            splits = text_splitter.split_documents(docs)

            # 在每个切分快内加入文件名称和页码，并拼接到内容中
            for i, doc in enumerate(splits):
                # 获取当前页码
                page_number = i + 1  # 页码从1开始
                # 文件名称
                file_name = os.path.basename(pdf_file)
                # 在 page_content 开头拼接文件名称和页码
                # doc.page_content = f"文件名称: {file_name}, 页码: {page_number}\n{doc.page_content}"
                doc.page_content = f"{doc.page_content}"
                all_splits.append(doc)

            print(f"{len(splits)} 个文档切片已从 {pdf_file} 加载完成")

        print(f"总共加载了 {len(all_splits)} 个文档切片")

        # 检查嵌入模型
        if embeddings_model_name == "bge-m3":
            logging.warning(
                "Embeddings接口不支持文件批量上传转换为索引值且有 QPS 限制，将尝试依次上传。"
                f"共 {len(all_splits)} 个文件切片，上传过程可能较慢。"
            )
            i = 0
            for doc in all_splits:
                print('正在上传文档切片:', doc)
                data['text_splitter'].append(doc)

                vectorstore.add_documents(
                    documents=[doc],
                    embedding=embeddings,
                    persist_directory=persist_directory
                )
                time.sleep(0.3)
                logging.info(f"当前进度：{i}/{len(all_splits)}")
                yield json.dumps({"output": f"进度 {i / len(all_splits) * 100:.2f}%", "status": "active"})
                i += 1
            logging.info("所有文档切片已上传完成")
        else:
            vectorstore.from_documents(
                documents=all_splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )

        # 保存到 CSV（可选）
        out_df = pd.DataFrame(data)
        out_df.to_csv("/Users/caozhixuan/Desktop/chatreport-main/Demo-BaiduQianfan-Chatbot-master/3.csv", index=False)

        logging.info(f"库中文本数量：{vectorstore._collection.count()}")
        test_text = "这是一条测试文本"
        test_embedding = embeddings.embed_query(test_text)
        logging.info(f"嵌入检查：{len(test_embedding)}")
        return json.dumps({"output": "uploaded", "status": "finished"})

    @staticmethod
    def process_pdf3():
        """
        处理并上传 PDF 文件到向量数据库，支持按标题切分并在每个切分快内加入文档名称。
        """
        import os
        import math
        import time
        import json
        import logging
        from langchain.schema import Document
        from langchain.vectorstores import Chroma
        from appbuilder.core.components.doc_parser.doc_parser import DocParser
        from appbuilder.core.components.doc_splitter.doc_splitter import DocSplitter

        # 初始化向量数据库
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        hashmap_json = {}
        data = {'text_splitter': []}

        # 读取现有的哈希记录，避免重复处理
        hash_path = os.path.join(".venv", "hash.json")
        if os.path.exists(hash_path):
            hashmap_json = json.load(open(hash_path))
        else:
            with open(hash_path, "w") as f:
                json.dump(hashmap_json, f)

        # 加载 PDF 文件
        folder_path = "/Users/caozhixuan/Desktop/chatreport-main/国家及省市级政策文件"
        pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
        if not pdf_files:
            raise ValueError("未找到任何 PDF 文件！")

        splits = []  # 存储切分后的文档块
        for pdf_path in pdf_files:
            print(f"正在处理 PDF 文件: {pdf_path}")
            try:
                # 按标题切分文档
                documents, res_paras = split_title(pdf_path)
                file_name = os.path.basename(pdf_path)  # 获取文件名
                for doc in documents:
                    # 在文档切分的 metadata 中添加文件名信息
                    doc.metadata["file_name"] = file_name
                    splits.append(doc)
            except Exception as e:
                logging.error(f"处理文件 {pdf_path} 时发生错误: {e}")
                continue

        print(f"总共切分出 {len(splits)} 个文档块")

        # 上传到向量数据库
        if embeddings_model_name == "bge-m3":
            logging.warning(
                f"Embeddings接口有 QPS 限制，逐条上传切分文档。共 {len(splits)} 个切片，预计需要 {math.ceil(len(splits) / 60)} 分钟。"
            )
            for i, doc in enumerate(splits):
                # 添加到数据库
                vectorstore.add_documents(documents=[doc], embedding=embeddings, persist_directory=persist_directory)
                logging.info(f"上传进度：{i + 1}/{len(splits)}")
                yield json.dumps({"output": f"进度 {i / len(splits) * 100:.2f}%", "status": "active"})
                time.sleep(0.3)
        else:
            vectorstore.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )

        # 持久化数据库
        vectorstore.persist()

        # 保存哈希记录
        with open(hash_path, "w") as f:
            json.dump(hashmap_json, f)

        logging.info(f"库中文本数量：{vectorstore._collection.count()}")

        # 嵌入检查
        test_text = "这是一条测试文本"
        test_embedding = embeddings.embed_query(test_text)
        logging.info(f"嵌入检查：嵌入维度 {len(test_embedding)}")

        return json.dumps({"output": "uploaded", "status": "finished"})

    @staticmethod
    def retrieve_all_documents():
        """
        静态方法：检索向量数据库中的所有文档并返回。
        """
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
    @staticmethod
    def check_vector_storage():
        """
        检查向量数据库的状态，包括库名和内容数。
        """
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        library_name = vectorstore._collection.name
        library_count = vectorstore._collection.count()

        logging.info(f"库名：{library_name}")
        logging.info(f"库内内容数：{library_count}")
        return json.dumps({"library_name": f"{library_name}", "library_count": f"{library_count}"})



    @staticmethod
    def response(query_content: str, is_chat: bool):
        """
        根据查询内容生成回答。
        """
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = vectorstore.as_retriever()
        # 定义系统提示信息
        system_prompt = [
            '''
            你是新产教有限公司的一个聊天机器人。你的工作是从给定的背景信息里找出答案。背景信息是公司的Wiki库。                                                 
            你必须根据给出的Wiki来完整回答，不可以参考其他信息源，包括互联网和你可能拥有的任何其他知识。                        
            回答尽量完整，不需要你对Wiki内容进行总结，请尽量原句复述。请尽量囊括足够多的Wiki信息。                                                                                                                  
            在回答的最后，你可以提醒用户在Wiki中再次确认信息以确保准确性。                                                       
            以下是背景信息：{context}\n
            这是问题：{question}
            ''',
            '''
            你是新产教有限公司的一个聊天机器人。你的工作是帮助公司的同事解决问题。                       
            回答尽量完整，尽量囊括足够多的背景信息。                                                                       
            在回答的最后，你可以提醒用户自行再次确认信息以确保准确性。                                                          
            \n
            '''
        ]
        if not is_chat:
            # 处理文档检索问答
            qa_chain_prompt = PromptTemplate(input_variables=["context", "question"], template=system_prompt[0])
            qa_chain = RetrievalQA.from_chain_type(
                llm=model,
                retriever=retriever,
                chain_type_kwargs={"prompt": qa_chain_prompt}
            )

            docs = retriever.invoke(input=query_content, search_kwargs={"k": 10})
            print(docs)
            context = "\n".join([doc.page_content for doc in docs])
            print(context)
            qa_chain_prompt = PromptTemplate(input_variables=["context", "question"], template=system_prompt[0])
            chain = qa_chain_prompt | model
            for chunk in chain.stream({"context": context, "question": query_content}):
                print({"output": chunk, "status": "active"})
        else:
            # 处理聊天模式的对话
            qa_chain_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt[1]),
                    MessagesPlaceholder(variable_name="message")
                ]
            )
            chain = qa_chain_prompt | model
            for chunk in chain.stream({"message": [HumanMessage(content=query_content)]}):
                print({"output": chunk, "status": "active"})
        print({"output": "", "status": "finished"})

# 调用示例
if __name__ == "__main__":
    # 示例: 清空向量数据库
    # print(ChatBot.clear_vector_database())
    #
    # import pdb
    # pdb.set_trace()

    # 调用函数并处理返回的生成器
    for output in ChatBot.process_pdf3():
        result = json.loads(output)
        print(f"当前状态: {result['status']}, 输出: {result['output']}")

    # # 示例: 检查向量数据库
    print(ChatBot.check_vector_storage())
    # print(ChatBot.retrieve_all_documents())
    # # 示例: 文档检索问答
    # ChatBot.response("碳排放", is_chat=False)
    #
    # # 示例: 聊天模式对话
    # ChatBot.response("你好", is_chat=True)
