# import PyPDF2
# from langchain.document_loaders import PyMuPDFLoader
# def pdf_to_text_page(pdf_path):
#     # 打开 PDF 文件
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text_by_page = []

#         # 遍历每一页，提取文本
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text = page.extract_text()
#             text_by_page.append(text)

#     return text_by_page

#基于 BERT 的文本切分方法
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

#初始化文本分割任务的pipeline
p = pipeline(task=Tasks.document_segmentation, model='damo/nlp_bert_document-segmentation_chinese-base')

#输入需要分割的长文本
documents = 'BM25是基于词频反向文档频率（TF-IDF）的经典检索模型，能够有效地处理文档和查询之间的词匹配，并且在处理短文本查询时非常有效，提供了精确度高的检索结果。Embedding能够捕捉到更加深层次的语义相似性，弥补了BM25对语义匹配的不足。通过词向量或句向量的相似度计算，Embedding模型可以找到语义上更接近的文档。通过BM25与Embedding的混合检索，系统同时结合了词频匹配和语义匹配的优点。这种混合模式既保证了检索的精确度，又提升了语义覆盖度。企业碳排放知识问答系统需要实时查询和分析大量数据，并与企业数据库高效对接。Text2SQL能够快速生成SQL查询，使得系统可以即时获取最新的碳排放数据，支持决策过程。然而传统的text2sql，难以捕捉文本中的关键信息、无法处理较为复杂的文本问题，针对此问题，本文重新构建了text2sql系统。首先利用大语言模型（LLM）解析用户问题，并行提取时间范围和定位数据表。接着，结合关系数据库的schema信息生成初步的SQL查询，并对生成的SQL语句进行优化。优化后的SQL经过语法检查后，最终执行查询，返回结果。整个流程通过多层次的信息提取和SQL优化，确保了查询的准确性和执行效率。其流程如图所示。'
#执行文本分割
result = p(documents=documents)

#输出分割后的文本结果
print(result[OutputKeys.TEXT])