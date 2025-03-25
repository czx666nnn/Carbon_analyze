# #基于 BERT 的文本切分方法
# from modelscope.outputs import OutputKeys
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks

# #初始化文本分割任务的pipeline
# p = pipeline(task=Tasks.document_segmentation, model='damo/nlp_bert_document-segmentation_chinese-base')

# #输入需要分割的长文本
# documents = 'BM25是基于词频反向文档频率（TF-IDF）的经典检索模型，能够有效地处理文档和查询之间的词匹配，并且在处理短文本查询时非常有效，提供了精确度高的检索结果。Embedding能够捕捉到更加深层次的语义相似性，弥补了BM25对语义匹配的不足。通过词向量或句向量的相似度计算，Embedding模型可以找到语义上更接近的文档。通过BM25与Embedding的混合检索，系统同时结合了词频匹配和语义匹配的优点。这种混合模式既保证了检索的精确度，又提升了语义覆盖度。企业碳排放知识问答系统需要实时查询和分析大量数据，并与企业数据库高效对接。Text2SQL能够快速生成SQL查询，使得系统可以即时获取最新的碳排放数据，支持决策过程。然而传统的text2sql，难以捕捉文本中的关键信息、无法处理较为复杂的文本问题，针对此问题，本文重新构建了text2sql系统。首先利用大语言模型（LLM）解析用户问题，并行提取时间范围和定位数据表。接着，结合关系数据库的schema信息生成初步的SQL查询，并对生成的SQL语句进行优化。优化后的SQL经过语法检查后，最终执行查询，返回结果。整个流程通过多层次的信息提取和SQL优化，确保了查询的准确性和执行效率。其流程如图所示。'
# #执行文本分割
# result = p(documents=documents)

# #输出分割后的文本结果
# print(result[OutputKeys.TEXT])

import requests  # 用于发送HTTP请求
from bs4 import BeautifulSoup  # 用于解析HTML内容
import re  # 用于正则表达式操作
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
url = ''
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
else:
    print(f"Error: {response.status_code}")

single_sentences_list = re.split(r'(?<=[。？！」])\s+', text)

print(f"{len(single_sentences_list)} sentences were found")

sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]
print(sentences[:6])

def determine_window_size(sentence):
    """
    根据句子长度动态确定窗口大小
    参数:
    sentence (str): 句子内容
    
    返回值:
    int: 建议的窗口大小
    """
    base_window = 
    max_window = 
    
    sentence_length = len(sentence)
    
    if sentence_length < 10:  # 非常短的句子
        return max_window
    elif sentence_length < 20:  # 较短的句子
        return max_window - 1
    elif sentence_length < 40:  # 中等长度的句子
        return base_window + 1
    else:  # 长句子
        return base_window
        
def combine_sentences_dynamic(sentences):
    """
    参数:
    sentences (list): 包含句子及其索引的字典列表，格式为 [{'sentence': '句子内容', 'index': 索引}]。
    
    返回值:
    list: 更新后的句子列表，每个句子字典新增一个键 'combined_sentence'，表示组合后的句子。
    """
    for sentence in sentences:
        sentence['window_size'] = determine_window_size(sentence['sentence'])
    
    for i in range(len(sentences)):
        buffer_size = sentences[i]['window_size']
        
        combined_sentence = ''

        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '

        combined_sentence += sentences[i]['sentence']

        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentence'] = combined_sentence

    return sentences
sentences = combine_sentences_dynamic(sentences)

print(sentences[:3])
model = SentenceTransformer(model_name_or_path='G:/pretrained_models/mteb/bge-m3')
embeddings = model.encode([x['combined_sentence'] for x in sentences])
print(embeddings)

for i, sentence in enumerate(sentences):
    sentence['combined_sentence_embedding'] = embeddings[i]

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']

        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity

        distances.append(distance)
        sentences[i]['distance_to_next'] = distance

    return distances, sentences

distances, sentences = calculate_cosine_distances(sentences)

# 提取所有句子用于TF-IDF和主题建模
all_sentences = [s['sentence'] for s in sentences]

# 使用TF-IDF向量化文本
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(all_sentences)

# 使用LDA进行主题建模
num_topics = 3  # 可根据文档内容调整主题数
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
topic_distributions = lda.fit_transform(tfidf_matrix)

# 为每个句子添加主题分布信息
for i, sentence in enumerate(sentences):
    if i < len(topic_distributions):
        sentence['topic_distribution'] = topic_distributions[i]

# 计算主题变化度量
topic_changes = []
for i in range(len(sentences) - 1):
    # 获取当前句子和下一个句子的主题分布
    if i < len(topic_distributions) - 1:
        current_topic = topic_distributions[i]
        next_topic = topic_distributions[i + 1]
        
        # 计算主题分布间的欧几里得距离
        topic_change = np.linalg.norm(current_topic - next_topic)
        topic_changes.append(topic_change)
        
        # 将主题变化保存到句子字典中
        sentences[i]['topic_change_to_next'] = topic_change

# 综合评分：结合余弦距离和主题变化
combined_scores = []
for i in range(len(distances)):
    if i < len(topic_changes):
        # 将嵌入距离和主题变化进行加权结合
        embedding_weight = 0.7
        topic_weight = 0.3
        
        combined_score = embedding_weight * distances[i] + topic_weight * topic_changes[i]
        combined_scores.append(combined_score)
        
        # 将组合分数保存到句子字典中
        sentences[i]['combined_score'] = combined_score

if len(combined_scores) < len(distances):
    for i in range(len(combined_scores), len(distances)):
        combined_scores.append(distances[i])
        sentences[i]['combined_score'] = distances[i]

plt.figure(figsize=(12, 6))
plt.plot(combined_scores)

y_upper_bound = max(combined_scores) * 1.2
plt.ylim(0, y_upper_bound)
plt.xlim(0, len(combined_scores))
plt.title('句子间内容变化得分 (嵌入距离 + 主题变化)')
plt.xlabel('句子索引')
plt.ylabel('变化得分')
breakpoint_percentile_threshold = 90
breakpoint_score_threshold = np.percentile(combined_scores, breakpoint_percentile_threshold)
plt.axhline(y=breakpoint_score_threshold, color='r', linestyle='-')

indices_above_thresh = [i for i, x in enumerate(combined_scores) if x > breakpoint_score_threshold]
num_blocks = len(indices_above_thresh) + 1
plt.text(x=(len(combined_scores) * .01), y=y_upper_bound / 50, s=f"{num_blocks} 块")

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
start_indices = [0] + [idx + 1 for idx in indices_above_thresh]
end_indices = indices_above_thresh + [len(combined_scores)]

for i in range(len(start_indices)):
    start_idx = start_indices[i]
    end_idx = end_indices[i]
    
    plt.axvspan(start_idx, end_idx, facecolor=colors[i % len(colors)], alpha=0.25)
    plt.text(x=np.average([start_idx, end_idx]),
             y=breakpoint_score_threshold + (y_upper_bound) / 20,
             s=f"块 #{i+1}", horizontalalignment='center',
             rotation='vertical')

start_index = 0

chunks = []

for index in indices_above_thresh:
    end_index = index

    group = sentences[start_index:end_index + 1]
    combined_text = ' '.join([d['sentence'] for d in group])
    chunks.append(combined_text)
    start_index = index + 1

if start_index < len(sentences):
    combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
    chunks.append(combined_text)

print(f"总共生成了 {len(chunks)} 个文本块:")
for i, chunk in enumerate(chunks):
    print(f"块 #{i+1}:")
    print(f"{chunk[:200]}...(总长度: {len(chunk)}字符)")
    print("-" * 50)
