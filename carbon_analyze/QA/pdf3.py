import os
from appbuilder.core.components.doc_parser.doc_parser import DocParser
from appbuilder.core.components.doc_splitter.doc_splitter import DocSplitter
from appbuilder.core.message import Message
import time
import pandas as pd
from langchain.schema import Document
os.environ["APPBUILDER_TOKEN"] = ""
# 单文档解析
def split_title(file_path):
    msg = Message(file_path)
    parser = DocParser()
    # KPI measure start
    start_time = time.time()  # 开始时间
    parse_result = parser(msg,return_raw=True)# 这里跟之前不同
    end_time = time.time()  # 结束时间
    # KPI
    elapsed_time = end_time - start_time  # 花费的时间
    elapsed_time = round(elapsed_time, 2)  # 保留两位小数
    print("Parse time {}s".format(elapsed_time))
    # 基于parser的结果切分段落
    # splitter = DocSplitter(splitter_type="split_by_chunk") # 基于chunk
    splitter = DocSplitter(splitter_type="split_by_title") # 基于 title
    res_paras = splitter(parse_result)
    documents = [Document(page_content=item['text'], metadata={'node_id': item['node_id']}) for item in res_paras.content['paragraphs']]
    return documents,res_paras


from concurrent.futures import ThreadPoolExecutor
def get_title_path(nodes, parent_id, cache):
    """
    通过路径压缩和迭代方式查找指定节点的标题层级。
    """
    if parent_id in cache:
        return cache[parent_id]

    titles = []
    current_id = parent_id

    # 迭代替代递归，逐层向上查找
    while current_id:
        node = nodes[current_id]
        titles.append(node.text)
        current_id = node.parent

    # 标题从根节点开始，需反转顺序
    titles.reverse()

    # 缓存路径
    cache[parent_id] = titles
    return titles


def process_subtree(nodes, subtree_ids, cache):
    """
    处理子树，计算每个节点的标题层级。
    """
    results = {}
    for node_id in subtree_ids:
        results[node_id] = get_title_path(nodes, node_id, cache)
    return results


def get_titles_with_parallel(nodes, use_parallel=False):
    """
    主函数：查找所有节点的标题层级，支持并行化。
    """
    # 初始化缓存
    cache = {}
    node_ids = list(nodes.keys())

    if use_parallel:
        # 切分节点树为多个子树
        num_workers = 4
        chunk_size = len(node_ids) // num_workers
        subtrees = [node_ids[i:i + chunk_size] for i in range(0, len(node_ids), chunk_size)]

        # 并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_subtree, nodes, subtree, cache) for subtree in subtrees]
            results = {}
            for future in futures:
                results.update(future.result())
        return results

    else:
        # 串行处理
        results = {}
        for node_id in node_ids:
            results[node_id] = get_title_path(nodes, node_id, cache)
        return results
