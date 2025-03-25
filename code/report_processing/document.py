import fitz
import re
import os
import time
import json
import requests
import io
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import langdetect

TOP_K = 10
BASE_CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_CHUNK_SIZE = 500

QUERIES = {
    'general': ["该报告中涉及的企业名称是什么？", "企业所属的行业类别是什么？", "企业的地理位置在哪里？"],
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

class Report:
    def __init__(self, path=None, url=None, title='', abs='', authers=[], store_path=None, top_k=TOP_K, db_path=None, retrieved_chunks_path=None):
        self.chunks = []
        self.page_idx = []
        self.path = path
        self.url = url
        assert ((path is None and url is not None) or (path is not None and url is None))
        self.store_path = store_path
        self.queries = QUERIES
        self.top_k = top_k
        self.section_names = []
        self.section_texts = {}
        self.db_path = db_path
        self.retrieved_chunks_path = retrieved_chunks_path
        if title == '':
            if self.path:
                self.pdf = fitz.open(self.path)
            else:
                self.parse_pdf_from_url(self.url)
            self.title = self.get_title()
            self.parse_pdf()
        else:
            self.title = title
        self.authers = authers
        self.abs = abs
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d + 1) for d in range(10)]
        self.first_image = ''

    def parse_pdf_from_url(self, url):
        response = requests.get(url)
        pdf = io.BytesIO(response.content)
        self.pdf = fitz.open(stream=pdf)

    def parse_pdf(self):
        if self.path:
            self.pdf = fitz.open(self.path)
        else:
            self.parse_pdf_from_url(self.url)
        self.text_list = [re.sub(r'\s+', ' ', page.get_text("text").strip()) for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        self.retriever, self.vector_db = self._get_retriever(self.db_path)
        self.section_text_dict = self._retrieve_chunks()
        if not os.path.exists(self.retrieved_chunks_path):
            os.makedirs(self.retrieved_chunks_path)
        with open(os.path.join(self.retrieved_chunks_path, 'retrieved.json'), 'w') as f:
            to_dump = {
                key: {chunk.metadata['source']: [chunk.page_content, chunk.metadata['page']] for chunk in chunk_list}
                for key, chunk_list in self.section_text_dict.items()
            }
            json.dump(to_dump, f)
        self.section_text_dict.update({"title": self.title})
        store_flag = True
        if self.store_path is not None and store_flag:
            self.pdf.save(self.store_path)

    def get_image_path(self, image_path=''):
        max_size = 0
        image_list = []
        with fitz.Document(self.path) as my_pdf_file:
            for page_number in range(1, len(my_pdf_file) + 1):
                page = my_pdf_file[page_number - 1]
                images = page.get_images()
                for image_number, image in enumerate(page.get_images(), start=1):
                    xref_value = image[0]
                    base_image = my_pdf_file.extract_image(xref_value)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    image = Image.open(io.BytesIO(image_bytes))
                    image_size = image.size[0] * image.size[1]
                    if image_size > max_size:
                        max_size = image_size
                    image_list.append(image)
        for image in image_list:
            image_size = image.size[0] * image.size[1]
            if image_size == max_size:
                image_name = f"image.{ext}"
                im_path = os.path.join(image_path, image_name)
                max_pix = 480
                origin_min_pix = min(image.size[0], image.size[1])
                if image.size[0] > image.size[1]:
                    min_pix = int(image.size[1] * (max_pix / image.size[0]))
                    newsize = (max_pix, min_pix)
                else:
                    min_pix = int(image.size[0] * (max_pix / image.size[1]))
                    newsize = (min_pix, max_pix)
                image = image.resize(newsize)
                image.save(open(im_path, "wb"))
                return im_path, ext
        return None, None

    def get_chapter_names(self):
        doc = fitz.open(self.path)
        text_list = [page.get_text() for page in doc]
        all_text = ''.join(text_list)
        chapter_names = []
        for line in all_text.split('\n'):
            line_list = line.split(' ')
            if '.' in line:
                point_split_list = line.split('.')
                space_split_list = line.split(' ')
                if 1 < len(space_split_list) < 5:
                    if 1 < len(point_split_list) < 5 and (
                            point_split_list[0] in self.roman_num or point_split_list[0] in self.digit_num):
                        chapter_names.append(line)
        return chapter_names

    def get_title(self):
        doc = self.pdf
        max_font_size = 0
        max_string = ""
        max_font_sizes = [0]
        for page in doc:
            text = page.get_text("dict")
            blocks = text["blocks"]
            for block in blocks:
                if block["type"] == 0 and len(block['lines']):
                    if len(block["lines"][0]["spans"]):
                        font_size = block["lines"][0]["spans"][0]["size"]
                        max_font_sizes.append(font_size)
                        if font_size > max_font_size:
                            max_font_size = font_size
                            max_string = block["lines"][0]["spans"][0]["text"]
        max_font_sizes.sort()
        cur_title = ''
        for page in doc:
            text = page.get_text("dict")
            blocks = text["blocks"]
            for block in blocks:
                if block["type"] == 0 and len(block['lines']):
                    if len(block["lines"][0]["spans"]):
                        cur_string = block["lines"][0]["spans"][0]["text"]
                        font_size = block["lines"][0]["spans"][0]["size"]
                        if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(font_size - max_font_sizes[-2]) < 0.3:
                            if len(cur_string) > 4:
                                if cur_title == '':
                                    cur_title += cur_string
                                else:
                                    cur_title += ' ' + cur_string
        return cur_title.replace('\n', ' ')

    def _detect_language(self, text):
        return langdetect.detect(text)

    def _get_structured_blocks(self):
        blocks = []
        page_numbers = []
        for i, page in enumerate(self.pdf):
            text_dict = page.get_text("dict")
            for block in text_dict["blocks"]:
                if block["type"] == 0 and len(block["lines"]):
                    block_text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]])
                    blocks.append(re.sub(r'\s+', ' ', block_text.strip()))
                    page_numbers.append(i + 1)
        return blocks, page_numbers

    def _split_long_chunk(self, chunk, embeddings, max_size=MAX_CHUNK_SIZE):
        if len(chunk) <= max_size:
            return [chunk]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_size // 2,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["。", "！", "？", ".", "!", "?", "\n", " "]
        )
        sub_chunks = text_splitter.split_text(chunk)
        chunk_embeddings = embeddings.embed_documents(sub_chunks)
        final_sub_chunks = []
        i = 0
        while i < len(sub_chunks):
            if i + 1 < len(sub_chunks):
                sim = cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[i + 1]])[0][0]
                if sim < 0.7:
                    final_sub_chunks.append(sub_chunks[i])
                    i += 1
                else:
                    final_sub_chunks.append(sub_chunks[i] + " " + sub_chunks[i + 1])
                    i += 2
            else:
                final_sub_chunks.append(sub_chunks[i])
                i += 1
        return final_sub_chunks

    def _get_retriever(self, db_path):
        embeddings = QianfanEmbeddingsEndpoint(
            qianfan_ak=os.environ["QIANFAN_AK"],
            qianfan_sk=os.environ["QIANFAN_SK"]
        )
        lang = self._detect_language(self.all_text)
        separators = ["\n\n", "\n", " ", ""] if lang.startswith("zh") else [". ", "! ", "? ", "\n\n", "\n", " "]
        initial_splitter = RecursiveCharacterTextSplitter(
            chunk_size=BASE_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=separators
        )
        blocks, page_numbers = self._get_structured_blocks()
        initial_chunks = []
        chunk_pages = []
        for block, page in zip(blocks, page_numbers):
            if block:
                block_chunks = initial_splitter.split_text(block)
                initial_chunks.extend(block_chunks)
                chunk_pages.extend([page] * len(block_chunks))
        chunk_embeddings = embeddings.embed_documents(initial_chunks)
        distances = []
        for i in range(len(chunk_embeddings) - 1):
            sim = cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[i + 1]])[0][0]
            distances.append(1 - sim)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        dynamic_threshold = mean_dist + std_dist
        split_indices = [i for i, dist in enumerate(distances) if dist > dynamic_threshold]
        final_chunks = []
        start_idx = 0
        for idx in split_indices + [len(initial_chunks)]:
            chunk_text = ' '.join(initial_chunks[start_idx:idx + 1])
            if len(chunk_text) > MAX_CHUNK_SIZE:
                sub_chunks = self._split_long_chunk(chunk_text, embeddings)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk_text)
            start_idx = idx + 1
        self.chunks = final_chunks
        self.page_idx = chunk_pages[:len(self.chunks)]
        if os.path.exists(db_path):
            doc_search = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        else:
            doc_search = FAISS.from_texts(
                self.chunks,
                embeddings,
                metadatas=[{"source": str(i), "page": str(page)} for i, page in enumerate(self.page_idx)]
            )
            doc_search.save_local(db_path)
        return doc_search.as_retriever(search_kwargs={"k": self.top_k}), doc_search

    def _retrieve_chunks(self):
        section_text_dict = {}
        for key in self.queries.keys():
            if key == 'general':
                docs_1 = self.retriever.invoke(self.queries[key][0])[:5]
                docs_2 = self.retriever.invoke(self.queries[key][1])[:5]
                docs_3 = self.retriever.invoke(self.queries[key][2])[:5]
                section_text_dict[key] = docs_1 + docs_2 + docs_3
            else:
                time.sleep(1)
                section_text_dict[key] = self.retriever.invoke(self.queries[key])
        return section_text_dict
