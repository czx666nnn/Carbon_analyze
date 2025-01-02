import fitz  # PyMuPDF
import re


def is_heading(text):
    """
    判断是否为“第几条”标题，不再限制加粗
    """
    heading_pattern = re.compile(r'^第[一二三四五六七八九十百千]+条')  # 匹配“第几条”
    return heading_pattern.match(text)  # 仅通过正则表达式判断是否为标题


def split_pdf_by_headings(pdf_path):
    """
    按照“第几条”标题切分 PDF 内容，不考虑加粗
    """
    doc = fitz.open(pdf_path)
    sections = []
    section_text = ""
    prev_page_last_line = ""  # 保存上一页的最后一行，处理跨页情况

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]  # 获取页面中的所有文本块
        page_text = ""

        for block in blocks:
            if block["type"] == 0:  # 确保是文本块
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        # 处理跨页情况：将上一页的最后一行合并到当前页开头
                        if prev_page_last_line and not is_heading(text):
                            text = prev_page_last_line + " " + text
                            prev_page_last_line = ""  # 清空上一页的最后一行

                        # 检测标题，如果是新的标题，保存上一部分并开始新部分
                        if is_heading(text):
                            if section_text:
                                sections.append(section_text)  # 保存前一个部分
                                section_text = ""
                        section_text += text + "\n"
                        page_text += text + "\n"

        # 如果页面最后几行不是标题，可能属于下一页的延续，保存到 prev_page_last_line
        if page_text:
            prev_page_last_line = page_text.split("\n")[-1]  # 保存最后一行内容

    if section_text:
        sections.append(section_text)  # 保存最后一部分

    return sections

# import pandas as pd
# from langchain.schema import Document
#
# data = {
#     'text_splitter': [],
# }
# # 使用示例
# pdf_path = "/Users/caozhixuan/Desktop/chatreport-main/QA/1.国家发改委关于公布《供电营业规则》的令.pdf"
# sections = split_pdf_by_headings(pdf_path)
#
# documents = [Document(page_content=section) for section in sections]
# print(documents)
# # 打印切分后的结果
# for idx, doc in enumerate(documents):
#     data['text_splitter'].append(doc)
#     print(f"Document {idx + 1}:")
#     print(doc.page_content)  # 打印每个 Document 对象的内容
#     print("-" * 80)
#
# out_df = pd.DataFrame(data)
# out_df.to_csv("/Users/caozhixuan/Desktop/chatreport-main/QA/2.csv", index=False)
