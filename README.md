# 基于大语言模型的企业碳排放分析与知识问答系统
## Directories
- data: Shown is our manually annotated dataset
  - data/RAG_data.xlsx: Dataset to test text retrieval
  - data/evaluation_SQL.txt: The data set used to test the Text2SQL system. Due to commercial confidentiality, we have not published the SQL execution results. Thank you.
  - analyze_result: Example of analysis results of corporate carbon emissions system
- code
  - QA/pdf1.py: Split documents according to custom rules
  - QA/pdf2.py: Text segmentation method based on BERT (semantics)
  - QA/pdf3.py: Text segmentation method based on document tree
  - code/report_processing/excel_analyze.py: method of table processing
  - code/report_processing/image_analyze.py: method of image processing
  - code/report_processing/script.py: Scripts to handle running the carbon emissions analysis system
  - code/report_processing/analyze_prompt.py、QA/QAprompt_zh.txt: Prompt Engineering for entire system design
  - QA/evaluate: All treatments used in our experiments
## Usage
Set up the environment
```shell
conda create --name carbon_analyze python=3.10
conda activate carbon_analyze 
pip install -r requirements.txt
```

1. Configure your own API key in reader.py, document.py, document.py, user_qa.py


2. Analyze a given report, for example: 2023-tesla-impact-report-highlights.pdf
```commandline
python app.py --pdf_path reports/2023-tesla-impact-report-highlights.pdf
```
- Analysis report will be stored at 2023-tesla-impact-report-highlights.html"

3. Conduct customized Question Answering
```shell
python app.py --pdf_path  reports/2023-tesla-impact-report-highlights.pdf --user_question " What key factors did the company consider when evaluating indirect reductions?" --answer_length 100
```
- user_question takes the user's question
- answer_length specifies the length of generation 
## 附录一Prompt
### 基于LLM的智能问答系统Prompt
```
P1: 你是双碳领域专家，根据背景信息，筛选出最适合回答问题的规则或内容，保持其完整性并直接引用原文作答，不要做修改，不需要你总结，没有适合回答的就不要生成。以下是候选信息：{context}\n，问题：{query}\n
```
```
P2: 你是双碳领域的资深学者。你的工作是帮助用户更好地理解问题，不要直接回答问题。请按照以下步骤简要分析问题：首先，概述与问题相关的背景信息；其次，列出可能的关键要素或影响因素；最后，总结潜在的挑战或考虑因素。字数控制在100字以内，尽量囊括足够多的背景信息。问题：{query}
```
```
P3: 从以下文本中提取关键句：{text},关键句应该是能够概括主要内容的单词或短语，选择反映核心主题或概念的语句，避免提取常见无关词汇（如：连词、副词等）。关键句数量不超过5个。
```
```
P4: 你是一个双碳领域专家，请根据以下提问判断其意图：
a.是否与国家碳排放政策相关（例如：政策内容、实施情况、政策影响等）。
b.是否需要查询企业数据库（例如：需要具体企业的碳排放数据、减排措施等结构化信息）。 {query}
输出以下选项之一：
1.直接检索(问题与国家碳排放政策无关，且无需企业具体数据，可通过公开信息或通用知识回答。)\n
2.需要查询向量数据库(问题与国家碳排放政策相关，但无需企业具体数据，可通过语义搜索政策文档或文本信息回答。)\n
3.需要查询关系数据库(问题需要具体企业的结构化数据（如碳排放量、合规情况），但与国家碳排放政策无关。)\n
4.都需要查询（问题既与国家碳排放政策相关，又需要具体企业的结构化数据。）
```
```
P5: 输入查询： {query}
请你按照以下输出要求，逐步完成。
输出要求：
1.将查询分解成5个简单、清晰的子查询。
2.识别并消除查询中的歧义，提供最可能的解释。
3.提炼出查询的核心意图和概念元素。
4.生成一个高层次的简化表示，保留查询的本质含义。
```
```
P6: 你是MySQL专家,对于给定问题和schema等信息，你的任务是创建一个语法准确、准确回答用户问题的高效MySQL查询，根据给定问题和已知信息生成满足要求的SQL。
\n 问题：{query}, 已知{time_info},直接使用此日期，无需推算。\n 已知信息: 
\n 数据表信息：{table_info}. \n 生成过程中参考如下信息：\n{remarks}. \n 
\n 参考示例:\n {sample} 
\n 要求：
\n 在给定数据表范围内查询回答问题所必须的列，注意使用列的单位，SQL必须正确使用表中含有的列名，根据schema和问题为查询的列名或计算结果设置中文别名。不明确查询日期但是表有含有日期的列时，默认是查询表中最新日期的数据。
\n 你可以一步步思考如何得到正确的SQL，但最终只给出MySQL语句。除了生成的SQL不要包含任何其他。
```
```
P7: 从问题中提取时间段，记录开始和结束的的时间。
\n 问题：{query}。 已知现在{now_date_info}
\n 返回一个json，格式为：{{ 问题中包含时间相关的描述，精确到日：yyyy-MM-dd～yyyy-MM-dd}}。
\n 注意：当问题中有如最近、目前、当前时，返回{{}}.
\n 示例：昨天下午4点怎么样？结果：{{昨天：2024-03-01～2024-03-01}}\n 示例：上个月初公司运行情况怎样？结果：{{上个月初：2024-03-01～2024-03-10}}。\n问题：今年的碳配额较去年如何？ 结果：{{"今年":2024-01-01～2024-12-31,"去年":"2023-01-01～2023年-12-31"}}.
\n 只输出json，没有时间相关返回{{}},不要解释也不要输出中间过程
```
```
P8: 当前查询未能匹配到任何碳排放数据。请确认您的问题是否包括以下要素：企业名称、年份、具体碳排放种类等。如果仍有问题，请重述或联系管理员获取帮助。
```
```
P9: 对Mysql语句进行纠错和优化，基于以下SQL查询：{query}。
使用提供的表信息：{table_info}，确保查询在语法和逻辑上正确。
将当前时间{time_info}直接纳入任何时间相关操作，无需进一步计算。
考虑附加备注：{remarks}，这是生成Mysql语句的思维链。
优化时，确保：

1.使用WHERE、JOIN和ORDER BY子句中的列索引。
2.选择必要的列，而不是使用SELECT *。
3.避免在WHERE子句中对列使用函数，以允许索引使用。
4.尽可能使用JOIN替代子查询。
5.确保日期函数正确使用提供的{time_info}。

此外，解释所做的更改及其为何能改善查询的正确性和性能。
最后，提供优化的SQL查询。
```
```
P10: 你是一名出色的碳排放相关性评估器，能够精准衡量分析维度与分析结果之间的相关性，并给出合理的分数。
【任务描述】
你的任务是根据企业碳排放分析的四个核心维度（如 Scope 1、Scope 2、Scope 3 和定制化维度）对分析结果的相关性进行评估。具体流程如下：
接收到两段文本：第一段是分析维度，第二段是分析结果。
通过分析两者间的逻辑关系，确定分析结果是否紧密围绕指定维度展开。
评分范围为 0 至 100，其中 0 表示完全不相关，100表示完全相关。
【打分标准】
分数根据分析结果覆盖的维度内容和深度进行评估。
如果分析结果全面覆盖维度要求，并提供详实、具体的分析，则得 10 分。
如果分析结果偏离维度内容或缺乏细节，分数将相应降低。
长答案与短答案应得到公平的评分，答案的准确性和全面性比长度更重要。
文本：{Dimension}\n{anwser}
```
```
P11: 你是一个专业的文档核查助手，请检查给定回答中的所有关于企业的信息，确保所有回答内容能够追溯到其原始来源，避免包含任何未经验证的推测性内容。以下是你的任务：
1.核对“报告内容”和“系统回答”，找出任何与报告不符或超出的信息，并将其标记为“幻觉”。
2.提供修正后的回答，确保其完全符合报告内容的事实和语义。
3.按照以下格式输出结果：
【报告内容】：{report_content}
【SQL执行结果】：{SQL_result}（若没有则是不需要调用企业数据库）
【系统回答】：{generated_answer}
【幻觉标记】：请标记不符合报告内容的部分并说明原因。
【修正后的回答】：请提供修正后的回答，确保其准确且符合报告内容。
【原因说明】：简要说明为何标记该内容为幻觉，以及修正依据。
```
## 附录二企业碳排放分析系统Prompt
```code/report_processing/analyze_prompt_zh.txt```

## 附录三效果展示
### 基于LLM的智能问答系统回答效果
| 方法       | 知识库召回列表       | 问题：根据《全国碳市场发展报告》，中国全国碳市场如何通过其独特的配额分配方法展现碳市场机制的灵活性和适用性优势，从而对全球碳市场机制创新做出贡献，并引起国际社会的高度关注？       |
|-----------|-----------|-----------|
| Qwen-Max  | 无  | 中国全国碳市场通过其独特的配额分配方法展现了碳市场机制的灵活性和适用性优势，主要体现在以下几个方面：结合历史强度法与基准线法：既考虑企业过去的排放强度，又参考行业最佳实践水平，确保公平性和激励性。差异化管理：根据不同行业的特点和减排潜力，实施差异化的配额分配策略，促进产业结构优化。动态调整机制：根据技术进步和社会经济发展情况，适时调整配额总量和分配规则，保持市场的长期有效性。引入拍卖制度：除了免费发放配额外，还通过拍卖方式出售部分配额，形成合理的价格信号，并将所得资金用于  |
| 普通RAG  | ['上海”，做好经验分享和互学互鉴，开展碳市场能力建设活动，… 推动国际合作，不断优化制度设计。\n全国碳市场国际关注度持续提升，… 为全球碳市场机制创新贡献“中国方案”。\n', '全国碳市场占全国二氧化碳排放的40%以上，… 强制和自愿两个市场共同构成全国碳市场体系。\n全国碳市场对全球碳价水平和交易机制具有重要影响力，… 为全球碳市场机制创新贡献“中国方案”。\n', '全国碳市场通过市场机制控制温室气体排放，… 助力经济社会绿色低碳转\n']  | 中国基于碳排放强度控制目标的配额分配方法既能够将温室气体控排责任压实到企业，又能为碳减排提供激励，展现了碳市场机制的灵活性和适用性优势，为全球碳市场机制创新贡献了“中国方案”。中国的碳市场对全球碳价水平和碳交易机制成效具有重要影响力，全国碳市场建设和运行情况受到国际社会高度关注。  |
| 本文方法  | ['文件名称: 工信部印发《重点工业产品碳足迹核算规则标准编制指南》.pdf, 页码: 7\n30 日，技术审查原则上应达成一致，… 鼓励社会团体在标准信息公共服务平台自我声明公开。\n', '文件名称: 碳排放权交易管理暂行条例（中华人民共和国国务院令 第775号）.otl.pdf, 页码: 13\n生态环境主管部门和其他负有监督管理职责的部门，… 被检查者应当如实反映情况、提供资料。\n', '文件名称: 全国碳市场发展报告（2024） .pdf, 页码: 6\n全国温室气体自愿减排交易市场正式启动，… 全国碳市场为全球碳市场机制创新贡献了“中国方案”。\n',']  | 中国基于碳排放强度控制目标的配额分配方法展现了碳市场机制的灵活性和适用性优势，为全球碳市场机制创新贡献了“中国方案”。中国的碳市场对全球碳价水平和碳交易机制成效具有重要影响力，全国碳市场建设和运行情况受到国际社会高度关注。中国基于碳排放强度控制目标的配额分配方法既能够将温室气体控排责任压实到企业，又能为碳减排提供激励，展现了碳市场机制的灵活性和适用性优势，为全球碳市场机制创新贡献了“中国方案”。  |
### 企业碳排放分析系统分析效果
```查看文件夹analyze_result```
## 附录四 评分结果展示
analyze_result/诺基亚_评分.html
