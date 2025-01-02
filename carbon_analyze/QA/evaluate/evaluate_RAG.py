from collections import Counter
import warnings
import pandas as pd
import jieba
import torch
from transformers import AutoTokenizer, AutoModel

class BertScorer:
    def __init__(self, model_name="bert-base-chinese"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def get_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0)

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算余弦相似度"""
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = y / torch.norm(y, dim=-1, keepdim=True)
        return torch.mm(x, y.T)

def segment_text(text: str) -> list[str]:
    return list(jieba.cut(text))

# 计算ROUGE-N
def rouge_n(reference: str, candidate: str, n: int) -> float:
    reference_tokens = segment_text(reference)
    candidate_tokens = segment_text(candidate)

    # 生成N-grams
    reference_ngrams = [tuple(reference_tokens[i:i + n]) for i in range(len(reference_tokens) - n + 1)]
    candidate_ngrams = [tuple(candidate_tokens[i:i + n]) for i in range(len(candidate_tokens) - n + 1)]

    # 统计N-grams的频率
    reference_counts = Counter(reference_ngrams)
    candidate_counts = Counter(candidate_ngrams)

    # 计算N-grams的重叠数量
    overlap = sum(min(reference_counts[ng], candidate_counts[ng]) for ng in candidate_counts)

    # 计算ROUGE-N得分
    if len(reference_ngrams) == 0:
        return 0
    return overlap / len(reference_ngrams)

# 计算最长公共子序列(LCS)
def lcs(X: list[str], Y: list[str]) -> int:
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n]

# 计算ROUGE-L
def rouge_l(reference: str, candidate: str) -> float:
    reference_tokens = segment_text(reference)
    candidate_tokens = segment_text(candidate)

    lcs_length = lcs(reference_tokens, candidate_tokens)

    if len(reference_tokens) == 0:
        return 0
    return lcs_length / len(reference_tokens)

# 提取信息并计算指标
def extract_info_from_excel(file_path: str, out_path: str) -> None:
    bert_scorer = BertScorer(model_name="bert-base-chinese")
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names

    data = {
        '答案': [],
        '输出': [],
        'rouge_1_score': [],
        'rouge_2_score': [],
        'rouge_l_score': [],
        'Precision': [],
        'Recall': [],
    }

    for sheet in sheet_names:
        df = excel_file.parse(sheet)
        for _, row in df.iterrows():
            a = row['答案']
            output0 = row['输出']

            # 计算ROUGE分数
            rouge_1_score0 = rouge_n(a, output0, 1)
            rouge_2_score0 = rouge_n(a, output0, 2)
            rouge_l_score0 = rouge_l(a, output0)

            # 获取BERT嵌入并计算精确度和召回率
            ref_emb = bert_scorer.get_embeddings(a)
            cand_emb = bert_scorer.get_embeddings(output0)
            sim_matrix = bert_scorer.cosine_similarity(ref_emb, cand_emb)

            # 修正Precision和Recall计算公式
            recall = sim_matrix.max(dim=1).values.mean().item()
            precision = sim_matrix.max(dim=0).values.mean().item()

            # 保存结果
            data['答案'].append(a)
            data['输出'].append(output0)
            data['rouge_1_score'].append(rouge_1_score0)
            data['rouge_2_score'].append(rouge_2_score0)
            data['rouge_l_score'].append(rouge_l_score0)
            data['Precision'].append(precision)
            data['Recall'].append(recall)

    # 将结果保存到CSV文件
    out_df = pd.DataFrame(data)
    out_df.to_csv(out_path, index=False, encoding='utf-8')
    print("数据已保存至:", out_path)

if __name__ == "__main__":
    file_path = ''
    out_file = ''
    extract_info_from_excel(file_path, out_file)
