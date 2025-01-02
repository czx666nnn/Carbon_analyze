import pandas as pd
import re

def load_table(file_path):
    df = pd.read_excel(file_path)
    if not {'question', 'pred_sql', 'gold_sql'}.issubset(df.columns):
        raise ValueError("表格缺少必要的列：'question', 'pred_sql', 'gold_sql'")
    return df

def extract_aggregates(sql):

    pattern = re.compile(r'(avg|min|max|sum|count)\s?\((\w+)\)', re.IGNORECASE)
    matches = pattern.findall(sql)
    return [(agg.lower(), col.lower()) for agg, col in matches]

def bag_match_evaluation(df):

    pred_bags = []
    gold_bags = []

    for _, row in df.iterrows():
        pred_sql = row['pred_sql']
        gold_sql = row['gold_sql']

        pred_bag = extract_aggregates(pred_sql)
        gold_bag = extract_aggregates(gold_sql)

        pred_bags.append(set(pred_bag))
        gold_bags.append(set(gold_bag))

    match_count = sum(1 for pred_bag, gold_bag in zip(pred_bags, gold_bags) if pred_bag == gold_bag)
    accuracy = match_count / len(df)
    return accuracy, match_count

def exact_match_evaluation(df):

    exact_matches = df['pred_sql'] == df['gold_sql']
    accuracy = exact_matches.mean()
    return accuracy, exact_matches

def partial_match_evaluation(df):

    clauses = ['SELECT', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING']
    partial_results = {clause: [] for clause in clauses}

    for _, row in df.iterrows():
        pred_sql = row['pred_sql'].upper()
        gold_sql = row['gold_sql'].upper()
        for clause in clauses:
            gold_contains = clause in gold_sql
            pred_contains = clause in pred_sql
            partial_results[clause].append(int(gold_contains == pred_contains))

    partial_accuracies = {clause: sum(matches) / len(matches) for clause, matches in partial_results.items()}
    return partial_accuracies

def evaluate(file_path):

    df = load_table(file_path)
    exact_accuracy, exact_results = exact_match_evaluation(df)
    print(f"{exact_accuracy:.2%}")

    partial_accuracies = partial_match_evaluation(df)
    for clause, accuracy in partial_accuracies.items():
        print(f"  {clause}: {accuracy:.2%}")

    bag_accuracy, match_count = bag_match_evaluation(df)
    print(f" {bag_accuracy:.2%} ({match_count}/{len(df)})")

if __name__ == "__main__":
    file_path = ""
    evaluate(file_path)
