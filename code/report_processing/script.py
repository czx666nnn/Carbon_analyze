import os
import json
import argparse
from document import Report
from reader import Reader
from user_qa import UserQA
import asyncio
import excel_analyze
TOP_K = 10
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, default=None)
    parser.add_argument("--pdf_url", type=str, default=None)
    parser.add_argument("--basic_info_dir", type=str, default='data/basic_info')
    parser.add_argument("--llm_name", type=str, default='')
    parser.add_argument("--answers_dir", type=str, default='data/answers')
    parser.add_argument("--assessment_dir", type=str, default='data/assessment')
    parser.add_argument("--vector_db_dir", type=str, default='data/vector_db')
    parser.add_argument("--retrieved_chunks_dir", type=str, default='data/retrieved_chunks')
    parser.add_argument("--user_qa_dir", type=str, default='data/user_qa')
    parser.add_argument("--user_question", type=str, default='')
    parser.add_argument("--answer_length", type=int, default=150)
    parser.add_argument("--detail", action='store_true', default=False)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    if args.pdf_path:
        report_name = os.path.basename(args.pdf_path)
    else:
        assert (args.pdf_url is not None)
        report_name = args.pdf_url.split('/')[-1]
    assert report_name.endswith('.pdf')
    report_name = report_name.replace('.pdf', '')
    # 创建目录
    directories = [
        args.basic_info_dir, args.answers_dir, args.assessment_dir,
        args.vector_db_dir, args.retrieved_chunks_dir, args.user_qa_dir, "data/pdf/"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    report = Report(
        path=args.pdf_path,
        url=args.pdf_url,
        store_path=os.path.join("data/pdf/", report_name + '.pdf'),
        db_path=os.path.join(args.vector_db_dir, report_name),
        retrieved_chunks_path=os.path.join(args.retrieved_chunks_dir, report_name)
    )
    # 
    # json_path = 'output_tables.json'  # 输出的 JSON 文件路径
    # excel_analyze.extract_tables_from_pdf(args.pdf_path, json_path)

    # # 数据库连接配置
    # db_config = {
    #     'host': 'localhost',
    #     'user': 'root',
    #     'password': 'your_password',
    #     'database': 'your_database_name'
    # }
    # # 调用函数将数据存入数据库
    # excel_analyze.store_tables_in_db(json_path, db_config)

    if args.user_question == '':
        try:
            reader = Reader(llm_name=args.llm_name, answer_length=str(args.answer_length))
            result_qa = asyncio.run(reader.qa_with_chat(report_list=[report]))
            result_analysis = asyncio.run(reader.analyze_with_chat(report_list=[report]))
        except Exception as e:
            if "This model's maximum context length is" in str(e):
                import pdb
                pdb.set_trace()
                report = Report(
                    path=os.path.join("data/pdf/", args.pdf_path.split('/')[-1]),
                    store_path=None,
                    top_k=TOP_K - 5,
                    db_path=os.path.join(args.vector_db_dir, report_name),
                    retrieved_chunks_path=os.path.join(args.retrieved_chunks_dir, report_name),
                )
                reader = Reader(llm_name=args.llm_name, answer_length=str(args.answer_length))
                result_qa = asyncio.run(reader.qa_with_chat(report_list=[report]))
                result_analysis = asyncio.run(reader.analyze_with_chat(report_list=[report]))
        html_path_qa = report_name + '_' + args.llm_name + '_qa.html'
        html_path_analysis = report_name + '_' + args.llm_name + '_analysis.html'
        with open(html_path_qa, 'w') as f:
            f.write(result_qa[0])
        with open(html_path_analysis, 'w') as f:
            f.write(result_analysis[0])
        with open(os.path.join(args.basic_info_dir, report_name + '_' + args.llm_name + '.json'), 'w') as f:
            json.dump(reader.basic_info_answers[0], f)
        with open(os.path.join(args.answers_dir, report_name + '_' + args.llm_name + '.json'), 'w') as f:
            json.dump(reader.answers[0], f)
        with open(os.path.join(args.assessment_dir, report_name + '_' + args.llm_name + '.json'), 'w') as f:
            json.dump(reader.assessment_results[0], f)
    else:
        qa = UserQA(llm_name=args.llm_name)
        answer, _ = qa.user_qa(
            args.user_question,
            report,
            basic_info_path=os.path.join(args.basic_info_dir, report_name + '_' + args.llm_name + '.json'),
            answer_length=args.answer_length,
        )
        with open(os.path.join(args.user_qa_dir, report_name + '_' + args.llm_name + '.jsonl'), 'a') as f:
            qa_json = json.dumps(answer)
            f.write(qa_json + '\n')
        print("Answer dictionary:", answer)
        return answer['ANSWER']

if __name__ == '__main__':
    main()
