# Enterprise Carbon Emission Analysis and Knowledge Question-Answering System Based on Large Language Models
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
  - code/report_processing/analyze_prompt.py、QA/prompt.txt: Prompt Engineering for entire system design
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

## Citation
Please cite our paper if you use CHATREPORT in your research.
```bibtex
@misc{ni2023chatreport,
      title={CHATREPORT: Democratizing Sustainability Disclosure Analysis through LLM-based Tools}, 
      author={Jingwei Ni and Julia Bingler and Chiara Colesanti-Senni and Mathias Kraus and Glen Gostlow and Tobias Schimanski and Dominik Stammbach and Saeid Ashraf Vaghefi and Qian Wang and Nicolas Webersinke and Tobias Wekhof and Tingyu Yu and Markus Leippold},
      year={2023},
      eprint={2307.15770},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

