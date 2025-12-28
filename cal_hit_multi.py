#%%
from util.retriever_utils import load_passages, validate, save_results
import pickle
import os
import csv 

#%%
def load_data_with_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def process_and_save_retrieval_results(top_docs, dataset_name, questions, question_answers, all_passages, num_threads, match_type, output_dir, output_no_text=False):
    recall_outfile = os.path.join(output_dir, 'recall_at_k.csv')
    result_outfile = os.path.join(output_dir, 'results.json')
    
    questions_doc_hits = validate(
        dataset_name,
        all_passages,
        question_answers,
        top_docs,
        num_threads,
        match_type,
        recall_outfile,
        use_wandb=False
    )
    
    save_results(
        all_passages,
        questions,
        question_answers,
        top_docs,
        questions_doc_hits,
        result_outfile,
        output_no_text=output_no_text
    )
    
    return questions_doc_hits


#%%
if __name__=='__main__':
    
    dataset_name = 'webq'
    num_threads = 1
    output_no_text = False
    ctx_file = './corpus/wiki_webq_corpus.tsv'


    match_type = 'string'
    input_file_path = './datas/webq-test.csv'
    with open(input_file_path,'r', encoding="utf-8") as file:
        query_data = csv.reader(file, delimiter='\t')
        questions, question_answers = zip(*[(item[0], eval(item[1])) for item in query_data])
        questions = questions
        question_answers = question_answers
    
    all_passages = load_passages(ctx_file)

    output_dir = './output/webq-test-result'

    
    top_docs_pkl_path = './output/result_str.pkl'

    top_docs = load_data_with_pickle(top_docs_pkl_path)
    
    os.makedirs(output_dir, exist_ok=True)
    questions_doc_hits = process_and_save_retrieval_results(
        top_docs,
        dataset_name,
        questions,
        question_answers,
        all_passages,
        num_threads,
        match_type,
        output_dir,
        output_no_text=output_no_text
    )

    print('Validation End!')
