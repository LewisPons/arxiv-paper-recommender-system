"""
This script is for testing in the CLI the paper Recommender
    Example:    
    python3 ./src/models/paperRecommender.py --modelname GrammarGuru
"""
import argparse
from random import choice
from utils.constants import user_requests_tests, TEST_INPUTS
from utils.mlutilities import get_recomendations_metadata,load_arxiv_parquet, load_model, load_dict, load_sparse_matrix, gensim_tokenizer
import time
import sys 


if __name__ == "__main__":

    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='ArXiv Paper Recommender CLI')
    parser.add_argument('--query', default=None, type=str, help='User query')
    parser.add_argument('--n', default=3, type=int, help='Number of Recommendations to retieve')
    parser.add_argument('--modelname', choices=["SemanticSherlock", "LanguageLiberator", "TextualTango", "GrammarGuru"],  default=None, type=str, help='Model Name')

    args = parser.parse_args()
    

    query = args.query
    model_name = args.modelname
    n = args.n
    
    if args.query is None:
        query = choice(TEST_INPUTS)
    
    query = gensim_tokenizer(query)
    print(query)
    
    if model_name is None:
        raise Exception('Please Select a model name to use: ["SemanticSherlock", "LanguageLiberator", "TextualTango", "GrammarGuru"]')
        
    start = time.time()
    parent_folder = f"models/{model_name}"
    
    parquet_file = f"{parent_folder}/data/{model_name}.parquet.gzip"
    dictionary = f"{parent_folder}/dictionaries/{model_name}.dict"
    model = f"{parent_folder}/tdidf/{model_name}.model"
    sparse_matrix = f"{parent_folder}/similarities_matrix/{model_name}"
    

    df = load_arxiv_parquet(parquet_file)
    dict_corpus = load_dict(dictionary)
    similarities = load_sparse_matrix(sparse_matrix)
    tfidf_model = load_model(model)
    print("------ working")
    
    results_df = get_recomendations_metadata(query=query, df=df, n=n, dictionary=dict_corpus, index=similarities, tfidf_model=tfidf_model)

    results = list(zip(
        results_df['title'].to_list(), 
        results_df['authors'].to_list(),  
        results_df['categories'].to_list(),
        results_df['abstract'].to_list()
        )
    )

    for abstract in results:
        print(f"User Request ---- : \n {query}")
        time.sleep(0.3)
        print(f"--------------------------")
        print(f"Title: {abstract[0]}")
        time.sleep(0.3)
        print(f"Author: {abstract[1]}")
        time.sleep(0.3)
        print(f"Categories: {abstract[2]}\n")
        time.sleep(0.3)
        print(f"Abstract: {abstract[3]}\n")
        print(f"--------------------------")
        time.sleep(1)
    
    end = time.time()
    total_time = end - start
    print(f"-------------- Execution Time: {total_time}")