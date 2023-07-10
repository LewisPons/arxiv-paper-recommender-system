import pandas as pd
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
from gensim.models import TfidfModel
from gensim.parsing import strip_tags, strip_numeric, \
    strip_multiple_whitespaces, strip_punctuation, \
    remove_stopwords, preprocess_string

import argparse
from typing import List
from re import sub
from random import choice
from utils.constants import user_requests_tests
import time
from functools import cache


if __name__ == "__main__":
    """
        Example: 
        python script.py --samples 3000 --corpus_dictionary_path "30Ktokens.dict" --arxiv_datasr_path "/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_papers.parquet.gzip" --save_dict --query "your query here"

    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='ArXiv Paper Recommender CLI')
    parser.add_argument('--query', default=None, type=str, help='User query')
    parser.add_argument('--model', default=None, type=str, help='User query')
    parser.add_argument('--discipline', default=None, type=str, help='User query')
    args = parser.parse_args()

    query = args.query
    if args.query is None:
        disciplines = ["Math", "Statistics", "Electrical Engineering", "QuantitativeBiology", "Economics"]
        discipline = choice(disciplines)
        query = choice(user_requests_tests[discipline])
        
    start = time.time()

    @cache
    def load_arxiv_parquet(path: str = "/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_papers.parquet.gzip"):
        df = pd.read_parquet(path)
        return df
        
    @cache  
    def load_dict(path: str = "/Users/luis.morales/Desktop/arxiv-paper-recommender/models/dictionaries/LanguageLiberator.dict"):
        dict_corpus = Dictionary.load(path)
        return dict_corpus
    
    @cache
    def load_model(path: str = "/Users/luis.morales/Desktop/arxiv-paper-recommender/models/tfidf/SemanticSherlock.model"):
        tfidf_model = TfidfModel.load(path)
        return tfidf_model
    
    @cache
    def load_sparse_matrix(path: str = "/Users/luis.morales/Desktop/arxiv-paper-recommender/models/similarities_matrix/LanguageLiberatorSimilarities/LanguageLiberator"):
        similarities = SparseMatrixSimilarity.load(path)
        return similarities
    
    df = load_arxiv_parquet()
    dict_corpus = load_dict()
    similarities = load_sparse_matrix()
    tfidf_model = load_model()
    
    results_df = get_recomendations_metadata(query=query, df=df, n=3, dictionary=dict_corpus, index=similarities, tfidf_model=tfidf_model)

    results = list(zip(
        results_df['title'].to_list(), 
        results_df['authors'].to_list(),  
        results_df['categories'].to_list(),
        results_df['abstract'].to_list()
        )
    )

    for abstract in results:
        print(f"User Request ---- : \n {query}")
        print(f"User Request Discipline: {discipline}")
        print(f"--------------------------")
        print(f"Title: {abstract[0]}")
        print(f"Author: {abstract[1]}")
        print(f"Categories: {abstract[2]}\n")
        print(f"Abstract: {abstract[3]}\n")
        print(f"--------------------------")
    
    end = time.time()
    total_time = end - start
    print(f"-------------- Execution Time: {total_time}")