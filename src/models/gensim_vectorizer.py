import pandas as pd
from gensim import corpora
from gensim import similarities
from gensim.models import TfidfModel
from gensim.parsing import strip_tags, strip_numeric, \
    strip_multiple_whitespaces, stem_text, strip_punctuation, \
    remove_stopwords, preprocess_string
import re

from typing import List
from utils.constants import TEST_INPUTS
import argparse
from random import choice

import sys



SAMPLES = 3000
CORPUS_DICTIONARY_PATH="30Ktokens"
ARXIV_DATASR_PATH = "/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_papers.parquet.gzip"
SAVE_DICT = False
QUERY = ""

transform_to_lower = lambda s: s.lower()
remove_single_char = lambda s: re.sub(r'\s+\w{1}\s+', '', s)

cleaning_filters = [
    strip_tags,
    strip_numeric,
    strip_punctuation, 
    strip_multiple_whitespaces, 
    transform_to_lower,
    remove_stopwords,
    remove_single_char
]

def gensim_tokenizer(docs: List[str]):
    """
    Tokenizes a list of strings using a series of cleaning filters.

    Args:
        docs (List[str]): A list of strings to be tokenized.

    Returns:
        List[List[str]]: A list of tokenized documents, where each document is represented as a list of tokens.
    """
    tokenized_docs = list()
    for doc in docs:
        processed_words = preprocess_string(doc, cleaning_filters)
        tokenized_docs.append(processed_words)
    
    return tokenized_docs


def cleaning_pipe(document):
    """
    Applies a series of cleaning steps to a document.

    Args:
        document (str): The document to be cleaned.

    Returns:
        list: A list of processed words after applying the cleaning filters.
    """
    # Invoking gensim.parsing.preprocess_string method with set of filters
    processed_words = preprocess_string(document, cleaning_filters)
    return processed_words


def get_gensim_dictionary(tokenized_docs: List[str], dict_name: str = "corpus", save_dict: bool = False):
    """
        Create dictionary of words in preprocessed corpus and saves the dict object
    """
    dictionary = corpora.Dictionary(tokenized_docs)
    if save_dict:    
        parent_folder = "/Users/luis.morales/Desktop/arxiv-paper-recommender/models/nlp_dictionaries"
        dictionary.save(f'{parent_folder}/{dict_name}.dict')
    return dictionary


def get_closest_n(query: str, n: int):
    '''
    Retrieves the top matching documents as per cosine similarity
    between the TF-IDF vector of the query and all documents.

    Args:
        query (str): The query string to find matching documents.
        n (int): The number of closest documents to retrieve.

    Returns:
        numpy.ndarray: An array of indices representing the top matching documents.
    '''
    # Clean the query document using cleaning_pipe function
    query_document = cleaning_pipe(query)

    # Convert the query document to bag-of-words representation
    query_bow = dictionary.doc2bow(query_document)

    # Calculate similarity scores between the query and all documents using TF-IDF model
    sims = index[tfidf_model[query_bow]]

    # Get the indices of the top n closest documents based on similarity scores
    top_idx = sims.argsort()[-1 * n:][::-1]

    return top_idx


def get_recomendations_metadata(query: str, df: pd.DataFrame, n: int):
    '''
    Retrieves metadata recommendations based on a query using cosine similarity.

    Args:
        query (str): The query string for which recommendations are sought.
        n (int): The number of recommendations to retrieve.
        df (pd.DataFrame): The DataFrame containing metadata information.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended metadata, reset with a new index.
    '''
    # Get the indices of the closest matching documents based on the query
    recommendations_idxs = get_closest_n(query, n)
    
    # Retrieve the recommended metadata rows from the DataFrame based on the indices
    recommendations_metadata = df.iloc[recommendations_idxs]
    
    # Reset the index of the recommended metadata DataFrame
    recommendations_metadata = recommendations_metadata.reset_index(drop=True)
    
    return recommendations_metadata

if __name__ == "__main__":
    """
        Example: 
        python script.py --samples 3000 --corpus_dictionary_path "30Ktokens.dict" --arxiv_datasr_path "/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_papers.parquet.gzip" --save_dict --query "your query here"

    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='ArXiv Paper Recommender CLI')
    parser.add_argument('--samples', default=30000, type=int, help='Number of samples to consider')
    parser.add_argument('--corpus_dictionary_path', default=None ,type=str, help='Path to the corpus dictionary')
    parser.add_argument('--save_dict', default=False, help='Flag to save the dictionary')
    parser.add_argument('--arxiv_dataset_path',
                        default="/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_papers.parquet.gzip",
                        type=str, help='Path to the ARXIV parquet source')
    parser.add_argument('--query', default=None, type=str, help='User query')
    args = parser.parse_args()

    num_samples = args.samples
    corpus_dictionary_path = args.corpus_dictionary_path
    arxiv_dataset_path = args.arxiv_dataset_path
    save_dict = args.save_dict
    query = args.query

    print("Parameters:")
    print(f"num_samples: {num_samples}, type: {type(num_samples)}")
    print(f"corpus_dictionary_path: {corpus_dictionary_path}, type: {type(corpus_dictionary_path)}")
    print(f"arxiv_dataset_path: {arxiv_dataset_path}, type: {type(arxiv_dataset_path)}")
    print(f"save_dict: {save_dict}, type: {type(save_dict)}")
    print(f"query: {query}, type: {type(query)}")
    

    if num_samples is None:
        df = pd.read_parquet(arxiv_dataset_path)
    df = pd.read_parquet(arxiv_dataset_path).sample(num_samples).reset_index(drop=True)
    

    corpus = df['cleaned_abstracts'].to_list()
    tokenized_corpus = gensim_tokenizer(corpus)

    dictionary = get_gensim_dictionary(
        tokenized_docs=tokenized_corpus, 
        dict_name=corpus_dictionary_path, 
        save_dict=save_dict
        )

    BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_corpus]

    tfidf_model = TfidfModel(BoW_corpus)

    index = similarities.SparseMatrixSimilarity(tfidf_model[BoW_corpus], num_features=len(dictionary))

    if query is None:
        query = choice(TEST_INPUTS)

    results_df = get_recomendations_metadata(query=query, df=df, n=3)


    for abstract in list(zip(results_df['abstract'].to_list(), results_df['title'].to_list())):
        print(f"User Request ---- : \n {query}")
        print(f"User Request ---- : \n ")
        print(f"Title: {abstract[1]}")
        print(f"Abstract: {abstract[0]}\n")
        print(f"--------------------------")