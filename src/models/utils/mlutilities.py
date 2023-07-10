import pandas as pd
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
from gensim.models import TfidfModel
from gensim.parsing import strip_tags, strip_numeric, \
    strip_multiple_whitespaces, stem_text, strip_punctuation, \
    remove_stopwords, preprocess_string

from re import sub
from typing import List
from functools import cache


transform_to_lower = lambda s: s.lower()
remove_single_char = lambda s: sub(r'\s+\w{1}\s+', '', s)

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


def get_closest_n(dictionary: Dictionary, index: SparseMatrixSimilarity, tfidf_model : TfidfModel, query: str, n: int):
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


def get_recomendations_metadata(query: str, df: pd.DataFrame, n: int, 
                                dictionary: Dictionary, index: SparseMatrixSimilarity, 
                                tfidf_model : TfidfModel):
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
    recommendations_idxs = get_closest_n(dictionary, index, tfidf_model, query, n)
    
    # Retrieve the recommended metadata rows from the DataFrame based on the indices
    recommendations_metadata = df.iloc[recommendations_idxs]
    
    # Reset the index of the recommended metadata DataFrame
    recommendations_metadata = recommendations_metadata.reset_index(drop=True)
    
    return recommendations_metadata

@cache
def load_arxiv_parquet(path: str):
    df = pd.read_parquet(path)
    return df
    
@cache  
def load_dict(path: str):
    dict_corpus = Dictionary.load(path)
    return dict_corpus

@cache
def load_model(path: str ):
    tfidf_model = TfidfModel.load(path)
    return tfidf_model

@cache
def load_sparse_matrix(path: str):
    similarities = SparseMatrixSimilarity.load(path)
    return similarities