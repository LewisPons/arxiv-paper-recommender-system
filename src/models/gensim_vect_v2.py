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

transform_to_lower = lambda s: s.lower()
remove_single_char = lambda s: re.sub(r'\s+\w{1}\s+', '', s)

class PaperRecommender:
    def __init__(self, 
                 num_samples=3000, 
                 corpus_dictionary_path="30Ktokens", 
                 arxiv_dataset_path="/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_papers.parquet.gzip", 
                 save_dict=False, 
                 query=""):
        self.num_samples = num_samples
        self.corpus_dictionary_path = corpus_dictionary_path
        self.arxiv_dataset_path = arxiv_dataset_path
        self.save_dict = save_dict
        self.query = query
        self.cleaning_filters = [
            strip_tags,
            strip_numeric,
            strip_punctuation, 
            strip_multiple_whitespaces, 
            transform_to_lower,
            remove_stopwords,
            remove_single_char
        ]
        self.dictionary = None
        self.index = None
        self.tfidf_model = None
        self.df = None

    def gensim_tokenizer(self, docs: List[str]):
        tokenized_docs = list()
        for doc in docs:
            processed_words = preprocess_string(doc, self.cleaning_filters)
            tokenized_docs.append(processed_words)
        return tokenized_docs

    def cleaning_pipe(self, document):
        processed_words = preprocess_string(document, self.cleaning_filters)
        return processed_words

    def get_gensim_dictionary(self, tokenized_docs: List[str], dict_name: str = "corpus"):
        dictionary = corpora.Dictionary(tokenized_docs)
        if self.save_dict:
            parent_folder = "/Users/luis.morales/Desktop/arxiv-paper-recommender/models/nlp_dictionaries"
            dictionary.save(f'{parent_folder}/{dict_name}.dict')
        return dictionary

    def get_closest_n(self, query: str, n: int):
        query_document = self.cleaning_pipe(query)
        query_bow = self.dictionary.doc2bow(query_document)
        sims = self.index[self.tfidf_model[query_bow]]
        top_idx = sims.argsort()[-1 * n:][::-1]
        return top_idx

    def get_recommendations_metadata(self, query: str, n: int):
        recommendations_idxs = self.get_closest_n(query, n)
        recommendations_metadata = self.df.iloc[recommendations_idxs]
        recommendations_metadata = recommendations_metadata.reset_index(drop=True)
        return recommendations_metadata

    def run_recommender(self):
        if self.num_samples is None:
            self.df = pd.read_parquet(self.arxiv_dataset_path)
            
        self.df = pd.read_parquet(self.arxiv_dataset_path).sample(self.num_samples).reset_index(drop=True)
        corpus = self.df['cleaned_abstracts'].to_list()
        
        tokenized_corpus = self.gensim_tokenizer(corpus)
        self.dictionary = self.get_gensim_dictionary(tokenized_docs=tokenized_corpus, dict_name=self.corpus_dictionary_path)
        
        BoW_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_corpus]
        
        self.tfidf_model = TfidfModel(BoW_corpus)
        self.index = similarities.SparseMatrixSimilarity(self.tfidf_model[BoW_corpus], num_features=len(self.dictionary))
        if self.query is None:
            self.query = choice(TEST_INPUTS)
        return self.results
