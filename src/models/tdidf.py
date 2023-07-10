import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import ColumnSelector
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.utilities import *
import sys
from pprint import pprint

CONFIG_FILE_PATH = "/Users/luis.morales/Desktop/arxiv-paper-recommender/models/configs.yaml"
config = read_yaml_config(CONFIG_FILE_PATH)
pprint(config)

@execution_time
def train_tfidf():
    df = pd.read_parquet("/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/arxiv_papers.parquet.gzip") \
        .sample(500000) \
        .reset_index(drop=True)
        
        
    vectorizer = TfidfVectorizer(**config["models"]["tfidf"]["tfidf_deffault"])
    pprint(config["models"]["tfidf"]["tfidf_deffault"])
    sys.exit()
    
    vectors = vectorizer.fit_transform(df['cleaned_abstracts'])

    tfidf_df = pd.DataFrame(vectors.toarray(), columns=[i for i in vectorizer.get_feature_names_out()])


    tfidf_df.to_parquet("/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_tfidf.parquet.gzip")
    
train_tfidf()

