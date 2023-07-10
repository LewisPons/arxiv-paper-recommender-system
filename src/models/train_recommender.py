import pandas as pd
from gensim.similarities import SparseMatrixSimilarity
import argparse
import logging
import time

from utils.utilities import read_yaml_config, validate_and_create_subfolders
from utils.mlutilities import *

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


model_configurations = read_yaml_config("/Users/luis.morales/Desktop/arxiv-paper-recommender/src/models/configs.yaml")


if __name__ == "__main__":
    """
        Example: 
        python3 ./src/models/train_recommender.py --modelsize Medium

    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='ArXiv Paper Recommender CLI')
    parser.add_argument('--modelsize',choices=["Large", "SubLarge", "Medium", "Small"],  default=None, type=str, help='Model Size')

    args = parser.parse_args()
    model_size = args.modelsize
    start = time.time()
    
    
    if model_size is None:
        raise Exception("The `modelsize` flag was no passed to the CLI.")
    
    
    model_config = model_configurations["GensimConfig"][model_size]
    model_name = model_configurations["GensimConfig"][model_size]["ModelName"]
    dataset_fraq_split = model_configurations["GensimConfig"][model_size]["DataSetFracSplit"]
    random_seed = model_configurations["GensimConfig"][model_size]["RandomSeedSplit"]
    logging.info(f"Started training of {model_name} Model.")
    

    validate_and_create_subfolders(
        model_name=model_name
    )
    logging.info(f"Model Folder `{model_name}` was created successfully.")
    
    
    if dataset_fraq_split is None:
        df = pd.read_parquet("/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_papers.parquet.gzip")
        logging.info(f"The full text Corpus was readed.")
        
    else :
        df = pd.read_parquet("/Users/luis.morales/Desktop/arxiv-paper-recommender/data/processed/reduced_arxiv_papers.parquet.gzip") \
            .sample(frac=dataset_fraq_split, random_state=random_seed) \
            .reset_index(drop=True)
        logging.info(f"A random split of {dataset_fraq_split}% was applied on the Text Corpus ")
    logging.info(f"Dimensions of the dataset: {df.shape}")
    
    df.to_parquet(f"/Users/luis.morales/Desktop/arxiv-paper-recommender/models/data/{model_name}.parquet.gzip", compression='gzip')
    logging.info(f"The Dataset used for this training was successfully saved in: `/Users/luis.morales/Desktop/arxiv-paper-recommender/models/data/{model_name}.parquet.gzip`.")
    
    

    corpus = df['cleaned_abstracts'].to_list()
    tokenized_corpus = gensim_tokenizer(corpus)
    logging.info(f"Dictionary Learned on the {model_name} corpus dataset.")
    
    
    dictionary = get_gensim_dictionary(tokenized_docs=tokenized_corpus, dict_name=model_name, save_dict=True)
    logging.info("Dictionary Saved Locally.")
    
    
    BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_corpus]
    tfidf_model = TfidfModel(BoW_corpus)
    logging.info(f"TD-IDF {model_name} Model was successfully trained.")
    
    
    tfidf_model.save(f"/Users/luis.morales/Desktop/arxiv-paper-recommender/models/tfidf/{model_name}.model")
    logging.info(f"Model: {model_name} was successfully saved.")


    index = SparseMatrixSimilarity(tfidf_model[BoW_corpus], num_features=len(dictionary))
    logging.info(f"The Similarities Sparse Matrix was successfully created.")
    index.save(f"/Users/luis.morales/Desktop/arxiv-paper-recommender/models/similarities_matrix/{model_name}")
    logging.info(f"The Similarities Matrix was successfully saved for the model: {model_name}.")
    
    end = time.time()
    total_time = end - start
    logging.info(f"Full Training of {model_size} model took {total_time} secs.")
    logging.info(f"The {model_name} Model was successfully trained! yei :)")
    
    