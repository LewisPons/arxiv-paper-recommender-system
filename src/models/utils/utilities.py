import yaml
import time
from typing import Dict, Union
import pandas as pd
import spacy
import os


def read_yaml_config(file_path: str) -> Dict:
    """
    Reads a YAML configuration file and returns the loaded configuration as a dictionary.

    Args:
        file_path (str): The path to the YAML configuration file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def validate_and_create_subfolders(
    model_name: str, 
    parent_subfolder: str = "/Users/luis.morales/Desktop/arxiv-paper-recommender/models"
):
    model_subfolders = ["data", "dictionaries", "similarities_matrix", "tdidf"]
    
    if not os.path.exists(f"{parent_subfolder}/{model_name}"):
        os.makedirs(f"{parent_subfolder}/{model_name}")
        for msubfolder in model_subfolders:
            os.makedirs(f"{parent_subfolder}/{model_name}/{msubfolder}")
                
                




def execution_time(func):
    """
    Decorator that measures the execution time of a function and prints the elapsed time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_seconds = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_seconds:.4f} seconds.")
        return result
    return wrapper


def cleanData(doc: Union[pd.Series, str], nlp = spacy.load('en_core_web_sm')):
    """
    TODO: Optimize NLP Object to only obtain stopwords, lemmas, and tokenize docs.
    
    Cleans and processes the input documents by performing various text cleaning operations.

    Args:
        doc (pd.Series): The documents to be cleaned, passed in a Series object.
        stemming (bool, optional): Specifies whether stemming should be applied. Defaults to False.

    Returns:
        str: The cleaned and processed document as a single string.
    """
    doc = doc.lower()
    doc = nlp(doc)
    tokens = [tokens.lower_ for tokens in doc]
    tokens = [tokens for tokens in doc if (tokens.is_stop == False)]
    tokens = [tokens for tokens in tokens if (tokens.is_punct == False)]
    final_token = [token.lemma_ for token in tokens]
    
    return " ".join(final_token)


