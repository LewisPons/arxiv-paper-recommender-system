"""
    This is a utility script for use in sagemaker
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm

# File paths
json_file_path = "/home/studio-lab-user/arxiv-paper-recommender-system/arxiv-metadata-oai-snapshot.json"
parquet_file_path = "/home/studio-lab-user/arxiv-paper-recommender-system/data/processed/arxiv_papers_raw.parquet.gzip"

# Batch size
batch_size = 10000

# Create the parent directory if it doesn't exist
parent_dir = os.path.dirname(parquet_file_path)
os.makedirs(parent_dir, exist_ok=True)

# Open the JSON file
with open(json_file_path, 'r') as file:
    # Initialize an empty list to store the data
    arxiv_data = []
    processed_count = 0

    # Iterate over each line in the file
    for line in tqdm(file):
        # Load the JSON data from each line and append it to the arxiv_data list
        arxiv_data.append(json.loads(line))

        processed_count += 1

        # Process a batch of data
        if processed_count % batch_size == 0:
            df = pd.DataFrame.from_records(arxiv_data)
            # Convert the batch to parquet and append it to the file
            # df.to_parquet(parquet_file_path, compression='gzip', engine='pyarrow', index=False, append=True)
            # Create a parquet table from your dataframe
            table = pa.Table.from_pandas(df)

            # Write direct to your parquet file
            pq.write_to_dataset(table , root_path=parquet_file_path)
            arxiv_data = []

    # Process the remaining data (if any)
    if arxiv_data:
        df = pd.DataFrame.from_records(arxiv_data)
        # Convert the remaining batch to parquet and append it to the file
        # df.to_parquet(parquet_file_path, compression='gzip', engine='pyarrow', index=False, append=True)
        pq.write_to_dataset(parquet_file_path , root_path=parquet_file_path)
