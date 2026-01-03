import os

from ingest.chunks import *
from ingest.loader import load_documents

DATASET_DIRECTORY = r"src/dataset/"

if __name__ == "__main__":
    try:
        text = list(load_documents(DATASET_DIRECTORY))
        for meta_data in text:
            print(process_document(meta_data))
    except Exception as e:
        print("The following exception occured", e)
