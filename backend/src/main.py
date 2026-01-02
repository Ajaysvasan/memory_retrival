import os

from ingest.loader import load_documents

DATASET_DIRECTORY = r"src/dataset/"

if __name__ == "__main__":
    try:
        print(load_documents(DATASET_DIRECTORY))
    except Exception as e:
        print("The following exception occured", e)
