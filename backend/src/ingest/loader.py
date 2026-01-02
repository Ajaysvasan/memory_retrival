import os


def load_documents(directory: str) -> list:
    documents = []
    for file_name in os.listdir(directory):
        path = os.path.join(directory, file_name)
        print(path)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        documents.append({"text": text, "source": path})
    return documents
