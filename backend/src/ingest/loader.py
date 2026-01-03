import re
import unicodedata
from pathlib import Path
from typing import Dict, List


def text_normalization(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\a")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(directory: str, extentions: tuple = (".txt", ".md")) -> List[Dict]:
    documents = []
    directory_path = Path(directory)
    if not directory_path.exists():
        raise Exception("The directory is not found")
    for file_path in directory_path.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() not in extentions:
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_text = file.read()

            normalize_text = text_normalization(raw_text)
            documents.append({"source": file_path.name, "text": normalize_text})
        except Exception as e:
            raise Exception(f"The following exception has occured {e}")
    return documents
