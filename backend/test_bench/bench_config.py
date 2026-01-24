from pathlib import Path


class Config:
    DATASET_DIR = Path(__file__).parent / "data" / "datasets" / "wikipedia_general"


config = Config()

if __name__ == "__main__":
    config = Config()
    print(config.DATASET_DIR)
