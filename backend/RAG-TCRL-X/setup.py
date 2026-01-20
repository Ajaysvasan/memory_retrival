from setuptools import setup, find_packages

setup(
    name="rag-tcrl-x",
    version="1.0.0",
    description="Topic-Conditioned RAG with RL Control",
    author="RAG-TCRL-X Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "rag-tcrl-x=main:main",
        ],
    },
)
