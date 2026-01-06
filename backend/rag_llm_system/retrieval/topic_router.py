import numpy as np
from sklearn.cluster import MiniBatchKMeans


class TopicRouter:
    """Embedding-based topic detection."""

    def __init__(self, n_topics: int = 50):
        self.n_topics = n_topics
        self.clusterer = MiniBatchKMeans(n_clusters=n_topics, random_state=42)
        self.fitted = False

    def fit(self, embeddings: np.ndarray):
        """Fit on document embeddings."""
        self.clusterer.fit(embeddings)
        self.fitted = True

    def predict_topic(self, query_embedding: np.ndarray) -> int:
        """Predict topic cluster."""
        if not self.fitted:
            return 0
        return int(self.clusterer.predict(query_embedding.reshape(1, -1))[0])
