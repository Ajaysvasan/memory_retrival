import numpy as np
from typing import List, Tuple
from pathlib import Path
from sklearn.cluster import KMeans
from logger import Logger
from config import Config


class TopicModeler:
    """Topic modeling using K-means clustering"""

    def __init__(self):
        self.logger = Logger().get_logger("TopicModeler")
        self.centroids = None
        self.num_topics = Config.NUM_TOPICS

    def train(self, embeddings: np.ndarray) -> np.ndarray:
        """Train topic model on embeddings"""

        if len(embeddings) == 0:
            raise ValueError("Cannot train topic model on empty embeddings")

        if len(embeddings) < Config.TOPIC_MIN_SAMPLES:
            raise ValueError(
                f"Insufficient samples for topic modeling: "
                f"{len(embeddings)} < {Config.TOPIC_MIN_SAMPLES}"
            )

        # Adjust number of topics if needed
        actual_topics = min(self.num_topics, len(embeddings))

        if actual_topics < self.num_topics:
            self.logger.warning(
                f"Reducing topics from {self.num_topics} to {actual_topics} "
                f"due to sample size"
            )

        self.logger.info(f"Training topic model with {actual_topics} topics...")

        # K-means clustering
        kmeans = KMeans(
            n_clusters=actual_topics, random_state=42, n_init=10, max_iter=300
        )

        kmeans.fit(embeddings)
        self.centroids = kmeans.cluster_centers_

        self.logger.info(f"Topic model trained: {self.centroids.shape}")
        return self.centroids

    def assign_topics(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign topics to embeddings using argmax cosine similarity"""

        if self.centroids is None:
            raise RuntimeError("Topic model not trained")

        # Compute cosine similarity: topic(e) = argmax_j cos(e, μ_j)
        similarities = np.dot(embeddings, self.centroids.T)
        topic_assignments = np.argmax(similarities, axis=1)

        self.logger.debug(f"Assigned topics to {len(embeddings)} embeddings")
        return topic_assignments

    def get_topic_distribution(self, topic_assignments: np.ndarray) -> dict:
        """Get distribution of chunks across topics"""
        unique, counts = np.unique(topic_assignments, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))

        self.logger.info("Topic distribution:")
        for topic_id, count in sorted(distribution.items()):
            self.logger.info(f"  Topic {topic_id}: {count} chunks")

        return distribution

    def save_centroids(self, filepath: Path):
        """Save topic centroids to disk"""
        if self.centroids is None:
            raise RuntimeError("No centroids to save")

        np.save(filepath, self.centroids)
        self.logger.info(f"Saved topic centroids to {filepath}")

    def load_centroids(self, filepath: Path):
        """Load topic centroids from disk"""
        if not filepath.exists():
            raise FileNotFoundError(f"Centroids file not found: {filepath}")

        self.centroids = np.load(filepath)
        self.logger.info(
            f"Loaded topic centroids from {filepath}: shape={self.centroids.shape}"
        )
