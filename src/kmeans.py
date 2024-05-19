import numpy as np
import pandas as pd
import random
from typing import List, Tuple

class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    def fit(self, data: np.ndarray):
        self._initialize_centroids(data)
        for _ in range(self.max_iter):
            clusters = self._create_clusters(data)
            new_centroids = self._calculate_new_centroids(data, clusters)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def predict(self, data: np.ndarray) -> List[int]:
        clusters = self._create_clusters(data)
        return self._get_labels_from_clusters(clusters, data)

    def _initialize_centroids(self, data: np.ndarray):
        indices = random.sample(range(data.shape[0]), self.n_clusters)
        self.centroids = data[indices]

    def _create_clusters(self, data: np.ndarray) -> List[List[int]]:
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, point in enumerate(data):
            closest_centroid = self._closest_centroid(point)
            clusters[closest_centroid].append(idx)
        return clusters

    def _closest_centroid(self, point: np.ndarray) -> int:
        distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
        return distances.index(min(distances))

    def _calculate_new_centroids(self, data: np.ndarray, clusters: List[List[int]]) -> np.ndarray:
        new_centroids = np.zeros((self.n_clusters, data.shape[1]))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(data[cluster], axis=0)
            new_centroids[cluster_idx] = cluster_mean
        return new_centroids

    def _get_labels_from_clusters(self, clusters: List[List[int]], data: np.ndarray) -> List[int]:
        labels = np.empty(data.shape[0], dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for data_idx in cluster:
                labels[data_idx] = cluster_idx
        return labels


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return data, labels

def save_results(labels: List[int], output_file: str):
    pd.DataFrame(labels, columns=["Cluster"]).to_csv(output_file, index=False)

def main(input_file: str, k: int, output_file: str):
    data, _ = load_data(input_file)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    save_results(labels, output_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-Means Clustering")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("k", type=int, help="Number of clusters")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file")

    args = parser.parse_args()

    main(args.input_file, args.k, args.output_file)