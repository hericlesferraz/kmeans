import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data(file_path: str) -> np.ndarray:
    df = pd.read_csv(file_path)
    data = df.iloc[:, :-1].values
    return data

def load_results(file_path: str) -> np.ndarray:
    df = pd.read_csv(file_path)
    return df["Cluster"].values

def visualize_clusters(data_file: str, results_file: str):
    data = load_data(data_file)
    labels = load_results(results_file)
    
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = reduced_data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
    
    plt.legend()
    plt.title('K-Means Clustering Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize K-Means Clustering Results")
    parser.add_argument("data_file", type=str, help="Path to the input CSV file with original data")
    parser.add_argument("results_file", type=str, help="Path to the CSV file with clustering results")

    args = parser.parse_args()

    visualize_clusters(args.data_file, args.results_file)
