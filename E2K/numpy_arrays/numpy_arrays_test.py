import numpy as np
import numexpr as ne

# Define parameters for experiments
K = 3  # Number of clusters
dimensionality = 2  # Dimensionality of data points (e.g., 2D)
dataset_size = 1000  # Size of the dataset (number of data points)


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids

def assign_clusters(data, centroids):
    distances = np.zeros((data.shape[0], len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.sum((data - centroid) ** 2, axis=1)
    cluster_indices = np.argmin(distances, axis=1)
    clusters = [data[cluster_indices == i] for i in range(len(centroids))]
    return clusters

def update_centroids(clusters):
    centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
    return centroids

def kmeans_numpy(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Example usage
data_points = np.array([(1, 2), (5, 8), (3, 6), (8, 1), (10, 5)])
k_value = 2
clusters, centroids = kmeans_numpy(data_points, k_value)
print("Clusters:", clusters)
print("Centroids:", centroids)
