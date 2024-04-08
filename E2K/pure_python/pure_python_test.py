import random
import math

# Define parameters for experiments
K = 3  # Number of clusters
dimensionality = 2  # Dimensionality of data points (2D)
dataset_size = 1000  # Size of the dataset (number of data points)


import random
import math

def euclidean_distance(point1, point2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))




def euclidean_distance(point1, point2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

def initialize_centroids(data, k):
    centroids = random.sample(data, k)
    return centroids

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters

def update_centroids(clusters):
    centroids = [tuple(map(lambda x: sum(x) / len(x), zip(*cluster))) for cluster in clusters]
    return centroids

def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters, centroids

# Example usage
data_points = [(1, 2), (5, 8), (3, 6), (8, 1), (10, 5)]
k_value = 2
clusters, centroids = kmeans(data_points, k_value)
print("Clusters:", clusters)
print("Centroids:", centroids)
