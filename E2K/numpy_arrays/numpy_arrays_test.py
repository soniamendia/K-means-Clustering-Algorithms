import numpy as np
import numexpr as ne
import cProfile
import pstats
import io

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

def profile_kmeans_numpy(k_values, dimensionalities, dataset_sizes):
    # Open log file
    log_file = open('log.txt', 'w')

    for k in k_values:
        for dim in dimensionalities:
            for size in dataset_sizes:
                log_file.write(f"Profiling for K={k}, Dimensionality={dim}, Dataset Size={size}\n")
                data = np.random.uniform(-10, 10, (size, dim))
                profiler = cProfile.Profile()
                profiler.enable()
                kmeans_numpy(data, k)
                profiler.disable()
                s = io.StringIO()
                sortby = 'cumulative'
                ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
                ps.print_stats()
                log_file.write(s.getvalue())

    # Close log file
    log_file.close()

k_values = [3, 4, 5]
dimensionalities = [2, 3, 4]
dataset_sizes = [100, 1000, 10000]

profile_kmeans_numpy(k_values, dimensionalities, dataset_sizes)



