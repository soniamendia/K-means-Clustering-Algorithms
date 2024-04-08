import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def euclidean_distance(np.ndarray[np.float64_t, ndim=1] point1, np.ndarray[np.float64_t, ndim=1] point2):
    cdef double distance = 0.0
    cdef int dim = point1.shape[0]
    for i in range(dim):
        distance += (point1[i] - point2[i]) ** 2
    return sqrt(distance)

def initialize_centroids(np.ndarray[np.float64_t, ndim=2] data, int k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids

def assign_clusters(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=2] centroids):
    cdef int n = data.shape[0]
    cdef int k = centroids.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] distances = np.zeros((n, k))
    cdef np.ndarray[np.int32_t, ndim=1] cluster_indices = np.zeros(n, dtype=np.int32)
    
    for i in range(k):
        for j in range(n):
            distances[j, i] = euclidean_distance(data[j], centroids[i])
    
    for j in range(n):
        cluster_indices[j] = np.argmin(distances[j])
    
    clusters = [data[cluster_indices == i] for i in range(k)]
    return clusters

def update_centroids(list clusters):
    cdef int k = len(clusters)
    cdef np.ndarray[np.float64_t, ndim=2] centroids = np.zeros((k, clusters[0].shape[1]))
    
    for i in range(k):
        centroids[i] = np.mean(clusters[i], axis=0)
    
    return centroids

def kmeans_cython(np.ndarray[np.float64_t, ndim=2] data, int k, int max_iterations=100):
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        
        if np.array_equal(new_centroids, centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids
