# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

def euclidean_distance(cnp.ndarray[cnp.float64_t, ndim=1] point1, cnp.ndarray[cnp.float64_t, ndim=1] point2):
    return sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(cnp.ndarray[cnp.float64_t, ndim=2] data, int k):
    return data[np.random.choice(data.shape[0], k, replace=False)]

def assign_clusters(cnp.ndarray[cnp.float64_t, ndim=2] data, cnp.ndarray[cnp.float64_t, ndim=2] centroids):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] distances = np.zeros((data.shape[0], centroids.shape[0]))
    cdef int i
    for i in range(centroids.shape[0]):
        distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cluster_indices = np.argmin(distances, axis=1)
    return [data[cluster_indices == i] for i in range(centroids.shape[0])]

def update_centroids(list clusters):
    return np.array([np.mean(cluster, axis=0) for cluster in clusters])

def kmeans_cython(cnp.ndarray[cnp.float64_t, ndim=2] data, int k, int max_iterations=100):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] centroids = initialize_centroids(data, k)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] new_centroids
    cdef list clusters
    cdef int _
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids
