import random
import math
import cProfile
import pstats
import io

# Function to calculate the Euclidean distance between two points in N-dimensional space
def euclidean_distance(point1, point2):
    # Compute the root of the sum of the squared differences
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

# Function to randomly initialize centroids from the dataset
def initialize_centroids(data, k):
    # Select k unique random indices for initial centroids
    centroids = random.sample(data, k)
    return centroids

# Function to assign each data point to the closest centroid
def assign_clusters(data, centroids):
    # Create a list of empty lists to hold data points for each centroid
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        # Calculate the distance from this point to each centroid
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        # Find the index of the centroid closest to this point
        cluster_index = distances.index(min(distances))
        # Append the point to the corresponding cluster
        clusters[cluster_index].append(point)
    return clusters

# Function to re-calculate centroids based on the mean of the points in each cluster
def update_centroids(clusters):
    # Calculate the new centroid as the mean of all points in the cluster
    centroids = [tuple(map(lambda x: sum(x) / len(x), zip(*cluster))) for cluster in clusters]
    return centroids

# The main K Means clustering algorithm
def kmeans(data, k, max_iterations=100):
    # Initialize centroids
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        # Assign all data points to clusters based on current centroids
        clusters = assign_clusters(data, centroids)
        # Calculate new centroids based on current clusters
        new_centroids = update_centroids(clusters)
        # Check for convergence (no change in centroids)
        if new_centroids == centroids:
            break
        # Update centroids for next iteration
        centroids = new_centroids
    return clusters, centroids

# Function to generate synthetic data for testing
def generate_data(size, dimensionality):
    # Generate data points with random coordinates in a specified dimensional space
    return [tuple(random.uniform(-10, 10) for _ in range(dimensionality)) for _ in range(size)]

# Function to run the K Means algorithm with profiling
def run_profiling(data, k):
    # Create a profiler object
    profiler = cProfile.Profile()
    profiler.enable()
    # Run K Means
    kmeans(data, k)
    profiler.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    # Create a Stats object to organize and print profiling results
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats()
    # Output profiling results
    print(s.getvalue())

# Main function to profile K Means with varying parameters
def profile_kmeans(k_values, dimensionalities, dataset_sizes):
    for k in k_values:
        for dim in dimensionalities:
            for size in dataset_sizes:
                print(f"Profiling for K={k}, Dimensionality={dim}, Dataset Size={size}")
                # Generate synthetic data
                data = generate_data(size, dim)
                # Profile the K Means algorithm
                run_profiling(data, k)

# Set parameters and begin profiling
k_values = [3, 4, 5]
dimensionalities = [2, 3, 4]
dataset_sizes = [100, 1000, 10000]


profile_kmeans(k_values, dimensionalities, dataset_sizes)

