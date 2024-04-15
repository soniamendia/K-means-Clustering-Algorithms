import numpy as np
import cProfile
import pstats
import io
from cython_test import kmeans_cython  # Adjust the import if the compiled name differs

# Function to generate synthetic data
def generate_data(size, dimensionality):
    return np.random.rand(size, dimensionality)

# Function to run kmeans and profile it, with output redirected to a file
def profile_kmeans_cython(k_values, dimensionalities, dataset_sizes):
    with open('log_cython.txt', 'w') as log_file:  # Open log file
        for k in k_values:
            for dim in dimensionalities:
                for size in dataset_sizes:
                    log_file.write(f"Profiling for K={k}, Dimensionality={dim}, Dataset Size={size}\n")
                    data = generate_data(size, dim)
                    profiler = cProfile.Profile()
                    profiler.enable()
                    kmeans_cython(data, k)
                    profiler.disable()
                    s = io.StringIO()
                    sortby = 'cumulative'
                    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
                    ps.print_stats()
                    log_file.write(s.getvalue())  # Write profiling results to file

# Parameters for profiling
k_values = [3, 4, 5]
dimensionalities = [2, 3, 4]
dataset_sizes = [100, 1000, 10000]

# Run the profiling
profile_kmeans_cython(k_values, dimensionalities, dataset_sizes)






