# K-means Clustering Algorithms

This repository explores various implementations of the K-means clustering algorithm, including pure Python, optimization with Numpy and Numexpr, and acceleration using Cython. The project aims to compare these implementations in terms of performance metrics such as execution time and memory usage, providing insights into the scalability and efficiency of each approach.

## Project Overview

The K-means clustering algorithm is a widely used technique in data science for grouping data points into a predefined number of clusters. This project implements the K-means algorithm in several versions to assess their performance across different dimensions and dataset sizes.

## Implementation Details

- **Pure Python:** A basic K-means algorithm implementation using only Python standard library functions.
- **Numpy Arrays and Numexpr:** An optimized version of the K-means algorithm using Numpy arrays for efficient computation and Numexpr for faster numerical expression evaluation.
- **Cython:** An accelerated version of the K-means algorithm using Cython to convert Python code into C, aiming for a significant speed-up in execution time.

## Experimental Setup

To compare the performance of each implementation, we define the following experiment parameters:

- **Number of Clusters (K):** The predefined number of clusters to partition the dataset.
- **Dimensionality of Data Points:** The number of features each data point has.
- **Size of the Dataset:** The total number of data points in the dataset.

The hardware and software environment are also detailed to ensure reproducibility of the experiments.

## Profiling

We employ profiling techniques to measure and compare the performance of each K-means implementation. The key metrics include:

- Execution Time
- Memory Usage

## Execution and Analysis

Each implementation is executed under varying parameters, such as different dataset sizes and dimensionalities. The results are then recorded and analyzed to understand the performance impact. This section discusses the scalability of each implementation and provides a comparative analysis based on the profiling metrics.


## Getting Started

Instructions on how to set up the project, including cloning the repository, installing dependencies, and running the examples, are provided to ensure that users can easily replicate the experiments and explore the K-means algorithm implementations.

## Contributing

Contributions are welcome! If you're interested in improving the K-means Clustering Algorithms project, please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for more details.


## Contact

For questions, feedback, or further discussion, please contact [2009048@upy.edu.mx].

