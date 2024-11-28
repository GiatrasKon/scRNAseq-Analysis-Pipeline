# Single-cell RNA-seq Analysis Pipeline

This repository contains a comprehensive data analysis pipeline for single-cell RNA sequencing (scRNA-seq) data. The pipeline is designed to facilitate the identification of distinct cellular states using high-dimensional data. It implements dimensionality reduction techniques, clustering algorithms, and visualization strategies to analyze and interpret scRNA-seq datasets effectively. The project is structured with object-oriented programming (OOP) principles, ensuring modularity and scalability.

This analysis was performed as part of the 3rd Assignment for the "Machine Learning in Computational Biology" graduate course of the MSc Data Science & Information Technologies Master's programme (Bioinformatics - Biomedical Data Science Specialization) of the Department of Informatics and Telecommunications department of the National and Kapodistrian University of Athens (NKUA), under the supervision of professor Elias Manolakos, in the academic year 2023-2024.

---

## Project Overview

### Project Description

The pipeline processes single-cell RNA-seq data to achieve the following:

1. **Dimensionality Reduction**: Reducing the complexity of the dataset using PCA, t-SNE, and UMAP, while optimizing parameters for optimal representation.
2. **Clustering**: Identifying cellular states using clustering algorithms such as Gaussian Mixture Models (GMM) and DBSCAN, with automatic selection of the optimal number of clusters using metrics like BIC.
3. **Visualization**: Generating intuitive plots to showcase clustering results, posterior probabilities, and the structure of reduced data embeddings.

### Key Features

- Parameter optimization for dimensionality reduction techniques (e.g., variance threshold for PCA, perplexity for t-SNE, and n_neighbors for UMAP).
- Automatic model selection for clustering based on Bayesian Information Criterion (BIC).
- Support for both probabilistic (GMM) and density-based (DBSCAN) clustering methods.
- Comprehensive visualizations for insights into data structure and clustering outcomes.

---

## Main Workflow

1. **Data Input**: Load your scRNA-seq dataset in `.csv` format.
2. **Dimensionality Reduction**: Perform PCA, t-SNE, and UMAP to project data into lower dimensions.
3. **Clustering**: Use GMM or DBSCAN to identify cell groups. Automatically select the optimal number of clusters for GMM based on BIC.
4. **Visualization**: Generate 2D/3D scatter plots, posterior probability heatmaps, and clustering visualizations.
5. **Export Results**: Save clustering labels as a `.csv` file.

---

## Results Overview

The pipeline has been tested with a sample scRNA-seq dataset containing 137 cells and 54,675 genes. It successfully reduces dimensionality, clusters cells into biologically meaningful groups, and provides intuitive visualizations. Example results can be viewed in the Jupyter Notebook (`scRNAseq_data_analysis.ipynb`).

---

## Installation and Usage

### Cloning the Repository

```sh
git clone https://github.com/GiatrasKon/scRNAseq-Analysis-Pipeline.git
```

### Package Dependencies

Ensure you have the following packages installed:

- pandas
- numpy
- sklearn
- matplotlib
- seaborn
- umap

Install dependencies using:

```sh
pip install pandas matplotlib seaborn numpy scikit-learn umap
```

### Repository Structure

- `codebase.py`: Python script implementing the `DataPipeline` class.
- `scRNAseq_data_analysis.ipynb`: Notebook demonstrating the usage of the `DataPipeline` class.
- `data/`: Placeholder for the input dataset (`RNA-seq.csv`) and output (`labels.csv`).
- `documents/`: Assignment description and professor's feedback.

### Usage

1. **Prepare Your Dataset**: Place your scRNA-seq dataset in a `.csv` file (e.g., `data/RNA-seq.csv`) with cells as rows and genes as columns.
2. **Run the Pipeline**: Use the provided files to analyze the data step-by-step:
    - `codebase.py`: Contains the main pipeline implementation. Key methods include:
        - `dimensionality_reduction()`: Perform PCA, t-SNE, and UMAP.
        - `clustering()`: Cluster the reduced data using GMM and DBSCAN.
        - `plot_dimensionality_reduction()`, `plot_clustering_results()`, and `plot_posterior_probabilities()`: Visualize results.
    - `scRNAseq_data_analysis.ipynb`: Demonstrates how to use the `DataPipeline` class in an interactive Jupyter Notebook. Modify and run this notebook for step-by-step guidance.
3. **Export Results**:
    - Clustering results, including labels, will be saved as `data/labels.csv`.
    - Visualizations will be displayed for inspection and can be saved manually.

---