# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# Function for calculating the trustworthiness of an embedding
def trustworthiness(X, X_embedded, n_neighbors=5):
    n_samples = X.shape[0] # getting the number of samples
    nn_X = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X) # fitting the nearest neighbors model on the original data
    nn_X_embedded = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_embedded) # fitting the nearest neighbors model on the embedded data
    ranks_X = nn_X.kneighbors(X, return_distance=False)[:, 1:] # getting the ranks of the nearest neighbors in the original data
    ranks_X_embedded = nn_X_embedded.kneighbors(X_embedded, return_distance=False)[:, 1:] # getting the ranks of the nearest neighbors in the embedded data
    t = 0.0 # initializing trustworthiness score
    # Iterating over all samples
    for i in range(n_samples):
        # Getting the ranks of the neighbors for the ith sample in the original and embedded data
        rank_X_i = ranks_X[i, :]
        rank_X_embedded_i = ranks_X_embedded[i, :]
        # Iterating over the neighbors
        for j in range(n_neighbors):
            # Checking if the neighbor in the embedded space is not a neighbor in the original space
            if rank_X_embedded_i[j] not in rank_X_i:
                t += 1.0
    # Calculating the trustworthiness score
    return 1.0 - t / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))

class DataPipeline:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, index_col=0) # loading the data from a CSV file
        # Initializing attributes to store dimensionality reduction results
        self.pca_result_2d = None
        self.pca_result_3d = None
        self.tsne_result_2d = None
        self.tsne_result_3d = None
        self.umap_result_2d = None
        self.umap_result_3d = None
        # Initializing dictionaries to store models and results
        self.gmm_models = {}  # for storing GMM models
        self.dbscan_models = {}  # for storing DBSCAN models
        self.optimal_clusters = {}  # for storing optimal number of clusters
        self.cell_posteriors = {}  # for storing posterior probabilities from GMM
        self.labels = {}  # for storing cluster labels

    def dimensionality_reduction(self, methods=['PCA', 'TSNE', 'UMAP'], variance_threshold=0.9, tsne_grid_params=None, umap_grid_params=None):
        """
        Perform dimensionality reduction using specified methods such as PCA, t-SNE, and UMAP.
        
        Parameters:
        - methods (list): List of dimensionality reduction methods to apply.
        - variance_threshold (float): Variance threshold for PCA.
        - tsne_grid_params (dict): Grid parameters for t-SNE optimization.
        - umap_grid_params (dict): Grid parameters for UMAP optimization.
        
        Returns:
        - None
        """
        # PCA
        if 'PCA' in methods:
            # Fitting PCA on the data to determine the number of components to reach the variance threshold
            pca = PCA().fit(self.data)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            print(f'Number of components selected for PCA: {n_components}')
            # Performing PCA with 2 components
            pca_2d = PCA(n_components=2)
            self.pca_result_2d = pca_2d.fit_transform(self.data)
            # Performing PCA with 3 components
            pca_3d = PCA(n_components=3)
            self.pca_result_3d = pca_3d.fit_transform(self.data)
        
        # t-SNE
        if 'TSNE' in methods:
            if tsne_grid_params is None:
                # Default parameter grid for t-SNE
                tsne_grid_params = {
                    'perplexity': [5, 10, 20, 30, 40, 50],
                    'learning_rate': [10, 100, 200, 300, 500, 1000],
                    'n_iter': [2000],
                    'random_state': [42]
                }
            self.tsne_result_2d = self.optimize_tsne(2, tsne_grid_params) # optimizing t-SNE with 2 components
            self.tsne_result_3d = self.optimize_tsne(3, tsne_grid_params) # optimizing t-SNE with 3 components
        
        # UMAP
        if 'UMAP' in methods:
            if umap_grid_params is None:
                # Default parameter grid for UMAP
                umap_grid_params = {
                    'n_neighbors': [5, 10, 15, 20, 25, 30],
                    'min_dist': [0.1, 0.25, 0.5, 0.75, 1.0],
                    'metric': ['euclidean', 'manhattan'],
                    'random_state': [42]
                }
            self.umap_result_2d = self.optimize_umap(2, umap_grid_params) # optimizing UMAP with 2 components
            self.umap_result_3d = self.optimize_umap(3, umap_grid_params) # optimizing UMAP with 3 components
    
    def optimize_tsne(self, n_components, tsne_grid_params):
        """
        Optimizes t-SNE embedding with specified components and parameter grid.
        
        Parameters:
            - n_components (int): Number of components for the t-SNE embedding.
            - tsne_grid_params (dict): Grid parameters for t-SNE optimization.
        
        Returns:
            - best_embedding (array): The optimized t-SNE embedding with the best parameters.
        """
        param_grid = ParameterGrid(tsne_grid_params) # generating parameter combinations from the grid
        # Initializing variables to store the best score and parameters
        best_score = -np.inf
        best_params = None
        best_embedding = None
        # Iterating over all parameter combinations
        for params in param_grid:
            tsne = TSNE(n_components=n_components, **params)  # initializing t-SNE with current parameters
            embedding = tsne.fit_transform(self.data)  # fitting and transforming the data
            score = trustworthiness(self.data, embedding)  # evaluating the trustworthiness of the embedding
            # Updating the best parameters and score if the current one is better
            if score > best_score:
                best_score = score
                best_params = params
                best_embedding = embedding
        # Printing the best parameters and their corresponding trustworthiness score
        print(f'Best t-SNE params for {n_components}D: {best_params} with trustworthiness score: {best_score}')
        return best_embedding

    def optimize_umap(self, n_components, umap_grid_params):
        """
        Optimizes UMAP embedding with specified components and parameter grid.
        
        Parameters:
            - n_components (int): Number of components for the UMAP embedding.
            - umap_grid_params (dict): Grid parameters for UMAP optimization.
        
        Returns:
            - best_embedding (array): The optimized UMAP embedding with the best parameters.
        """
        param_grid = ParameterGrid(umap_grid_params) # generating parameter combinations from the grid
        # Initializing variables to store the best score and parameters
        best_score = -np.inf
        best_params = None
        best_embedding = None
        # Iterating over all parameter combinations
        for params in param_grid:
            umap_model = umap.UMAP(n_components=n_components, **params)  # initializing UMAP with current parameters
            embedding = umap_model.fit_transform(self.data)  # fitting and transforming the data
            score = trustworthiness(self.data, embedding)  # evaluating the trustworthiness of the embedding
            # Updating the best parameters and score if the current one is better
            if score > best_score:
                best_score = score
                best_params = params
                best_embedding = embedding
        # Printing the best parameters and their corresponding trustworthiness score
        print(f'Best UMAP params for {n_components}D: {best_params} with trustworthiness score: {best_score}')
        return best_embedding
    
    def plot_dimensionality_reduction(self, methods=['PCA', 'TSNE', 'UMAP']):
        """
        Plots the 2D and 3D embeddings for the specified dimensionality reduction methods.

        Parameters:
            - methods (list): List of dimensionality reduction methods to plot.
        
        Returns:
            - None
        """
        # Iterating over the specified methods
        for method in methods:
            # Getting the 2D and 3D results of the current method
            result_2d = getattr(self, f'{method.lower()}_result_2d')
            result_3d = getattr(self, f'{method.lower()}_result_3d')
            # Plotting the results if both 2D and 3D embeddings are available
            if result_2d is not None and result_3d is not None:
                fig = plt.figure(figsize=(16, 6))
                
                # Plotting the 2D embedding
                ax1 = fig.add_subplot(121)
                sns.scatterplot(x=result_2d[:, 0], y=result_2d[:, 1], ax=ax1)
                ax1.set_title(f'{method} 2D Result')
                ax1.set_xlabel(f'{method} 1')
                ax1.set_ylabel(f'{method} 2')
                
                # Plotting the 3D embedding
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(result_3d[:, 0], result_3d[:, 1], result_3d[:, 2])
                ax2.set_title(f'{method} 3D Result')
                ax2.set_xlabel(f'{method} 1')
                ax2.set_ylabel(f'{method} 2')
                ax2.set_zlabel(f'{method} 3')
                
                plt.show()

    def clustering(self, max_clusters=10, clustering_algorithms=['GMM', 'DBSCAN'], random_seed=42):
        """
        Performs clustering on reduced data embeddings using the specified clustering algorithms.

        Parameters:
            max_clusters (int): The maximum number of clusters to consider.
            clustering_algorithms (list): List of clustering algorithms to use, e.g., ['GMM', 'DBSCAN'].
            random_seed (int): The random seed for reproducibility.

        Returns:
            None
        """
        # Creating a dictionary of reduced data embeddings
        reduced_data = {
            'PCA_2D': self.pca_result_2d,
            'PCA_3D': self.pca_result_3d,
            'TSNE_2D': self.tsne_result_2d,
            'TSNE_3D': self.tsne_result_3d,
            'UMAP_2D': self.umap_result_2d,
            'UMAP_3D': self.umap_result_3d
        }
        # Initializing variables to store the best clustering results
        best_silhouette_score = -np.inf
        best_labels = None
        best_method_dim = None
        best_clustering_algorithm = None
        # Iterating over each reduced dataset
        for method, data in reduced_data.items():
            if data is not None:
                print(f'Clustering on {method}')

                # GMM Clustering
                if 'GMM' in clustering_algorithms:
                    lowest_bic = np.inf # Initialize the lowest BIC as infinity
                    best_gmm = None
                    # Iterating over different covariance types and number of components
                    for cv_type in ['spherical', 'tied', 'diag', 'full']:
                        for n_components in range(1, max_clusters + 1):
                            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=random_seed)
                            gmm.fit(data) # fitting the GMM model to the data
                            bic = gmm.bic(data) # calculating the BIC for the current model
                            # Update the best GMM model if the current BIC is lower than the lowest BIC
                            if bic < lowest_bic:
                                lowest_bic = bic
                                self.optimal_clusters[method] = n_components
                                best_gmm = gmm
                    # Storing the best GMM results if a GMM model was found
                    if best_gmm:
                        self.gmm_models[method] = best_gmm
                        self.cell_posteriors[method] = best_gmm.predict_proba(data)
                        self.labels[method] = best_gmm.predict(data)
                        print(f'Optimal number of clusters for {method} (GMM): {self.optimal_clusters[method]}')
                        # Calculating the silhouette score for the best GMM model
                        silhouette_avg = silhouette_score(data, self.labels[method])
                        print(f'Silhouette Score for {method} (GMM): {silhouette_avg}')
                        # Updating the best clustering results if the current silhouette score is better
                        if silhouette_avg > best_silhouette_score:
                            best_silhouette_score = silhouette_avg
                            best_labels = self.labels[method]
                            best_method_dim = method
                            best_clustering_algorithm = 'GMM'

                # DBSCAN Clustering
                if 'DBSCAN' in clustering_algorithms:
                    best_dbscan_score = -np.inf
                    best_dbscan_labels = None
                    # Defining the parameter grid for DBSCAN
                    dbscan_params = {
                        'eps': [0.1, 0.5, 1.0, 1.5, 2.0],
                        'min_samples': [5, 10, 15, 20]
                    }
                    # Iterating over parameter combinations
                    for params in ParameterGrid(dbscan_params):
                        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples']) # initializing DBSCAN with current parameters
                        labels = dbscan.fit_predict(data) # fitting and predicting the labels
                        # Calculating the silhouette score for the current DBSCAN model
                        if len(set(labels)) > 1:  # at least two clusters to calculate silhouette score
                            silhouette_avg = silhouette_score(data, labels)
                            # Updating the best DBSCAN results if the current silhouette score is better
                            if silhouette_avg > best_dbscan_score:
                                best_dbscan_score = silhouette_avg
                                best_dbscan_labels = labels
                    # Storing the best DBSCAN results if any were found
                    if best_dbscan_labels is not None:
                        self.labels[method + '_DBSCAN'] = best_dbscan_labels # storing the labels for the best DBSCAN model
                        self.optimal_clusters[method + '_DBSCAN'] = len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0) # storing the optimal number of clusters
                        print(f'Optimal number of clusters for {method} (DBSCAN): {self.optimal_clusters[method + '_DBSCAN']}') 
                        print(f'Silhouette Score for {method} (DBSCAN): {best_dbscan_score}')
                        # Updating the best clustering results if the current silhouette score is better
                        if best_dbscan_score > best_silhouette_score:
                            best_silhouette_score = best_dbscan_score
                            best_labels = best_dbscan_labels
                            best_method_dim = method
                            best_clustering_algorithm = 'DBSCAN'

        # Saving the best labels to a CSV file
        if best_labels is not None:
            labels_df = pd.DataFrame(best_labels, index=self.data.index, columns=['labels'])
            labels_df.to_csv('../data/labels.csv')
            print(f'Best labels saved from {best_method_dim} method using {best_clustering_algorithm} algorithm.')

    def plot_clustering_results(self, methods=['PCA', 'TSNE', 'UMAP'], clustering_algorithms=['GMM', 'DBSCAN']):
        """
        Iterates over specified methods and clustering algorithms, plots clustering results in 2D and 3D if data is available, 
        differentiating between GMM and DBSCAN clustering algorithms, and displays the figure if valid plots exist.
        
        Parameters:
        - methods (list): List of dimensionality reduction methods to apply.
        - clustering_algorithms (list): List of clustering algorithms to use.
        
        Returns:
        - None
        """
        # Iterating over the specified methods and clustering algorithms
        for method in methods:
            for clustering_algorithm in clustering_algorithms:
                fig = plt.figure(figsize=(16, 6))
                has_2d = False
                has_3d = False
                # Iterating over 2D and 3D embeddings
                for dim in ['2D', '3D']:
                    key = f'{method}_{dim}'
                    data = getattr(self, f'{method.lower()}_result_{dim.lower()}')
                    # Plotting the clustering results if data is available
                    if data is not None:
                        # Plotting the GMM clustering results
                        if clustering_algorithm == 'GMM' and self.labels.get(key) is not None:
                            if dim == '2D':
                                ax1 = fig.add_subplot(121)
                                scatter = sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=self.labels[key], palette=sns.color_palette("hsv", len(np.unique(self.labels[key]))), ax=ax1)
                                ax1.set_title(f'{method} 2D Clustering Results (GMM)')
                                ax1.set_xlabel(f'{method} 1')
                                ax1.set_ylabel(f'{method} 2')
                                ax1.legend(title='Cluster')
                                has_2d = True
                            elif dim == '3D':
                                ax2 = fig.add_subplot(122, projection='3d')
                                scatter = ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c=self.labels[key], cmap='hsv')
                                ax2.set_title(f'{method} 3D Clustering Results (GMM)')
                                ax2.set_xlabel(f'{method} 1')
                                ax2.set_ylabel(f'{method} 2')
                                ax2.set_zlabel(f'{method} 3')
                                unique_labels = np.unique(self.labels[key])
                                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.hsv(i / len(unique_labels)), markersize=10) for i in range(len(unique_labels))]
                                ax2.legend(handles, unique_labels, title='Cluster')
                                has_3d = True
                        # Plotting the DBSCAN clustering results
                        if clustering_algorithm == 'DBSCAN' and self.labels.get(key + '_DBSCAN') is not None:
                            if dim == '2D':
                                ax1 = fig.add_subplot(121)
                                scatter = sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=self.labels[key + '_DBSCAN'], palette=sns.color_palette("hsv", len(np.unique(self.labels[key + '_DBSCAN']))), ax=ax1)
                                ax1.set_title(f'{method} 2D Clustering Results (DBSCAN)')
                                ax1.set_xlabel(f'{method} 1')
                                ax1.set_ylabel(f'{method} 2')
                                ax1.legend(title='Cluster')
                                has_2d = True
                            elif dim == '3D':
                                ax2 = fig.add_subplot(122, projection='3d')
                                scatter = ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c=self.labels[key + '_DBSCAN'], cmap='hsv')
                                ax2.set_title(f'{method} 3D Clustering Results (DBSCAN)')
                                ax2.set_xlabel(f'{method} 1')
                                ax2.set_ylabel(f'{method} 2')
                                ax2.set_zlabel(f'{method} 3')
                                unique_labels = np.unique(self.labels[key + '_DBSCAN'])
                                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.hsv(i / len(unique_labels)), markersize=10) for i in range(len(unique_labels))]
                                ax2.legend(handles, unique_labels, title='Cluster')
                                has_3d = True
                # Displaying the figure if there are valid 2D or 3D plots
                if has_2d or has_3d:
                    fig.tight_layout()
                    plt.show()

    def plot_posterior_probabilities(self, methods=['PCA', 'TSNE', 'UMAP'], clustering_algorithms=['GMM']):
        """
        Iterates over the specified methods, plots the posterior probabilities heatmap for GMM clustering if available.

        Parameters:
            - self: The object instance.
            - methods (list): List of dimensionality reduction methods to apply.
            - clustering_algorithms (list): List of clustering algorithms to use.
        
        Returns:
            - None
        """
        # Iterating over the specified methods
        for method in methods:
            for dim in ['2D', '3D']:
                key = f'{method}_{dim}'
                # Plotting the posterior probabilities heatmap for GMM clustering
                if 'GMM' in clustering_algorithms and self.cell_posteriors.get(key) is not None:
                    plt.figure(figsize=(12, 8))
                    ax = sns.heatmap(self.cell_posteriors[key], cmap="viridis", cbar=True)
                    plt.title(f'Posterior Probabilities Heatmap for {key} (GMM)')
                    plt.xlabel('Clusters')
                    plt.ylabel('Cells')
                    ax.set_yticklabels([])  # hiding y-axis labels
                    plt.show()

# Usage Example:
# data_pipeline = DataPipeline('data/RNA-seq.csv')
# data_pipeline.dimensionality_reduction()
# data_pipeline.plot_dimensionality_reduction()
# data_pipeline.clustering(clustering_algorithms=['GMM', 'DBSCAN'])
# data_pipeline.plot_clustering_results(clustering_algorithms=['GMM', 'DBSCAN'])
# data_pipeline.plot_posterior_probabilities(clustering_algorithms=['GMM', 'DBSCAN'])