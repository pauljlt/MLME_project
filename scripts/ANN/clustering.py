import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from scripts.ANN.data_management import load_all_data


def preprocess_data(data, visuals_dir, plot=False):
    """
    Preprocess the data by scaling and applying PCA.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the loaded data.
    
    Returns:
    pd.DataFrame: The preprocessed data.
    """

    # Scale the features
    features = data.drop(columns=['trajectory_id', 'timestamp'], errors='ignore')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_features)
    
    # Create a new DataFrame with PCA features
    pca_data = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])

    # Reattach trajectory_id if it exists
    if 'trajectory_id' in data.columns:
        pca_data['trajectory_id'] = data['trajectory_id'].values

    # Reattach timestamp if it exists
    if 'timestamp' in data.columns:
        pca_data['timestamp'] = data['timestamp'].values
    
    # Plot PCA results if requested
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_data['PC1'], pca_data['PC2'], alpha=0.5)
        plt.title('PCA Results')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.savefig(os.path.join(visuals_dir, "pca_results.svg"))

    return pca_data


def cluster_data(data, eps=0.3, min_samples=5):
    """
    Cluster the data using DBSCAN.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the preprocessed data.
    eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    pd.Series: The cluster labels for each sample.
    """

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data[['PC1', 'PC2']])
    data_clustered = data.copy()
    data_clustered['Cluster'] = cluster_labels

    return data_clustered


def plot_clusters(data, visuals_dir, cluster_labels):
    """
    Plot the clusters in a 2D space.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the preprocessed data.
    cluster_labels (pd.Series): The cluster labels for each sample.

    Returns:
    None: Displays the plot.
    """

    plt.figure(figsize=(10, 6))

    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels != -1])

    colors = plt.get_cmap('tab10', n_clusters)

    for idx, label in enumerate(unique_labels):
        if label == -1:
            # outliers/noise in grey
            color = 'lightgrey'
            label_name = 'Noise'
        else:
            color = colors(idx % n_clusters)
            label_name = f'Cluster {label}'
        mask = (cluster_labels == label)
        plt.scatter(
            data.loc[mask, 'PC1'],
            data.loc[mask, 'PC2'],
            c=[color],
            label=label_name,
            s=50,
            alpha=0.5,
            edgecolors=[color],
            linewidths=0.5
        )

    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    plt.savefig(f"{visuals_dir}/dbscan_clustering_results.svg")


def main():
    # Load the data
    file_path = "./release/Data"
    data = load_all_data(file_path)

    # Create visuals directory if it doesn't exist
    visuals_dir = "./visuals/clustering"
    if not os.path.exists(visuals_dir):
        os.makedirs(visuals_dir)

    # Check if data is loaded successfully
    if not data.empty:
        print("Data loaded successfully. Preprocessing data...")

        # Preprocess the data
        preprocessed_data = preprocess_data(data, visuals_dir, plot=True)
        print("Preprocessed Data:")
        print(preprocessed_data.head())
        print("Data preprocessing completed. Proceeding with clustering...")

        # Perform clustering
        data_clustered = cluster_data(preprocessed_data, eps=0.3, min_samples=5) # tune parameters as wished
        cluster_labels = data_clustered["Cluster"]
        print("Cluster Labels:")
        print(cluster_labels.value_counts())
        print("Clustered Data:")
        print(data_clustered.head())

        # Plot the clusters
        print("Plotting clusters...")
        plot_clusters(preprocessed_data, visuals_dir, cluster_labels)
        print("Clustering completed and plotted.")

        # Merge the cluster labels back to the original data
        raw_data_with_clusters = data.merge(data_clustered[['trajectory_id', 'timestamp', 'Cluster']], on=['trajectory_id', 'timestamp'], how='left')
        raw_data_with_clusters = raw_data_with_clusters.sort_values(by=['trajectory_id', 'timestamp'])

        # Save the clustered data to a CSV file
        raw_data_with_clusters.to_csv(os.path.join(visuals_dir, "clustered_raw_data.csv"), index=False)
    else:
        print("No data found or loaded.")



if __name__ == "__main__":
    main()