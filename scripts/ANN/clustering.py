import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from data_management import load_all_data


def preprocess_data(data, plot=False):
    """
    Preprocess the data by scaling and applying PCA.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the loaded data.
    
    Returns:
    pd.DataFrame: The preprocessed data.
    """

    # Scale the features
    features = data.drop(columns=['trajectory_id'], errors='ignore')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_features)
    
    # Create a new DataFrame with PCA features
    pca_data = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])

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
    cluster_labels = dbscan.fit_predict(data)
    data_clustered = data.copy()
    data_clustered['Cluster'] = cluster_labels

    return data_clustered


def plot_clusters(data, cluster_labels):
    """
    Plot the clusters in a 2D space.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the preprocessed data.
    cluster_labels (pd.Series): The cluster labels for each sample.

    Returns:
    None: Displays the plot.
    """

    visuals_dir = "C:/Users/paulj/Documents/Uni/12. Semester/MLME/MLME_project/visuals"

    plt.figure(figsize=(10, 6))

    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    # Wähle eine qualitative Colormap mit genügend Farben
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
            alpha=0.8
        )

    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    plt.savefig(f"{visuals_dir}/dbscan_clustering_results.png")
    plt.show()



if __name__ == "__main__":
    # Example usage
    file_path = "./release/Data"
    data = load_all_data(file_path)
    if not data.empty:
        preprocessed_data = preprocess_data(data, plot=True)
        print("Preprocessed Data:")
        print(preprocessed_data.head())
        data_clustered = cluster_data(preprocessed_data, eps=0.3, min_samples=5) # tune parameters as wished
        cluster_labels = data_clustered["Cluster"]
        print("Cluster Labels:")
        print(cluster_labels.value_counts())
        plot_clusters(preprocessed_data, cluster_labels)
        print("Clustering completed and plotted.")
    else:
        print("No data found or loaded.")