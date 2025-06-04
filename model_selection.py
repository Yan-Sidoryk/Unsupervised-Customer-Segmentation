import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from minisom import MiniSom
from collections import Counter
import warnings
warnings.filterwarnings('ignore')




def find_optimal_k(data, max_k=10):

    scores = []
    k_range = range(2, max_k + 1)

    print("Finding optimal k using silhouette score...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        scores.append(silhouette_score(data, labels))
    
    return k_range[np.argmax(scores)]



def run_som(data, map_size=(10, 10), n_clusters=None):

    # Run SOM clustering with KMeans on winner neuron
    if n_clusters is None:
        n_clusters = find_optimal_k(data)
    
    # Initialize and train SOM
    som = MiniSom(map_size[0], map_size[1], data.shape[1], 
                  sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(data)
    som.train_random(data, 1000)
    
    # Get winner neurons for each data point
    winner_coordinates = np.array([som.winner(x) for x in data])
    
    # Cluster the winner coordinates
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    som_labels = kmeans.fit_predict(winner_coordinates)
    
    return som_labels



def run_all_clustering(data):

    # Determine optimal number of clusters
    n_clusters = find_optimal_k(data)
    print(f"Optimal number of clusters determined: {n_clusters}")
    results = {}

    # Hierarchical
    print("Running Hierarchical Clustering...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    results['Hierarchical'] = hierarchical.fit_predict(data)
    
    # K-Means
    print("Running K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    results['K-Means'] = kmeans.fit_predict(data)

    # SOM
    print("Running Self-Organizing Map clustering...")
    results['SOM'] = run_som(data, n_clusters=n_clusters)
    
    # DBSCAN (auto eps)
    print("Running DBSCAN...")
    neighbors = NearestNeighbors(n_neighbors=4)
    distances, _ = neighbors.fit(data).kneighbors(data)
    eps = np.percentile(np.sort(distances[:, 3]), 90)
    
    dbscan = DBSCAN(eps=eps, min_samples=5)
    results['DBSCAN'] = dbscan.fit_predict(data)

    # Mean Shift
    print("Running Mean Shift...")
    mean_shift = MeanShift()
    results['Mean Shift'] = mean_shift.fit_predict(data)
    
    # Gaussian Mixture
    print("Running Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    results['GMM'] = gmm.fit_predict(data)
    
    return results



def evaluate_results(data, results):

    scores = []
    
    for name, labels in results.items():
        # Skip if only one cluster or contains noise points with insufficient clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            continue
            
        # Handle DBSCAN noise points
        if name == 'DBSCAN' and -1 in labels:
            mask = labels != -1
            if np.sum(mask) < 10:  # Need enough points for evaluation
                continue
            eval_data = data[mask]
            eval_labels = labels[mask]
            noise_count = np.sum(labels == -1)
        else:
            eval_data = data
            eval_labels = labels
            noise_count = 0
        
        if len(np.unique(eval_labels)) < 2:
            continue
        
        # Calculate metrics
        sil_score = silhouette_score(eval_data, eval_labels)
        ch_score = calinski_harabasz_score(eval_data, eval_labels)
        db_score = davies_bouldin_score(eval_data, eval_labels)
        
        scores.append({
            'Algorithm': name,
            'Silhouette': round(sil_score, 3),
            'Calinski-Harabasz': round(ch_score, 1),
            'Davies-Bouldin': round(db_score, 3),
            'Clusters': len(np.unique(eval_labels)),
            'Noise Points': noise_count
        })
    
    return pd.DataFrame(scores)



def plot_results(data, results):
    print("Plotting clustering results...")

    n_plots = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, labels) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', alpha=0.7, s=30)
        
        n_clusters = len(np.unique(labels))
        noise_info = f" (Noise: {np.sum(labels == -1)})" if -1 in labels else ""
        
        # Add cluster sizes to title
        title = f'{name}\nClusters: {n_clusters}{noise_info}'
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()



def plot_cluster_sizes(results):

    n_algorithms = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Get diverse colors
    
    for i, (name, labels) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        counter = Counter(labels)
        
        # Sort by cluster label, but put noise (-1) at the end if it exists
        sorted_items = sorted(counter.items())
        if -1 in counter:
            noise_count = sorted_items.pop(0)
            sorted_items.append(noise_count)
        
        # Prepare data for plotting
        cluster_names = []
        cluster_counts = []
        bar_colors = []
        
        for j, (label, count) in enumerate(sorted_items):
            if label == -1:
                cluster_names.append('Noise')
                bar_colors.append('red')  # Noise in red
            else:
                cluster_names.append(f'C{label}')
                bar_colors.append('skyblue')
            cluster_counts.append(count)
        
        # Create bar chart
        bars = ax.bar(cluster_names, cluster_counts, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add count labels on bars
        for bar, count in zip(bars, cluster_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(cluster_counts)*0.01,
                   f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Clusters', fontsize=10)
        ax.set_ylabel('Number of Points', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels if many clusters
        if len(cluster_names) > 5:
            ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(n_algorithms, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Cluster Size Distribution by Algorithm', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()



def clustering_comparison(data):
    
    # Run all algorithms
    results = run_all_clustering(data)
    
    # Plot results
    plot_results(data, results)
    
    # Plot cluster sizes as bar charts
    plot_cluster_sizes(results)
    
    # Evaluate and show results
    scores_df = evaluate_results(data, results)
    
    if not scores_df.empty:
        print("\nClustering Comparison Results:")
        print("=" * 83)
        
        # Display with better formatting for the cluster sizes column
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        print(scores_df.to_string(index=False))
        
        print(f"\nBest Silhouette Score: {scores_df.loc[scores_df['Silhouette'].idxmax(), 'Algorithm']}")
    
    return results, scores_df
