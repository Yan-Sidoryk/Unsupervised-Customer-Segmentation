# ---------- Importing the libraries ---------- #
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.graph_objects as go
import plotly.express as px


# Get unique clusters and colors
unique_clusters = sorted(info_df_with_cluster['cluster'].unique())
colors = px.colors.qualitative.G10

# Create figure with individual traces for each cluster
fig = go.Figure()

# Add a separate trace for each cluster
for i, cluster in enumerate(unique_clusters):
    cluster_data = info_df_with_cluster[info_df_with_cluster['cluster'] == cluster]
    
    # Create hover text
    hover_text = [
        f"Customer ID: {cid}<br>Cluster: {cluster}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
        for cid, lat, lon in zip(
            cluster_data['customer_id'], 
            cluster_data['latitude'], 
            cluster_data['longitude']
        )
    ]
    
    fig.add_trace(go.Scattermapbox(
        lat=cluster_data['latitude'],
        lon=cluster_data['longitude'],
        mode='markers',
        marker=dict(
            size=8,
            color=colors[i % len(colors)],
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text',
        name=f'Cluster {cluster} ({len(cluster_data)} customers)',
        visible=True
    ))

# Calculate center point
center_lat = info_df_with_cluster['latitude'].mean()
center_lon = info_df_with_cluster['longitude'].mean()


# Individual cluster buttons
for i, cluster in enumerate(unique_clusters):
    visible_array = [False] * len(unique_clusters)
    visible_array[i] = True
    

# Update layout with all your original settings plus new interactive features
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=11
    ),
    height=600,
    width=1000,
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title={
        'text': f'Customer Clusters - {len(info_df_with_cluster)} customers',
        'x': 0.5,
        'xanchor': 'center'
    },
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left", 
        x=1.02,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1
    )
)

# Show the map
fig.show()

# Set style for consistency
plt.style.use('default')
sns.set_palette("husl")

# ---------- Visualizations ---------- #

def plot_pca_explained_variance(info_df_scaled, figsize=(10, 6)):
    
    # Calculate explained variance for different numbers of components
    explained_variances = []
    n_components_range = range(1, info_df_scaled.shape[1])  # Up to max features

    for n in n_components_range:
        pca_tmp = PCA(n_components=n)
        pca_tmp.fit(info_df_scaled.drop(columns=['customer_id'], axis=1))  # Exclude customer_id from PCA
        explained_variances.append(pca_tmp.explained_variance_ratio_.sum() * 100)  # Convert to percent

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create line plot with enhanced styling
    line = ax.plot(n_components_range, explained_variances, marker='o', 
                   linewidth=2.5, markersize=8, color='#2E86AB', 
                   markerfacecolor='#2E86AB', markeredgecolor='white', 
                   markeredgewidth=2, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Number of PCA Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax.set_title('PCA: Explained Variance vs Number of Components', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid and clean styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Add horizontal reference lines
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, 
               label='80% Variance')
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.7, 
               label='95% Variance')
    
    
    # Set axis limits
    ax.set_xlim([0.5, max(n_components_range) + 0.5])
    ax.set_ylim([0, 105])
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Format y-axis to show percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    
    plt.tight_layout()
    plt.show()



def plot_dbscan_counts(info_df_clustered, cluster_column='DBScan', figsize=(10, 6)):

    # Get cluster counts
    cluster_counts = info_df_clustered.groupby([cluster_column]).size()
    unique_labels = sorted(cluster_counts.index)
    
    # Enhanced color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars
    bars = ax.bar(range(len(unique_labels)), cluster_counts.values, 
                  color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Customize the plot
    ax.set_xlabel('DBSCAN Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Number of Customers in Each DBSCAN Cluster', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels (handle noise points as -1)
    x_labels = []
    for label in unique_labels:
        if label == -1:
            x_labels.append('Outliers ')
        else:
            x_labels.append(f'C{label}')
    
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(x_labels, fontweight='bold')
    
    # Add grid and clean styling
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, count in zip(bars, cluster_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(cluster_counts.values) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim([0, max(cluster_counts.values) * 1.1])
    
    plt.tight_layout()
    plt.show()



def plot_elbow_and_silhouette(info_df_scaled, max_k=10, step=1):
    
    # Setup for tracking both metrics
    k_range = range(2, max_k+1, step)  # Silhouette score requires at least 2 clusters
    inertia_values = []
    silhouette_scores = []

    # Calculate both metrics for each k value
    for k in k_range:
        # Create and fit KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(info_df_scaled.drop(columns=['customer_id']))
        
        # Store inertia (sum of squared distances to nearest centroid)
        inertia_values.append(kmeans.inertia_)
        
        # Calculate and store silhouette score
        silhouette_avg = silhouette_score(info_df_scaled.drop(columns=['customer_id']), cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Enhanced styling colors
    inertia_color = '#6e80a9'
    silhouette_color = '#86b637'

    # Plot 1: Elbow Method on the left y-axis
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (Sum of Squared Distances)', color=inertia_color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(k_range, inertia_values, 'o-', color=inertia_color, 
                     linewidth=2.5, markersize=8, markerfacecolor=inertia_color,
                     markeredgecolor='white', markeredgewidth=2,
                     label='Inertia', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=inertia_color, labelsize=10)

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Silhouette Score', color=silhouette_color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(k_range, silhouette_scores, 's-', color=silhouette_color,
                     linewidth=2.5, markersize=8, markerfacecolor=silhouette_color,
                     markeredgecolor='white', markeredgewidth=2,
                     label='Silhouette Score', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=silhouette_color, labelsize=10)

    # Add title
    ax1.set_title('K-Means Optimization: Elbow Method & Silhouette Analysis', 
                  fontsize=14, fontweight='bold', pad=20)

    # Clean styling
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Set x-ticks
    ax1.set_xticks(k_range)
    ax1.tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    plt.show()



def plot_cluster_counts(info_df_clustered, cluster_column='cluster'):
    
    # Get cluster counts
    cluster_counts = info_df_clustered.groupby([cluster_column]).size()
    unique_labels = sorted(cluster_counts.index)
    
    # Enhanced color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))[::-1]  # Reverse the order for better visibility
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars = ax.bar(range(len(unique_labels)), cluster_counts.values, 
                  color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Customize the plot
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Number of Customers in Each Cluster', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels([f'C{c}' for c in unique_labels], fontweight='bold')
    
    # Add grid and clean styling
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, count in zip(bars, cluster_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(cluster_counts.values) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim([0, max(cluster_counts.values) * 1.1])
    
    plt.tight_layout()
    plt.show()



def plot_silhouette_analysis(info_df_clustered, cluster_column='cluster'):

    # Use all numeric columns except the cluster column
    feature_columns = info_df_clustered.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_column in feature_columns:
        feature_columns.remove(cluster_column)

    X = info_df_clustered[feature_columns].values
    cluster_labels = info_df_clustered[cluster_column].values
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(f'Silhouette Analysis for Clustered Data\nOverall Average Score: {silhouette_avg:.3f}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # === LEFT PLOT: Silhouette Plot ===
    y_lower = 10
    unique_labels = sorted(np.unique(cluster_labels))[::-1]  # Sort labels in descending order for better visualization
    
    # Enhanced color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    cluster_stats = {}
    
    for i, cluster_label in enumerate(unique_labels):
        # Aggregate silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == cluster_label]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = colors[i]
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor='white', alpha=0.8, linewidth=0.5)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, f'C{cluster_label}', 
                fontweight='bold', fontsize=10, ha='right')
        
        # Store stats for bar plot
        cluster_avg = np.mean(ith_cluster_silhouette_values)
        cluster_stats[cluster_label] = {
            'avg_score': cluster_avg,
            'size': size_cluster_i,
            'color': color
        }
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
    
    # Customize silhouette plot
    ax1.set_xlabel('Silhouette Coefficient Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cluster Label', fontsize=12, fontweight='bold')
    ax1.set_title('Silhouette Plot by Cluster', fontsize=14, fontweight='bold', pad=20)
    
    # Add vertical line for average silhouette score
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
               label=f'Average: {silhouette_avg:.3f}')
    ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    
    # Customize grid and appearance
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_ylim([0, len(X) + (len(unique_labels) + 1) * 10])
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # === RIGHT PLOT: Bar Chart ===
    cluster_nums = list(cluster_stats.keys())[::-1]  # Reverse order for bar chart
    avg_scores = [cluster_stats[c]['avg_score'] for c in cluster_nums]
    sizes = [cluster_stats[c]['size'] for c in cluster_nums]
    bar_colors = [cluster_stats[c]['color'] for c in cluster_nums]
    
    # Create bars
    bars = ax2.bar(range(len(cluster_nums)), avg_scores, color=bar_colors, 
                  alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add average line
    ax2.axhline(y=silhouette_avg, color='red', linestyle='--', linewidth=2, 
               label=f'Overall Average: {silhouette_avg:.3f}')
    
    # Customize bar chart
    ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Average Silhouette Score by Cluster', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(cluster_nums)))
    ax2.set_xticklabels([f'C{c}' for c in cluster_nums], fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # Add value labels on bars
    for i, (bar, score, size) in enumerate(zip(bars, avg_scores, sizes)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}\n(n={size})', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Set y-axis limits for better visualization
    y_min = min(0, min(avg_scores) - 0.1)
    y_max = max(avg_scores) + 0.1
    ax2.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.show()

    # Return average cluster silhouette scores as a sorted dictionary
    avg_cluster_scores = {key : cluster_stats[key]['avg_score'] for key in cluster_stats.keys()}
    return dict(sorted(avg_cluster_scores.items(), key=lambda item: item[1], reverse=True))



def plot_feature_importance(df, cluster_col='cluster'):

    # Calculate feature importance
    features = df.drop(columns=[cluster_col]).columns
    importance = {}
    
    for feature in features:
        # Calculate between-cluster variance (weighted by cluster size)
        cluster_means = df.groupby(cluster_col)[feature].mean()
        overall_mean = df[feature].mean()
        n_clusters = len(cluster_means)
        n_samples = len(df)
        
        between_variance = sum(
            df[cluster_col].value_counts()[cluster] * 
            (cluster_means[cluster] - overall_mean)**2 
            for cluster in range(n_clusters)
        ) / (n_clusters - 1) if n_clusters > 1 else 0
        
        # Calculate within-cluster variance
        within_variance = sum(
            sum((df.loc[df[cluster_col] == cluster, feature] - cluster_means[cluster])**2)
            for cluster in range(n_clusters)
        ) / (n_samples - n_clusters) if (n_samples - n_clusters) > 0 else 1
        
        # F-statistic is the ratio of between-variance to within-variance
        importance[feature] = between_variance / within_variance if within_variance > 0 else float('inf')
    
    importance_series = pd.Series(importance).sort_values(ascending=False)
    
    # Create color gradient based on importance values
    colors = []
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(importance_series)), importance_series.values, color='#DBB714',
                    alpha=0.8, edgecolor='white', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('F-statistic (Feature Importance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title('Feature Importance for Clustering', fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis labels
    ax.set_yticks(range(len(importance_series)))
    ax.set_yticklabels(importance_series.index)
    
    # Add grid and clean styling
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, importance_series.values)):
        width = bar.get_width()
        if np.isfinite(value):
            ax.text(width + max(importance_series.values) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}', ha='left', va='center', fontsize=9)
        else:
            ax.text(width * 0.5, bar.get_y() + bar.get_height()/2,
                    '∞', ha='center', va='center', fontsize=12, color='white')
    
    # Invert y-axis to show most important features at the top
    ax.invert_yaxis()
    
    # Set x-axis limits with some padding
    max_val = max([v for v in importance_series.values if np.isfinite(v)])
    ax.set_xlim([0, max_val * 1.15])
    
    plt.tight_layout()
    plt.show()



def plot_clusters_map(info_df_with_cluster, cluster_col='cluster'):
    # Get unique clusters and colors
    unique_clusters = sorted(info_df_with_cluster['cluster'].unique())
    colors = px.colors.qualitative.G10

    # Create figure with individual traces for each cluster
    fig = go.Figure()

    # Add a separate trace for each cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_data = info_df_with_cluster[info_df_with_cluster['cluster'] == cluster]
        
        # Create hover text
        hover_text = [
            f"Customer ID: {cid}<br>Cluster: {cluster}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
            for cid, lat, lon in zip(
                cluster_data['customer_id'], 
                cluster_data['latitude'], 
                cluster_data['longitude']
            )
        ]
        
        fig.add_trace(go.Scattermapbox(
            lat=cluster_data['latitude'],
            lon=cluster_data['longitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text',
            name=f'Cluster {cluster} ({len(cluster_data)} customers)',
            visible=True
        ))

    # Calculate center point
    center_lat = info_df_with_cluster['latitude'].mean()
    center_lon = info_df_with_cluster['longitude'].mean()


    # Individual cluster buttons
    for i, cluster in enumerate(unique_clusters):
        visible_array = [False] * len(unique_clusters)
        visible_array[i] = True
        

    # Update layout with all your original settings plus new interactive features
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11
        ),
        height=600,
        width=1000,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={
            'text': f'Customer Clusters - {len(info_df_with_cluster)} customers',
            'x': 0.5,
            'xanchor': 'center'
        },
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left", 
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )

    # Show the map
    fig.show()