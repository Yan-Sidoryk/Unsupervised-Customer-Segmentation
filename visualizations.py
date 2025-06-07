# ---------- Importing the libraries ---------- #
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch

from umap import UMAP

from collections import Counter




# ---------- Data Visualizations ---------- #

def general_visualization(info_df):

    # Define the columns to plot
    columns = ['customer_gender', 'kids_home', 'teens_home', 'number_complaints', 
            'distinct_stores_visited', 'degree_level', 'typical_hour'] 

    # Enhanced color palette
    color_palette = plt.cm.Set2(np.linspace(0, 1, 8))[::-1]  # Reverse the order for better visibility

    # Create a grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot general bar plots for each column in the list
    for i, col in enumerate(columns):
        # Get value counts for proper ordering and colors
        if col == 'degree_level':
            # Custom order for degree levels
            degree_order = ['None', 'Bsc', 'Msc', 'Phd']
            value_counts = info_df[col].value_counts().reindex(degree_order).fillna(0).astype(int)
        else:
            value_counts = info_df[col].value_counts().sort_index()
        
        # Create bars with enhanced styling
        bars = axes[i].bar(range(len(value_counts)), value_counts.values, 
                        color=color_palette[i % len(color_palette)], alpha=0.8, 
                        edgecolor='white', linewidth=1.5)
        
        # Customize the plot
        axes[i].set_xlabel(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Count', fontsize=10, fontweight='bold')
        axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}', 
                        fontsize=11, fontweight='bold', pad=15)
        
        # Set x-axis labels
        axes[i].set_xticks(range(len(value_counts)))
        axes[i].set_xticklabels(value_counts.index, fontweight='bold')
        
        # Add grid and clean styling
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
        # Add value labels on bars
        for bar, count in zip(bars, value_counts.values):
            height = bar.get_height()
            if col != 'typical_hour':   # Skip value labels for 'typical_hour'
                axes[i].text(bar.get_x() + bar.get_width()/2., height + max(value_counts.values) * 0.01,
                            f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9, rotation=0)
        
        # Set y-axis limits
        axes[i].set_ylim([0, max(value_counts.values) * 1.1])

    # Make female bar pink and male bar blue (keeping your specific styling)
    bars = axes[0].patches  
    bars[0].set_color('pink')  # Female
    bars[0].set_alpha(0.8)
    bars[0].set_edgecolor('white')
    bars[0].set_linewidth(1.5)
    bars[1].set_color([0.55294118, 0.62745098, 0.79607843, 1])  # Male
    bars[1].set_alpha(0.8)
    bars[1].set_edgecolor('white')
    bars[1].set_linewidth(1.5)

    # Plot time preferences separately with enhanced styling
    time_columns = ['morning_shopper', 'afternoon_shopper', 'evening_shopper']
    time_preferences = info_df[time_columns].sum()

    bars = axes[7].bar(range(len(time_preferences)), time_preferences.values, 
                    color=color_palette[7 % len(color_palette)], alpha=0.8, 
                    edgecolor='white', linewidth=1.5)

    axes[7].set_xlabel('Typical Shopping Time', fontsize=10, fontweight='bold')
    axes[7].set_ylabel('Count', fontsize=10, fontweight='bold')
    axes[7].set_title('Shopping Time Preferences', fontsize=11, fontweight='bold', pad=15)
    axes[7].set_xticks(range(len(time_preferences)))
    axes[7].set_xticklabels([col.replace('_shopper', '').replace('_', ' ').title() 
                            for col in time_columns], fontweight='bold')

    # Add grid and clean styling
    axes[7].grid(True, alpha=0.3, axis='y')
    axes[7].spines['top'].set_visible(False)
    axes[7].spines['right'].set_visible(False)

    # Add value labels on bars
    for bar, count in zip(bars, time_preferences.values):
        height = bar.get_height()
        axes[7].text(bar.get_x() + bar.get_width()/2., height + max(time_preferences.values) * 0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    axes[7].set_ylim([0, max(time_preferences.values) * 1.1])

    # Plot ages with bins - enhanced histogram styling
    n, bins, patches = axes[8].hist(info_df['age'], bins=range(21, 86, 3), 
                                color=color_palette[0], alpha=0.8, 
                                edgecolor='white', linewidth=1.5)

    axes[8].set_xlabel('Age', fontsize=10, fontweight='bold')
    axes[8].set_ylabel('Count', fontsize=10, fontweight='bold')
    axes[8].set_title('Distribution of Age', fontsize=11, fontweight='bold', pad=15)
    axes[8].set_xticklabels(range(21, 86, 5), fontsize=9, fontweight='bold')
    axes[8].grid(True, alpha=0.3, axis='y')
    axes[8].spines['top'].set_visible(False)
    axes[8].spines['right'].set_visible(False)

    # Plot customer_for - enhanced histogram styling
    n, bins, patches = axes[9].hist(info_df['customer_for'], bins=15, 
                                color=color_palette[1], alpha=0.8, 
                                edgecolor='white', linewidth=1.5)

    axes[9].set_xlabel('Years as Customer', fontsize=10, fontweight='bold')
    axes[9].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[9].set_title('Distribution of Customer Tenure', fontsize=11, fontweight='bold', pad=15)
    axes[9].set_xticklabels(axes[9].get_xticklabels(), fontsize=9, fontweight='bold')
    axes[9].grid(True, alpha=0.3, axis='y')
    axes[9].spines['top'].set_visible(False)
    axes[9].spines['right'].set_visible(False)

    # Plot the pie chart of loyalty card distribution with enhanced styling
    loyalty_counts = info_df['loyalty_card'].map({1: 'Yes', 0: 'No'}).value_counts()
    wedges, texts, autotexts = axes[10].pie(loyalty_counts.values, labels=loyalty_counts.index, 
                                        autopct='%1.1f%%', startangle=90, 
                                        colors=['darkseagreen', 'tomato'],
                                        wedgeprops=dict(edgecolor='white', linewidth=2))

    axes[10].set_title('Loyalty Card Distribution', fontsize=11, fontweight='bold', pad=15)

    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    for text in texts:
        text.set_fontweight('bold')
        text.set_fontsize(10)

    # Enhanced histogram for promotion percentage
    n, bins, patches = axes[11].hist(info_df['percentage_of_products_bought_promotion'], 
                                    bins=30, color=color_palette[2], alpha=0.8, 
                                    edgecolor='white', linewidth=1.5)

    axes[11].set_xlabel('Percentage of Products Bought on Promotion', fontsize=10, fontweight='bold')
    axes[11].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[11].set_title('Distribution of Promotion Purchase Percentage', 
                    fontsize=11, fontweight='bold', pad=15)
    axes[11].grid(True, alpha=0.3, axis='y')
    axes[11].spines['top'].set_visible(False)
    axes[11].spines['right'].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.show()



def spending_visualization(info_df, spend_columns):
    # Enhanced color palette
    color_palette = plt.cm.Set2(np.linspace(0, 1, len(spend_columns)))[::-1]  # Reverse the order for better visibility

    # Create a grid of subplots
    n_cols = 3  # Set the number of columns in the grid
    n_rows = (len(spend_columns) + n_cols - 1) // n_cols  # Calculate the number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))  
    axes = axes.flatten() 

    # Plot each column in a subplot with enhanced styling
    for i, col in enumerate(spend_columns):
        # Create KDE plot with enhanced styling
        sns.kdeplot(data=info_df, x=col, ax=axes[i], 
                    color=color_palette[i], linewidth=3, alpha=0.8, fill=True)
                    # fillalpha=0.3)
        
        # Customize the plot
        axes[i].set_xlabel(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Density', fontsize=10, fontweight='bold')
        axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}', 
                        fontsize=11, fontweight='bold', pad=15)
        
        # Add grid and clean styling
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
        # Make x-axis labels bold and rotate
        axes[i].tick_params(axis='x', rotation=45, labelsize=9)
        axes[i].tick_params(axis='y', labelsize=9)
        # Set font weight for tick labels
        for label in axes[i].get_xticklabels():
            label.set_fontweight('bold')
        for label in axes[i].get_yticklabels():
            label.set_fontweight('bold')

    # Remove any unused subplots
    for j in range(len(spend_columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_map(df, lat_col='latitude', lon_col='longitude',
                         hover_name_col='customer_name', zoom=11, 
                         height=600, width=800, mapbox_style="open-street-map"):
    
    # Plot scatter map 
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        hover_name=hover_name_col,
        zoom=zoom,
        height=height,
        width=width
    )

    # Remove margins and white background
    fig.update_layout(
        mapbox_style=mapbox_style,
        margin=dict(l=0, r=0, t=0, b=0),  # Removes all margins
        paper_bgcolor='rgba(0,0,0,0)',     # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'       # Transparent plot area
    )
    
    return fig

def plot_correlation_heatmap(info_df, columns):
    # Calculate the correlation matrix for the selected columns
    correlation_matrix = info_df[columns].corr()

    # Create a mask to remove duplicate correlations but keep the diagonal
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

    # Plot the heatmap with enhanced styling
    plt.figure(figsize=(16, 13))

    # Create heatmap with enhanced styling
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                    mask=mask, square=True, linewidths=1, linecolor='white',
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                    cbar_kws={'shrink': 0.8, 'aspect': 30})

    # Customize the plot with consistent styling
    plt.title('Correlation Heatmap of Numeric Customer Features', 
            fontsize=14, fontweight='bold', pad=20)

    # Style the axes
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')

    # Make tick labels bold and improve readability
    ax.tick_params(axis='x', labelsize=10, rotation=90)
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')

    # Clean up the spines (though they're mostly hidden by the heatmap)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Enhance colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.show()



def dendogram_visualization(info_df, columns):
    # Calculate the correlation matrix for the selected columns
    correlation_matrix = info_df[columns].corr()

    # Convert the correlation matrix to a distance matrix
    distance_matrix = 1 - correlation_matrix

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(distance_matrix, method='ward')

    # Plot the dendrogram with enhanced styling
    plt.figure(figsize=(14, 8))

    # Create dendrogram with color threshold at distance 2
    dendrogram = sch.dendrogram(linkage_matrix, labels=columns, leaf_rotation=90, 
                            leaf_font_size=10, color_threshold=2.0,  # Set threshold at distance 2
                            above_threshold_color='black')

    # Get current axes
    ax = plt.gca()

    # Customize the plot with consistent styling
    plt.title('Dendrogram of Numeric Customer Features', 
            fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Distance', fontsize=12, fontweight='bold')

    # Set tick label size
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    # Make tick labels bold
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Add grid and clean styling
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Enhance the dendrogram lines
    for line in ax.get_lines():
        line.set_linewidth(2)

    # Make x-axis labels more readable
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontweight='bold')

    # Add a horizontal line at distance 2 to show the color threshold
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, linewidth=2)

    plt.tight_layout()
    plt.show()


def plot_top_purchased_items(all_items, n_items=25):
    # Count the occurrences of each item
    item_counts = Counter([item for sublist in all_items['all_purchased_items'] for item in sublist])

    # Get the top most common items
    top_items = item_counts.most_common(n_items)

    # Separate the items and their counts for plotting
    items, counts = zip(*top_items)
    items = [item.replace('_', ' ').title() for item in items]  # Format item names
    counts = list(counts)

    # Enhanced color palette
    # colors = plt.cm.Set2(np.linspace(0, 1, len(items)))[::-1]  # Reverse the order for better visibility

    # Plot the top items with enhanced styling
    plt.figure(figsize=(15, 6))

    # Create bars with enhanced styling
    bars = plt.bar(range(len(items)), counts, color=[0.4, 0.76078431, 0.64705882, 1.0], alpha=0.8, 
                edgecolor='white', linewidth=1.5)

    # Customize the plot
    plt.xlabel('Items', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(f'Top {len(items)} Purchased Items', fontsize=14, fontweight='bold', pad=20)

    # Set x-axis labels
    plt.xticks(range(len(items)), items, rotation=45, ha='right', fontweight='bold', fontsize=10)
    plt.yticks(fontweight='bold', fontsize=10)

    # Add grid and clean styling
    plt.grid(True, alpha=0.3, axis='y')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Set y-axis limits
    plt.ylim([0, max(counts) * 1.1])

    plt.tight_layout()
    plt.show()

def plot_least_purchased_items(all_items, n_items=25):
    # Count the occurrences of each item
    item_counts = Counter([item for sublist in all_items['all_purchased_items'] for item in sublist])

    # Get the least common items (sorted in ascending order by count)
    least_items = item_counts.most_common()[:-n_items-1:-1]  # Get last n_items in reverse

    # Separate the items and their counts for plotting
    items, counts = zip(*least_items)
    items = [item.replace('_', ' ').title() for item in items]  # Format item names
    counts = list(counts)

    # Plot the least purchased items with enhanced styling
    plt.figure(figsize=(15, 6))

    # Create bars with enhanced styling 
    bars = plt.bar(range(len(items)), counts, color=[1.0, 0.6, 0.4, 1.0], alpha=0.8, edgecolor='white', linewidth=1.5)

    # Customize the plot
    plt.xlabel('Items', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(f'{len(items)} Least Purchased Items', fontsize=14, fontweight='bold', pad=20)

    # Set x-axis labels
    plt.xticks(range(len(items)), items, rotation=45, ha='right', fontweight='bold', fontsize=10)
    plt.yticks(fontweight='bold', fontsize=10)

    # Add grid and clean styling
    plt.grid(True, alpha=0.3, axis='y')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Set y-axis limits
    plt.ylim([0, max(counts) * 1.1])

    plt.tight_layout()
    plt.show()

# ---------- Model Visualizations ---------- #

def plot_pca_explained_variance(info_df_scaled, figsize=(10, 6)):
    
   # Prepare data (exclude customer_id)
    pca_data = info_df_scaled.drop(columns=['customer_id'], axis=1)
    
    # Calculate explained variance for different numbers of components
    explained_variances = []
    individual_variances = []
    n_components_range = range(1, pca_data.shape[1] + 1)  # Include all features

    for n in n_components_range:
        pca_tmp = PCA(n_components=n)
        pca_tmp.fit(pca_data)
        explained_variances.append(pca_tmp.explained_variance_ratio_.sum() * 100)
        individual_variances.append(pca_tmp.explained_variance_ratio_[-1] * 100)  # Last component's variance

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Cumulative Explained Variance
    ax1.plot(n_components_range, explained_variances, marker='o', 
             linewidth=2.5, markersize=6, color='#2E86AB', 
             markerfacecolor='#2E86AB', markeredgecolor='white', 
             markeredgewidth=1.5)
    
    # Add reference lines and annotations
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Variance')
    ax1.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
    ax1.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% Variance')
    
    # Find and annotate key points
    idx_80 = next((i for i, v in enumerate(explained_variances) if v >= 80), None)
    idx_90 = next((i for i, v in enumerate(explained_variances) if v >= 90), None)
    idx_95 = next((i for i, v in enumerate(explained_variances) if v >= 95), None)
    
    if idx_80: ax1.annotate(f'{idx_80+1} components', xy=(idx_80+1, 80), xytext=(idx_80+3, 75),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    if idx_90: ax1.annotate(f'{idx_90+1} components', xy=(idx_90+1, 90), xytext=(idx_90+3, 85),
                           arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))
    
    ax1.set_xlabel('Number of PCA Components', fontweight='bold')
    ax1.set_ylabel('Cumulative Explained Variance (%)', fontweight='bold')
    ax1.set_title('Cumulative Explained Variance', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    
    # Plot 2: Scree Plot (Individual Component Variance)
    ax2.plot(n_components_range, individual_variances, marker='o', 
             linewidth=2.5, markersize=6, color='#A23B72',
             markerfacecolor='#A23B72', markeredgecolor='white', 
             markeredgewidth=1.5)
    ax2.set_xlabel('Component Number', fontweight='bold')
    ax2.set_ylabel('Individual Explained Variance (%)', fontweight='bold')
    ax2.set_title('Scree Plot (Individual Components)', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Plot 3: Elbow Detection (Rate of Change)
    rate_of_change = [explained_variances[i] - explained_variances[i-1] 
                     for i in range(1, len(explained_variances))]
    
    ax3.plot(range(2, len(n_components_range)+1), rate_of_change, marker='s', 
             linewidth=2.5, markersize=6, color='#F18F01',
             markerfacecolor='#F18F01', markeredgecolor='white', 
             markeredgewidth=1.5)
    ax3.set_xlabel('Number of PCA Components', fontweight='bold')
    ax3.set_ylabel('Marginal Variance Explained (%)', fontweight='bold')
    ax3.set_title('Marginal Variance Added per Component', fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Plot 4: Kaiser Criterion Visualization
    pca_full = PCA()
    pca_full.fit(pca_data)
    eigenvalues = pca_full.explained_variance_
    
    ax4.bar(range(1, len(eigenvalues)+1), eigenvalues, 
            color=['#C73E1D' if ev >= 1 else 'lightgray' for ev in eigenvalues],
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Kaiser Criterion (λ=1)')
    ax4.set_xlabel('Component Number', fontweight='bold')
    ax4.set_ylabel('Eigenvalue', fontweight='bold')
    ax4.set_title('Kaiser Criterion (Eigenvalues > 1)', fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax4.legend()
    
    # Style all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Print recommendations
    print("PCA Component Selection Recommendations:")
    print("="*50)
    kaiser_components = sum(1 for ev in eigenvalues if ev >= 1)
    print(f"Kaiser Criterion: {kaiser_components} components (eigenvalues > 1)")
    if idx_80: print(f"80% Variance: {idx_80+1} components")
    if idx_90: print(f"90% Variance: {idx_90+1} components")
    if idx_95: print(f"95% Variance: {idx_95+1} components")
    
    return None 



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
    inertia_color = [0.55294118, 0.62745098, 0.79607843, 1.0]  # Soft blue
    silhouette_color = [0.65098039, 0.84705882, 0.32941176, 1.0]  # Soft green

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
    bars = ax.barh(range(len(importance_series)), importance_series.values, color=[1.0, 0.85098039, 0.18431373, 1.0],  # Soft yellow
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



def plot_clusters_map(info_df, cluster_col='cluster'):

    # Get unique clusters and colors
    unique_clusters = sorted(info_df['cluster'].unique())
    colors = px.colors.qualitative.G10

    # Create figure with individual traces for each cluster
    fig = go.Figure()

    # Add a separate trace for each cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_data = info_df[info_df['cluster'] == cluster]
        
        # Create hover text
        hover_text = [
            f"Customer ID: {cid}<br>Cluster: {cluster}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}<br>Age: {age}"
            for cid, lat, lon, age in zip(
                cluster_data['customer_id'], 
                cluster_data['latitude'], 
                cluster_data['longitude'],
                cluster_data['age']
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
    center_lat = info_df['latitude'].mean()
    center_lon = info_df['longitude'].mean()


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
            'text': f'Customer Clusters - {len(info_df)} customers',
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



# ---------- After Clustering Visualizations ---------- #

def plot_umap_2d(info_df_clustered):

    # Apply UMAP for dimensionality reduction
    umap_model = UMAP(n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(info_df_clustered.drop(columns=['cluster'] , axis=1))

    # Create the visualization with enhanced styling
    plt.figure(figsize=(10, 8))

    # Create scatter plot with enhanced styling
    scatter = sns.scatterplot(
        x=umap_result[:, 0], 
        y=umap_result[:, 1], 
        hue=info_df_clustered['cluster'], 
        palette='Set2',
        alpha=0.8,
        s=80,
        edgecolors='white',
        linewidth=0.5
    )

    # Get current axes
    ax = plt.gca()

    # Customize the plot with consistent styling
    plt.title('UMAP Projection of Customer Segments', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')

    # Make tick labels bold
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Improve legend with bold styling
    legend = plt.legend(title='Cluster', title_fontsize=12, fontsize=10, 
                    frameon=True, fancybox=True, shadow=True)
    legend.get_title().set_fontweight('bold')
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Add grid and clean styling
    ax.grid(True, alpha=0.3, axis='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Set background color to white for better contrast
    ax.set_facecolor('white')

    # Tight layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()



def plot_cluster_boxplots(info_df_with_cluster):
    # Select numeric columns (excluding customer_id and cluster)
    numeric_cols = info_df_with_cluster.select_dtypes(include=[np.number]).columns.difference(['cluster'])

    # Plot boxplots for each numeric variable grouped by cluster
    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        # Create boxplot with enhanced styling
        box_plot = sns.boxplot(
            data=info_df_with_cluster,
            x='cluster',
            y=col,
            ax=axes[i],
            palette='Set2',  # Use Set2 to match other visualizations
            linewidth=1.5,
            flierprops=dict(marker='o', markersize=4, alpha=0.6, markeredgecolor='white', markeredgewidth=0.5)
        )
        
        # Customize the plot
        axes[i].set_xlabel('Cluster', fontsize=10, fontweight='bold')
        axes[i].set_ylabel(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        axes[i].set_title(f'Boxplot of {col.replace("_", " ").title()} by Cluster', 
                        fontsize=11, fontweight='bold', pad=15)
        
        # Make tick labels bold
        axes[i].tick_params(axis='x', labelsize=9)
        axes[i].tick_params(axis='y', labelsize=9)
        for tick in axes[i].get_xticklabels():
            tick.set_fontweight('bold')
        for tick in axes[i].get_yticklabels():
            tick.set_fontweight('bold')
        
        # Add grid and clean styling
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_linewidth(1.5)
        axes[i].spines['bottom'].set_linewidth(1.5)
        
        # Set background color to white
        axes[i].set_facecolor('white')
        
        # Format x-axis labels as C0, C1, etc.
        cluster_labels = [f'C{int(label.get_text())}' for label in axes[i].get_xticklabels()]
        axes[i].set_xticklabels(cluster_labels, fontweight='bold')

    # Remove unused subplots
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()








