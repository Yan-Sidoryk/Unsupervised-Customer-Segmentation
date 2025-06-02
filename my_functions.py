# ---------- Importing the libraries ---------- #
import pandas as pd
import numpy as np

import datetime as dt
import reverse_geocoder as rg
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans, DBSCAN, MeanShift
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from minisom import MiniSom

from collections import Counter, defaultdict



# ---------- Feature Engineering Functions ---------- #

def feature_engineering_info(data, k=5):

    # Drop the index column 
    data.drop('Unnamed: 0', axis=1, inplace=True)

    # Replace loyalty card number with binary column
    data['loyalty_card'] = data['loyalty_card_number'].notna().astype(int)
    data.drop('loyalty_card_number', axis=1, inplace=True)

    # Get ages of customers from birthdate
    data['customer_birthdate'] = pd.to_datetime(data['customer_birthdate'])
    data['age'] = dt.datetime.now().year - data['customer_birthdate'].dt.year
    # Drop origional birthdate column
    data.drop('customer_birthdate', axis=1, inplace=True)


    # Impute missing values before more feature engineering
    # Separate categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=[np.number]).columns.difference(['customer_id']).tolist()

    # KNN Imputation for numerical columns
    imputer = KNNImputer(n_neighbors=k)
    data_imputed = pd.DataFrame(imputer.fit_transform(data[numerical_cols]), columns=numerical_cols, index=data.index)

    # Build back a dataframe with the imputed numerical columns and original categorical columns
    data = pd.concat([data[['customer_id']], data_imputed, data[categorical_cols]], axis=1)


    # Add shopping time patterns 
    data['morning_shopper'] = data['typical_hour'].between(6, 11).astype(int)
    data['afternoon_shopper'] = data['typical_hour'].between(12, 17).astype(int)
    data['evening_shopper'] = data['typical_hour'].between(18, 23).astype(int)
    # data.drop('typical_hour', axis=1, inplace=True)  >>> Not dropping the original column for now, might need for visualization <<<

    # Add total lifetime spend column
    spend_columns = [col for col in data.columns if 'lifetime_spend_' in col]
    data['total_lifetime_spend'] = data[spend_columns].sum(axis=1)

    # Add columns for percentage of total spend
    for col in spend_columns:     
        data[f'spend_{"_".join(col.split("_")[2:])}_percent'] = (data[col] / data['total_lifetime_spend']) * 100
     # >>> Keeping the original spend columns for the visualization <<<

    # Add column for education level 
    data['degree_level'] = data['customer_name'].str.extract(r'^([^\.]+)\.').fillna('None')

    # Add total childern column
    data['total_children'] = data['kids_home'] + data['teens_home']   

    # Add customer_for column
    data['customer_for'] = dt.datetime.now().year - data['year_first_transaction']
    data.drop('year_first_transaction', axis=1, inplace=True)

    # Remove the negative values from percentage of products bought on promotion (probably a typing error)
    data['percentage_of_products_bought_promotion'] = data['percentage_of_products_bought_promotion'].abs()

    return data



def feature_engineering_basket(data):

    # Remove brackets and quotes from string of goods
    data['list_of_goods'] = data['list_of_goods'].apply(lambda x: x.replace('[', '').replace(']', '').replace('\'', ''))

    # Change list of goods to be a list
    data['list_of_goods'] = data['list_of_goods'].apply(lambda x: x.split(', '))

    # Add column for number of items in the basket
    data['n_items'] = data['list_of_goods'].apply(lambda x: len(x))

    return data



def all_purchased_items(basket_df):
    return basket_df.groupby('customer_id').apply(
        lambda group: pd.Series({
            'all_purchased_items': [item for sublist in group['list_of_goods'] for item in sublist]
        })
    ).reset_index()

    

# ---------- Clustering Functions ---------- #

def extra_preprocessing(info_df):

    # Drop the customers from Marko
    info_df = info_df[
        ~(
            (info_df['longitude'] >= -9.214894) & (info_df['longitude'] <= -9.213011) &
            (info_df['latitude'] >= 38.72212) & (info_df['latitude'] <= 38.72405)
        )]

    # Drop the customers name
    info_df.drop(columns=['customer_name'], inplace=True)

    # Drop the columns that were only kept for visualization
    info_df.drop(columns=['latitude', 'longitude'], inplace=True)
    # info_df.drop(columns=['lifetime_spend_groceries',
    #     'lifetime_spend_electronics', 'lifetime_spend_vegetables',
    #     'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
    #     'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
    #     'lifetime_spend_videogames', 'lifetime_spend_petfood'], inplace=True)

    return info_df



def encoding_scaling(info_df, k=5):

    # Separate categorical and numerical columns
    categorical_cols = info_df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = info_df.select_dtypes(include=[np.number]).columns.difference(['customer_id']).tolist()

    # One-hot encode categorical columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    info_df_cat_encoded = pd.DataFrame(
        encoder.fit_transform(info_df[categorical_cols].fillna('missing')),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=info_df.index
    )

    # Scale numerical columns with RobustScaler
    scaler = RobustScaler()
    info_df_num_scaled = pd.DataFrame(scaler.fit_transform(info_df[numerical_cols]), columns=numerical_cols, index=info_df.index)

    # Combine all features
    info_df_scaled = pd.concat([info_df[['customer_id']], info_df_num_scaled, info_df_cat_encoded], axis=1)

    return info_df_scaled
    


def k_distance_graph(info_df_scaled, k):

    # Compute the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(info_df_scaled)
    distances, indices = nbrs.kneighbors(info_df_scaled)

    # Sort the distances in ascending order
    distances = np.sort(distances[:, k-1])  # Get the distance to the kth nearest neighbor

    # Create the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {k}th nearest neighbor')
    plt.title(f'K-Distance Graph (k={k})')
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def remove_outliers(info_df_scaled, eps, min_samples):

    # Create a temporary DataFrame to store cluster labels 
    info_df_clustered = info_df_scaled.copy()

    info_df_clustered['DBScan'] = DBSCAN(
        eps=eps, 
        min_samples=min_samples
        ).fit_predict(info_df_scaled.drop(columns=['customer_id'], axis=1))
        

    # Plot the number of customers in each cluster
    info_df_clustered.groupby(['DBScan']).size().plot(kind='bar', color='skyblue')
    plt.show()

    # Remove outliers, if there are any
    info_df_scaled = info_df_scaled[info_df_clustered['DBScan'] != -1]
    info_df_scaled.reset_index(drop=True, inplace=True)

    # Save outliers to a separate DataFrame
    outliers_df = info_df_clustered[info_df_clustered['DBScan'] == -1]

    print(f"Number of outliers removed: {len(outliers_df)}")

    return info_df_scaled, outliers_df



def pca_graph(info_df_scaled):

    # Plot explained variance ratio for different numbers of PCA components 
    explained_variances = []
    n_components_range = range(1, info_df_scaled.shape[1])  # Up to max features

    for n in n_components_range:
        pca_tmp = PCA(n_components=n)
        pca_tmp.fit(info_df_scaled.drop(columns=['customer_id'], axis=1))  # Exclude customer_id from PCA
        explained_variances.append(pca_tmp.explained_variance_ratio_.sum() * 100)  # Convert to percent

    plt.figure(figsize=(8, 5))
    plt.plot(n_components_range, explained_variances, marker='o')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('PCA: Explained Variance vs Number of Components')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def dimensionality_reduction(info_df_scaled, n_comp):

    # Apply PCA for dimensionality reduction
    info_df_pca = info_df_scaled[['customer_id']].copy()
    pca = PCA(n_components=n_comp, random_state=42)
    pca_result = pca.fit_transform(info_df_scaled.drop(columns=['customer_id'], axis=1))
    for i in range(n_comp):
        info_df_pca[f'pca{i+1}'] = pca_result[:, i]


    explained_variance = pca.explained_variance_ratio_
    print(f"Variance explained by each component: {explained_variance}")
    print(f"Total variance explained by {n_comp} components: {sum(explained_variance):.2%}")

    return info_df_pca



def elbow_and_silhouette(info_df_scaled, max_k, step=1):

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
        
    # Create a figure with two subplots
    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Plot 1: Elbow Method on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Sum of Squared Distances)', color=color, fontsize=12)
    ax1.plot(k_range, inertia_values, 'o-', color=color, markersize=8, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
    ax2.plot(k_range, silhouette_scores, 'o-', color=color, markersize=8, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title
    plt.title('Elbow Method and Silhouette Score for Optimal k', fontsize=14)
    plt.xticks(k_range)  # Set x-ticks to match k_range

    # Adjust layout
    plt.tight_layout()
    plt.show()



def kmeans_clustering(info_df_pca, info_df_scaled, k):

    # Fit KMeans with your chosen number of clusters 
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(info_df_pca.drop(columns=['customer_id']))

    # Create a new DataFrame to hold the scaled data and cluster labels
    info_df_clustered = info_df_scaled.copy()
    info_df_clustered['cluster'] = clusters

    # Create a DataFrame to hold the cluster profiles for each cluster
    # This will give you the mean values of each feature for each cluster
    cluster_profiles = info_df_clustered.drop(columns=['customer_id']).groupby('cluster').mean().round(2)

    return info_df_clustered, cluster_profiles



def generate_cluster_names(cluster_profiles, z_threshold, max_features=4):

    labels = {}
    
    # Calculate global mean and std for normalization reference
    global_mean = cluster_profiles.mean().mean()
    global_std = cluster_profiles.values.std()
    
    for i, row in cluster_profiles.iterrows():
        # Calculate z-scores for each feature
        z_scores = (row - global_mean) / global_std
        
        # Get significant high features (z-score > threshold)
        high_features = z_scores[z_scores > z_threshold].sort_values(ascending=False)
        high_features = high_features.head(max_features).index.tolist()
        
        # Get significant low features (z-score < -threshold)
        low_features = z_scores[z_scores < -z_threshold].sort_values(ascending=True)
        low_features = low_features.head(max_features).index.tolist()
        
        # Create labels
        high_label = "HIGH : " + ", ".join(f.replace('_', ' ') for f in high_features) if high_features else "No significant high features"
        low_label = "LOW : " + ", ".join(f.replace('_', ' ') for f in low_features) if low_features else "No significant low features"
        
        labels[i] = f"{high_label} | {low_label}"
    
    return labels



def calculate_f_statistic_importance(df, cluster_col='cluster'):

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

    # Visualization of feature importance
    plt.figure(figsize=(15, 7))
    sns.barplot(y=importance_series.index, x=importance_series.values, color='skyblue')
    plt.title('Feature Importance (F-statistic)', fontsize=16)
    plt.ylabel('')
    plt.tight_layout()
    plt.show()



# ---------- Association Rules ---------- #

def get_association_rules_by_cluster(basket_df, info_df_clustered, min_lift=1.3):
    """
    Find association rules: "If customer buys A, they're likely to buy B"
    Returns top 5 rules per cluster with highest lift
    """
    # Step 1: Merge with cluster info
    df = basket_df.merge(info_df_clustered, on='customer_id', how='left')
    df['cluster'] = df['cluster'].fillna(-1).astype(int)
    
    results = {}
    
    # Step 2: Process each cluster
    for cluster in df['cluster'].unique()[df['cluster'].unique() != -1]:  # Exclude -1 cluster (if any)
        
        cluster_data = df[df['cluster'] == cluster]
        
        # Get transactions (baskets) for this cluster
        transactions = []
        for _, row in cluster_data.iterrows():
            items = row['list_of_goods']
            if isinstance(items, list):
                transactions.append(set(items))
            else:
                transactions.append({items})
        
        total_transactions = len(transactions)
        
        if total_transactions < 10:
            print(f"  Skipping cluster {cluster} - too few transactions")
            continue
        
        # Count individual items
        item_counts = Counter()
        for basket in transactions:
            for item in basket:
                item_counts[item] += 1
        
        # Only consider items that appear in at least 5 transactions
        min_support_count = max(5, int(0.05 * total_transactions))
        frequent_items = [item for item, count in item_counts.items() 
                         if count >= min_support_count]
        
        if len(frequent_items) < 2:
            print(f"  Skipping cluster {cluster} - not enough frequent items")
            continue
        
        # Find association rules: A → B
        rules = []
        
        for item_a in frequent_items:
            for item_b in frequent_items:
                if item_a != item_b:
                    # Count transactions
                    has_a = 0          # P(A)
                    has_b = 0          # P(B)  
                    has_both = 0       # P(A ∩ B)
                    
                    for basket in transactions:
                        if item_a in basket:
                            has_a += 1
                            if item_b in basket:
                                has_both += 1
                        if item_b in basket:
                            has_b += 1
                    
                    if has_a > 0 and has_both > 0:
                        # Calculate metrics
                        support_a = has_a / total_transactions
                        support_b = has_b / total_transactions
                        support_ab = has_both / total_transactions
                        confidence = has_both / has_a  # P(B|A)
                        lift = confidence / support_b if support_b > 0 else 0
                        
                        # Only keep rules with decent confidence and lift
                        if confidence >= 0.1 and lift > min_lift:
                            rules.append({
                                'antecedent': item_a,      # If customer buys this...
                                'consequent': item_b,      # ...they're likely to buy this
                                'support': support_ab,
                                'confidence': confidence,
                                'lift': lift,
                                'count_a': has_a,
                                'count_both': has_both
                            })
        
        # Sort by lift and get top 5
        rules.sort(key=lambda x: x['lift'], reverse=True)
        top_rules = rules[:5]
        
        results[cluster] = {
            'rules': top_rules,
            'total_transactions': total_transactions,
            'frequent_items_count': len(frequent_items)
        }
        
    return df, results



def print_recommendations(cluster_results, cluster_names):

    print("\n" + "="*80)
    print("ASSOCIATION RULES RECOMMENDATIONS BY CLUSTER")
    print("="*80)
    
    for cluster in sorted(cluster_results.keys()):
        print(f"\n🔹 CLUSTER {cluster}")
        print("   " + cluster_names[cluster])
        print(f"   Total transactions: {cluster_results[cluster]['total_transactions']}")
        
        rules = cluster_results[cluster]['rules']
        if not rules:
            print("   No strong association rules found")
            continue
        
        print("   Top recommendations:")
        for i, rule in enumerate(rules, 1):
            print(f"   {i}. If customer buys '{rule['antecedent']}'")
            print(f"      → {rule['confidence']:.0%} likely to also buy '{rule['consequent']}'")
            print(f"      → {rule['lift']:.1f}x more likely than average")
            print()


