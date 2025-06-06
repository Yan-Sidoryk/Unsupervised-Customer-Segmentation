# ---------- Importing the libraries ---------- #
import pandas as pd
import numpy as np

import datetime as dt
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans, DBSCAN, MeanShift
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from minisom import MiniSom

from collections import Counter, defaultdict

from visualizations import plot_dbscan_counts



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

    # Make descrete numerical columns int again
    data['kids_home'] = data['kids_home'].astype(int)
    data['teens_home'] = data['teens_home'].astype(int)
    data['distinct_stores_visited'] = data['distinct_stores_visited'].astype(int)
    data['number_complaints'] = data['number_complaints'].astype(int)
    data['typical_hour'] = data['typical_hour'].astype(int)


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
    # Define the mapping for degree_level to ordinal values
    degree_level_mapping = {
        'None': 0,
        'Bsc': 1,
        'Msc': 2,
        'Phd': 3
    }

    # Apply the mapping to the degree_level column
    data['degree_level_ordinal'] = data['degree_level'].map(degree_level_mapping)

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

def extra_preprocessing(data, k=5):

    # Copy the original DataFrame to preserve the original data
    info_df = data.copy()

    # Drop the customers name
    info_df.drop(columns=['customer_name'], inplace=True)

    # Drop the columns that were only kept for visualization
    info_df.drop(columns=['morning_shopper', 'afternoon_shopper', 'evening_shopper', 'degree_level'], inplace=True)
    # info_df.drop(columns=['lifetime_spend_groceries',
    #     'lifetime_spend_electronics', 'lifetime_spend_vegetables',
    #     'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
    #     'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
    #     'lifetime_spend_videogames', 'lifetime_spend_petfood'], inplace=True)


    # Separate categorical columns
    categorical_cols = info_df.select_dtypes(include=['object']).columns.tolist()
    

    # One-hot encode categorical columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    info_df_cat_encoded = pd.DataFrame(
        encoder.fit_transform(info_df[categorical_cols].fillna('missing')),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=info_df.index
    )

    # Get the Marko clients
    marko_clients_id_list = info_df[
        (
            (info_df['longitude'] >= -9.214894) & (info_df['longitude'] <= -9.213008) &
            (info_df['latitude'] >= 38.72212) & (info_df['latitude'] <= 38.72405)
        )]['customer_id'].tolist()

    # Drop latitude and longitude here
    info_df.drop(columns=['latitude', 'longitude'], inplace=True)

    # Separate numerical columns after dropping latitude and longitude
    numerical_cols = info_df.select_dtypes(include=[np.number]).columns.difference(['customer_id']).tolist()

    # Scale numerical columns with RobustScaler
    scaler = RobustScaler()
    info_df_num_scaled = pd.DataFrame(scaler.fit_transform(info_df[numerical_cols]), columns=numerical_cols, index=info_df.index)

    # Combine all features
    info_df_scaled = pd.concat([info_df[['customer_id']], info_df_num_scaled, info_df_cat_encoded], axis=1)

    return info_df_scaled[~info_df_scaled['customer_id'].isin(marko_clients_id_list)], info_df_scaled[info_df_scaled['customer_id'].isin(marko_clients_id_list)]



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



def remove_outliers(info_df_scaled, eps, min_samples):

    # Create a temporary DataFrame to store cluster labels 
    info_df_clustered = info_df_scaled.copy()

    info_df_clustered['DBScan'] = DBSCAN(
        eps=eps, 
        min_samples=min_samples
        ).fit_predict(info_df_scaled.drop(columns=['customer_id'], axis=1))
        

    # Plot the number of customers in each cluster
    plot_dbscan_counts(info_df_clustered)

    # Save outliers to a separate DataFrame
    outliers_df = info_df_scaled[info_df_clustered['DBScan'] == -1]

    # Remove outliers, if there are any
    info_df_scaled = info_df_scaled[info_df_clustered['DBScan'] != -1]
    info_df_scaled.reset_index(drop=True, inplace=True)

    print(f"Number of outliers removed: {len(outliers_df)}")

    return info_df_scaled, outliers_df



def kmeans_clustering(info_df_pca, outliers_df, info_df_scaled, k):

    # Fit KMeans with your chosen number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(info_df_pca.drop(columns=['customer_id']))

    # Get the cluster labels for the main data
    info_df_pca['cluster'] = kmeans.predict(info_df_pca.drop(columns=['customer_id']))

    # Get the cluster labels for the outliers
    outliers_df['cluster'] = kmeans.predict(outliers_df.drop(columns=['customer_id']))

    # Combine the lables from both DataFrames with the info_df_scaled for final output
    info_df_clustered = info_df_scaled.merge(
            pd.concat([info_df_pca[['customer_id', 'cluster']],
                        outliers_df[['customer_id', 'cluster']]]),
            on='customer_id',
            how='right'
            )

    return info_df_clustered






def generate_cluster_profiles(info_df_clustered):
    # Create a DataFrame to hold the cluster profiles for each cluster
    # This will give you the mean values of each feature for each cluster
    return info_df_clustered.drop(columns=['customer_id']).groupby('cluster').mean().round(2)




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




from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def get_association_rules_mlxtend(basket_df, info_df_clustered, 
                                  min_support=0.05, min_threshold=1.3):
    """
    Using MLxtend - the most popular association rules library.
    Very efficient and well-documented.
    """
    # Merge data
    df = basket_df.merge(info_df_clustered, on='customer_id', how='inner')
    df = df.dropna(subset=['cluster', 'list_of_goods'])
    
    results = {}
    
    for cluster in df['cluster'].unique():
        print(f"\nProcessing cluster {cluster}...")
        cluster_data = df[df['cluster'] == cluster]
        
        # Prepare transactions
        transactions = []
        for items in cluster_data['list_of_goods']:
            if isinstance(items, list):
                transactions.append(items)
            else:
                transactions.append([str(items)])
        
        if len(transactions) < 10:
            continue
            
        # Convert to one-hot encoded format
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            continue
            
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
        
        if len(rules) == 0:
            continue
            
        # Format results
        rules_list = []
        for _, rule in rules.iterrows():
            rules_list.append({
                'antecedent': ', '.join(list(rule['antecedents'])),
                'consequent': ', '.join(list(rule['consequents'])),
                'support': rule['support'],
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'conviction': rule['conviction']
            })
        
        # Sort by lift and take top 5
        rules_list.sort(key=lambda x: x['lift'], reverse=True)
        
        results[cluster] = {
            'rules': rules_list[:5],
            'total_transactions': len(transactions),
            'total_rules_found': len(rules_list)
        }
        
        print(f"  Found {len(rules_list)} rules from {len(transactions)} transactions")
    
    return df, results