# ---------- Importing the libraries ---------- #
import pandas as pd
import numpy as np

import datetime as dt
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.decomposition import PCA
from minisom import MiniSom
import scipy.cluster.hierarchy as sch

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

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

    # Add total lifetime spend column
    spend_columns = [col for col in data.columns if 'lifetime_spend_' in col]
    data['total_lifetime_spend'] = data[spend_columns].sum(axis=1)

    # Add columns for percentage of total spend
    for col in spend_columns:     
        data[f'spend_{"_".join(col.split("_")[2:])}_percent'] = (data[col] / data['total_lifetime_spend']) * 100

    # Add column for education level 
    data['degree_level'] = data['customer_name'].str.extract(r'^([^\.]+)\.').fillna('None')
    # Define the mapping for degree_level to ordinal values
    degree_level_mapping = {'None': 0, 'Bsc': 1, 'Msc': 2, 'Phd': 3}
    # Apply the mapping to the degree_level column
    data['degree_level_ordinal'] = data['degree_level'].map(degree_level_mapping)

    # Add total childern column
    data['total_children'] = data['kids_home'] + data['teens_home']   

    # Add customer_for column
    data['customer_for'] = (dt.datetime.now().year - data['year_first_transaction']).clip(0)
    data.drop('year_first_transaction', axis=1, inplace=True)

    # Remove the negative values from percentage of products bought on promotion (probably a typing error)
    data['percentage_of_products_bought_promotion'] = data['percentage_of_products_bought_promotion'].abs() 
    data.loc[data['percentage_of_products_bought_promotion'] > 1, 'percentage_of_products_bought_promotion'] -= 1

    # Add no_kids column
    data['no_kids'] = (data['total_children'] == 0).astype(int)

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

    # Get a list of all items purchased by each customer (with repetitions)
    return basket_df.groupby('customer_id').apply(
        lambda group: pd.Series({
            'all_purchased_items': [item for sublist in group['list_of_goods'] for item in sublist]
        })
    ).reset_index()

    

# ---------- Clustering Functions ---------- #

def extra_preprocessing(data, k=5):

    # Copy the original DataFrame to preserve the original data
    info_df = data.copy()

    # Drop the irrelevant columns
    info_df.drop(columns=['customer_name', 'customer_for', 'customer_gender'], inplace=True) 
    info_df.drop(columns=['teens_home', 'kids_home'], inplace=True)      # , 'kids_home'      

    # Drop the columns that were only kept for visualization
    info_df.drop(columns=['morning_shopper', 'afternoon_shopper', 'evening_shopper', 'degree_level'], inplace=True)

    # Separate categorical columns
    categorical_cols = info_df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    info_df_cat_encoded = pd.DataFrame(
        encoder.fit_transform(info_df[categorical_cols].fillna('missing')),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=info_df.index)

    # Get the Makro clients
    makro_clients_id_list = info_df[
        (
            (info_df['longitude'] >= -9.214894) & (info_df['longitude'] <= -9.213008) &
            (info_df['latitude'] >= 38.72212) & (info_df['latitude'] <= 38.72405)
        )]['customer_id'].tolist()

    # Drop latitude and longitude
    info_df.drop(columns=['latitude', 'longitude'], inplace=True)

    # Separate numerical columns after dropping latitude and longitude
    numerical_cols = info_df.select_dtypes(include=[np.number]).columns.difference(['customer_id']).tolist()

    # Scale numerical columns with RobustScaler
    scaler = RobustScaler()
    info_df_num_scaled = pd.DataFrame(scaler.fit_transform(info_df[numerical_cols]), columns=numerical_cols, index=info_df.index)
    # Manually rescale some columns 
    info_df_num_scaled['age'] = info_df_num_scaled['age'] * 3
    info_df_num_scaled['number_complaints'] = info_df_num_scaled['number_complaints'] * 2
    info_df_num_scaled['spend_vegetables_percent'] = info_df_num_scaled['spend_vegetables_percent'] * 0.5

    # Combine all features
    info_df_scaled = pd.concat([info_df_num_scaled, info_df[['customer_id']], info_df_cat_encoded], axis=1)

    return info_df_scaled[~info_df_scaled['customer_id'].isin(makro_clients_id_list)], info_df_scaled[info_df_scaled['customer_id'].isin(makro_clients_id_list)]



def dimensionality_reduction(info_df_scaled, outliers_df, n_comp):

    # Apply PCA for dimensionality reduction
    info_df_pca = info_df_scaled[['customer_id']].copy()
    pca = PCA(n_components=n_comp, random_state=42).fit(info_df_scaled.drop(columns=['customer_id'], axis=1))
    pca_result = pca.transform(info_df_scaled.drop(columns=['customer_id'], axis=1))
    for i in range(n_comp):
        info_df_pca[f'pca{i+1}'] = pca_result[:, i]

    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Variance explained by each component: {explained_variance}")
    print(f"Total variance explained by {n_comp} components: {sum(explained_variance):.2%}")

    # Repeat for the outliers DataFrame
    outliers_df_pca = outliers_df[['customer_id']].copy()
    outliers_pca = pca.transform(outliers_df.drop(columns=['customer_id'], axis=1))
    for i in range(n_comp):
        outliers_df_pca[f'pca{i+1}'] = outliers_pca[:, i]

    return info_df_pca, outliers_df_pca



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



def kmeans_clustering(info_df_pca, info_df_scaled, k):
    np.random.seed(42)  # For reproducibility
    
    # Fit KMeans with your chosen number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(info_df_pca.drop(columns=['customer_id']))

    # Get the cluster labels for the main data
    info_df_pca['cluster'] = kmeans.predict(info_df_pca.drop(columns=['customer_id']))

    # Combine the lables from both DataFrames with the info_df_scaled for final output
    info_df_clustered = info_df_scaled.merge(
            info_df_pca[['customer_id', 'cluster']],
            on='customer_id',
            how='right'
            )

    return info_df_clustered



def generate_cluster_profiles(info_df_clustered):

    # Create a DataFrame to hold the cluster profiles for each cluster
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



def assign_outliers_to_clusters(outliers_df, clustered_df):

    # Get the feature columns 
    feature_columns = [col for col in clustered_df.columns if col not in ['cluster', 'customer_id']]
    
    # Calculate centroids
    centroids = clustered_df.groupby('cluster')[feature_columns].mean()
    
    # Calculate distances using sklearn
    distances = euclidean_distances(outliers_df[feature_columns], centroids)

    # Assign to nearest cluster
    result_df = outliers_df.copy()
    result_df['cluster'] = centroids.index[np.argmin(distances, axis=1)]

    return result_df



# ---------- Association Rules ---------- #

def get_association_rules(df, min_support):

    # Convert item lists to one-hot encoded format
    te = TransactionEncoder()
    te_ary = te.fit_transform(df['list_of_goods'])
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    print(min_support)

    # Generate frequent itemsets with minimum support
    frequent_itemsets = apriori(
        encoded_df, 
        min_support=min_support,  # Adjust based on data density
        use_colnames=True,
        max_len=2
    )

    if len(frequent_itemsets) > 5:
        # Extract association rules with LIFT metric
        rules = association_rules(
            frequent_itemsets,
            metric="lift",      
            min_threshold=1.0   
        )

        # Add a frozenset key of combined items to identify symmetric rules
        rules['pair_key'] = rules.apply(lambda x: frozenset(x['antecedents']).union(x['consequents']), axis=1)

        # Sort by confidence (or other metrics if you prefer)
        rules = rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])

        # Drop duplicates keeping the highest-confidence version of each rule pair
        rules = rules.drop_duplicates(subset='pair_key', keep='first')


        # Filter and sort relevant columns
        result = rules[[
            'antecedents', 
            'consequents', 
            'lift',
            'support',        # Support for full rule (antecedents + consequents)
            'confidence',
        ]].sort_values(by=['lift', 'confidence'], ascending=[False, False])

        return result

    return ""




