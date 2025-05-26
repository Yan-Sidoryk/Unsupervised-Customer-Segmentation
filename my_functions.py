# ---------- Importing the libraries ---------- #
import pandas as pd
import numpy as np

import datetime as dt
import reverse_geocoder as rg
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans, DBSCAN, MeanShift
import scipy.cluster.hierarchy as sch
from minisom import MiniSom



# ---------- Feature Engineering Functions ---------- #

# Transform latitude and longitude into city names
def get_cities(df, lat_col='latitude', lon_col='longitude'):

    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Filter rows with valid coordinates
    valid_mask = (~df[lat_col].isna()) & (~df[lon_col].isna())
    valid_coords = df[valid_mask]
    
    if len(valid_coords) == 0:
        result_df['city'] = np.nan
        return result_df
    
    # Convert coordinates to list of tuples (required by reverse_geocoder)
    coord_tuples = [(row[lat_col], row[lon_col]) for _, row in valid_coords.iterrows()]
    
    # Batch geocode all coordinates at once
    results = rg.search(coord_tuples)
    
    # Create a city column filled with NaN
    result_df['city'] = np.nan
    
    # Populate city for valid coordinates
    for i, idx in enumerate(valid_coords.index):
        # Extract city name from results
        result_df.at[idx, 'city'] = results[i]['name']
           
    return result_df['city']



def feature_engineering_info(data):

    data.drop('Unnamed: 0', axis=1, inplace=True)

    # Get ages of customers from birthdate
    data['customer_birthdate'] = pd.to_datetime(data['customer_birthdate'])
    data['age'] = dt.datetime.now().year - data['customer_birthdate'].dt.year
    # Drop origional birthdate column
    data.drop('customer_birthdate', axis=1, inplace=True)

    # Add shopping time patterns 
    data['morning_shopper'] = data['typical_hour'].between(6, 11).astype(int)
    data['afternoon_shopper'] = data['typical_hour'].between(12, 17).astype(int)
    data['evening_shopper'] = data['typical_hour'].between(18, 23).astype(int)
    # data.drop('typical_hour', axis=1, inplace=True)  >>> Not dropping the original column for now, might need for visualization <<<

    # Add city column
    data['city'] = get_cities(data)
    # Drop coordinates columns
    # data.drop(['latitude', 'longitude'], axis=1, inplace=True)    >>>Drop them later, now kulled for visualization<<<

    # Add total lifetime spend column
    spend_columns = [col for col in data.columns if 'lifetime_spend_' in col]
    data['total_lifetime_spend'] = data[spend_columns].sum(axis=1)

    # Add columns for percentage of total spend
    for col in spend_columns:     
        data[f'spend_{"_".join(col.split("_")[2:])}_percent'] = (data[col] / data['total_lifetime_spend']) * 100

    # Add column for education level 
    data['degree_level'] = data['customer_name'].str.extract(r'^([^\.]+)\.').fillna('None')

    # >>> I think we should drop the original spend columns, but not doing it for now <<<

    # Replace loyalty card number with binary column
    data['loyalty_card'] = data['loyalty_card_number'].notna().astype(int)
    data.drop('loyalty_card_number', axis=1, inplace=True)

    # Add total childern column
    data['total_children'] = data['kids_home'] + data['teens_home']   # >>> I think we should drop the original columns, but not doing it for now <<<

def feature_engineering_basket(data):

    # Remove brackets and quotes from string of goods
    data['list_of_goods'] = data['list_of_goods'].apply(lambda x: x.replace('[', '').replace(']', '').replace('\'', ''))

    # Change list of goods to be a list
    data['list_of_goods'] = data['list_of_goods'].apply(lambda x: x.split(', '))

    # Add column for number of items in the basket
    data['n_items'] = data['list_of_goods'].apply(lambda x: len(x))



def all_purchased_items(basket_df):
    return basket_df.groupby('customer_id').apply(
        lambda group: pd.Series({
            'all_purchased_items': [item for sublist in group['list_of_goods'] for item in sublist]
        })
    ).reset_index()

    

# ---------- Clustering Functions ---------- #

def imputation_scaling(info_df):

    # Create a copy of the DataFrame to avoid modifying the original
    info_df_scaled = info_df.copy()
    
    # Get the numeric, discrete and categorical columns
    info_col_continuous = [
                    'lifetime_spend_groceries',
                    'lifetime_spend_electronics',
                    'lifetime_spend_vegetables',
                    'lifetime_spend_nonalcohol_drinks',
                    'lifetime_spend_alcohol_drinks',
                    'lifetime_spend_meat',
                    'lifetime_spend_fish',
                    'lifetime_spend_hygiene',
                    'lifetime_spend_videogames',
                    'lifetime_spend_petfood',
                    'lifetime_total_distinct_products',
                    'percentage_of_products_bought_promotion',
                    'total_lifetime_spend',
                    'spend_groceries_percent',
                    'spend_electronics_percent',
                    'spend_vegetables_percent',
                    'spend_nonalcohol_drinks_percent',
                    'spend_alcohol_drinks_percent',
                    'spend_meat_percent',
                    'spend_fish_percent',
                    'spend_hygiene_percent',
                    'spend_videogames_percent',
                    'spend_petfood_percent']

    info_col_discrete  = [
                    'kids_home',
                    'teens_home',
                    'number_complaints',
                    'distinct_stores_visited',
                    'total_children', 
                    'age',
                    'year_first_transaction',
                    'typical_hour']

    info_col_categorical = [ 
                    'customer_gender', 
                    'city', 
                    'degree_level']


    # Not included 'customer_id', 'customer_name' and 'typical_hour'
    # Also 'loyalty_card' cannot have missing values, so it is not included 
    # We may have situations where 'morning_shopper', 'afternoon_shopper' and 'evening_shopper' are all 0 for a customer, but mb it's fine ??
    # >>> We also have negative values in 'percentage_of_products_bought_promotion' <<< !!!


    # Fill missing values for continuous columns with the mean and scale them
    for col in info_col_continuous:
        info_df_scaled[col].fillna(info_df_scaled[col].mean(), inplace=True)

        scaler = RobustScaler()
        info_df_scaled[col] = scaler.fit_transform(info_df_scaled[[col]])


    # Fill missing values for discrete columns with the median and scale them
    for col in info_col_discrete:
        info_df_scaled[col].fillna(info_df_scaled[col].median(), inplace=True)

        scaler = RobustScaler()
        info_df_scaled[col] = scaler.fit_transform(info_df_scaled[[col]])


    # Fill missing values for categorical columns with mode
    for col in info_col_categorical:
        info_df_scaled[col].fillna(info_df_scaled[col].mode()[0], inplace=True)


    # One-hot encode the categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(info_df_scaled[info_col_categorical])

    # Create DataFrame with encoded column names
    feature_names = encoder.get_feature_names_out(info_col_categorical)
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=info_df_scaled.index)

    # Remove original categorical columns and add encoded ones
    info_df_scaled.drop(columns=info_col_categorical, inplace=True)
    info_df_scaled = pd.concat([info_df_scaled, encoded_df], axis=1) 

    return info_df_scaled



def k_distance_graph(info_df_scaled, k):

    X = info_df_scaled[['total_lifetime_spend', 'lifetime_spend_groceries',
       'lifetime_spend_electronics', 'lifetime_spend_vegetables',
       'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
       'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
       'lifetime_spend_videogames', 'lifetime_spend_petfood', 'total_children']].values  # Adjust specific columns

    # Compute the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)

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

    # Implementing the DBSCAN algorithm
    info_df_clustered['DBScan'] = DBSCAN(
        eps=eps, 
        min_samples=min_samples
        ).fit_predict(info_df_scaled[['total_lifetime_spend', 'lifetime_spend_groceries',
        'lifetime_spend_electronics', 'lifetime_spend_vegetables',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
        'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
        'lifetime_spend_videogames', 'lifetime_spend_petfood', 'total_children']])

    # Plot the number of customers in each cluster
    info_df_clustered.groupby(['DBScan']).size().plot(kind='bar', color='skyblue')
    plt.show()

    # And print for better readability
    print(info_df_clustered.groupby(['DBScan']).size())

    # Even with such a high eps, we still have outliers, so we can remove them
    info_df_scaled = info_df_scaled[info_df_clustered['DBScan'] != -1]
    info_df_scaled.reset_index(drop=True, inplace=True)

    # Save outliers to a separate DataFrame
    outliers_df = info_df_clustered[info_df_clustered['DBScan'] == -1]

    return info_df_scaled, outliers_df



def dimensionality_reduction(info_df_scaled):

    # Reducing the dimensionality of the data to the most important features

    # info_df_scaled.drop(columns=['spend_groceries_percent',
    #        'spend_electronics_percent', 'spend_vegetables_percent',
    #        'spend_nonalcohol_drinks_percent', 'spend_alcohol_drinks_percent',
    #        'spend_meat_percent', 'spend_fish_percent', 'spend_hygiene_percent',
    #        'spend_videogames_percent', 'spend_petfood_percent'], inplace=True)

    # Useless columns
    info_df_scaled.drop(columns=['customer_name', 'latitude', 'longitude'], inplace=True)

    # Redundant columns
    info_df_scaled.drop(columns=['lifetime_spend_groceries',
        'lifetime_spend_electronics', 'lifetime_spend_vegetables',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
        'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
        'lifetime_spend_videogames', 'lifetime_spend_petfood'], inplace=True)


    # Irrelevant columns
    info_df_scaled.drop(columns=['city_Almada', 'city_Amadora', 'city_Bobadela',
        'city_Cacilhas', 'city_Camarate', 'city_Famoes', 'city_Lisbon',
        'city_Moscavide', 'city_Odivelas', 'city_Olival do Basto',
        'city_Pontinha', 'city_Pragal', 'city_Sacavem'], inplace=True)

    info_df_scaled.drop(columns=['degree_level_Msc', 'degree_level_Phd', 
        'morning_shopper', 'evening_shopper', 'afternoon_shopper'], inplace=True)            

    
    # TESTING

    # info_df_scaled.drop(columns=['kids_home', 'teens_home'], inplace=True) , 'typical_hour'   'customer_gender_male', 
    info_df_scaled.drop(columns=['total_children', 'typical_hour', 'year_first_transaction', 'customer_gender_male'], inplace=True)



def elbow_and_silhouette(info_df_scaled, max_k=21):

    # Setup for tracking both metrics
    k_range = range(2, max_k)  # Silhouette score requires at least 2 clusters
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



def kmeans_clustering(info_df_scaled, k):

    # Fit KMeans with your chosen number of clusters 
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(info_df_scaled.drop(columns=['customer_id']))

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


