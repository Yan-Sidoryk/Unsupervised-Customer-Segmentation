import pandas as pd
import numpy as np
import datetime as dt
import reverse_geocoder as rg



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



def feature_engineering(data):

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

    # Add city column
    data['city'] = get_cities(data)
    # Drop coordinates columns
    data.drop(['latitude', 'longitude'], axis=1, inplace=True)

    # Add total lifetime spend column
    spend_columns = [col for col in data.columns if 'lifetime_spend_' in col]
    data['total_lifetime_spend'] = data[spend_columns].sum(axis=1)

    # Add columns for percentage of total spend
    for col in spend_columns:     
        data[f'spend_{"_".join(col.split("_")[2:])}_percent'] = (data[col] / data['total_lifetime_spend']) * 100

    # >>> I think we should drop the original spend columns, but not doing it for now <<<

    # Replace loyalty card number with binary column
    data['loyalty_card'] = data['loyalty_card_number'].notna().astype(int)
    data.drop('loyalty_card_number', axis=1, inplace=True)

    # Add total childern column
    # data['total_children'] = data['kids_home'] + data['teens_home']   >>> Not sure if this is a good variable <<<

    
