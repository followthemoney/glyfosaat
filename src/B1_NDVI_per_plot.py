# Written by Collin van Rooij in collaboration with Dimitri Tokmetzis for Follow the Money (FTM)
# This script is used to analyze the Sentinel-2 data per plot, as calculated in A1_get_satellite_data.py and as described in the  
# methodology (Technische Verantwoording) that accompanies FTM's articly on glyphosate use in the Netherlands.

import geopandas as gpd
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import os

outputfolder = os.path.join(os.getcwd(), "output")
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

# read local parquet files starting with this path and concatenate them to one gdf
chunk_output_directory = outputfolder + '/ExtractedData'
filepath = f'{chunk_output_directory}/Result22oct_chunk_'

# Read the parquet files from local files starting with this path and concatenate them to one gdf
VIGDF = gpd.GeoDataFrame(pd.concat([gpd.read_parquet(f'{filepath}{i}.parquet') for i in range(1, 193)], ignore_index=True))
# Set the CRS to EPSG:4326 if it is not already set
if VIGDF.crs is None:
    VIGDF.set_crs(epsg=4326, inplace=True)
else:
    VIGDF.to_crs(epsg=4326, inplace=True)
# Define the columns that are relevant for the analysis
collistBRP = ['geometry', 'gewas', 'gewascode', 'locid']

## Convert the column names
referense_date_str = '2020_02_01'
# Function to convert the column name to the number of days since the reference date
def days_since_reference(date_str, reference_date_str):
    reference_date = pd.to_datetime(reference_date_str, format='%Y_%m_%d')
    date_obj = pd.to_datetime(date_str, format='%Y_%m_%d')
    delta = (date_obj - reference_date).days
    return f"day_{delta}"

# Function to process the GDF to return the updated GDF with days as columnnames for a specific VI
def process_geodataframe(endswith_arg, gdf, columnlist, reference_date_str):
    # Extract columns that end with the specified argument
    specific_cols = [col for col in gdf.columns if col.endswith(f'_{endswith_arg}')]
    specific_colsVIGDF = VIGDF.drop(columns='geometry')
    specific_colsVIGDF = [col for col in specific_colsVIGDF.columns if col.endswith(f'_{endswith_arg}') or col == 'gewas']	
    # Update the column list with the specific columns
    columnlist += specific_cols
    # Create a new GeoDataFrame with the updated column list
    valdata_specific = gdf[columnlist]
    VIGDF_specific = VIGDF[specific_colsVIGDF]
    # Rename the columns to remove the specific suffix
    valdata_specific.columns = [re.sub(f'_{endswith_arg}', '', col) for col in valdata_specific.columns]
    VIGDF_specific.columns = [re.sub(f'_{endswith_arg}', '', col) for col in VIGDF_specific.columns]
    # Iterate over the columns, applying the date conversion only to columns that match the date format
    new_columns = []
    for col in valdata_specific.columns:
        try:
            # Try to convert the column name to a date
            pd.to_datetime(col, format='%Y_%m_%d')
            # If successful, calculate the days since reference_date and append the result
            new_columns.append(days_since_reference(col, reference_date_str))
        except ValueError:
            # If the column is not a date, keep the original name
            new_columns.append(col)
    valdata_specific.columns = new_columns
    new_columns = []
    for col in VIGDF_specific.columns:
        try:
            # Try to convert the column name to a date
            pd.to_datetime(col, format='%Y_%m_%d')
            # If successful, calculate the days since reference_date and append the result
            new_columns.append(days_since_reference(col, reference_date_str))
        except ValueError:
            # If the column is not a date, keep the original name
            new_columns.append(col)
    VIGDF_specific.columns = new_columns
    # Extract the numeric part of the column names that start with 'day_'
    day_columns = [col for col in valdata_specific.columns if col.startswith('day_')]
    # Convert day column names to numeric values (e.g., 'day_4' -> 4)
    day_numbers = [int(col.split('_')[1]) for col in day_columns]
    # take the average of each VI column per gewas and add it as a new column called, meanVIday
    # Calculate the mean VI values for each 'gewas', so it can be used to inspect anomalies
    meanVIday = VIGDF_specific[[col for col in VIGDF_specific.columns if col.startswith('day_') or col == 'gewas']].groupby('gewas').mean()
    meanVIday = meanVIday.add_prefix(f'mean_{endswith_arg}_')
    meanVIday = meanVIday.reset_index()
    valdata_withmean = valdata_specific.merge(meanVIday, on='gewas', how='left')

    # create an array which has n rows (where n is number of rows in valdata_specific) of day_numbers
    day_numbers_array = np.array([day_numbers] * len(valdata_specific))	
    return valdata_specific, valdata_withmean, day_numbers_array

# Run the funtion for the NDVI, for all the plots
allplots, allplots_withmean, day_num_array_all = process_geodataframe('NDVI', VIGDF, collistBRP, '2020_02_01')
# Export the data to a parquet file
allplots.to_parquet(outputfolder + '/allplotsNDVI22oct.parquet')