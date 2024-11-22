# Written by Collin van Rooij in collaboration with Dimitri Tokmetzis for Follow the Money (FTM)
# This script is used to retrieve the Sentinel-2 satellite data as described in the 
# methodology (technical addendum) that accompanies FTM's articly on glyphosate use in the Netherlands.

import os
import requests
import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import geojson
import json
import owslib
from owslib.wfs import WebFeatureService
from owslib.fes import *
from owslib.etree import etree
from io import BytesIO
from pyarrow import parquet
import ee
import geemap
import pyproj
# make sure pyproj is new; https://github.com/geopandas/geopandas/issues/2874
import time
from shapely.geometry import mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

## Get the BRP and pre process it
# create output folder, specify the path if desirable
outputfolder = os.path.join(os.getcwd(), "output")
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

# Define the WFS URL of the BRP
url = "https://service.pdok.nl/rvo/brpgewaspercelen/atom/v1_0/downloads/brpgewaspercelen_definitief_2020.gpkg"
filepath = 'brp2020.gpkg'

# Fetch the file
response = requests.get(url)
with open(filepath, 'wb') as f:
    f.write(response.content)

print(f"Downloaded GeoPackage file to {os.getcwd()}\{filepath}")

# convert it to a geopandas dataframe
brp_gdf = gpd.read_file(filepath)
# drop columns named category and status, export gdf to a parquet file
brp_gdf= brp_gdf.drop(columns=['category', 'status'])
brp_gdf.to_parquet(os.path.join(outputfolder, "brp_NL.parquet"))

# read brp_NL.parquet from local file and simplify the geometry
brp_NL_28992 = gpd.read_parquet(os.path.join(outputfolder, "brp_NL.parquet"))
brp_NL_28992['geometry'] = brp_NL_28992['geometry'].simplify(tolerance=10)
# create the locid so it is easier to identify the plots
brp_NL_28992['locid'] = range(1, len(brp_NL_28992) + 1)
# convert to WGS84 to be able to retrieve GEE data
brp_NL = brp_NL_28992.to_crs(epsg=4326)
# read in the elligible crops, which is a csv file of the 88 most grown crops in the Netherlands
# based on the BRP data, more information can be found in the methodology 
top95crops = pd.read_csv('ElligibleCrops.csv')
# create a list of gewascode where the validation is no (which means it is elligible)
elligiblecrops = top95crops['gewascode'][top95crops['validation'] == 'no']
# select the plots where the gewascodes are in the elligiblecrops
elligibleplots = brp_NL_28992[brp_NL_28992['gewascode'].isin(elligiblecrops)]

# calculate the total area of brp_NL_28992 and the elligibleplots, to see how many percent of the plots are elligible
totalarea = brp_NL_28992['geometry'].area.sum()
elligiblearea = elligibleplots['geometry'].area.sum()
percentelligible = elligiblearea / totalarea * 100

## Create the calibration and validation data
# read in the waarneming data, which was requested from waarneming.nl
# This data is used to generate our calibration and (partly)validation data
waarnemingdf = pd.read_csv('ValDataWaarnemingNL.csv')
# clean the DF
waarnemingdf = waarnemingdf[['lat', 'lng', 'date', 'country division', 'location', 'local x', 
                            'local y', 'has photos', 'link']]   
# filter on the year 2020 in the 'date' column
waarnemingdf = waarnemingdf[waarnemingdf['date'].str.contains('2020')]
waarnemingGdf = gpd.GeoDataFrame(waarnemingdf, 
                              geometry=gpd.points_from_xy(waarnemingdf['local x'], 
                                                          waarnemingdf['local y'], crs= 'EPSG:28992')) 
# first create a buffer of 200 meters around the points
waarnemingGdf['geometryBuffer'] = waarnemingGdf['geometry'].buffer(200)
# take all the plots that intersect with the buffer
plotsnearwaarnemingen = brp_NL_28992[brp_NL_28992.intersects(waarnemingGdf['geometryBuffer'].unary_union)]
# get the plots larger than 50000 m2 and add the crop information
plotsnearwaarnemingen['area'] = plotsnearwaarnemingen['geometry'].area
biggerplots = plotsnearwaarnemingen[plotsnearwaarnemingen['area'] > 50000]
biggerplots = biggerplots.merge(top95crops[['gewascode', 'validation']], on='gewascode', how='left')
# finally, filter out the crops that where not analyzed at all
finalvalplots = biggerplots[biggerplots['validation'].isin(['no', 'yes'])]

# initialize a random seed, so the results are reproducible
# create 2 random subsets of the finalvalplots, for calibration
# The indices for the first subset, were not omitted from the second subset, hence the 274 calibration plots
np.random.seed(7)
validationplots1 = finalvalplots.sample(150)
validationplots2 = finalvalplots.sample(150)
# remove duplicates
locidsval = pd.concat([validationplots1['index'], validationplots2['index']])
locidsval = locidsval.drop_duplicates()
restplots = finalvalplots[~finalvalplots['index'].isin(locidsval)]
# create a validation set of 200 plots, which was used for the final validation
validationAfterIter = restplots.sample(200)

# create the complete random sample
allvallocids = pd.concat([locidsval, validationAfterIter['index']])

# create multiple batches, excluding the plots from the previous batches
finalvalplots_batch1 = restplots[~restplots['locid'].isin(allvallocids)].sample(100)
finalvalplots_batch2 = restplots[~restplots['locid'].isin(finalvalplots_batch1['locid'])].sample(200)

# export these files to parquet, if desirable
validationAfterIter.to_parquet(outputfolder + '\ValidationAfterIter.parquet')
validationplots1.to_parquet(outputfolder + '\Validationplots1.parquet')
validationplots2.to_parquet(outputfolder + '\Validationplots2.parquet')
finalvalplots_batch1.to_parquet(outputfolder + '\FinalValPlotsBatch1.parquet')
finalvalplots_batch2.to_parquet(outputfolder + '\FinalValPlotsBatch2.parquet')

#reproject validationplots to epsg 4326, to be used with GEE
validationplots1 = validationplots1.to_crs(epsg=4326)
validationplots2 = validationplots2.to_crs(epsg=4326)
validationAfterIter = validationAfterIter.to_crs(epsg=4326)	
finalvalplots_batch1 = finalvalplots_batch1.to_crs(epsg=4326)
finalvalplots_batch2 = finalvalplots_batch2.to_crs(epsg=4326)

## Get the Sentinel-2 data from GEE

# Authenticae and initialize the Earth Engine API
ee.Authenticate()
ee.Initialize()

# function to mask clouds, from https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#colab-python
def mask_s2_clouds(image):
  
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )
  return image.updateMask(mask).divide(10000)

# Get NL outline
countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") 
NL = countries.filter(ee.Filter.eq("country_na", "Netherlands"))

# Define the start and end date and the maximum cloud cover percentage
start_date = '2020-02-01'
end_date = '2020-05-31'
CPP = 20
# Get the Sentinel 2 imagery for the relevant months in 2020, filter on Drenthe and max 10 percent cloud cover
S2_NL = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','QA60'])
    .filterDate(start_date, end_date)
    .filterBounds(NL)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CPP))
    .map(mask_s2_clouds)
    .map(lambda image: image.clip(NL))
    )

# Function to add a 'date' property to each image based on the 'system:index', since the date is not directly available in the metadata
def add_date_property(image):
    # Extract the first part of 'system:index' (YYYYMMDD) from the index string
    index = image.get('system:index')
    date_str = ee.String(index).slice(0, 8)
    
    # Format the date as 'YYYY-MM-DD'
    formatted_date = ee.String(date_str.slice(0, 4))\
                     .cat('-')\
                     .cat(date_str.slice(4, 6))\
                     .cat('-')\
                     .cat(date_str.slice(6, 8))
    
    # Return the image with the new 'date' property
    return image.set('date', formatted_date)

# Map the function over the collection to add the 'date' property
collection_with_dates = S2_NL.map(add_date_property)

# Function to mosaic images by date
def mosaic_by_date(date):
    # Filter images by the given date
    filtered = collection_with_dates.filter(ee.Filter.eq('date', date))
    
    # Mosaic the images for that date
    mosaic = filtered.mosaic()
    
    # Set the 'date' property on the mosaic
    return mosaic.set('date', date)

# Get the list of unique dates and create a collection with imagery from each day
unique_dates = ee.List(collection_with_dates.aggregate_array('date')).distinct()
mosaicked_collection = ee.ImageCollection(unique_dates.map(mosaic_by_date))

# Create a list of dates so they can be used to name the bands later on
dates_list = unique_dates.getInfo()

# create a list of VIs, so possible other Indices could be used 
# create a list of visible bands, for the validation data 
vi_list = ['NDVI', 'PSSRa', 'IRECI', 'GNDVI']
val_list = ['B2', 'B3', 'B4']

def calculate_indices(image):
    B3 = image.select('B3')  # Green
    B4 = image.select('B4')  # Red
    B5 = image.select('B5')  # Red Edge 1
    B6 = image.select('B6')  # Red Edge 2    
    B7 = image.select('B7')  # Red Edge 3
    B8 = image.select('B8')  # NIR
    # NDVI
    NDVI = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI')
    # PSSRa (Proportion of Soil-Adjusted Vegetation Index)
    PSSRa = B8.divide(B4).rename('PSSRa')
    # IRECI (Infrared Relative Extracted Chlorophyll Index)
    IRECI = B7.subtract(B4).divide(B5.divide(B6)).rename('IRECI')
    # GNDVI (Green Normalized Difference Vegetation Index)
    GNDVI = B8.subtract(B3).divide(B8.add(B3)).rename('GNDVI')
    
    return image.addBands([NDVI, PSSRa, IRECI, GNDVI])

S2_indices = mosaicked_collection.map(calculate_indices)
S2_indices = S2_indices.select('NDVI', 'PSSRa', 'IRECI', "GNDVI")
S2_Validation = mosaicked_collection.select('B2', 'B3', 'B4')

# convert it to a single image for processing
mosaickedimage = S2_indices.toBands()
S2_valImage = S2_Validation.toBands()

# Create a list of new band names by combining each date with each VI name
new_band_namesVI = []
for date in dates_list:
    for vi in vi_list:
        new_band_name = date.replace('-', '_') + '_' + vi
        new_band_namesVI.append(new_band_name)

new_band_namesval = []
for date in dates_list:
    for val in val_list:
        new_band_name = date.replace('-', '_') + '_' + val
        new_band_namesval.append(new_band_name)

# Rename bands in the image using the new names
def rename_bands(image, new_band_names):
    band_names = image.bandNames().getInfo()
    renamed_image = image.select(band_names).rename(new_band_names)
    return renamed_image

# Apply renaming
renamed_image = rename_bands(mosaickedimage, new_band_namesVI)
renamed_valImage = rename_bands(S2_valImage, new_band_namesval)

## Export the NDVI image data to Google Drive, this takes a couple of hours
# split up the GDF into chunks of max 10mb
def split_gdf(gdf, chunk_size):
    return np.array_split(gdf, len(gdf) / chunk_size)
    
def gdf_to_ee_featurecollection(gdf_chunk):
    geojson_dict = json.loads(gdf_chunk.to_json())
    return ee.FeatureCollection(geojson_dict)

# Function to apply reduceRegions over FeatureCollection, i.e. calculate the average NDVI for each plot
def apply_reduce_regions(fc, image, scale=10):
    return image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale
    )


# convert gdf to ee.featurecollection
chunk_size = 4000
gdf_NL_chunks = split_gdf(brp_NL, chunk_size)

# Function to export each chunk directly to local storage in parquet format
def export_chunk_to_local(fc, chunk_idx, directory, file_prefix):
    """
    Export a small chunk of a feature collection to local storage.
    
    Parameters:
    fc : ee.FeatureCollection
        The feature collection to export.
    chunk_idx : int
        The index of the chunk being exported (for file naming).
    directory : str
        The local directory where the export should be saved.
    file_prefix : str
        The prefix for the exported file.
    """
    # Define a unique file name for the chunk export
    file_name = f'{file_prefix}_chunk_{chunk_idx}.parquet'
    file_path = os.path.join(directory, file_name)
    
    # Convert ee.FeatureCollection to GeoJSON
    geojson_dict = fc.getInfo()
    
    # Convert GeoJSON to GeoDataFrame
    gdf_chunk = gpd.GeoDataFrame.from_features(geojson_dict['features'])
    
    # Write the GeoDataFrame to a Parquet file
    gdf_chunk.to_parquet(file_path, index=False)
    
    print(f'Chunk {chunk_idx} exported to {file_path}')

# Function to read the last processed chunk index from a file, in case the script is interrupted
def read_last_processed_chunk(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return int(file.read().strip())
    return 0

# Function to write the last processed chunk index to a file
def write_last_processed_chunk(file_path, chunk_idx):
    with open(file_path, 'w') as file:
        file.write(str(chunk_idx))

# Path to the progress file
progress_file = 'progress.txt'
# Read the last processed chunk index
last_processed_chunk = read_last_processed_chunk(progress_file)
# Initialize an empty list to store processed results
all_results = []

# specify the directory and prefix for the output
chunk_output_directory = outputfolder + '/ExtractedData'
# Loop over each chunk, convert to ee.FeatureCollection, apply reduceRegions
for idx, gdf_chunk in enumerate(gdf_NL_chunks):
    if idx + 1 <= last_processed_chunk:
        continue  # Skip already processed chunks

    print(f'Processing chunk {idx+1}/{len(gdf_NL_chunks)}')

    try:
        # Filter out invalid geometries
        gdf_chunk = gdf_chunk[gdf_chunk.is_valid]

        # Convert GeoDataFrame chunk to ee.FeatureCollection
        fc = gdf_to_ee_featurecollection(gdf_chunk)

        # Apply reduceRegions function to the feature collection
        S2stats = apply_reduce_regions(fc, renamed_image)
        
        # Export the processed chunk to local storage, specify directory and prefix
        export_chunk_to_local(S2stats, idx + 1, chunk_output_directory, 'Result22oct')

        # Optional: Store the results in memory if you need them locally (beware of large datasets)
        all_results.append(S2stats)

        # Update the progress file
        write_last_processed_chunk(progress_file, idx + 1)

    except Exception as e:
        print(f'Error processing chunk {idx+1}: {e}')
        break  # Exit the loop on error

print("Processing completed.")

## Export the validation data to Google Drive, this takes (per set) multiple hours as well
# function that creates a folder for each plot, for easy checking
def create_folder_for_plot(output_folder, index_value):
    plot_folder = os.path.join(output_folder, f"FinalValidation{index_value}")
    os.makedirs(plot_folder, exist_ok=True)
    return plot_folder

# function that export the images, you will get a lot of errors like 
# "Error processing date {date} for plot {index_value}: {e}" since no plots have data for all possible dates
# other errors should not be ignored
def export_rgb_images_for_plot(plot_folder, plotgeom, image, index_value, dates_list):
    for date in dates_list:
        # Construct the band names for the given date
        bands = [f"{date}_B4", f"{date}_B3", f"{date}_B2"]
        try:
            # Check if the bands contain valid data
            data_check = image.select(bands).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=plotgeom,
                scale=10
            ).getInfo()

            # If all band sums are zero, skip this date
            if all(value == 0 for value in data_check.values()):
                print(f"No data for date {date} and plot {index_value}, skipping.")
                continue

            # Create an image with the plot outline
            outline = ee.Image().byte().paint(
                featureCollection=ee.FeatureCollection([ee.Feature(plotgeom)]),
                color=1,
                width=2  # Adjust the width of the outline as needed
            ).visualize(palette=['red'])

            # Blend the outline with the original image
            rgb_image = image.select(bands).visualize(min=0, max=0.3).blend(outline)

            thumb_url = rgb_image.getThumbURL({
                'region': plotgeom.bounds().getInfo(),
                'dimensions': '512x512',
                'format': 'png'
            })
            response = requests.get(thumb_url)
            if response.status_code == 200:
                image_path = os.path.join(plot_folder, f"{index_value}_{date}_validation.png")
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download image for date {date} and plot {index_value}")
        except ee.ee_exception.EEException as e:
            print(f"Error processing date {date} for plot {index_value}: {e}")

# function that combines the previous functions and also creates a plot geometry
def process_plot(index, row, image, output_folder, dates_list):
    plot_geometry = row['geometry']
    plot_geojson = mapping(plot_geometry)
    plotgeom = ee.Geometry(plot_geojson)
    index_value = row['index']
    plot_folder = create_folder_for_plot(output_folder, index_value)
    export_rgb_images_for_plot(plot_folder, plotgeom, image, index_value, dates_list)

num_cores = multiprocessing.cpu_count()

# function that distributes the processing over the cores
def process_validation_dataset(validation_dataset, image, output_folder, dates_list):
    with ThreadPoolExecutor(max_workers= num_cores) as executor:
        futures = [
            executor.submit(process_plot, index, row, image, output_folder, dates_list)
            for index, row in validation_dataset.iterrows()
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing plot: {e}")

# Adjust the validationdataset accordingly
validationdataset = validationplots1
output_folder_validation = os.path.join(outputfolder, str(validationdataset))
if not os.path.exists(output_folder_validation):
    os.makedirs(output_folder_validation)
# the image and dates_list are the same for each validation set
image = renamed_valImage
dates_list_val = [date.replace('-', '_') for date in dates_list]
process_validation_dataset(validationAfterIter, image, output_folder_validation, dates_list_val)