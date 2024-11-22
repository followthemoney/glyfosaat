# Written by Collin van Rooij in collaboration with Dimitri Tokmetzis for Follow the Money (FTM)
# This script is used to analyze all the Areas of Interest (AOIs) and how they overlap with the glyphosate fields and as described in the  
# methodology (technical addendum) that accompanies FTM's articly on glyphosate use in the Netherlands.

from pyarrow import parquet
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
from owslib.wfs import WebFeatureService
from owslib.fes import *
from owslib.etree import etree
import os
import json
import xml.etree.ElementTree as ET
import html5lib
from shapely.geometry import Point
from shapely.ops import unary_union

outputfolder = os.path.join(os.getcwd(), "output")
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

# yellowfields parquet file inladen
GlyphosateFields = gpd.read_parquet(outputfolder + '/Glypho_W16_DT05_FT045_IT06/FinalplotsGlypho_W16_DT05_FT045_IT06.parquet')
# reproject yellowfields to epsg 28992 to be able to calculate overlapping geometries
GlyphosateFields = GlyphosateFields.to_crs(epsg=28992)
# calculate the area of the yellowfields
GlyphosateFields['area'] = GlyphosateFields['geometry'].area
# calculate the total area of the yellowfields in ha
GlyphosateFields_area = GlyphosateFields['area'].sum()/10000
#  buffer the yellowfields with 250m and 1000m, dissolve the geometries to get a new gdf
GlyphosateFields['buffer_250'] = GlyphosateFields.buffer(250)
GlyphosateFields['buffer_1000'] = GlyphosateFields.buffer(1000)
GlyphosateFields['buffer_500'] = GlyphosateFields.buffer(500)

#Define function that can retrieve WFS data
def fetch_wfs_data(wfs_url, wfslayer, output_filename, crs="EPSG:28992"):
    wfs = WebFeatureService(url=wfs_url, version="2.0.0")
    response = wfs.getfeature(typename=wfslayer, outputFormat="json")
    gdf = gpd.read_file(response)
    
    if gdf.crs is None:
        gdf.set_crs(crs, inplace=True)
    
    parquet_path = os.path.join(outputfolder, f'{output_filename}.parquet')
    gdf.to_parquet(parquet_path)
    
    return gdf

# Fetch the Natura2000 data
wfs_url_natura2000 = "https://service.pdok.nl/rvo/natura2000/wfs/v1_0"
wfslayer_natura2000 = "natura2000"
output_filename_natura2000 = "Natura2000"
natura2000gdf = fetch_wfs_data(wfs_url_natura2000, wfslayer_natura2000, output_filename_natura2000)

# calculate the area of the veluwe
natura2000gdf['area'] = natura2000gdf['geometry'].area
# Some areas are split up in multiple geometries, so we need to dissolve them
dissolved_natura2000 = natura2000gdf.dissolve(by='naamN2K')

# Get grondwaterbeschermingsgebieden from wfs
urlWWG = "https://ogc-geoservices.bij12.nl/geoserver/IMNa_IKN/wfs?request=GetCapabilities"
layer_wwg = "grondwater_bescherming"
output_filename_wwg = "Grondwaterbeschermingsgebieden"
wwggdf = fetch_wfs_data(urlWWG, layer_wwg, output_filename_wwg)

# dissolve the polygons that touch each other, to get the actual GWBgebieden
mergedgeom = wwggdf.unary_union
new_wwggdf = gpd.GeoDataFrame(geometry=[mergedgeom], crs=wwggdf.crs)
new_wwggdf = new_wwggdf.explode(index_parts=False).reset_index(drop=True)

# Perform a spatial join to attach attributes from the original wwggdf
joined = gpd.sjoin(new_wwggdf, wwggdf, how='left', predicate='intersects')

# Group by the index of the new geometries and take the first match
new_wwggdf = joined.groupby(joined.index).first().reset_index(drop=True)
new_wwggdf.crs = wwggdf.crs

# get the provincies
urlProvincies = 'https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?request=GetCapabilities&service=WFS'
layer_provincies = 'Provinciegebied'
output_filename_provincies = 'provincies'
provgdf = fetch_wfs_data(urlProvincies, layer_provincies, output_filename_provincies)
# create a list of the provinces where it is prohibited to use pesticides in
#  https://natuurenmilieu.nl/app/uploads/Bestrijdingsmiddelen-in-Nederlandse-natuur-en-water-versie-7-PDF.pdf#page=76
provincies_verboden = ['Fryslân', 'Drenthe', 'Overijssel',  'Noord-Holland', 'Noord-Brabant', 'Groningen']
# create a column in the provincies gdf where it is True if the statcode is in provincies_verboden
provgdf['verboden'] = provgdf['naam'].isin(provincies_verboden)

# Import the KRW wateren which is in XML format, via web
def getGML(url, outputfolder, output_filename):
    response = requests.get(url)
    response.raise_for_status()
    localfilename = os.path.join(outputfolder, f'{output_filename}.gml')
    with open(localfilename, 'wb') as f:
        f.write(response.content)

    gdf = gpd.read_file(localfilename)
    
    parquet_filename = os.path.join(outputfolder, f'{output_filename}.parquet')
    gdf.to_parquet(parquet_filename)
    
    print(f"File downloaded and saved as {parquet_filename}")
    return gdf
KRWurl = 'https://service.pdok.nl/ihw/gebiedsbeheer/krw-oppervlaktewaterlichamen/atom/downloads/INSPIRESurfaceWaterBody.gml'
output_filename = 'KRWwateren'
KRWwateren = getGML(KRWurl, outputfolder, output_filename)

# reproject the KRWwateren to RD
KRWwateren = KRWwateren.to_crs(epsg=28992)


# import the buurten, which was not possible with the WFS	
def download_geopackage(url, outputfolder, output_filename, layer_name):
    response = requests.get(url)
    response.raise_for_status()
    
    # Save the GeoPackage file locally
    localfilename = os.path.join(outputfolder, f'{output_filename}.gpkg')
    with open(localfilename, 'wb') as f:
        f.write(response.content)
    
    # Convert the GeoPackage file to a GeoDataFrame
    gdf = gpd.read_file(localfilename, layer = layer_name)
    parquet_filename = os.path.join(outputfolder, f'{output_filename}.parquet')
    gdf.to_parquet(parquet_filename)
    return gdf

# Example usage
Buurtenurl = "https://service.pdok.nl/cbs/wijkenbuurten/2020/atom/downloads/wijkenbuurten_2020_v3.gpkg"
output_filename = 'Buurten'
layername = 'buurten'
buurten_gdf = download_geopackage(Buurtenurl, outputfolder, output_filename, layername)

# put the column names in a text file and export
with open(outputfolder + '/buurten_columns.txt', 'w') as f:
    f.write('\n'.join(buurten_gdf.columns))

smallerBuurtenset = buurten_gdf[[
        'geometry',
        'aantal_inkomensontvangers',
        'gemiddeld_inkomen_per_inkomensontvanger',
        'gemiddeld_inkomen_per_inwoner',
        'percentage_personen_met_laag_inkomen',
        'percentage_personen_met_hoog_inkomen',
        'percentage_huishoudens_met_laag_inkomen',
        'percentage_huishoudens_met_hoog_inkomen',
        'percentage_huishoudens_onder_of_rond_sociaal_minimum',
        'percentage_huishoudens_met_lage_koopkracht',
        'aantal_personen_met_een_ao_uitkering_totaal',
        'aantal_personen_met_een_ww_uitkering_totaal',
        'aantal_personen_met_een_alg_bijstandsuitkering_tot',
        'aantal_personen_met_een_aow_uitkering_totaal',
        'gemiddeld_gestandaardiseerd_inkomen_van_huishoudens',
        'opleidingsniveau_hoog',
        'opleidingsniveau_middelbaar',
        'opleidingsniveau_laag',
        'geboorte_totaal',
        'geboortes_per_1000_inwoners',
        'sterfte_totaal',
        'sterfte_relatief',
        'percentage_huishoudens_zonder_kinderen',
        'percentage_huishoudens_met_kinderen',
        'percentage_personen_0_tot_15_jaar',
        'percentage_personen_15_tot_25_jaar',
        'percentage_personen_25_tot_45_jaar',
        'percentage_personen_45_tot_65_jaar',
        'percentage_personen_65_jaar_en_ouder',
        'bevolkingsdichtheid_inwoners_per_km2',
        'aantal_inwoners',
        'buurtcode',
        'buurtnaam',
        'wijkcode',
        'gemeentecode',
        'gemeentenaam'
    ]]

## scholen and buitenspeeldata available on request
scholen = gpd.GeoDataFrame(columns=['geometry'])
buitenspeelplekken = gpd.GeoDataFrame(columns=['geometry'])

# function that calculates the amount of intersecting geometries, either directly overlapping or within a buffer

def add_intersection_columns(gdf, DirectIntersect = None, buffer_250=None, buffer_500=None, buffer_1000=None):
    if DirectIntersect is not None:
        gdf['intersects'] = gdf['geometry'].apply(lambda x: DirectIntersect.intersects(x).any())
    if buffer_250 is not None:
        gdf['intersects_buffer_250'] = gdf['geometry'].apply(lambda x: buffer_250.intersects(x).any())
    if buffer_500 is not None:
        gdf['intersects_buffer_500'] = gdf['geometry'].apply(lambda x: buffer_500.intersects(x).any())
    if buffer_1000 is not None:
        gdf['intersects_buffer_1000'] = gdf['geometry'].apply(lambda x: buffer_1000.intersects(x).any())

    return gdf

# Run the function to see where the buffers or the plots themselves intersect with the AOIs
dissolved_natura2000 = add_intersection_columns(dissolved_natura2000, GlyphosateFields['geometry'], GlyphosateFields['buffer_250'], 
                                            GlyphosateFields['buffer_500'], GlyphosateFields['buffer_1000'])
scholen = add_intersection_columns(scholen, DirectIntersect= None, buffer_250= GlyphosateFields['buffer_250'], 
                                            buffer_500= GlyphosateFields['buffer_500'], buffer_1000= GlyphosateFields['buffer_1000'])
new_wwggdf = add_intersection_columns(new_wwggdf, DirectIntersect= GlyphosateFields['geometry'], buffer_250= None, buffer_500=None, buffer_1000=None)	
KRWwateren = add_intersection_columns(KRWwateren, GlyphosateFields['geometry'], GlyphosateFields['buffer_250'], 
                                            GlyphosateFields['buffer_500'], GlyphosateFields['buffer_1000'])
smallerBuurtenset = add_intersection_columns(smallerBuurtenset, GlyphosateFields['geometry'], GlyphosateFields['buffer_250'],
                                            GlyphosateFields['buffer_500'], GlyphosateFields['buffer_1000'])
buitenspeelplekken = add_intersection_columns(buitenspeelplekken, DirectIntersect= None, buffer_250= GlyphosateFields['buffer_250'], 
                                            buffer_500= GlyphosateFields['buffer_500'], buffer_1000= GlyphosateFields['buffer_1000'])

# Export the data to parquet
scholen.to_parquet(outputfolder + '/scholen.parquet')
buitenspeelplekken.to_parquet(outputfolder + '/buitenspeelplekken.parquet')
smallerBuurtenset.to_parquet(outputfolder + '/Buurten.parquet')
dissolved_natura2000.to_parquet(outputfolder + '/Natura2000.parquet')
new_wwggdf.to_parquet(outputfolder + '/Grondwaterbeschermingsgebieden.parquet')
KRWwateren.to_parquet(outputfolder + '/KRWwateren.parquet')


# Perform spatial intersection to get the population in the buffers
buffers = [250, 500, 1000]
buffer_results = {}
total_populations = {}

for buffer in buffers:
    buffer_gdf = gpd.GeoDataFrame(geometry=GlyphosateFields[f'buffer_{buffer}'], crs=GlyphosateFields.crs)
    buffer_gdf = buffer_gdf.dissolve()
    
    wijken_influ = gpd.overlay(smallerBuurtenset, buffer_gdf, how='intersection')
    
    # Replace negative values with NaN
    wijken_influ['bevolkingsdichtheid_inwoners_per_km2'] = wijken_influ['bevolkingsdichtheid_inwoners_per_km2'].apply(lambda x: np.nan if x < 0 else x)
    
    # Calculate population
    wijken_influ['population'] = wijken_influ['bevolkingsdichtheid_inwoners_per_km2'] * (wijken_influ['geometry'].area / 1000000)
    
    # Sum the total population in the wijken
    total_populations[f'total_population_{buffer}'] = wijken_influ['population'].sum()
    buffer_results[buffer] = wijken_influ 

# Export the results to a csv
pd.DataFrame(total_populations, index=[0]).to_csv(outputfolder + '/total_populations.csv')

## Below is the analysis, which calculates how many AOIs intersect with the glyphosate sprayed fields
# Function that creates an overview of the data, including the total amount of plots and 
# the amount of plots that intersect with the AOIs. A clip can be supplied (will be used for the provinces)
def create_overview(gdfs_dict, clip_gdf=None):
    data = []
    for name, gdf in gdfs_dict.items():
        if clip_gdf is not None:
            # Clip the GeoDataFrame to the provided clip_gdf
            gdf = gpd.clip(gdf, clip_gdf)
        
        row = {}
        row['naam'] = name
        row['total'] = len(gdf)
        if 'intersects' in gdf.columns:
            row['intersects'] = gdf['intersects'].sum()
        if 'intersects_buffer_250' in gdf.columns:
            row['intersects_buffer_250'] = gdf['intersects_buffer_250'].sum()
        if 'intersects_buffer_500' in gdf.columns:
            row['intersects_buffer_500'] = gdf['intersects_buffer_500'].sum()
        if 'intersects_buffer_1000' in gdf.columns:
            row['intersects_buffer_1000'] = gdf['intersects_buffer_1000'].sum()
        data.append(row)
    return pd.DataFrame(data)

# Create a dictionary to map the GeoDataFrames to their names
gdfs_dict = {
    'Natura2000': dissolved_natura2000,
    'ScholenNL': scholen,
    'Grondwaterbeschermingsgebieden': new_wwggdf,
    'KRWwateren': KRWwateren,
    'Buitenspeelplekken': buitenspeelplekken
}

# Create the overview for the entire dataset and export it
overview = create_overview(gdfs_dict)
overview.to_csv(outputfolder + '/overviewbuffersNL.csv')

## Create overviews for each province, the pachtdata is not available for public use, so the code below is commented out
# At the bottom of the script, the code is included to show how the pachtdata was processed

# Create a folder to store the province data
provinciefolder = outputfolder + '/provincies'
# Create the overview per province
os.makedirs(provinciefolder, exist_ok=True)
provincielist = provgdf['naam'].unique()
for provincie in provincielist:
    # Get the provincie
    provincie_gdf = provgdf[provgdf['naam'] == provincie]
    provincie_gdf = provincie_gdf.to_crs(epsg=28992)
    
    # Use create_overview to get the overview of the data clipped to the province
    overview = create_overview(gdfs_dict, clip_gdf=provincie_gdf)
    
    # calculate the number of people living nearby a glyphosate field in each province
    provincie_populations = {}
    for buffer, wijken_influ in buffer_results.items():
        clipped_wijken_influ = gpd.clip(wijken_influ, provincie_gdf)
        provincie_populations[f'total_population_{buffer}'] = clipped_wijken_influ['population'].sum()
    for buffer in buffers:
        overview[f'total_population_{buffer}'] = provincie_populations[f'total_population_{buffer}']
    # Export the overview to a CSV file
    overview.to_csv(provinciefolder + f'/overviewbuffers_{provincie}.csv')
    
    # Export the GlyphosateFields to a geojson for each province, first clip on the province
    clippedGlyphosateFields = gpd.clip(GlyphosateFields, provincie_gdf)
    clippedGlyphosateFields = clippedGlyphosateFields.drop(columns=['buffer_250', 'buffer_1000', 'buffer_500'])
    clippedGlyphosateFields.to_file(provinciefolder + f'/GlyfosaatVelden_{provincie}.geojson', driver='GeoJSON')
    
    # Clip the pachtdata to the province
    # clippedpachtdata = gpd.clip(GlyphosateFields_pacht, provincie_gdf)
    # provinciedata = gpd.clip(GlyphosateFields, provincie_gdf)
    
    # Calculate the total area of the plots in the province
    provincie_area = provinciedata['area'].sum() / 10000
    
    # Count the number of plots in the province
    provincie_count = len(provinciedata)
    
    # Calculate the total area of the plots where Catagorie_PBL equals Publiek_provincie
    # provinciebezit = clippedpachtdata[clippedpachtdata['Catagorie_PBL'] == 'Publiek_provincie']
    # provinciebezit_area = provinciebezit['area'].sum() / 10000
    
    # Make a DataFrame where we include the provincie_area, provincie_count, provinciebezit_area, and provinciebezit
    provinciedata = pd.DataFrame({
        'provincie_area': provincie_area,
        'provincie_count': provincie_count,
        #'provinciebezit_area': provinciebezit_area,
        #'provinciebezit': len(provinciebezit)
    }, index=[0])
    
    # Export the provinciedata to a CSV file
    provinciedata.to_csv(provinciefolder + f'/PachtEnPlotdata_{provincie}.csv')


## Get the names of the AOIs which are nearby the glyphosate sprayed fields

# Function that creates a DataFrame with the names of the AOIs that intersect with the glyphosate sprayed fields
def create_true_names_df(gdfs_dict, columns_dict, name_columns_dict):
    true_names = {}
    for name, gdf in gdfs_dict.items():
        name_column = name_columns_dict[name]
        for col in columns_dict[name]:
            if col in gdf.columns:
                # Convert numpy.ndarray to tuple if necessary
                if isinstance(gdf[name_column].iloc[0], np.ndarray):
                    gdf[name_column] = gdf[name_column].apply(tuple)
                true_names[f'{name}_{col}'] = gdf[gdf[col] == True][name_column].unique()
    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in true_names.items()]))

# Define the columns to check for each GeoDataFrame
columns_dict = {
    'natura2000gdf': ['intersects', 'intersects_buffer_250', 'intersects_buffer_500', 'intersects_buffer_1000'],
    'scholen': ['intersects', 'intersects_buffer_250', 'intersects_buffer_500', 'intersects_buffer_1000'],
    'wwggdf': ['intersects', 'intersects_buffer_250', 'intersects_buffer_500', 'intersects_buffer_1000'],
    'KRWwateren': ['intersects', 'intersects_buffer_250', 'intersects_buffer_500', 'intersects_buffer_1000'],
    'buitenspeelplekken': ['intersects', 'intersects_buffer_250', 'intersects_buffer_500', 'intersects_buffer_1000']
}

# Define the name column for each GeoDataFrame
name_columns_dict = {
    'natura2000gdf': 'naamN2K',
    'scholen': 'naam',
    'wwggdf': 'identificatie',
    'KRWwateren': 'localId', 
    'buitenspeelplekken': 'geometry.coordinates'  
}

# Add the name column to the dissolved Natura2000 data
dissolved_natura2000['naamN2K'] = dissolved_natura2000.index

# Create a dictionary to map the GeoDataFrames to their names
gdfs_dict = {
    'natura2000gdf': dissolved_natura2000,
    'scholen': scholen,
    'wwggdf': new_wwggdf,
    'KRWwateren': KRWwateren,
    'buitenspeelplekken': buitenspeelplekken
}

# Create the DataFrame for the entire dataset
true_names_df = create_true_names_df(gdfs_dict, columns_dict, name_columns_dict)
# Export the DataFrame to a CSV file
true_names_df.to_csv(outputfolder + '/true_names.csv')

# create the same overview for the provinces
# Create the DataFrame for each province
for provincie in provincielist:
    provincie_gdf = provgdf[provgdf['naam'] == provincie]
    provincie_gdf = provincie_gdf.to_crs(epsg=28992)
    
    clipped_gdfs_dict = {name: gpd.clip(gdf, provincie_gdf) for name, gdf in gdfs_dict.items()}
    true_names_df = create_true_names_df(clipped_gdfs_dict, columns_dict, name_columns_dict)
    
    true_names_df.to_csv(provinciefolder + f'/namendata_{provincie}.csv')



# the pachtdata cannot be published, but the code below
# indicates how the data was processed
# read in the geopackage with the pachtdata
pachtdata = gpd.read_file()
# reproject the pachtlocaties to RD
pachtdata = pachtdata.to_crs(epsg=28992)
# create a gdf, where the column Catagorie_PBL is joined to GlyphosateFields  from the pachtdata
# drop irrelevant columns
pachtdatagdf = pachtdata[['geometry', 'Catagorie_PBL']]
GlyphosateFields_pacht = gpd.sjoin(GlyphosateFields, pachtdatagdf, how='left', predicate='intersects')
GlyphosateFields_pacht.to_parquet(outputfolder + '/GlyphosateFields_pacht.parquet')
#Calculate the total area of the plots where Catagorie_PBL equals Publiek_provincie
provinciebezitNL = GlyphosateFields_pacht[GlyphosateFields_pacht['Catagorie_PBL'] == 'Publiek_provincie']
provinciebezit_areaNL = provinciebezitNL['area'].sum() / 10000