# Written by Collin van Rooij in collaboration with Dimitri Tokmetzis for Follow the Money (FTM)
# This script is used to acutally categorize the plots (from B1_NDVI_per_plot) as either sprayed or not sprayed and as described in the  
# methodology (Technische Verantwoording) that accompanies FTM's articly on glyphosate use in the Netherlands.

from pyarrow import parquet
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

outputfolder = os.path.join(os.getcwd(), "output")
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

# Import allplotsNDVI.parquet as a geodataframe
allplots = gpd.read_parquet(outputfolder + '/allplotsNDVI22oct.parquet')
#read in the ElligibleCrops.csv file
top95crops = pd.read_csv('ElligibleCrops.csv')
# merge the columns begin_days and end_days from top95crops to sprayedfields based on gewascode
# so we can use this information to construct the "window"
allplots = allplots.merge(top95crops[['gewascode','begin_days', 'end_days', 'validation']], on='gewascode', how='left')

# Define the NDVI columns and the day numbers for the analysis
ndvicols = [col for col in allplots.columns if col.startswith('day')]
day_numbers = [int(col.split('_')[-1]) for col in ndvicols]

# filter out the unlikely, and bottom percent crops -> so where the value in column validation equals no
elligibleplots = allplots[allplots['validation'] == 'no']


# The actual algorithm, which checks for a gradual decrease in NDVI values
# For a more comprehensive explanation, see the methodology (Technische Verantwoording) 
# that accompanies FTM's article on glyphosate use in the Netherlands.
def check_gradual_decrease(row, day_numbers, window=21, drop_threshold=0.5, final_treshold=0.5, initial_treshold = 0.6):	
    # Get the NDVI values and day numbers for the row (ignoring NaN values)
    ndvi_values = row[ndvicols].values
    ndvi_values = np.array(ndvi_values, dtype=float)
    valid_idx = ~np.isnan(ndvi_values)  # Ignore NaN values
    valid_days = np.array(day_numbers)[valid_idx]
    valid_ndvi = ndvi_values[valid_idx]
    
    final_sow = row.get('end_days', None)
    if not np.isnan(final_sow):
        valid_days = valid_days[valid_days <= final_sow]
        valid_ndvi = valid_ndvi[:len(valid_days)]
    # Loop through all windows within the 21-day time frame
    for i in range(len(valid_days)):
        # Check if the initial value of the window is higher than the initial_threshold
        if valid_ndvi[i] < initial_treshold:
            # Look for a value within the window that meets the threshold
            found = False
            for k in range(i, min(i + window, len(valid_ndvi))):
                if valid_ndvi[k] >= initial_treshold:
                    i = k  # Update the starting index to the found value
                    found = True
                    break
            if not found:
                continue  # Skip to the next window if no value meets the threshold
        for j in range(i+1, len(valid_days)):
            if valid_days[j] - valid_days[i] <= window:
                X = np.array(valid_days[i:j+1]).reshape(-1, 1)  # Day numbers as X
                y = valid_ndvi[i:j+1]  # NDVI values as Y
                
                # Fit linear regression to check for overall trend
                model = LinearRegression().fit(X, y)
                
                # Check if the slope is negative (overall trend is downward)
                if model.coef_[0] < 0:
                    # Check for sharp drops
                    for k in range(i + 1, j + 1):
                        if valid_ndvi[k] <= drop_threshold * valid_ndvi[k - 1]:
                            return False
                    
                    # Check if the final value is half or less than the initial value
                    if valid_ndvi[j] <= final_treshold * valid_ndvi[i]:
                        # If all conditions met (negative trend, no sharp drops, final drop significant), return True
                        return True
    return False  # No gradual decrease found

# parameters after the iterations on the first val set
modelparams29oct = [
    ('Glypho_W16_DT05_FT045_IT06', {'window': 16, 'drop_threshold': 0.5, 'final_treshold': 0.45, 'initial_treshold': 0.6}),
    ('Glypho_W16_DT05_FT05_IT06', {'window': 16, 'drop_threshold': 0.5, 'final_treshold': 0.5, 'initial_treshold': 0.6}),
    ('Glypho_W16_DT05_FT055_IT06', {'window': 16, 'drop_threshold': 0.5, 'final_treshold': 0.55, 'initial_treshold': 0.6}),
    ('Glypho_W16_DT055_FT045_IT06', {'window': 16, 'drop_threshold': 0.55, 'final_treshold': 0.45, 'initial_treshold': 0.6}),
    ('Glypho_W16_DT055_FT05_IT06', {'window': 16, 'drop_threshold': 0.55, 'final_treshold': 0.5, 'initial_treshold': 0.6}),
    ('Glypho_W16_DT055_FT055_IT06', {'window': 16, 'drop_threshold': 0.55, 'final_treshold': 0.55, 'initial_treshold': 0.6}),
    ('Glypho_W16_DT06_FT045_IT06', {'window': 16, 'drop_threshold': 0.6, 'final_treshold': 0.45, 'initial_treshold': 0.6}),
    ('Glypho_W16_DT06_FT05_IT06', {'window': 16, 'drop_threshold': 0.6, 'final_treshold': 0.5, 'initial_treshold': 0.6}),
    ('Glypho_W16_DT06_FT055_IT06', {'window': 16, 'drop_threshold': 0.6, 'final_treshold': 0.55, 'initial_treshold': 0.6}),
]

# actual model
finalmodel = modelparams29oct[0]

# Apply the check_gradual_decrease function for each column in the sampled dataset
# NOTE: this will take multiple hours, instead, the finalmodel could be run
for col_name, params in modelparams29oct:
    elligibleplots[col_name] = elligibleplots.apply(
        check_gradual_decrease, axis=1, day_numbers=day_numbers, **params
    )
    # Print the number of True and False in the new column using value_counts
    print(elligibleplots[col_name].value_counts())

# Export the elligibleplots to parquet  
elligibleplots.to_parquet(outputfolder + '/YellowFields13nov.parquet')

# Export the actual sprayed plots to parquet, and the gewassenlijstYF to CSV
for col_name, _ in modelparams29oct:
    # Create a subfolder to stach all the results in 
    subfolder = f"{outputfolder}/{col_name}"
    os.makedirs(subfolder, exist_ok=True)
    # Print number of True and False in the Sprayed column
    sprayed_plots = elligibleplots[elligibleplots[col_name] == True]
    gewassenlijstYF = sprayed_plots['gewas'].value_counts()
    
    # Export the gewassenlijstYF to CSV
    gewassenlijstYF = pd.DataFrame(gewassenlijstYF)
    gewassenlijstYF.to_csv(f"{subfolder}/gewassenlijstFinalYF_{col_name}.csv")
    
    # Export the DataFrame to Parquet
    sprayed_plots.to_parquet(f"{subfolder}/Finalplots{col_name}.parquet")

## Make sure the validationplots are exported correctly, so they can be checked in excel
# This code can be adjusted to export the initial calibration datasets as well    
# read in the validationsets
validationset1 = gpd.read_parquet(outputfolder + '\FinalValPlotsBatch1.parquet')
validationset2 = gpd.read_parquet(outputfolder + '\FinalValPlotsBatch2.parquet')
# get the locids, and concatenate them
locidsbatch1 = validationset1['locid'].values
locidsbatch2 = validationset2['locid'].values
alllocids = locidsbatch1 + locidsbatch2
# filter allplots for the locids
validationplots = allplots[allplots['locid'].isin(alllocids)]
# add the sprayedcolumns to the validationplots
sprayedcolumns = ['Glypho_W16_DT05_FT045_IT06']
validationplots = validationplots.merge(elligibleplots[['locid'] + sprayedcolumns], on='locid', how='left')
#only export the gewas, locid and sprayedcolumns to a csv
validationplots[['gewas', 'locid'] + sprayedcolumns].to_csv(outputfolder + '/FinalValidationBatch12.csv', index=False)
