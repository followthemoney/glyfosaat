# README glyfosaat

This repository contains the code that was used for our story on glyphosate-use detection in the Netherlands in the early spring of 2020. The data is downloaded and analyzed in Python scripts. Note: you need an account at Google Earth Engine, to be able to analyze the Sentinel-2 data.

## Used data

For our analysis we used several open source datasets:

1. [Basisregristratie Gewaspercelen](https://service.pdok.nl/rvo/brpgewaspercelen/atom/v1_0/basisregistratie_gewaspercelen_brp.xml) (BRP) from the Rijksdienst voor Ondernemend Nederland (RVO) via PDOK.
2. [Sentinel-2 satellite data](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) from the European Space Agency (ESA) via the Google Earth Engine (GEE) Data Catalog  
3. [Groundwater protection areas](https://service.pdok.nl/provincies/grondwaterbeschermingsgebieden/atom/index.xml) by BIJ12 via PDOK
4. [Natura-2000 areas](PDOK) via PDOK
5. [Primary schools](https://duo.nl/open_onderwijsdata/primair-onderwijs/scholen-en-adressen/hoofdvestigingen-basisonderwijs.jsp) from Dienst Uitvoering Onderwijs (DUO)
6. [Kaderrichtlijn Water](https://service.pdok.nl/ihw/gebiedsbeheer/krw-oppervlaktewaterlichamen/atom/index.xml) from Informatiehuis Water via PDOK
7. [Outside playgrounds](https://buitenspeelkaart.nl/) from buitenspeelkaart.nl
8. [Population information](https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2020) on neighbourhood-level via the Centraal Bureau voor Statistiek (CBS)

## Methodology and workflow

The BRP is based on a yearly declaration, which all farmers have to do in the Netherlands and shows which crops are grown at which plots. The dataset was pre-processed to exclude winter crops, perennial crops, and tulips, since it would be either highly unlikely these are sprayed with glyphosate during this time of year, or because our method would not be able to accurately detect glyphosate application. A complete overview of which crops were and were not included can be found [here](data/ElligibleCrops.csv).

Our analysis was run for 58 crop-categories, which comprise 1.594,816 hectares, or 86% of the total acreage in the BRP. Of these 58 crops, we investigated what their probable final sowing date is. If available, this is also included in the overview.

The code should be run in chronological order. Below is indicated what each script does briefly. For a more elaborate explanation, please refer to the [technical addendum](docs/technical%20addendum.pdf) (in Dutch).

### [A1_get_satellite_data](src/A1_get_satellite_data.py)

We used open-source satellite data for this analysis. As indicated in [Bloem et al., 2020](https://www.mdpi.com/2073-4395/10/9/1409), the NDVI-values of glyphosate-sprayed plots becomes evident within 1 to 2 weeks. Detecting such fields with satellite imagery thus requires a high temporal resolution, i.e. you need enough images of consecutive days to be able to detect glyphosate use. This is why we opted to analyze the early spring of 2020 in the Netherlands. This was a sufficient sunny, but not extremely dry, early spring. We use Sentinel-2 data because of its spatial and temporal resolution, legacy, and availability through Google Earth Engine.

The code retrieves the BRP set, which is freely available. We pre-processed this by selecting the top 88 crop-categories to be used for calibration- and validation-set creation. The 58 crop-categories that were eligible for analysis is a subset of these 88 crop-categories.

We use the Python implementation of Google Earth Engine (GEE), to retrieve both NDVI (for analysis) as RGB (for validation) data for the plots. The data is retrieved in such a way that all the available imagery are mosaicked by the date. We constructed the code in such a way, that other indices could also be exported. This information was however never used.

Since GEE processes the data requests on their servers, there is a limit when you try to export the data to a local drive. Therefore the BRP had to be cut up in chunks. The data is exported as parquet files to a specific folder.

The code from line 365 onward, is used to generate RGB images for each plot for validation. Adjust the variable validationset in line 442 accordingly.

### [B1_NDVI_per_plot](src/B1_NDVI_per_plot.py)

The parquet chunks are loaded in as a geodataframe. Make sure the range in line 22 corresponds to the number of chunks that were generated in A1. The rest of the script renames the columns to day_xxx for easier analysis. The code can be adjusted to analyze other indices. It also calculates the mean index for each crop. This information was eventually not used in our analysis.

### [B2_Algorithm](src/B2_Algorithm.py)

The complete parquet file is merged with the crop data and the geodataframe now contains all plots with, among others:

- the mean NDVI value for each available day
- the final sowing date
- the crop that was grown in 2020

The algorithm was constructed based on the results from [Bloem et al., (2020)](https://www.mdpi.com/2073-4395/10/9/1409) & [Pause et al., (2019)](https://www.mdpi.com/2072-4292/11/21/2541). Both papers typified crop's responses to glyphosate application. The NDVI of crops gradually decreases between 1 and 3 weeks.

We encoded this gradual decrease in to a series of nested if-statements in the function check_gradual_decrease. It can be viewed as a decision-tree through which a plot can go multiple times. For a more in-depth explanation see the technical addendum.

This function was first run for a set of parameters as indicated in the technical addendum. We checked the results based on the validationset1 and validationset2. We calibrated the model with different parameters on this combined set (of a total of 274 plots). We then ran the algorithm for the nine parameter-combinations as described in the list modelparams29oct. This gave us our final model with the next configuration:

- Window: 16
- Drop_treshold: 0.5
- Final_treshold: 0.45
- Initial_treshold: 0.6

The geodataframe with all the sprayed plots was exported as a geoparquet, and used in the next python script.

### [C1_Spatial_analysis](src/C1_Spatial_analysis.py)

The sprayed plots are reprojected to RD New (EPSG: 28992) so overlapping geometries are correctly calculated. We created 3 buffers (of 250, 500, and 1000 meters), roughly based on the findings in the [OBO-research](https://www.rivm.nl/documenten/onderzoeksrapport-obo) by the RIVM. The next datasets of our Areas Of Interest (AOIs) are free and openly available, mostly through PDOK:

- Natura-2000 areas
- Groundwater-protection areas
- Province geometries
- KRW-waterbodies
- CBS-neighborhood data

The data on primary schools, playgrounds and land-ownership (pachtdata) was retrieved through external sources and cannot be made available on request.

The Groundwater-protection areas were split up in separate polygons, although belonging to the same groundwater-protection area as defined by the provinces. See for example [this map](https://www.schoon-water.nl/bewoners/plattegrond%20winningen.pdf) of Noord-Brabant, note how each groundwater-protection area has multiple different zones. The groundwater-protection data did not have a column that defined this shared geometry, so the polygons were merged based on their geometric overlap and adjacency.

These AOIs were then intersected either with direct overlap (for the groundwater-protection areas) or with the buffers (for the other AOIs). The neighborhood-data was also intersected with each buffer, to calculate the amount of people living inside each buffer.

These analyses were run for the Netherlands as a whole, and each province specifically.