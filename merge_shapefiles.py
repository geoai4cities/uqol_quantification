import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from natsort import natsorted

shapefiles_dir = Path('../Bhopal/tile_polygon_index_UQoL/')
shapefiles = shapefiles_dir.glob('*.shp')

gdf = pd.concat([gpd.read_file(shp) for shp in tqdm(shapefiles, desc="[Merging…]", ascii=False, ncols=75)]).pipe(gpd.GeoDataFrame)
gdf.to_file(f'../Bhopal/Bhopal_Shapefile/Bhopal_Shapefile_UQoL_100m.shp')