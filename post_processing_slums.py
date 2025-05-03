import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape
import numpy as np
import fiona

# Load your raster and shapefile
# raster_path = '../Bhopal/Bhopal_LULC_Map_Georef.tif'
# shapefile_path = '../../MS_Thesis/QGIS/Shapefiles/informal_settlements_region_dissolve.shp'

raster_path = '../Datasets/Bhopal/roi_7/nDSM/roi_7_nDSM_resampled_scaled.tif'
shapefile_path = '../Bhopal/temp/slums.shp'

# # Open the raster and read its metadata
# with rasterio.open(raster_path) as src:
#     raster_meta = src.meta.copy()

#     # Read the shapefile
#     shapes = [shape(feature['geometry']) for feature in fiona.open(shapefile_path)]

#     # Get raster dimensions
#     height, width = src.height, src.width

#     # Create a mask of the raster using the shapefile's geometry
#     mask = rasterize(shapes, out_shape=(height, width), transform=src.transform, fill=0, all_touched=True)

#     # Read the raster data
#     raster_data = src.read(1)

#     # Update the values in the masked area
#     raster_data[mask == 1] = np.where(raster_data[mask == 1] == 0, 5, raster_data[mask == 1])

# # Update metadata for the output raster
# raster_meta.update(dtype=rasterio.uint8)  # Change data type if needed


# Open the DSM and read its metadata
with rasterio.open(raster_path) as src:
    raster_meta = src.meta.copy()

    # Read the shapefile
    shapes = [shape(feature['geometry']) for feature in fiona.open(shapefile_path)]

    # Get DSM dimensions
    height, width = src.height, src.width

    # Create a mask of the DSM using the shapefile's geometry
    mask = rasterize(shapes, out_shape=(height, width), transform=src.transform, fill=0, all_touched=True)

    # Read the DSM data
    raster_data = src.read(1)

    # Update the values in the masked area based on the condition
    condition = np.logical_and(raster_data >= 133.0, raster_data <= 135.0)
    raster_data[np.logical_and(condition, mask == 1)] = 140.0
    # raster_data[mask == 1] = 140

# Update metadata for the output DSM
raster_meta.update(count=1, dtype=rasterio.float32)  # Change data type if needed

# Write the updated raster to a new file
# output_path = '../Bhopal/Bhopal_LULC_Map_Georef_Modified.tif'
output_path = '../Bhopal/temp/nDSM_modified_v2.tif'

with rasterio.open(output_path, 'w', **raster_meta) as dst:
    dst.write(raster_data, 1)
