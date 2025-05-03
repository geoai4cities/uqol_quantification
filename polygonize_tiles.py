import os
from tqdm import tqdm
from natsort import natsorted

def generate_tile_index(tiles_dir, tiles_polygon_dir):
    filenames = natsorted(os.listdir(tiles_dir))
    for file in tqdm(filenames, ncols=70):
        shp_file = file.split('.tif')[0]
        os.system(f'gdaltindex {tiles_polygon_dir}{shp_file}.shp {tiles_dir}{file}')

generate_tile_index(tiles_dir='../Bhopal/tile_polygon/', tiles_polygon_dir='../Bhopal/tile_polygon_index/')
