import sys
import os
import csv

import geopandas as gpd

def main():
    data = gpd.read_file(sys.argv[1])
    data['lat'] = data.geometry.apply(lambda x: x.centroid.coords[0][0])
    data['lon'] = data.geometry.apply(lambda x: x.centroid.coords[0][1])
    data = data.to_crs('epsg:2163')
    data['length'] = data.geometry.apply(lambda x: x.length)
    data = data.to_crs('epsg:4326')
    part_path = os.path.basename(sys.argv[1])
    what, _ = os.path.splitext(part_path)
    dir_path = os.path.dirname(sys.argv[1])
    data.to_csv(os.path.join(dir_path, what + "_network.csv"))

if __name__ == "__main__":
    main()
