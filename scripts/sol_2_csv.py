import sys

import csv

import geopandas as gpd

def main():
    data = gpd.read_file(sys.argv[1])
    data['lat'] = data.geometry.apply(lambda x: x.centroid.coords[0][0])
    data['lon'] = data.geometry.apply(lambda x: x.centroid.coords[0][1])
    data = data.to_crs('epsg:2163')
    data['length'] = data.geometry.apply(lambda x: x.length)
    data = data.to_crs('epsg:4326')
    data.to_csv("network.csv")

if __name__ == "__main__":
    main()