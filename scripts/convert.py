import sys

import csv

import geopandas as gpd


def main():
    data = gpd.read_file(sys.argv[1])
    data_m = data.to_crs(epsg='2163')

    with open("converted.csv", 'w') as convert:
        writer = csv.writer(convert)

        for row in data_m[data['geometry'].geom_type == 'LineString'].iterrows():
            writer.writerow([row[-1][1].wkt])





if __name__ == "__main__":
    main()