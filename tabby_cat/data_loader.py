"""
The place where the data comes from!
"""
import os
import zipfile

import requests
from bs4 import BeautifulSoup
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

class DataLoader():
    """
    The DataLoader object for getting spatial data and storing it
    """
    known_regions = {
        "Alabama": "al",
        "Alaska": "ak",
        "Arizona": "ar",
        "Arkansas": "az",
        "California": "ca",
        "Colorado": "co",
        "Connecticut": "ct",
        "Delaware": "de",
        "Florida": "fl",
        "Georgia": "ga",
        "Hawaii": "hi",
        "Idaho": "id",
        "Illinois": "il",
        "Indiana": "in",
        "Iowa": "ia",
        "Kansas": "ks",
        "Kentucky": "ky",
        "Louisiana": "la",
        "Maine": "me",
        "Maryland": "md",
        "Massachusetts": "ma",
        "Michigan": "mi",
        "Minnesota": "mn",
        "Mississippi": "ms",
        "Missouri": "mo",
        "Montana": "mt",
        "Nebraska": "ne",
        "Nevada": "nv",
        "New Hampshire": "nh",
        "New Jersey": "nj",
        "New Mexico": "nm",
        "New York": "ny",
        "North Carolina": "nc",
        "North Dakota": "nd",
        "Ohio": "oh", # Do this one first
        "Oklahoma": "ok",
        "Oregon": "or",
        "Pennsylvania": "pa",
        "Rhode Island": "ri",
        "South Carolina": "sc",
        "South Dakota": "sd",
        "Tennessee": "tn",
        "Texas": "tx",
        "Utah": "ut",
        "Vermont": "vt",
        "Virginia": "va",
        "Washington": "wa",
        "West Virginia": "wv",
        "Wisconsin": "wi",
        "Wyoming": "wy",
    }

    californias = ["norcal", "socal"]
    geofabrik_url = "http://download.geofabrik.de/north-america/us/"
    geofabrik_ending_url_string = "-latest-free.shp.zip"
    street_file_name = "gis_osm_roads_free_1.shp"

    # need to get this page and then go through each states cities
    openaddress = "http://results.openaddresses.io/?runs=all#runs"
    where = []
    add_files = []
    data_cols = ['OA:x', 'OA:y']

    streets_df = None
    address_df = None

    def __init__(self):
        pass

    def download_data_geofabrik(self, region):
        if not os.path.isdir(region):
            if region is "California":
                for cal in californias:
                    download_data_geofabrik(cal)
            location = region.lower().replace(" ", "-")
            full_url = f"{self.geofabrik_url}{location}{self.geofabrik_ending_url_string}"
            page = requests.get(full_url, stream=True)
            with open(f"{region}.zip", "wb") as fd:
                for chunk in page.iter_content(chunk_size=128):
                    fd.write(chunk)

            with zipfile.ZipFile(f"{region}.zip", 'r') as zip_ref:
                zip_ref.extractall(f"{region}")
        
        return region

    def download_data_openaddress(self, region):

        page = requests.get(self.openaddress)
        soup = BeautifulSoup(page.content, 'html.parser')
        links = soup.find_all('a', href=True)
        files = 0
        zips = 0
        statewide = [l for l in links if l.text[:15] == f"us/{self.known_regions[region]}/statewide"]
        if statewide:
            links = statewide
        for l in links:
            if l.text[:5] == f"us/{self.known_regions[region]}":
                link = l.attrs.get("href")
                if link[-3:] == "zip":
                    data = requests.get(link, stream=True)
                    full_file_name = f"./{region}/{self.known_regions[region]}_{zips}.zip"
                    zips += 1
                    if not os.path.exists(full_file_name):
                        with open(full_file_name, "wb") as output:
                            for chunk in data.iter_content(chunk_size=128):
                                output.write(chunk)
                    with zipfile.ZipFile(full_file_name) as zip_ref:
                        extracted = zip_ref.namelist()
                        zip_ref.extractall(f"{region}/{zips}")
                        for e in extracted:
                            if e[-3:] == "shp":
                                self.add_files.append(f"{region}/{zips}/{e}")

                            if e[-3] == "csv":
                                self.add_files.append(f"{region}/{zips}/{e}")

                if link[-3:] == "csv":
                    data = requests.get(link, stream=True)
                    full_file_name = f"./{region}/{self.known_regions[region]}_{files}.csv"
                    self.add_files.append(full_file_name)
                    if not os.path.exists(full_file_name):
                        with open(full_file_name, "wb") as output:
                            for chunk in data.iter_content(chunk_size=128):
                                output.write(chunk)
                    files += 1



    def read_csv(self, file_name):
        return pd.read_csv(file_name, usecols=self.data_cols)

    def read_shp(self, file_name):
        return gpd.read_file(file_name)

    def read_street_data(self, region):
        streets = self.read_shp(f"./{region}/{self.street_file_name}")
        self.streets_df = streets[streets['fclass'].isin([
            "residential",
            "primary",
            "secondary",
            "tertiary",
            "service",
            "unclassified",
            "trunk",
            "motorway",
            "motorway_link",
            "service"])]

    def read_address_data(self, region):
        for file_name in self.add_files:
            if file_name[-3:] == 'csv':
                df = self.read_csv(file_name)
                if 'X' in df.columns:
                    df['OA:x'] = df['X']
                    df['OA:y'] = df['Y']
                if 'XCoord' in df.columns:
                    df['OA:x'] = df['XCoord']
                    df['OA:y'] = df['YCoord']

                gdf = gpd.GeoDataFrame(
                    df.drop(['OA:x', 'OA:y'], axis=1),
                        crs={'init': 'epsg:4326'},
                        geometry=[Point(xy) for xy in zip(df['OA:x'], df['OA:y'])])
            if file_name[-3:] == 'shp':
                gdf = gpd.read_file(file_name)
                gdf = gdf.to_crs("epsg:4326")
            if self.address_df is None:
                self.address_df = gdf
            else:
                self.address_df = self.address_df.append(gdf, ignore_index=True)

        self.address_df = self.address_df.drop_duplicates()
            


def main():
    where = "Alaska"
    dl = DataLoader()
    dl.download_data_geofabrik(where)
    dl.read_street_data(where)
    dl.download_data_openaddress(where)
    dl.read_address_data(where)

    import pdb; pdb.set_trace()



if __name__ == "__main__":
    main()