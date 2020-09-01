"""
The place where the data comes from!
"""
import os
import zipfile
import time

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
        "District of Columbia": 'dc',
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

    californias = ["california/norcal", "california/socal"]
    geofabrik_url = "http://download.geofabrik.de/north-america/us/"
    geofabrik_ending_url_string = "-latest-free.shp.zip"
    street_file_name = "gis_osm_roads_free_1.shp"

    # need to get this page and then go through each states cities
    openaddress = "http://results.openaddresses.io/?runs=all#runs"
    where = []
    add_files = []
    data_cols = ['OA:x', 'OA:y']

    microsoft_buildings_url = "https://usbuildingdata.blob.core.windows.net/usbuildings-v1-1/"

    streets_df = None
    address_df = None

    def __init__(self):
        pass

    def download_data_geofabrik(self, region, url_location=None, index=0):
        if not os.path.isdir(f"{region}_{index}"):
            location = region.lower().replace(" ", "-")
            if url_location is not None:
                location = url_location
            if region == "California" and url_location is None:
                for i, cal in enumerate(self.californias):
                    self.download_data_geofabrik(region, cal, i)
                return region
            full_url = f"{self.geofabrik_url}{location}{self.geofabrik_ending_url_string}"
            if not os.path.exists(f"{region}_{index}.zip"):
                page = requests.get(full_url, stream=True)
                with open(f"{region}_{index}.zip", "wb") as fd:
                    for chunk in page.iter_content(chunk_size=128):
                        fd.write(chunk)

            with zipfile.ZipFile(f"{region}_{index}.zip", 'r') as zip_ref:
                zip_ref.extractall(f"{region}_{index}")
        
        return region

    def download_data_openaddress(self, region):

        page = requests.get(self.openaddress)
        soup = BeautifulSoup(page.content, 'html.parser')
        links = soup.find_all('a', href=True)
        files = 0
        zips = 0
        state_region = 'statewide'

        if not os.path.isdir(region):
            os.mkdir(region)

        if region in ["Texas", "Mississippi"]:
            state_region  = "statewide-partial"
        url_link_region = f"us/{self.known_regions[region]}/{state_region}.zip"
        statewide = [l for l in links if l.attrs.get("href")[-len(url_link_region):] == url_link_region]
        if statewide:
            links = statewide
        time.sleep(5)
        for l in links:
            what_region_url = f"us{self.known_regions[region]}"
            link = l.attrs.get("href")
            if "".join(link.split('/')[-3:][:2]) == what_region_url:
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
                            if e[-3:] == 'csv':
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

    def download_data_microsoft_buildings(self, region):
        if not os.path.exists(f"{region}/{region}.geojson"):
            location = region.replace(" ", "")
            full_url = f"{self.microsoft_buildings_url}{location}.zip"
            if not os.path.exists(f"{region}.zip"):
                page = requests.get(full_url, stream=True)
                with open(f"{region}.zip", "wb") as fd:
                    for chunk in page.iter_content(chunk_size=128):
                        fd.write(chunk)

            with zipfile.ZipFile(f"{region}.zip", 'r') as zip_ref:
                extracted = zip_ref.namelist()
                zip_ref.extractall(f"{region}")
                for e in extracted:
                    if e[-4:] == 'json':
                        self.add_files.append(f"{region}/{region}.geojson")
        
        return region

    def read_csv(self, file_name):
        return pd.read_csv(file_name)

    def read_shp(self, file_name):
        return gpd.read_file(file_name)

    def read_geojson(self, file_name):
        return gpd.read_file(file_name)

    def read_street_data(self, region):
        if region == 'California':
            streets_1 = self.read_shp(f"./{region}_0/{self.street_file_name}")
            streets_2 = self.read_shp(f"./{region}_1/{self.street_file_name}")
            streets = streets_1.append(streets_2)
        else:
            streets = self.read_shp(f"./{region}_0/{self.street_file_name}")
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
                df = df[(df.LON >= -180) & (df.LON <= 180) & (df.LAT >= -90) & (df.LAT <= 90)]
                gdf = gpd.GeoDataFrame(
                        df,
                        crs={'init': 'epsg:4326'},
                        geometry=[Point(xy) for xy in zip(df['LON'], df['LAT'])])
            elif file_name[-3:] == 'shp':
                gdf = gpd.read_file(file_name)
                gdf = gdf.to_crs("epsg:4326")
            elif file_name[-4:] == 'json':
                gdf = gpd.read_file(file_name)
                gdf.geometry = gdf.geometry.apply(lambda x: Point(x.centroid.coords[0]))
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
