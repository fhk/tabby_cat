"""
A PoC for the healthsite.io data
"""
import os
import logging
import sys

import geopandas as gpd

from tabby_cat.data_loader import DataLoader
from tabby_cat.processor import Processor
from tabby_cat.solver import PCSTSolver

def main():
    logging.basicConfig(filename='log.log',level=logging.DEBUG)

    logging.info("Started DataLoader")
    dl = DataLoader()
    hospital_sites = gpd.read_file(sys.argv[1])
    hospital_sites = hospital_sites[hospital_sites['amenity'] == 'hospital']
    hospital_sites['geometry'] = hospital_sites.apply(lambda x: x.geometry.centroid, axis=1)
    streets = gpd.read_file(sys.argv[2])
    #for where in dl.known_regions:
    logging.info(f"Running on {hospital_sites}")
    pr = Processor("Senegal")
    logging.info("Snapping addresses to streets")
    pr.snap_points_to_line(streets, hospital_sites)
    logging.info("Converting GIS to graph")
    pr.geom_to_graph()
    logging.info("Writing intermediate files")
    pr.store_intermediate()

    logging.info("Create solver")
    sl = PCSTSolver(pr.edges, pr.look_up, pr.demand_nodes)
    logging.info("Running solve")
    sl.solve()

    pr.graph_to_geom(sl.s_edges)

    pr.solution.to_crs("epsg:4326").to_file(f"{where}/output/solution.shp")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
