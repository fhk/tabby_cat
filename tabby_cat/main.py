"""
Main entry point for tabby cat
"""
import os
import logging
import sys

from tabby_cat.data_loader import DataLoader
from tabby_cat.processor import Processor
from tabby_cat.solver import PCSTSolver

def main():
    logging.basicConfig(filename='log.log',level=logging.DEBUG)

    logging.info("Started DataLoader")
    dl = DataLoader()
    where = sys.argv[1]
    #for where in dl.known_regions:
    logging.info(f"Running on {where}")

    logging.info("Getting data from geofabrik")
    dl.download_data_geofabrik(where)
    logging.info("Reading street data")
    dl.read_street_data(where)
    logging.info("Getting data from microsoft buildings")
    dl.download_data_microsoft_buildings(where)
    logging.info("Reading address data")
    dl.read_address_data(where)

    logging.info("Starting processing")
    pr = Processor(where)
    logging.info("Snapping addresses to streets")
    pr.snap_points_to_line(dl.streets_df, dl.address_df)
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
