"""
Main entry point for tabby cat
"""
import os
import logging

from tabby_cat.data_loader import DataLoader
from tabby_cat.processor import Processor
from tabby_cat.solver import PCSTSolver

def main():
    logging.basicConfig(filename='log.log',level=logging.DEBUG)
    where = "Vermont"
    logging.info(f"Running on {where}")
    logging.info("Started DataLoader")
    dl = DataLoader()

    if not os.path.isfile("geom_to_graph.pickle"):

        logging.info("Getting data from geofabrik")
        dl.download_data_geofabrik(where)
        logging.info("Reading street data")
        dl.read_street_data(where)
        logging.info("Getting data from openaddress") 
        dl.download_data_openaddress(where)
        logging.info("Reading address data")
        dl.read_address_data(where)
        #dl.address_df.to_file("address.shp")

    logging.info("Starting processing")
    pr = Processor(where)
    logging.info("Snapping addresses to streets")
    pr.snap_points_to_line(dl.streets_df, dl.address_df)
    logging.info("Converting GIS to graph")
    pr.geom_to_graph()

    logging.info("Create solver")
    sl = PCSTSolver(pr.edges, pr.look_up, pr.demand_nodes)
    logging.info("Running solve")
    sl.solve()

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
