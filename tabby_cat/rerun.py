"""
Main entry point for tabby cat
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
    where = sys.argv[1]

    logging.info("Starting processing")
    pr = Processor(where)
    test_lines = gpd.read_file(f"{where}/output/test_lines.shp")
    pr.add_test_line_edges(test_lines)
    logging.info("Snapping addresses to streets")
    pr.load_intermediate()
    pr.loaded = True
    logging.info("Converting GIS to graph")
    pr.geom_to_graph(rerun=True)
    #logging.info("Writing intermediate files")
    #pr.store_intermediate()

    logging.info("Create solver")
    sl = PCSTSolver(pr.edges, pr.look_up, pr.demand_nodes)
    logging.info("Running solve")
    sl.solve()

    pr.graph_to_geom(sl.s_edges)

    pr.solution.to_crs("epsg:4326").to_file(f"{where}/output/solution.shp")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
