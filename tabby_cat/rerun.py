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
    if len(sys.argv) > 2:
        traverse = int(sys.argv[2]) # 3
        node_gap = int(sys.argv[3]) # 9
        two_edge_cost = int(sys.argv[4]) # 3
        four_edge_cost = int(sys.argv[5]) # 1
        n_edge_cost = int(sys.argv[6]) # 2
        nearest_cost = int(sys.argv[7]) # 1
    logging.info("Starting processing")
    pr = Processor(where)
    logging.info("Snapping addresses to streets")
    pr.load_intermediate()

    test_lines = gpd.read_file(f"{where}/output/test_lines.shp")
    pr.add_test_line_edges(test_lines)

    pr.loaded = True
    logging.info("Converting GIS to graph")
    if len(sys.argv) > 2:
        pr.geom_to_graph(
            rerun=True,
            traverse=traverse,
            node_gap=node_gap,
            two_edge_cost=two_edge_cost,
            four_edge_cost=four_edge_cost,
            n_edge_cost=n_edge_cost,
            nearest_cost=nearest_cost)
    else:
        pr.geom_to_graph(rerun=True)
    #logging.info("Writing intermediate files")
    #pr.store_intermediate()

    logging.info("Create solver")
    sl = PCSTSolver(pr.edges, pr.look_up, pr.demand_nodes)
    logging.info("Running solve")
    sl.solve()

    pr.graph_to_geom(sl.s_edges)
    if len(sys.argv) > 2:
        pr.solution.to_crs("epsg:4326").to_file(
            f"{where}/output/{traverse}_{node_gap}_{two_edge_cost}_{four_edge_cost}_{n_edge_cost}_{nearest_cost}_solution.shp")
    else:
        pr.solution.to_crs("epsg:4326").to_file(
            f"{where}/output/solution.shp")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
