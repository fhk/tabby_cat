from pathlib import Path
import sys

import geopandas as gpd

from tabby_cat.processor import Processor
from tabby_cat.solver import PCSTSolver


def main():
    streets = gpd.read_file(sys.argv[1])
    buildings = gpd.read_file(sys.argv[2])
    buildings['geometry'] = buildings.apply(lambda x: x.geometry.centroid, axis=1)
    Path('./test/output').mkdir(parents=True, exist_ok=True)
    pr = Processor(f"test")
    pr.snap_points_to_line(streets, buildings)
    pr.geom_to_graph()
    sl = PCSTSolver(pr.edges, pr.look_up, pr.demand_nodes)
    sl.solve()
    pr.graph_to_geom(sl.s_edges)
    pr.solution.to_crs("epsg:4326").to_file(f"test/output/solution.shp")



if __name__ == "__main__":
    main()
