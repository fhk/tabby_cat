"""
Run the DataLoader dataframes through processing
"""
import os
import pickle
import logging
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
import networkx as nx
from shapely import wkt
from shapely.ops import split
from shapely.geometry import LineString, MultiPoint, MultiLineString
from pyproj import Proj, transform


class Processor():
    def __init__(self, where):
        self.where = where
        self.snap_lines = None
        self.all_lines = None
        self.cut_lines = None
        self.demand = set()
        self.demand_nodes = defaultdict(int)
        self.edge_to_geom = {}
        self.convert_ids = None
        self.inProj = Proj(init='epsg:3857')
        self.outProj = Proj(init='epsg:4326')
        self.loaded = False
        self.look_up = {}
        self.edges = OrderedDict()
        self.index = 0

    def _parallelize(self, points, lines):
        """
        Concept taken from here: https://swanlund.space/parallelizing-python
        """
        cpus = 4 # mp.cpu_count()
        
        intersection_chunks = np.array_split(points, cpus)
        
        pool = mp.Pool(processes=cpus)
        
        chunk_processes = [pool.apply_async(self._snap_part, args=(chunk, lines)) for chunk in intersection_chunks]

        intersection_results = [chunk.get() for chunk in chunk_processes]
        
        intersections_dist = pd.concat(intersection_results)

        return intersections_dist

    def _snap_part(self, gdf_chunk, lines):
        offset = 1000

        bbox = gdf_chunk.bounds + [-offset, -offset, offset, offset]
        hits = bbox.apply(lambda row: list(lines.sindex.intersection(row)), axis=1)

        tmp = pd.DataFrame({
            # index of points table
            "pt_idx": np.repeat(hits.index, hits.apply(len)),    # ordinal position of line - access via iloc later
            "line_i": np.concatenate(hits.values)
        })
        # Join back to the lines on line_i; we use reset_index() to 
        # give us the ordinal position of each line
        tmp = tmp.join(lines.reset_index(drop=True), on="line_i")
        # Join back to the original points to get their geometry
        # rename the point geometry as "point"
        tmp = tmp.join(gdf_chunk.geometry.rename("point"), on="pt_idx")
        # Convert back to a GeoDataFrame, so we can do spatial ops
        tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=gdf_chunk.crs)

        tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
        # Discard any lines that are greater than tolerance from points
        tolerance = 100
        #tmp = tmp.loc[tmp.snap_dist <= tolerance]
        # Sort on ascending snap distance, so that closest goes to top
        tmp = tmp.sort_values(by=["snap_dist"])
        # group by the index of the points and take the first, which is the
        # closest line
        closest = tmp.groupby("pt_idx").first()

        # construct a GeoDataFrame of the closest lines
        return  gpd.GeoDataFrame(closest, geometry="geometry")

    def points_to_multipoint(self, data):
        coords = set()
        for p in data.snapped:
            coords.add(p.coords[0])
            self.demand.add(p.coords[0])

        return data.geometry.iloc[0].difference(MultiPoint(list(coords)).buffer(1e-7))

    def project_array(self, coordinates):
        """
        Project a numpy (n,2) array in projection srcp to projection dstp
        Returns a numpy (n,2) array.
        """

        fx, fy = pyproj.transform(self.inProj, self.outProj, coordinates[:,0], coordinates[:,1])
        # Re-create (n,2) coordinates
        return np.dstack([fx, fy])[0]

    def snap_points_to_line(self, lines, points, write=True):
        """
        Taken from here: https://medium.com/@brendan_ward/how-to-leverage-geopandas-for-faster-snapping-of-points-to-lines-6113c94e59aa
        """
        # this creates and also provides us access to the spatial index
        if os.path.isfile(f'{self.where}/output/edge_to_geom.pickle'):
            self.load_intermediate()
            self.loaded = True
            return
        lines = lines.to_crs('epsg:3857')
        points = points.to_crs('epsg:3857')
        self.lines = lines

        closest = self._parallelize(points, lines)
        # Position of nearest point from start of the line
        series = gpd.GeoSeries(closest.point)
        series.crs = {'init': 'epsg:3857'}
        pos = closest.geometry.project(series)
        # Get new point location geometry
        new_pts = closest.geometry.interpolate(pos)
        # Identify the columns we want to copy from the closest line to the point, such as a line ID.
        line_columns = ['line_i', 'osm_id', 'code', 'fclass']
        # Create a new GeoDataFrame from the columns from the closest line and new point geometries (which will be called "geometries")
        snapped = gpd.GeoDataFrame(
            closest[line_columns], geometry=new_pts, crs="epsg:3857")
        closest['snapped'] = snapped.geometry
        split_lines = closest.groupby(closest["line_i"]).apply(lambda x: self.points_to_multipoint(x))
        split_lines_df = pd.DataFrame({"geom": split_lines}, index=[i for i in range(len(split_lines))])
        self.cut_lines = gpd.GeoDataFrame(split_lines_df, geometry="geom", crs="epsg:3857")
 
        # Join back to the original points:
        updated_points = points.drop(columns=["geometry"]).join(snapped)
        # You may want to drop any that didn't snap, if so: 
        updated_points.dropna(subset=["geometry"]).geometry.apply(lambda x: self.get_demand_nodes(x))
        points.dropna(subset=["geometry"]).geometry.apply(lambda x: self.get_demand_nodes(x))
        self.snap_lines = closest.apply(lambda x: LineString([x.point.coords[0], x.snapped.coords[0]]), axis=1)
        self.snap_lines = pd.DataFrame({"geom": self.snap_lines}, index=[i for i in range(len(self.snap_lines))])
        self.snap_lines = gpd.GeoDataFrame(self.snap_lines, geometry="geom", crs="epsg:3857")
        self.snap_lines = self.snap_lines.dropna()
        self.snap_lines['length'] = self.snap_lines.apply(lambda x: x.geom.length, axis=1)
        if write:
            if not os.path.isdir(f"{self.where}/output"):
                os.mkdir(f"{self.where}/output")
            updated_points.to_file(f"{self.where}/output/updated.shp")
            snap_gdf = self.snap_lines.to_crs('epsg:4326')
            snap_gdf['lat'] = snap_gdf.geometry.apply(lambda x: x.coords[0][0])
            snap_gdf['lon'] = snap_gdf.geometry.apply(lambda x: x.coords[0][1])
            snap_gdf[["lat", "lon", "length"]].to_csv(f"{self.where}/output/connections.csv")
            snap_gdf.to_file(f"{self.where}/output/test_lines.shp")

    def get_demand_nodes(self, geometry):
        coords = geometry.coords[0]
        coord_string = f'[{coords[0]:.0f}, {coords[1]:.0f}]'
        self.demand_nodes[coord_string] = 1

    def set_node_ids(self, geometry):
        start = None
        end = None
        if geometry.geom_type == "LineString":
            coords = geometry.coords[:]
            s = coords[0]
            e = coords[-1]
            s_coord_string = f'[{s[0]:.0f}, {s[1]:.0f}]'
            start = self.look_up.get(s_coord_string, None)
            if start is None:
                self.look_up[s_coord_string] = self.index
                start = self.index
                self.index += 1
            e_coord_string = f'[{e[0]:.0f}, {e[1]:.0f}]'
            end = self.look_up.get(e_coord_string, None)
            if end is None:
                self.look_up[e_coord_string] = self.index
                end = self.index
                self.index += 1
            self.edges[(start, end)] = geometry.length
            self.edge_to_geom[(start, end)] = geometry.wkt
        else:
            for line in geometry:
                coords = line.coords[:]
                s = coords[0]
                e = coords[-1]
                s_coord_string = f'[{s[0]:.0f}, {s[1]:.0f}]'
                start = self.look_up.get(s_coord_string, None)
                if start is None:
                    self.look_up[s_coord_string] = self.index
                    start = self.index
                    self.index += 1
                e_coord_string = f'[{e[0]:.0f}, {e[1]:.0f}]'
                end = self.look_up.get(e_coord_string, None)
                if end is None:
                    self.look_up[e_coord_string] = self.index
                    end = self.index
                    self.index += 1
                self.edges[(start, end)] = line.length
                self.edge_to_geom[(start, end)] = line.wkt

    def expand_lines(self, geom):
        new_lines = []
        if geom.geom_type == "LineString":
            coords = geom.coords[:]
            for i in range(len(coords) - 1):
                new_lines.append([coords[i], coords[i + 1]])
        else:
            for sub_geom in geom:
                coords = sub_geom.coords[:]
                for i in range(len(coords) - 1):
                    new_lines.append([coords[i], coords[i + 1]])
        if len(new_lines) > 1:
            return MultiLineString(new_lines)
        else:
            return LineString(new_lines[0])

    def add_inter_demand_connections(self, largest):
        flip_node = {v: k for k, v in self.convert_ids.items()}
        id_conn = OrderedDict()
        for n in self.nodes_to_connect:
            node = self.flip_look_up[n]
            path = nx.single_source_shortest_path(self.g, n, 3)
            for next_node in list(path.keys())[2:]:
                if self.demand_nodes[next_node] and self.g.degree(next_node) == 1:
                    if (n, next_node) in self.edges or (next_node, n) in self.edges:
                        continue
                    nn_coord = self.flip_look_up[next_node]
                    line = LineString([eval(node), eval(nn_coord)])
                    self.edge_to_geom[flip_node[n], flip_node[next_node]] = line.wkt
                    cost = line.length
                    if len(path) == 3:
                        id_conn[n, next_node] = cost * 3  # Increase cost to prefer drop
                    edge_mid = tuple(list(path.values())[2][1:])
                    edge_mid_flip = edge_mid[::-1]
                    edge_length = self.edges.get(edge_mid, False)
                    if not edge_length:
                        edge_length = self.edges[edge_mid_flip]
                    if len(path) == 4 and edge_length < 9:
                        id_conn[n, next_node] = cost
                    else:
                        id_conn[n, next_node] = cost * 2
        return id_conn

    def add_test_line_edges(self, test_lines):
        test_lines = test_lines.to_crs("epsg:3857")
        max_node_full_graph = max(self.convert_ids.keys())
        flip_node = {v: k for k, v in self.convert_ids.items()}
        self.nodes_to_connect = set()
        for line in test_lines.geometry:
            demand, node = line.coords[:]
            s_coord_string = f'[{demand[0]:.0f}, {demand[1]:.0f}]'
            e_coord_string = f'[{node[0]:.0f}, {node[1]:.0f}]'
            end = self.convert_ids.get(self.look_up.get(e_coord_string, None), None)
            if end is None:
                continue

            start = self.look_up.get(s_coord_string, None)

            if start:
                continue

            if start is None:
                start = self.index
                max_node_full_graph += 1
                self.convert_ids[max_node_full_graph] = start
                self.look_up[s_coord_string] = start
                self.index += 1

            if (start, end) in self.edges:
                continue

            self.demand_nodes[start] = 1
            self.demand_nodes[end] = 1
            self.nodes_to_connect.add(start)
            self.edge_to_geom[(max_node_full_graph, flip_node[end])] = line.wkt
            self.edges[(start, end)] = line.length
        self.flip_look_up = {v: k for k, v in self.look_up.items()}

    def geom_to_graph(self, rerun=False):
        if not self.edges:
            self.cut_lines = self.cut_lines.dropna()
            self.cut_lines.geometry = self.cut_lines.geometry.apply(lambda x: self.expand_lines(x))
            self.lines.geometry = self.lines.geometry.apply(lambda x: self.expand_lines(x))
            self.cut_lines.geometry.apply(lambda x: self.set_node_ids(x))
            self.lines.geometry.apply(lambda x: self.set_node_ids(x))
            self.snap_lines.geometry.apply(lambda x: self.set_node_ids(x))

        self.g = nx.Graph()
        self.g.add_edges_from(self.edges)
        largest_cc = max(nx.connected_components(self.g), key=len)
        self.add_inter_demand_connections(largest_cc)

        if not rerun:
            self.flip_look_up = {v: k for k, v in self.look_up.items()}
            self.convert_ids = {n: i for i, n in enumerate(largest_cc)}
            self.edges = OrderedDict(((self.convert_ids[k[0]], self.convert_ids[k[1]]), v) for k, v in self.edges.items() if k[0] in largest_cc)
            self.look_up = {k:self.convert_ids[v] for k, v in self.look_up.items() if v in largest_cc}    
            self.demand_nodes = defaultdict(int, {v:self.demand_nodes[self.flip_look_up[k]] for k, v in self.convert_ids.items()})
            
        demand_not_on_graph = len(self.demand) - len(self.demand_nodes)
        logging.info(f"Missing {demand_not_on_graph} points on connected graph")

    def graph_to_geom(self, s_edges):
        edge_keys = list(self.edges)
        flip_node = {v:k for k, v in self.convert_ids.items()}
        s_frame = pd.DataFrame([[i, self.edge_to_geom.get(
            (flip_node[edge_keys[s][0]], flip_node[edge_keys[s][1]]),
            LineString([eval(self.flip_look_up[edge_keys[s][0]]), eval(self.flip_look_up[edge_keys[s][1]])]).wkt)]
            for i, s in enumerate(s_edges)], columns=['id', 'geom'])
        s_frame['geom'] = s_frame.geom.apply(wkt.loads)
        self.solution = gpd.GeoDataFrame(s_frame, geometry='geom', crs='epsg:3857')

    def load_intermediate(self):
        with open(f'{self.where}/output/demand_nodes.pickle', 'rb') as handle:
            self.demand_nodes = pickle.load(handle)

        with open(f'{self.where}/output/look_up.pickle', 'rb') as handle:
            self.look_up = pickle.load(handle)
            self.flip_look_up = {v: k for k, v in self.look_up.items()}
 
        with open(f'{self.where}/output/edges.pickle', 'rb') as handle:
            self.edges = pickle.load(handle)

        with open(f'{self.where}/output/edge_to_geom.pickle', 'rb') as handle:
            self.edge_to_geom = pickle.load(handle)

        with open(f'{self.where}/output/convert_ids.pickle', 'rb') as handle:
            self.convert_ids = pickle.load(handle)
            self.index = max(self.convert_ids.values()) + 1

    def store_intermediate(self):
        with open(f'{self.where}/output/demand_nodes.pickle', 'wb') as handle:
            pickle.dump(self.demand_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.where}/output/look_up.pickle', 'wb') as handle:
            pickle.dump(self.look_up, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.where}/output/edges.pickle', 'wb') as handle:
            pickle.dump(self.edges, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.where}/output/edge_to_geom.pickle', 'wb') as handle:
            pickle.dump(self.edge_to_geom, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{self.where}/output/convert_ids.pickle', 'wb') as handle:
            pickle.dump(self.convert_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
