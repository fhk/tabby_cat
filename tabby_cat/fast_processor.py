import graph_assembler_cpp
import pandas as pd
import os

class FastProcessor:
    def __init__(self):
        self.assembler = graph_assembler_cpp.GraphAssembler()

    def load_data(self, osm_pbf_path, demand_points_path):
        if osm_pbf_path:
            self.assembler.load_osm_pbf(osm_pbf_path)
        if demand_points_path:
            self.assembler.load_demand_points(demand_points_path)

    def process(self):
        self.assembler.snap_demand_to_graph()

    def get_nodes_df(self):
        nodes = self.assembler.get_nodes()
        data = []
        for n in nodes:
            data.append({
                'id': n.id,
                'x': n.x,
                'y': n.y,
                'prize': n.prize
            })
        return pd.DataFrame(data)

    def get_edges_df(self):
        edges = self.assembler.get_edges()
        data = []
        for e in edges:
            data.append({
                'u': e.u,
                'v': e.v,
                'length': e.length
            })
        return pd.DataFrame(data)

    def save_to_parquet(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        nodes_df = self.get_nodes_df()
        edges_df = self.get_edges_df()

        nodes_df.to_parquet(os.path.join(output_dir, 'nodes.parquet'))
        edges_df.to_parquet(os.path.join(output_dir, 'edges.parquet'))

        return nodes_df, edges_df
