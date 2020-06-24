"""
Run the DataLoader dataframes through processing
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import split
from shapely.geometry import LineString, MultiPoint

class Processor():
    def __init__(self):
        self.snap_lines = None
        self.all_lines = None
        self.cut_lines = None

    def points_to_multipoint(self, data):
        coords = set()
        for p in data.snapped:
            coords.add(p.coords[0])

        return data.geometry.iloc[0].difference(MultiPoint(list(coords)).buffer(1e-7))

    def snap_points_to_line(self, lines, points):
        """
        Taken from here: https://medium.com/@brendan_ward/how-to-leverage-geopandas-for-faster-snapping-of-points-to-lines-6113c94e59aa
        """
        # this creates and also provides us access to the spatial index
        lines = lines.to_crs('epsg:3857')
        self.lines = lines
        points = points.to_crs('epsg:3857')
        offset = 500
        bbox = points.bounds + [-offset, -offset, offset, offset]
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
        tmp = tmp.join(points.geometry.rename("point"), on="pt_idx")
        # Convert back to a GeoDataFrame, so we can do spatial ops
        tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=points.crs)

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
        closest = gpd.GeoDataFrame(closest, geometry="geometry")
        # Position of nearest point from start of the line
        series = gpd.GeoSeries(closest.point)
        series.crs = {'init': 'epsg:3857'}
        pos = closest.geometry.project(series)
        # Get new point location geometry
        new_pts = closest.geometry.interpolate(pos)
        #Identify the columns we want to copy from the closest line to the point, such as a line ID.
        line_columns = ['line_i', 'osm_id', 'code', 'fclass']
        # Create a new GeoDataFrame from the columns from the closest line and new point geometries (which will be called "geometries")
        snapped = gpd.GeoDataFrame(
            closest[line_columns],geometry=new_pts, crs="epsg:3857")
        closest['snapped'] = snapped.geometry
        split_lines = closest.groupby(closest["line_i"]).apply(lambda x: self.points_to_multipoint(x))
        split_lines_df = pd.DataFrame({"geom": split_lines})
        self.cut_lines = gpd.GeoDataFrame(split_lines_df, geometry="geom", crs="epsg:3857")
        snap_lines = closest.apply(lambda x: LineString([x.point.coords[0], x.snapped.coords[0]]), axis=1)
        snap_df = pd.DataFrame({"geom": snap_lines})
        snap_gdf = gpd.GeoDataFrame(snap_df, geometry="geom", crs="epsg:3857")
        self.snap_lines = snap_gdf
        # gdf.to_crs("epsg:4326").to_file("test_lines.shp")
        # Join back to the original points:
        # updated_points = points.drop(columns=["geometry"]).join(snapped)
        # You may want to drop any that didn't snap, if so: 
        # updated_points = updated_points.dropna(subset=["geometry"]).to_crs('epsg:4326')
        #updated_points.to_file("updated.shp")

    def geom_to_graph(self):
        self.lines["start"] = self.lines.geometry.apply(lambda x: x.coords[0])
        self.lines["end"] = self.lines.geometry.apply(lambda x: x.coords[-1])
        self.cut_lines["start"] = self.cut_lines.geometry.apply(lambda x: [geom.coords[0] for geom in x] if x.geom_type == "MultiLineString" else x.coords[0])
        self.cut_lines["end"] = self.cut_lines.geometry.apply(lambda x: [geom.coords[-1] for geom in x] if x.geom_type == "MultiLineString" else x.coords[-1])

    def graph_to_geom(self):
        pass
