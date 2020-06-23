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
        pass

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
        pos.crs = {'init': 'epsg:3857'}
        # Get new point location geometry
        new_pts = closest.geometry.interpolate(pos)
        #Identify the columns we want to copy from the closest line to the point, such as a line ID.
        line_columns = ['line_i', 'osm_id', 'code', 'fclass']
        # Create a new GeoDataFrame from the columns from the closest line and new point geometries (which will be called "geometries")
        snapped = gpd.GeoDataFrame(
            closest[line_columns],geometry=new_pts)
        closest['snapped'] = snapped.geometry
        split_df = closest.groupby(closest["line_i"]).apply(lambda x: self.points_to_multipoint(x))
        test = closest.apply(lambda x: LineString([x.point.coords[0], x.snapped.coords[0]]), axis=1)
        df = pd.DataFrame({"geom": test})
        gdf = gpd.GeoDataFrame(df, geometry="geom")
        gdf.crs = {'init': 'epsg:3857'}
        # gdf.to_crs("epsg:4326").to_file("test_lines.shp")
        # Join back to the original points:
        updated_points = points.drop(columns=["geometry"]).join(snapped)
        # You may want to drop any that didn't snap, if so: 
        updated_points = updated_points.dropna(subset=["geometry"]).to_crs('epsg:4326')
        #updated_points.to_file("updated.shp")