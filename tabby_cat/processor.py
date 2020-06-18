"""
Run the DataLoader dataframes through processing
"""
import numpy as np
import pandas as pd
import geopandas as gpd


class Processor():
    def __init__(self):
        pass

    def snap_points_to_line(self, lines, points):
        """
        Taken from here: https://medium.com/@brendan_ward/how-to-leverage-geopandas-for-faster-snapping-of-points-to-lines-6113c94e59aa
        """
        # this creates and also provides us access to the spatial index
        lines = lines.to_crs(epsg=3857)
        points = points.to_crs(epsg=3857)
        offset = 100
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
        tmp = tmp.loc[tmp.snap_dist <= tolerance]# Sort on ascending snap distance, so that closest goes to top
        tmp = tmp.sort_values(by=["snap_dist"])
        # group by the index of the points and take the first, which is the
        # closest line
        closest = tmp.groupby("pt_idx").first()
        del tmp
        # construct a GeoDataFrame of the closest lines
        closest = gpd.GeoDataFrame(closest, geometry="geometry")
        # Position of nearest point from start of the line
        pos = closest.geometry.project(gpd.GeoSeries(closest.point))
        # Get new point location geometry
        new_pts = closest.geometry.interpolate(pos)
        #Identify the columns we want to copy from the closest line to the point, such as a line ID.
        line_columns = ['line_i', 'osm_id', 'code', 'fclass']
        # Create a new GeoDataFrame from the columns from the closest line and new point geometries (which will be called "geometries")
        snapped = gpd.GeoDataFrame(
            closest[line_columns],geometry=new_pts)
        # Join back to the original points:
        updated_points = points.drop(columns=["geometry"]).join(snapped)
        # You may want to drop any that didn't snap, if so:
        updated_points = updated_points.dropna(subset=["geometry"])
