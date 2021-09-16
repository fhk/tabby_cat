"""
Some tests to see why things dont always get connected
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

from tabby_cat.processor import Processor


def test_snapping():
    points = gpd.GeoDataFrame(
        {'geometry': [Point(0.00000, 0.00000)]}, geometry='geometry', crs='epsg:4326')
    lines = gpd.GeoDataFrame(
        {'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [LineString([[0.00001, 0.00001], [0.00002, 0.00002]])]}, geometry='geometry', crs='epsg:4326')

    p = Processor('test')
    p.snap_points_to_line(lines, points, write=False)

    assert len(p.cut_lines) == 2
