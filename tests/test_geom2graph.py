"""
Some tests to see why things dont always get connected
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString

from tabby_cat.processor import Processor


def test_snapping():
    points = gpd.GeoDataFrame(
        {'geometry': [Point(0.00000, 0.00000)]}, geometry='geometry', crs='epsg:4326')
    lines = gpd.GeoDataFrame(
        {'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [LineString([[0.00001, 0.00001], [0.00002, 0.00002]])]}, geometry='geometry', crs='epsg:4326')

    p = Processor('test')
    p.snap_points_to_line(lines, points, write=False)

    assert len(p.cut_lines) == 1


def test_snapping_more_treads():
    points = gpd.GeoDataFrame(
        {'geometry': [
            Point(0.00000, 0.00000),
            Point(0.00001, 0.00002),
            Point(0.00002, 0.00003),
            Point(0.00003, 0.00004),
            Point(0.00005, 0.00006)
            ]}, geometry='geometry', crs='epsg:4326')
    lines = gpd.GeoDataFrame(
        {'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [LineString([[0.00001, 0.00001], [0.00002, 0.00002]])]}, geometry='geometry', crs='epsg:4326')

    p = Processor('test')
    p.snap_points_to_line(lines, points, write=False)

    assert len(p.cut_lines) == 6



def test_snap_part():
    points = gpd.GeoDataFrame(
        {'geometry': [Point(0, 0)]}, geometry='geometry', crs='epsg:3857')
    lines = gpd.GeoDataFrame(
        {'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [LineString([[1, 1], [2, 2]])]}, geometry='geometry', crs='epsg:3857')

    p = Processor('test')
    gdf = p._snap_part(points, lines)

    assert gdf['line_i'].tolist() == [0]


def test_points_to_multipoint():
    points = gpd.GeoDataFrame(
        {'geometry': [Point(0, 0)]}, geometry='geometry', crs='epsg:3857')
    lines = gpd.GeoDataFrame(
        {'line_i': 1, 'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [LineString([[1, 1], [2, 2]])]}, geometry='geometry', crs='epsg:3857')

    p = Processor('test')

    pos = lines.geometry.project(points.iloc[0].geometry)
    # Get new point location geometry
    new_pts = lines.geometry.interpolate(pos)
    line_columns = ['line_i', 'osm_id', 'code', 'fclass', 'geometry']
    snapped = gpd.GeoDataFrame(
        lines[line_columns], geometry='geometry', crs="epsg:3857")
    snapped['snapped'] = snapped.geometry

    result = p.points_to_multipoint(snapped)

    assert result.coords[:] == [(1., 1.), (2., 2.)]


def test_points_to_multipoint_two_lines():
    points = gpd.GeoDataFrame(
        {'geometry': [Point(1.5, 1.5)]}, geometry='geometry', crs='epsg:3857')
    lines = gpd.GeoDataFrame(
        {'line_i': 1, 'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [LineString([[1, 1], [2, 2]])]}, geometry='geometry', crs='epsg:3857')

    p = Processor('test')

    pos = lines.geometry.project(points.iloc[0].geometry)
    # Get new point location geometry
    new_pts = lines.geometry.interpolate(pos)
    #import pdb; pdb.set_trace()
    line_columns = ['line_i', 'osm_id', 'code', 'fclass', 'geometry']
    snapped = gpd.GeoDataFrame(
        lines[line_columns], geometry=new_pts, crs="epsg:3857")
    snapped['snapped'] = snapped.geometry

    result = p.points_to_multipoint(snapped)

    assert result.coords[:] == [(1.5, 1.5)]


def test_points_to_multipoint_two_lines_long():
    points = gpd.GeoDataFrame(
        {'geometry': [Point(1.9, 2.1)]}, geometry='geometry', crs='epsg:3857')
    lines = gpd.GeoDataFrame(
        {'line_i': 1, 'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [LineString([[1, 1], [3, 3]])]}, geometry='geometry', crs='epsg:3857')

    p = Processor('test')

    pos = lines.geometry.project(points.iloc[0].geometry)
    # Get new point location geometry
    new_pts = lines.geometry.interpolate(pos)
    #import pdb; pdb.set_trace()
    line_columns = ['line_i', 'osm_id', 'code', 'fclass', 'geometry']
    snapped = gpd.GeoDataFrame(
        lines[line_columns], geometry=new_pts, crs="epsg:3857")
    snapped['snapped'] = snapped.geometry
    result = p.points_to_multipoint(snapped)

    assert result.coords[:] == [(2., 2.)]


def test_points_to_multipoint_two_lines_example():
    points = gpd.GeoDataFrame(
        {'geometry': [Point(-119.26930592423289568, 45.85674703498899163)]}, geometry='geometry', crs='epsg:4326')
    lines = gpd.GeoDataFrame(
        {'line_i': 1, 'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [MultiLineString(
            [LineString([(-119.26738070000004655, 45.85672950000000725),
                (-119.26934599999999875, 45.85674739999998906)])])]}, geometry='geometry', crs='epsg:4326')
    lines = lines.to_crs('epsg:3857')
    points = points.to_crs('epsg:3857')
    p = Processor('test')

    pos = lines.geometry.project(points.iloc[0].geometry)
    # Get new point location geometry
    new_pts = lines.geometry.interpolate(pos)
    #import pdb; pdb.set_trace()
    line_columns = ['line_i', 'osm_id', 'code', 'fclass', 'geometry']
    snapped = gpd.GeoDataFrame(
        lines[line_columns], geometry=new_pts, crs="epsg:3857")
    snapped['snapped'] = snapped.geometry
    result = p.points_to_multipoint(snapped)

    assert result.coords[:] == [(-13276998.402752772, 5757422.475012177)]


def test_snapping_real_one():
    points = gpd.GeoDataFrame(
        {'geometry': [Point(-119.26930592423289568, 45.85674703498899163)]}, geometry='geometry', crs='epsg:4326')
    lines = gpd.GeoDataFrame(
        {'id': 1, 'osm_id': 1, 'code': '', 'fclass': '',
         'geometry': [
            LineString([(-119.26738070000004655, 45.85672950000000725),
                (-119.26934599999999875, 45.85674739999998906)])]}, geometry='geometry', crs='epsg:4326')

    p = Processor('test')
    p.snap_points_to_line(lines, points, write=False)

    assert len(p.cut_lines) == 2