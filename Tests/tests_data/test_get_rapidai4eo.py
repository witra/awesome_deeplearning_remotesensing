import os

import pandas as pd
import shapely.geometry
import pytest
import geopandas as gpd
import torch

from src.data.get_RapidAI4EO import RapidAI4EO
from shapely.geometry import Point, Polygon
from typing import List
@pytest.fixture(scope='class', name='RapidAI4EO_data1')
def fixture_RapidAI4EO_data1():
    point = Point(13.25, 52.50)
    start_date = pd.to_datetime("2018-01-01T00:00:00Z")
    end_date = pd.to_datetime("2019-01-01T00:00:00Z")
    date_range = (start_date, end_date)
    data1 = RapidAI4EO(point, date_range)
    return data1


@pytest.fixture(scope='class', name='RapidAI4EO_data2')
def fixture_RapidAI4EO_data2():
    box = Polygon(shapely.geometry.box(13.05, 52.35, 13.72, 52.69))
    start_date = pd.to_datetime("2018-01-01T00:00:00Z")
    end_date = pd.to_datetime("2019-01-01T00:00:00Z")
    date_range = (start_date, end_date)
    data2 = RapidAI4EO(box, date_range)
    return data2

@pytest.fixture(scope='class', name='geom_RapidAI4EO_data2')
def fixture_geom_RapidAI4EO_data2(RapidAI4EO_data2):
    directory = "../data_test/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    RapidAI4EO_data2.get_geometries(path=f"{directory}rapidai4eo_geometries.geojson.gz")
    return RapidAI4EO_data2.load_geometries()

def test_RapidAI4EO(RapidAI4EO_data1, RapidAI4EO_data2):
    geom1 = RapidAI4EO_data1.geometry
    geom2 = RapidAI4EO_data2.geometry

    geom1_true = Point(13.25, 52.50)
    geom2_true = Polygon(shapely.geometry.box(13.05, 52.35, 13.72, 52.69))

    date_range1 = RapidAI4EO_data1.date_range
    date_range2 = RapidAI4EO_data2.date_range

    start_date = pd.to_datetime("2018-01-01T00:00:00Z")
    end_date = pd.to_datetime("2019-01-01T00:00:00Z")
    date_range_true = (start_date, end_date)

    assert geom1 == geom1_true
    assert geom2 == geom2_true
    assert date_range1 == date_range_true
    assert date_range2 == date_range_true

def test_get_geometries(RapidAI4EO_data1, mocker):
    mocker.patch('src.utils.download_file', return_value=2)
    not_exist_path = './downloaded'
    exist_path = './'

    assert RapidAI4EO_data1.get_geometries(path=not_exist_path) == 2
    assert RapidAI4EO_data1.get_geometries(path=exist_path) == exist_path

def test_get_labels(RapidAI4EO_data1, mocker):
    mocker.patch('src.utils.download_file', return_value=2)
    not_exist_path = './downloaded'
    exist_path = './'

    assert RapidAI4EO_data1.get_labels(path=not_exist_path) == 2
    assert RapidAI4EO_data1.get_labels(path=exist_path) == exist_path


def test_get_labels(RapidAI4EO_data1, mocker):
    mocker.patch('src.utils.download_file', return_value=2)
    not_exist_path = './downloaded'
    exist_path = './'

    assert RapidAI4EO_data1.get_labels_mapping(path=not_exist_path) == 2
    assert RapidAI4EO_data1.get_labels_mapping(path=exist_path) == exist_path


def test_load_geometries(RapidAI4EO_data1):
    directory = "../data_test/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    RapidAI4EO_data1.get_geometries(path=f"{directory}rapidai4eo_geometries.geojson.gz")
    geometries = RapidAI4EO_data1.load_geometries()
    assert isinstance(geometries, gpd.geodataframe.GeoDataFrame)


def test_filter_hrefs_on_geom(RapidAI4EO_data2, geom_RapidAI4EO_data2):
    hrefs = RapidAI4EO_data2.filter_hrefs_on_geom(geometries=geom_RapidAI4EO_data2)
    assert isinstance(hrefs, List)

def test_datapipe_img_only(RapidAI4EO_data2):
    hrefs_planet = ['link1', 'link2']
    dp = RapidAI4EO_data2.datapipe_img_only(hrefs_planet[:1],
                                            input_dims={'x': 100, 'y': 100},
                                            input_overlap={'x': 50, 'y': 50},
                                            batch_size=10)

    assert isinstance(dp, torch.utils.data.datapipes.iter.callable.CollatorIterDataPipe)