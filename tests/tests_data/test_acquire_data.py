import pytest
import pandas as pd
import shapely.geometry

from src.data.acquire_data import AcquireData
from shapely.geometry import Point, Polygon



@pytest.fixture(scope='class', name='acquire_data1')
def fixture_acquire_data1():
    point = Point(4.89, 52.37)
    start_date = pd.to_datetime("2018-01-01T00:00:00Z")
    end_date = pd.to_datetime("2019-01-01T00:00:00Z")
    date_range = (start_date, end_date)
    data = AcquireData(point, date_range)
    return data


@pytest.fixture(scope='class', name='acquire_data2')
def fixture_acquire_data2():
    box = Polygon(shapely.geometry.box(13.05, 52.35, 13.72, 52.69))
    start_date = pd.to_datetime("2018-01-01T00:00:00Z")
    end_date = pd.to_datetime("2019-01-01T00:00:00Z")
    date_range = (start_date, end_date)
    data = AcquireData(box, date_range)
    return data


def test_acquire_data(acquire_data1, acquire_data2):
    geom1 = acquire_data1.geometry
    geom2 = acquire_data2.geometry

    geom1_true = Point(4.89, 52.37)
    geom2_true = Polygon(shapely.geometry.box(13.05, 52.35, 13.72, 52.69))

    date_range1 = acquire_data1.date_range
    date_range2 = acquire_data2.date_range

    start_date = pd.to_datetime("2018-01-01T00:00:00Z")
    end_date = pd.to_datetime("2019-01-01T00:00:00Z")
    date_range_true = (start_date, end_date)

    assert geom1 == geom1_true
    assert geom2 == geom2_true
    assert date_range1 == date_range_true
    assert date_range2 == date_range_true


def test_overwrite_loc(acquire_data1):
    coords = ((13., 52.), (13., 53.), (14., 53.), (14., 52.), (13., 52.))
    geom = Polygon(coords)
    acquire_data1.overwrite_loc(geom)
    assert acquire_data1.geometry == geom


def test_overwrite_date_range(acquire_data1):
    start_date = pd.to_datetime("2017-01-01T00:00:00Z")
    end_date = pd.to_datetime("2020-01-01T00:00:00Z")
    data_range = (start_date, end_date)
    acquire_data1.overwrite_date_range(data_range)
    assert acquire_data1.date_range == data_range
