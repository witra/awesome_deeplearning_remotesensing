# -*- coding: utf-8 -*-
import os.path

import hydra
import gzip
import geopandas as gpd
from src.data.rapidai4eo import get_asset_hrefs
from src import utils


class AcquireData:
    """
    This class is intended to acquire satellite data from various sources.

    the final return is list of links of href
    """

    def __init__(self, geometry, date_range):
        """

        Parameters
        ----------
        geometry : default location to acquire the data
        date_range : default time to acquire the data

        Returns
        -------

        """
        self.geometry = geometry
        self.time_range = date_range

    def overwrite_loc_time(self, geometry, date_range):
        """
        This method define geometry that will use either default or new one
        Parameters
        ----------
        geometry :
        date_range :

        Returns
        -------

        """
        if geometry is None:
            geometry = self.geometry
        if date_range is None:
            date_range = self.time_range
        return geometry, date_range


class RapidAI4EO(AcquireData):
    def __init__(self, geometry, date_range, **kwargs):
        """
        This class aims to acquire the dataset from rapdAI4EO repository
        Parameters
        ----------
        geometry :
        time_range (pd.Timestamp) : tuple of start-end date. Have to follow the structure: YYYY-MM-DDT00:00:00Z
        kwargs :
        """
        super().__init__(geometry, date_range)

        # pull configuration
        hydra.initialize(version_base="1.1", config_path="../../config", job_name="RapidAI4EO")
        cfg = hydra.compose(config_name="conf_dataSrc.yaml")

        self.geometries_file_url = cfg.RapidAI4EO.geometries_file_url
        self.labels_file_url = cfg.RapidAI4EO.labels_file_url
        self.labels_mapping_file_url = cfg.RapidAI4EO.labels_mapping_file_url

        self.geometries_filename = cfg.RapidAI4EO.geometries_filename
        self.labels_filename = cfg.RapidAI4EO.labels_filename
        self.labels_mapping_filename = cfg.RapidAI4EO.labels_mapping_filename

    def get_geometries(self, path=None):
        """

        Parameters
        ----------
        path : has to be in ".gz" extension

        Returns
        -------

        """
        if path is None:
            path = self.geometries_filename
        else:
            self.geometries_filename = path

        if not os.path.exists(path):
            utils.download_file(self.geometries_file_url, path)
        else:
            print(f'{self.geometries_file_url} is already downloaded to {path}')

    def get_labels(self, path=None):
        """

        Parameters
        ----------
        path : has to be in ".gz" extension

        Returns
        -------

        """
        if path is None:
            path = self.labels_filename
        else:
            self.labels_filename = path

        if not os.path.exists(path):
            utils.download_file(self.labels_file_url, path)
        else:
            print(f'{self.labels_file_url} is already downloaded to {path}')

    def get_labels_mapping(self, path=None):
        """

        Parameters
        ----------
        path : path of the downloaded file is stored. it has to be in ".csv" extension

        Returns
        -------

        """
        if path is None:
            path = self.labels_mapping_filename
        else:
            self.labels_mapping_filename = path

        if not os.path.exists(path):
            utils.download_file(self.labels_mapping_file_url, path)
        else:
            print(f'{self.labels_mapping_file_url} is already downloaded to {path}')

    def load_geometries(self):
        """
        Get the available geometry and indexes

        Returns (GeoDataFrame):  Gpd of downloaded geometries from Planet
        -------

        """
        with gzip.open(self.geometries_filename) as f:
            print("loading geometry indexes...")
            geometries = gpd.read_file(f).set_index("sample_id")
        return geometries

    def filter_hrefs(self, geometries, filter_type=None, products=None):
        """
        This function filters the hrefs eiter based on location or labels (currently still location)
        Parameters
        ----------
        geometries (GeoDataFrame): Gpd of downloaded geometries from Planet
        products (list) : by default ['pfsr', 'pfqa', 's2']
        filter_type (str) : filtered  by location or label,

        Returns; List of hrefs
        -------

        """

        if products is None:
            products = ['pfsr', 'pfqa', 's2']

        # geometries = self.load_geometries()
        spatially_filtered_ids = geometries[geometries.geometry.intersects(self.geometry)].index
        hrefs = get_asset_hrefs(spatially_filtered_ids,
                                products=products,
                                temporal_filter=self.time_range)
        print(f'obtained {len(hrefs)} images for {products}')
        return hrefs



