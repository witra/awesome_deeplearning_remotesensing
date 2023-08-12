# -*- coding: utf-8 -*-
import os.path

import hydra
import gzip
import geopandas as gpd
import pandas as pd
import torchdata
import torch
import xarray as xr
import zen3geo


from typing import Tuple, Union, List
from src.data.acquire_data import AcquireData
from src.data.rapidai4eo import get_asset_hrefs
from src import utils
from shapely.geometry import Point, Polygon

class RapidAI4EO(AcquireData):
    def __init__(self, geometry: Union[Point, Polygon], date_range: Tuple[pd.Timestamp, pd.Timestamp]):
        """
        This class aims to acquire the dataset from rapdAI4EO repository
        Parameters
        ----------
        geometry (Point or Polygon) :
        time_range (pd.Timestamp) : tuple of start-end date. Have to follow the structure: YYYY-MM-DDT00:00:00Z
        """
        super().__init__(geometry, date_range)

        # pull configuration
        hydra.initialize(version_base="1.1", config_path="../../config", job_name="RapidAI4EO")
        cfg = hydra.compose(config_name="conf_dataSrc.yaml")

        # cfg.clear()

        self.geometries_file_url = cfg.RapidAI4EO.geometries_file_url
        self.labels_file_url = cfg.RapidAI4EO.labels_file_url
        self.labels_mapping_file_url = cfg.RapidAI4EO.labels_mapping_file_url

        self.geometries_filename = cfg.RapidAI4EO.geometries_filename
        self.labels_filename = cfg.RapidAI4EO.labels_filename
        self.labels_mapping_filename = cfg.RapidAI4EO.labels_mapping_filename
        hydra.core.global_hydra.GlobalHydra.instance().clear()


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
            return utils.download_file(self.geometries_file_url, path)
        else:
            print(f'{self.geometries_file_url} is already downloaded to {path}')
        return self.geometries_filename

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
            return utils.download_file(self.labels_file_url, path)
        else:
            print(f'{self.labels_file_url} is already downloaded to {path}')
        return self.labels_filename
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
            return utils.download_file(self.labels_mapping_file_url, path)
        else:
            print(f'{self.labels_mapping_file_url} is already downloaded to {path}')
        return self.labels_mapping_filename

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

    def filter_hrefs_on_geom(self, geometries, products=None):
        """
        This function filters the hrefs eiter based on location
        ----------
        geometries (GeoDataFrame): Gpd of downloaded geometries from Planet
        products (list) : by default ['pfsr', 'pfqa', 's2']
                        pfsr = planet data
                        pfag = mask
                        s2 = sentinel
        filter_type (str) : filtered  by location or label,

        Returns; List of hrefs
        -------

        """

        if products is None:
            products = ['pfsr', 'pfqa', 's2']

        spatially_filtered_ids = geometries[geometries.geometry.intersects(self.geometry)].index
        hrefs = get_asset_hrefs(spatially_filtered_ids,
                                products=products,
                                temporal_filter=self.date_range)
        print(f'obtained {len(hrefs)} images for {products}')
        return hrefs

    def datapipe_img_only(self,
                          img_hrefs: List,
                          input_dims=None,
                          input_overlap=None,
                          batch_size = 16) -> torch.utils.data.datapipes.iter.callable.CollatorIterDataPipe:
        """
        Build a basic datapipe for image only.
        Parameters
        ----------
        img_hrefs (list) : list of hrefs
        input_dims: x and y sizes
        input_overlap : default is 0 for x and y dims (there is no overlap, stride = input dims)

        Returns: datapipe
        -------

        """
        if input_overlap is None:
            input_overlap = {'y': 0, 'x': 0}
        if input_dims is None:
            input_dims = {'y': 128, 'x': 128}

        def imageset_to_tensor(chip_samples: xr.DataArray) -> (list[torch.Tensor]):
            """
            Coverts the xr.DataArray of satellite image to tensor
            Parameters
            ----------
            samples :

            Returns
            -------

            """
            img_tensor = [torch.as_tensor(chip_sample.data) for chip_sample in chip_samples]
            img_tensor = torch.stack(tensors=img_tensor)
            return img_tensor
        dp = torchdata.datapipes.iter.IterableWrapper(iterable=img_hrefs)
        dp = dp.read_from_rioxarray()
        dp = dp.slice_with_xbatcher(input_dims=input_dims, input_overlap=input_overlap)
        dp = dp.batch(batch_size=batch_size)
        dp = dp.collate(collate_fn=imageset_to_tensor)
        return dp

    def show_graph(self, dp):
        torchdata.datapipes.utils.to_graph(dp=dp)



