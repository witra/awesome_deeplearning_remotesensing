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
from src.data.custom_pipeline import rioxarray_dp
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

    @staticmethod
    def datapipe_img_only(img_hrefs: List,
                          input_dims=None,
                          input_overlap=None,
                          batch_size=16,
                          select_bands=None) -> torch.utils.data.datapipes.iter.callable.CollatorIterDataPipe:
        """
        Build a basic datapipe for image only.
        Parameters
        ----------
        img_hrefs (list) : list of hrefs
        input_dims: x and y sizes of the images
        input_overlap : default is 0 for x and y dims (there is no overlap, stride = input dims)
        batch_size (int): batch size
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
            img_tensor = [torch.as_tensor(chip_sample.data, dtype=torch.float32) for chip_sample in chip_samples]
            img_tensor = torch.stack(tensors=img_tensor)
            return img_tensor

        dp = torchdata.datapipes.iter.IterableWrapper(iterable=img_hrefs)
        dp = dp.read_from_rioxarray()
        if select_bands:
            dp = dp.filter_rioxarray(label="band", selected_list=select_bands)
        dp = dp.slice_with_xbatcher(input_dims=input_dims, input_overlap=input_overlap)
        dp = dp.batch(batch_size=batch_size)
        dp = dp.collate(collate_fn=imageset_to_tensor)
        return dp

    @staticmethod
    def datapipe_img_with_label(img_hrefs: List,
                                label_hrefs: List,
                                input_dims: dict = None,
                                input_overlap: dict = None,
                                batch_size=10) -> torch.utils.data.datapipes.iter.callable.CollatorIterDataPipe:
        """
        Build a basic datapipe pairing the images and its label.
        Parameters
        ----------
        img_hrefs :
        label_hrefs :
        input_dims :
        input_overlap :

        Returns
        -------

        """
        if input_overlap is None:
            input_overlap = {'y': 50, 'x': 50}
        if input_dims is None:
            input_dims = {'y': 100, 'x': 100}

        def xr_da_to_ds(img_and_label: tuple) -> xr.Dataset:
            """
            Pack the images and labels into xr.Dataset
            Parameters
            ----------
            img_and_label :

            Returns
            -------

            """

            img, mask = img_and_label
            dataset: xr.Dataset = xr.merge([img.rename('img'), mask.isel(band=0).rename('mask')],
                                           join='override')  # arbitrary select band 0 mask
            return dataset

        def dataset_to_tensor(chip_samples: xr.Dataset) -> (list[torch.Tensor], list[torch.Tensor]):
            """
            Coverts the xr.Dataset of satellite image and its label to tensor
            Parameters
            ----------
            samples :

            Returns
            -------

            """
            img_tensor: list[torch.Tensor] = [torch.as_tensor(chip.img.data) for chip in chip_samples]
            label_tensor: list[torch.Tensor] = [torch.as_tensor(chip.mask.data) for chip in chip_samples]

            img_tensor = torch.stack(tensors=img_tensor)
            label_tensor = torch.stack(tensors=label_tensor)
            return img_tensor, label_tensor

        # Init the pipelines
        dp_planet = torchdata.datapipes.iter.IterableWrapper(iterable=img_hrefs)
        dp_label = torchdata.datapipes.iter.IterableWrapper(iterable=label_hrefs)

        # read the files
        dp_planet = dp_planet.read_from_rioxarray()
        dp_label = dp_label.read_from_rioxarray()

        # zip two the data and the labels
        dp = dp_planet.zip(dp_label)

        # convert the zip into xr.Dataset using collate
        dp = dp.map(fn=xr_da_to_ds)

        # slice the image and mask
        dp = dp.slice_with_xbatcher(input_dims=input_dims, input_overlap=input_overlap)

        # take per batch
        dp = dp.batch(batch_size=batch_size)

        dp = dp.collate(collate_fn=dataset_to_tensor)
        return dp

    @staticmethod
    def show_graph(dp):
        return torchdata.datapipes.utils.to_graph(dp=dp)
