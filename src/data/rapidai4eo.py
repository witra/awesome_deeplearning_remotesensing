"""Helper classes and functions for accessing data from the RapidAI4EO corpus.

Source: https://beta.source.coop/planet/rapidai4eo

"""

from typing import List, Optional, Tuple, Union

import pandas as pd

__all__ = ["get_asset_hrefs", "PRODUCTS", "PF_TIMESTEPS", "S2_TIMESTEPS"]

PF_TIMESTEPS = pd.date_range("2018-01-03T10:30:00Z", "2019-12-29T10:30:00Z", freq="5D")
S2_TIMESTEPS = pd.date_range("2018-01-01T10:30:00Z", "2018-12-31T10:30:00Z", freq="1M")
PRODUCTS = ["pfsr", "pfqa", "s2"]


class ProductAssetFilter:
    """ProductAssetFilter.

    Attributes:
        product_code (str): The product code string as it appears in asset URLs.
        available_dates (pd.date_range): The dates available for the product type.
        date_format (str): The formatting string for converting dates to basenames.
    """

    _href_root: str = (
        "https://radiantearth.blob.core.windows.net/mlhub/rapidai4eo/imagery"
    )

    def __init__(
        self, product_code: str, available_dates: pd.date_range, date_format: str
    ):
        """Initialize the ProductAssetFilter class.

        Args:
            product_code (str): The product code string as it appears in asset URLs.
            available_dates (pd.date_range): The dates available for the product type.
            date_format (str): The formatting string for converting dates to basenames.
        """
        self.product_code = product_code
        self.available_dates = available_dates
        self.date_format = date_format

    def _filter_dates(
        self, temporal_filter: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    ) -> pd.date_range:
        """Filter dates by an optional temporal query."""
        dates = self.available_dates
        if temporal_filter is not None:
            dates = dates[(dates > temporal_filter[0]) & (dates < temporal_filter[1])]
        return dates

    def _dates_to_basenames(self, dates: pd.date_range) -> List[str]:
        """Convert dates to file basenames."""
        return [f"{date.strftime(self.date_format)}.tif" for date in dates]

    def _build_hrefs(self, sample_ids: pd.Series, basenames: List[str]) -> List[str]:
        """Build the hrefs to the assets given sample IDs and a file basenames."""
        hrefs = []
        for sample_id in sample_ids.values:
            utmz, tile_offset, _ = sample_id.split("_", 2)
            path_substring = f"{self._href_root}/{utmz}/{tile_offset}/{sample_id}/{self.product_code}"
            for basename in basenames:
                hrefs.append(f"{path_substring}/{basename}")

        return hrefs

    def get_hrefs(
        self,
        sample_ids: pd.Series,
        temporal_filter: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> List[str]:
        """Get a list of URLs for corpus data and metadata assets.

        Args:
            sample_ids (pd.Series): sample_ids
            temporal_filter (Optional[Tuple[pd.Timestamp, pd.Timestamp]]): temporal_filter

        Returns:
            List[str]:
        """
        dates = self._filter_dates(temporal_filter)
        basenames = self._dates_to_basenames(dates)
        hrefs = self._build_hrefs(sample_ids, basenames)
        return hrefs


PRODUCT_FILTERS = {
    "pfsr": ProductAssetFilter("PF-SR", PF_TIMESTEPS, "%Y-%m-%d"),
    "pfqa": ProductAssetFilter("PF-QA", PF_TIMESTEPS, "%Y-%m-%d"),
    "s2": ProductAssetFilter("S2-SR", S2_TIMESTEPS, "%Y-%m"),
}


def get_asset_hrefs(
    sample_ids: pd.Series,
    products: Optional[Union[str, List[str]]] = None,
    temporal_filter: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> List[str]:
    """Get a list of URLs for corpus data and metadata assets.

    Get the URLs for assets based on sampling location, product type, and a temporal query.
    This function replicates a RapidAI4EO STAC search without crawling the Catalog.

    Args:
        sample_ids (pd.Series): The IDs of the samples we are querying.
        products (Optional[Union[str, List[str]]]): The product type(s) we are querying.
            If none is specified, the URLs for all product types will be returned.
        temporal_filter (Optional[Tuple[pd.Timestamp, pd.Timestamp]]):
            An optional two-tuple containing the start and end date for a temporal filter.

    Returns:
        List[str]: The URLs to the filtered data and metadata files.
    """
    if products is None:
        products = PRODUCTS
    elif type(products) is str:
        products = [products]

    hrefs = []
    for product in products:
        product_filter = PRODUCT_FILTERS[product]
        hrefs += product_filter.get_hrefs(sample_ids, temporal_filter)

    return hrefs