import pandas as pd


from typing import Tuple, Union
from shapely.geometry import Point, Polygon

class AcquireData:
    """
    This class is intended to acquire satellite data from various sources.

    the final return is list of links of href
    """

    def __init__(self, geometry: Union[Point, Polygon], date_range: Tuple[pd.Timestamp, pd.Timestamp]):
        """

        Parameters
        ----------
        geometry : default location to acquire the data
        date_range : default time to acquire the data

        Returns
        -------

        """
        self.geometry = geometry
        self.date_range = date_range

    def overwrite_loc(self, geometry: Union[Point, Polygon]) -> Union[Point, Polygon]:
        """
        This method overwrites geometry that will use either default or new one
        Parameters
        ----------
        geometry :

        Returns
        -------

        """
        if geometry is None:
            geometry = self.geometry
        else:
            self.geometry = geometry
        return geometry

    def overwrite_date_range(self, date_range: Tuple[pd.Timestamp, pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        This method overwrites date_range that will use either default or new one
        Parameters
        ----------
        date_range :

        Returns
        -------

        """
        if date_range is None:
            date_range = self.date_range
        else:
            self.date_range = date_range
        return date_range
