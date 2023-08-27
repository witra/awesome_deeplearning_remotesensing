"""
Tests for rioxarray datapipes
"""
import zen3geo
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from src.data.custom_pipeline.rioxarray_dp import FilterRioXarrayIterDataPipe


def test_filter_rioxarray():
    """

    Returns
    -------

    """
    url = "https://radiantearth.blob.core.windows.net/mlhub/" \
          "rapidai4eo/imagery/33N/16E-243N/33N_16E-243N_39_05/PF-SR/2018-01-03.tif"
    dp = IterableWrapper(iterable=[url])
    dp = dp.read_from_rioxarray()
    dp_class = FilterRioXarrayIterDataPipe(source_dp=dp, label="band", selected_list=[1, 2, 3])
    dp_func = dp.filter_rioxarray(label="band", selected_list=[1, 2, 3])

    it_class = iter(dp_class)
    it_func = iter(dp_func)

    img_class = next(it_class)
    img_func = next(it_func)

    assert len(dp_class) == 1
    assert len(dp_func) == 1
    assert len(img_class.band) == 3
    assert len(img_func.band) == 3

