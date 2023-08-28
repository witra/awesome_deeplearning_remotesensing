"""
Datapipe for rioxarray
"""

from typing import List
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("filter_rioxarray")
class FilterRioXarrayIterDataPipe(IterDataPipe):
    """
    Filter/select the dimension of the rx.DataArray
    """
    def __init__(self, source_dp:IterDataPipe, label:str, selected_list: List) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.label = label
        self.selected_list = selected_list

    def __iter__(self):
        for img in self.source_dp:
            yield img.sel({self.label:self.selected_list})

    def __len__(self)-> int:
        return len(self.source_dp)


