from functools import partial

import pandas as pd
import shapely
import torch
import torch.nn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data.get_RapidAI4EO import RapidAI4EO
from src.models.mae.models_mae import MaskedAutoencoderViT


def batch_into_image():
    pass


def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # 1. load  model
    model = MaskedAutoencoderViT(img_size=64, in_chans=4,
                                 patch_size=8, embed_dim=60, depth=2, num_heads=12,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(device)
    model.load_state_dict(torch.load('../../../models/my_mae/model_test_mae.pt'))
    model.eval()

    # 0. define dataset
    aoi_geom = shapely.geometry.box(11.30, 48.00, 11.62, 48.30)  # Munich
    start_date = pd.to_datetime("2018-01-01T00:00:00Z")
    end_date = pd.to_datetime("2019-01-01T00:00:00Z")
    time_stamp = (start_date, end_date)

    # 2. init and gather data from rapidai4eo project
    rapidaieo_data = RapidAI4EO(aoi_geom, date_range=time_stamp)
    rapidaieo_data.get_geometries(path="./rapidai4eo_geometries.geojson.gz")
    geometries = rapidaieo_data.load_geometries()
    hrefs_planet = rapidaieo_data.filter_hrefs_on_geom(geometries=geometries, products=['pfsr'])
    planet_datapipe = rapidaieo_data.datapipe_img_only(hrefs_planet[:1],
                                                       input_dims={'x': 64, 'y': 64},
                                                       input_overlap={'x': 32, 'y': 32},
                                                       batch_size=64,
                                                       )

    # 3. Define data loader
    data_loader_test = torch.utils.data.DataLoader(dataset=planet_datapipe,
                                                   batch_size=None,
                                                   shuffle=True)

    # 4. Inference
    preds_targets = []
    with torch.inference_mode():
        for step, batch in enumerate(data_loader_test):
            batch = batch.to(device)
            loss, pred, _ = model(batch, mask_ratio=0.75)
            preds_targets.append([pred, batch])
            print(f'length of pred {len(pred)}, \n length batch {len(batch)}')


if __name__ == '__main__':
    """
    This script is to run the self-supervised experiment of MAE on the Planet dataset. 
    """
    main()
