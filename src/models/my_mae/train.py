import torch
import torch.nn
import wandb


import shapely
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from functools import partial
from src.data.get_RapidAI4EO import RapidAI4EO
from src.models.mae.models_mae import MaskedAutoencoderViT
from src.models.mae.util.misc import NativeScalerWithGradNormCount as NativeScaler
from keys import wandb_props

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # 0. define dataset
    aoi_geom = shapely.geometry.box(13.05, 52.35, 13.72, 52.69)
    start_date = pd.to_datetime("2018-01-01T00:00:00Z")
    end_date = pd.to_datetime("2019-01-01T00:00:00Z")
    time_stamp = (start_date, end_date)

    # 1. init and gather data from rapidai4eo project
    rapidaieo_data = RapidAI4EO(aoi_geom, date_range=time_stamp)
    rapidaieo_data.get_geometries(path="./rapidai4eo_geometries.geojson.gz")
    geometries = rapidaieo_data.load_geometries()
    hrefs_planet = rapidaieo_data.filter_hrefs_on_geom(geometries=geometries, products=['pfsr'])
    planet_datapipe = rapidaieo_data.datapipe_img_only(hrefs_planet[:50],
                                                       input_dims={'x': 64, 'y': 64},
                                                       input_overlap={'x': 32, 'y': 32},
                                                       batch_size=32
                                                       )

    # 2. Define data loader
    data_loader_train = torch.utils.data.DataLoader(dataset=planet_datapipe,
                                                    batch_size=None,
                                                    shuffle=True)

    # 3. define  model
    model = MaskedAutoencoderViT(img_size=64, in_chans=4,
                                 patch_size=8, embed_dim=60, depth=4, num_heads=12,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(device)

    # 4. init loss, and optimizer
    lr=0.001
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_scaler = NativeScaler()

    # 5. Init WnB
    wandb.login(key=wandb_props.acc)

    run = wandb.init(
        project="first test wandb.py",
        config={'lr': lr}
    )
    # 5. Create training loop
    for epoch in range(10):
        print(f'epoch: {epoch}')
        for data_iter_step, batch in enumerate(data_loader_train):
            print(f'iter step: {data_iter_step}')
            # move data to device
            batch = batch.to(device)

            # some args
            accum_iter = 1

            # zero gradient per batch
            optimizer.zero_grad()

            # run the model and get the
            loss, _, _ = model(batch, mask_ratio=0.75)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"]

            wandb.log({'loss': loss, 'lr':lr})
            # print(f"loss is {loss}")
            # print(f"lr is {lr} \n")

if __name__ == '__main__':
    """
    This script is to run the self-supervised experiment of MAE on the Planet dataset. 
    """
    main()