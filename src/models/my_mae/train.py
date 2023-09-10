import time
from datetime import datetime
from functools import partial

import pandas as pd
import shapely
import torch
import torch.nn
import torch.nn as nn
import timm.optim.optim_factory as optim_factory
import wandb
from torch.utils.tensorboard import SummaryWriter

from keys import wandb_props
from src.data.get_RapidAI4EO import RapidAI4EO
from src.models.mae.models_mae import MaskedAutoencoderViT

def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    best_loss = 1_000_000

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
    planet_datapipe = rapidaieo_data.datapipe_img_only(hrefs_planet[:100],
                                                       input_dims={'x': 64, 'y': 64},
                                                       input_overlap={'x': 32, 'y': 32},
                                                       batch_size=64
                                                       )

    # 2. Define data loader
    data_loader_train = torch.utils.data.DataLoader(dataset=planet_datapipe,
                                                    batch_size=None,
                                                    shuffle=True)

    # 3. define  model
    model = MaskedAutoencoderViT(img_size=64, in_chans=4,
                                 patch_size=8, embed_dim=60, depth=2, num_heads=12,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(device)

    # 4. init loss, and optimizer
    lr = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 5. Init WnB
    wandb.login(key=wandb_props.acc)

    run = wandb.init(
        project="train my mae",
        config={'lr': lr}
    )
    # 5. Create training loop
    for epoch in range(50):
        print(f'\n\nepoch: {epoch}')
        batch_loss = torch.tensor(0.0).to(device)
        model.train(True)

        lr_before = optimizer.param_groups[0]["lr"]
        run.log({'lr': lr_before})

        for data_iter_step, batch in enumerate(data_loader_train):
            # move data to the specified device
            batch.to(device)

            # zero gradient per batch
            optimizer.zero_grad()

            # run the model and get the
            loss, _, _ = model(batch, mask_ratio=0.75)

            # backprop loss
            loss.backward()

            # step the optimizer
            optimizer.step()
            lr_after = optimizer.param_groups[0]["lr"]

            batch_loss += loss
            run.log({'loss': loss})

            print(f'current running loss is {loss}, step {data_iter_step}')
            print(f'lr after: {lr_after}, lr_before{lr_before}')

        scheduler.step()

        # Track the best performance, and save the model's state
        if batch_loss.item()/data_iter_step < best_loss:
            print('save the model')
            best_loss = batch_loss.item()
            model_path = '../../../models/my_mae/model_test_mae.pt'
            torch.save(model.state_dict(), model_path)

        # log data to
        run.log({'batch loss': batch_loss})


if __name__ == '__main__':
    """
    This script is to run the self-supervised experiment of MAE on the Planet dataset. 
    """
    main()
