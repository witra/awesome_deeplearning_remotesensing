import argparse
import os
from functools import partial

import pandas as pd
import shapely
import torch
import torch.nn
import torch.nn as nn
import wandb
from torch.utils.tensorboard import SummaryWriter

from keys import wandb_props
from src import utils
from src.data.get_RapidAI4EO import RapidAI4EO
from src.models.mae.models_mae import MaskedAutoencoderViT


def get_args_parser():
    parser = argparse.ArgumentParser('Basic MAE trainer')
    parser.add_argument('config_file', type=str, help="""path to the config file""")
    parser.add_argument('--verbose', default=True, type=bool, help='print some statements')
    return parser


def main(args):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    best_loss = args.init_best_loss

    # 0. define dataset
    aoi_geom = shapely.geometry.box(args.lon_w, args.lat_s, args.lon_e, args.lat_n)
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    time_stamp = (start_date, end_date)

    # 1. init and gather data from rapidai4eo project
    rapidaieo_data = RapidAI4EO(aoi_geom, date_range=time_stamp)
    rapidaieo_data.get_geometries(path=args.planet_geomatry_path)
    geometries = rapidaieo_data.load_geometries()
    hrefs_planet = rapidaieo_data.filter_hrefs_on_geom(geometries=geometries, products=['pfsr'])
    planet_datapipe = rapidaieo_data.datapipe_img_only(hrefs_planet[:100],
                                                       input_dims={'x': args.x_input_dims, 'y': args.y_input_dims},
                                                       input_overlap={'x': args.x_overlap, 'y': args.x_overlap},
                                                       batch_size=args.batch_size
                                                       )

    # 2. Define data loader
    data_loader_train = torch.utils.data.DataLoader(dataset=planet_datapipe,
                                                    batch_size=None,
                                                    shuffle=True)

    # 3. define  model
    model = MaskedAutoencoderViT(img_size=64, in_chans=4,
                                 patch_size=8, embed_dim=60, depth=args.attention_depth, num_heads=12,
                                 decoder_embed_dim=512, decoder_depth=args.decoder_depth, decoder_num_heads=16,
                                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(device)

    # 4. init loss, and optimizer
    lr = args.lr
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
            batch = batch.to(device)

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
        if batch_loss.item() / data_iter_step < best_loss:
            print('save the model')
            best_loss = batch_loss.item()
            model_path = os.path.join(args.output_dir, args.model_name)
            torch.save(model.state_dict(), model_path)

        # log data to
        run.log({'batch loss': batch_loss})


if __name__ == '__main__':
    """
    This script is to run the self-supervised experiment of MAE on the Planet dataset. 
    """
    args = get_args_parser()
    args = args.parse_args()

    config_path = args.config_file
    config_data = utils.parse_config(config_path)
    config_merge = utils.merge_config_with_parse(config_data, args)
    args = argparse.Namespace(**config_merge)

    main(args)
