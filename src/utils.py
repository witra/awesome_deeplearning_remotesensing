import requests
import yaml
from tqdm import tqdm


def download_file(url, output_path):
    r = requests.get(url, stream=True)
    block_size = 1024  # 1KB
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    progress = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)
    if r.status_code == 200:
        with open(output_path, "wb") as f:
            for data in r.iter_content(block_size):
                f.write(data)
                progress.update(len(data))
            print(f"Downloaded {url} to {output_path}")
    else:
        print(f"failed to download {url}. Status code: {r.status_code}")


def parse_config(config_file):
    """
    Yaml only config parsing
    Parameters
    ----------
    config_file :

    Returns
    -------

    """
    try:
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        return {}


def merge_config_with_parse(config, args):
    config.update({k: v for k, v in vars(args).items() if v is not None})
    return config