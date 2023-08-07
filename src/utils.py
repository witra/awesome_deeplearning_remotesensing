import requests
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