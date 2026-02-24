import os
import random
import time

import requests as rq
from tqdm import main, tqdm


def download_file(url, save_path, overwrite=False):
    if overwrite or not os.path.exists(save_path):
        resp = rq.get(url)
        if len(resp.content) > 10:
            if resp.content[0] == b"<":
                return False
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
    return False


def download_all(links, save_dir):
    random.shuffle(links)
    for url in tqdm(links, desc="images"):
        try:
            filename = url.replace(":", "_").replace("/", "_")
            if download_file(url, f"{save_dir}/{filename}"):
                time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(10)
