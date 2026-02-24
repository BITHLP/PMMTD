import json
import os
import shutil

from tqdm import tqdm

images = set()

for sp in ("train", "valid", "test"):
    with open(f"./data/{sp}.jsonl") as f:
        examples = [json.loads(line) for line in f]
        for example in examples:
            images.add(example["img"])

for img in tqdm(images):
    if img is not None:
        shutil.copy("imgs_all/" + img, "imgs/" + img)

print(len(images))
