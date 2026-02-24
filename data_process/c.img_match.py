import json
import os
import re
from pprint import pp

from PIL import Image


def link2filename(img_link):
    def rep(x):
        return x.replace("/", "_").replace(":", "_")

    if re.match(r"\d+", img_link):
        return f"coco/COCO_train2014_{img_link.zfill(12)}.jpg"
    elif re.match(r"https://img\d+.360buyimg.com/", img_link):
        image_id = img_link.split("/", maxsplit=6)[-1]
        return "jd/" + rep(image_id)
    elif re.match(r"https?://static.xiangpi.com/", img_link):
        return "xiangpi/" + rep(img_link)
    elif re.match(r"https://www.fxhaoke.com/", img_link):
        return "course2/" + rep(img_link)
    elif re.match(r"https://icon.jiandan100.cn", img_link):
        return "jd100/" + rep(img_link)
    elif re.match(r"https://bkimg.cdn.bcebos.com/", img_link):
        return "bkimg/" + rep(img_link)
    elif re.match(r"https://t1.chei.com.cn/", img_link):
        return "occupation/" + rep(img_link)
    elif re.match(r"https?://i\d+.meishichina.com/", img_link):
        return "meishi/" + rep(img_link)
    else:
        return ""


def is_image_good(path):
    try:
        img = Image.open(path)
        img.close()
        return True
    except:
        return False


def main():
    alll, good = 0, 0
    with open("./data/b.0.jsonl") as f:
        examples = [json.loads(line) for line in f]
    for kg, dialog in examples:
        for goal, utts in dialog:
            for utt in utts:
                links = [link2filename(l) for l in utt[2]]
                alll += len(links)
                # DONE: check existance and format
                # TODO: remove invalid example
                links = [l for l in links if l != "" and is_image_good("imgs/" + l)]
                good += len(links)
                utt[2] = links
    print(good, alll, good / alll)
    print(len(examples), good / len(examples))
    with open("./data/b.1.jsonl", "w") as f:
        for r in examples:
            print(json.dumps(r, ensure_ascii=False), file=f)


if __name__ == "__main__":
    main()
