import json
import random
from pprint import pp

# from structure to model input/output
# predict:
#   topic
#   entity
#   remain turns
#   image ??????


def gen(example, irrelevant_imgs):
    prompt = "### 知识:\n"
    for s, p, o in example["kg"]:
        if not o.startswith("http"):
            prompt += f"{s}:{p}:{o}\n"

    if example["prev_goal"] is None:
        example["prev_goal"] = ["无"]
    example["prev_goal"].append("无")
    example["goal"].append("无")

    prompt += "\n### 对话:\n\n"
    img = None
    for r, utt, links in example["history"]:
        utt = utt.replace("\n", " ")
        if r == "user":
            prompt += f"问: {utt}\n"
        else:
            prompt += f"答: {utt}\n"
        if links:
            img = links[0]
        else:
            img = None

    response_img = None
    if example["links"]:
        response_img = example["links"][0]
        img = response_img

    prompt += "### 目标:\n"
    prompt += f"\n上一目标: {example['prev_goal'][0]}\n"
    prompt += f"上一实体: {example['prev_goal'][1]}\n"

    lbl = f"当前目标: {example['goal'][0]}\n"
    lbl += f"当前实体: {example['goal'][1]}\n"
    lbl += f"剩余回合: {example['remain']}\n"
    lbl += f"回复图片: {'是' if response_img else '否'}\n"

    # DONE?: 图片改成选择题
    if response_img:
        yield prompt, response_img, lbl + f"图片合适: 是\n"
        for ir_img in irrelevant_imgs:
            if ir_img != response_img:
                yield prompt, ir_img, lbl + f"图片合适: 否\n"

    lbl += "\n### 答: " + example["response"].replace("\n", " ").strip()
    # TODO: 推荐改成选择题?
    yield prompt, img, lbl


N_NEGATIVE = 3
for sp in ("train", "valid", "test"):
    with open(f"./data/out.{sp}.jsonl") as f:
        examples = [json.loads(line) for line in f]

        candidate_imgs = set()
        for example in examples:
            candidate_imgs.update(example["links"])
        candidate_imgs = list(candidate_imgs)

    a, b = 0, 0
    with open(f"./data/{sp}.jsonl", "w") as f:
        for e in examples:
            for prompt, img, lbl in gen(e, random.sample(candidate_imgs, N_NEGATIVE)):
                data = {"img": img, "prompt": prompt, "label": lbl}
                b += 1
                if img:
                    a += 1
                print(json.dumps(data, ensure_ascii=False), file=f)
    print(a, b, a / b)
