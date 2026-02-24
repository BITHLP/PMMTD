import json
import random

with open("./data/b.1.jsonl") as f:
    examples = [json.loads(line) for line in f]


random.shuffle(examples)
N = len(examples)

train = examples[: int(N * 0.7)]
valid = examples[int(N * 0.7) : int(N * 0.8)]
test = examples[int(N * 0.8) :]

with open("./data/b.train.jsonl", "w") as f:
    for r in train:
        print(json.dumps(r, ensure_ascii=False), file=f)

with open("./data/b.valid.jsonl", "w") as f:
    for r in valid:
        print(json.dumps(r, ensure_ascii=False), file=f)

with open("./data/b.test.jsonl", "w") as f:
    for r in test:
        print(json.dumps(r, ensure_ascii=False), file=f)
