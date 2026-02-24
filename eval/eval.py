import json
import sys
from collections import defaultdict

import evaluate
import jieba
import nltk

meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def evaluate_file(file):
    with open(file) as f:
        examples = [json.loads(line) for line in f]

    def average(xs):
        return sum(xs) / len(xs) if xs else None

    def exact_match(a, b):
        return 1 if a == b else 0

    def distinct(texts, n):
        if not isinstance(texts, str):
            texts = " ".join(texts)
        tokens = texts.split()
        ngram_counts = list(nltk.ngrams(tokens, n))
        unique_ngrams = set(ngram_counts)
        return len(unique_ngrams) / len(ngram_counts) + 1e-5

    stats = defaultdict(list)
    gold_responses = []
    pred_responses = []

    fields = ("当前目标", "剩余回合", "回复图片", "当前实体")
    for ex in examples:
        field = fields[0]

        for field in fields:
            if field != "当前实体" or ex[field] != "无":
                stats[field].append(
                    exact_match(ex[field], ex["prediction"][field].strip())
                )

        field = "剩余回合"
        try:
            remain_turn_int = int(ex["prediction"][field])
            remain_turn_std_int = ex[field]
        except:
            remain_turn_int = 20
            remain_turn_std_int = 0

        stats["剩余回合_abs_diff"].append(abs(int(remain_turn_std_int) - remain_turn_int))

        if "pos_image" in ex["prediction"]:

            def score(yn):
                return yn[0] / (yn[0] + yn[1])

            scores = [score(ex["prediction"]["pos_image"])] + [
                score(yn) for yn in ex["prediction"]["neg_images"]
            ]
            s0 = scores[0]
            scores.sort(reverse=True)
            stats["img_rank"].append(scores.index(s0) + 1)
            hit = 1 if s0 >= scores[0] - 1e-5 else 0
            stats["img_hit_rate"].append(hit)

        field = "\n### 答"
        gold_response = ex[field].strip()
        pred_response = ex["prediction"][field].strip()
        if gold_response:
            gold_response = " ".join(jieba.cut(gold_response, HMM=False))
            pred_response = " ".join(jieba.cut(pred_response, HMM=False))
            gold_responses.append(gold_response)
            pred_responses.append(pred_response)

    def get_result(metric, predictions, references, **kwargs):
        result = metric.compute(
            predictions=predictions, references=references, **kwargs
        )
        assert result is not None
        return result

    results = {"img_rank": 999, "img_hit_rate": -1}

    for field in stats:
        results[field] = average(stats[field])

    results["meteor"] = get_result(
        meteor, predictions=pred_responses, references=gold_responses
    )["meteor"]

    for i in range(1, 5):
        results[f"bleu-{i}"] = get_result(
            bleu, predictions=pred_responses, references=gold_responses, max_order=i
        )["bleu"]

    results["dist-1"] = distinct(pred_responses, 1)
    results["dist-2"] = distinct(pred_responses, 2)

    for rg, v in get_result(
        rouge, predictions=pred_responses, references=gold_responses
    ).items():
        results[rg] = v

    def get_res(field):
        r = f"{results[field]:.2f}"
        r += " " * max(0, (8 - len(r)))
        return r

    result_txt = ""
    result_txt += "task:\n"

    result_txt += " & ".join(
        [file]
        + list(
            map(
                get_res,
                ["当前目标", "当前实体", "剩余回合", "剩余回合_abs_diff", "回复图片", "img_hit_rate"],
            )
        )
    )

    result_txt += "response:\n"
    result_txt += " & ".join(
        [file]
        + list(
            map(
                get_res,
                [
                    "meteor",
                    "bleu-1",
                    "bleu-2",
                    "bleu-3",
                    "bleu-4",
                    "dist-1",
                    "dist-2",
                    "rouge1",
                    "rouge2",
                    "rougeL",
                    "rougeLsum",
                ],
            )
        )
    )
    return result_txt


for fn in sys.argv[1:]:
    res = evaluate_file(fn)
    print(res)
