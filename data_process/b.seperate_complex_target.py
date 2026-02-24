import json

from fuzzywuzzy import fuzz
from goal_to_structure import goal_to_structure


def sep(goals, kg, dialog, thres=50):
    goal_structs = [goal_to_structure(goal) for goal in goals]
    result = []
    for i, g in enumerate(goal_structs, start=1):
        ds = [d[1:] for d in dialog if d[0] == i]
        if g[0] in ("美食推荐", "餐馆推荐") and len(g) == 3:
            semi = [fuzz.partial_ratio(g[2], d[2]) for d in dialog]
            idx = semi.index(max(semi))
            assert max(semi) > thres
            result.append(((g[0], g[1]), ds[:idx]))
            result.append(((g[0], g[2]), ds[idx:]))
        else:
            result.append((g, ds))
    return kg, result


def main():
    with open("./data/a.jsonl") as f:
        examples = [json.loads(line) for line in f]

    results = []
    err = 0
    for goals, kg, dialog in examples:
        try:
            results.append(sep(goals, kg, dialog))
        except:
            err += 1

    with open("./data/b.0.jsonl", "w") as f:
        for r in results:
            print(json.dumps(r, ensure_ascii=False), file=f)

    print(err, len(examples))


if __name__ == "__main__":
    main()
