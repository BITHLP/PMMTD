import enum
import json
from pprint import pp

from goal_to_structure import goal_to_structure

# goal => 任务和结构和实体,
#
# 对话历史
# 为每个回复生成结构体：
# 结构化的目标
# 相关(历史/候选)实体+ 知识
# 剩余轮数 + 本目标是否结束
# 回复+ 图片


def main(sp):
    with open(f"./data/b.{sp}.jsonl") as f:
        examples = [json.loads(line) for line in f]

    data = []
    for kg, dialog in examples:
        history, prev_goal = [], None
        prev_goal = None
        for goal, utts in dialog:
            for i, (role, text, links) in enumerate(utts):
                text = text.replace("\n", " ").strip()
                if role == "bot":
                    remain = len(utts) - i - 1
                    # TODO: 粗知识/粗图片/粗候选推荐
                    d = {
                        "kg": kg,
                        "history": history.copy(),
                        "prev_goal": prev_goal,
                        "goal": goal,
                        "remain": remain,
                        "response": text,
                        "links": links,
                    }
                    data.append(d)
                history.append((role, text, links))
                prev_goal = goal

    with open(f"./data/out.{sp}.jsonl", "w") as f:
        for r in data:
            print(json.dumps(r, ensure_ascii=False), file=f)


if __name__ == "__main__":
    main("train")
    main("valid")
    main("test")
