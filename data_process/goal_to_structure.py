import json
import re
from pprint import pp
from re import findall

# goal => 任务和结构和实体,
#
# history
# bot response
# image
#
# 对话历史
# 为每个bot回复生成结构体：
# 目标（结构化的目标）
# 相关(历史/候选)实体+ 知识
# 回复+ 图片
# 轮数 + 本目标是否结束


def school_recom(s: str):
    school = findall("推荐大学『(.*?)』", s)[0]

    accepted = "接受" in s
    refused = "拒绝" in s
    assert refused != accepted

    return "推荐学校", school  # , "接受" if accepted else "拒绝"


def chit_chat(s: str):
    topic = findall("给定的主题『(.*?)』", s)[0]
    return ("寒暄",)  # , topic


def que(s: str):
    if "学科题目" in s:
        q = findall("使用学科题目『(.*?)』", s)[0]
        wrong = "错误答案" in s
        if q.endswith("..."):
            q = q[:-3]
        return "提问题目", q  # , wrong
    else:
        e = findall("『(.*?)』", s)[0]
        if "职业" in s:
            return "提问喜欢的职业", e
        elif "大学" in s:
            return "提问喜欢的大学", e
        else:
            raise RuntimeError(s)


def course_reco(s: str):
    name = findall("『(.*?)』", s)[0]

    accepted = "接受" in s
    refused = "拒绝" in s
    assert refused != accepted

    return "网课推荐", name  # , "接受" if accepted else "拒绝"


def book_reco(s: str):
    name = findall("『(.*?)』", s)[0]

    accepted = "接受" in s
    refused = "拒绝" in s
    assert refused != accepted

    return "辅导书推荐", name  # , "接受" if accepted else "拒绝"


def qa(s):
    name = findall("『(.*?)』", s)[0]
    if name.endswith("..."):
        name = name[:-3]
    return "答疑问答", name


def study_question(s):
    name = findall("『(.*?)』", s)[0]
    if name.endswith("..."):
        name = name[:-3]
    return "题目问答", name


def vqa(s):
    q, a = re.findall("【(.*?)】", s)
    return ("视觉问答",)  # , q, a


def qa2(s):
    name = re.findall("『(.*?)』", s)[0]
    if "职业" in s:
        return "职业问答", name
    if "大学" in s:
        return "大学问答", name
    raise RuntimeError(s)


def food_reco(s):
    names = re.findall("【(.*?)】", s)
    if "不喜欢" in s:
        return "美食推荐", names[0], names[1]
    else:
        return "美食推荐", names[0]


def rest_reco(s):
    names = re.findall("餐厅【(.*?)】", s)
    return "餐馆推荐", *names


def major_reco(s):
    name = re.findall("专业『(.*?)』", s)[0]
    return "专业推荐", name


def job_reco(s):
    name = re.findall("职业『(.*?)』", s)[0]
    return "职业推荐", name


def chat_school(s):
    name = re.findall("『(.*?)』", s)[0]
    return "关于大学的聊天", name


def goal_to_structure(desc) -> tuple:
    gt = desc.split("(")[0]
    if gt == "大学推荐":
        return school_recom(desc)
    elif gt == "寒暄":
        return chit_chat(desc)
    elif gt == "提问":
        return que(desc)
    elif gt == "网课推荐":
        return course_reco(desc)
    elif gt == "辅导书推荐":
        return book_reco(desc)
    elif gt in ("辅导书购买", "再见", "辅助决策"):
        return (gt,)
    elif gt == "答疑问答":
        return qa(desc)
    elif gt == "题目问答":
        return study_question(desc)
    elif gt == "用户提问":
        return vqa(desc)
    elif gt == "美食推荐":
        return food_reco(desc)
    elif gt == "餐馆推荐":
        return rest_reco(desc)
    elif gt == "问答":
        return qa2(desc)
    elif gt == "关于大学的聊天":
        return chat_school(desc)
    elif gt == "关于职业的聊天":
        name = re.findall("『(.*?)』", desc)[0]
        return "关于职业的聊天", name
    elif gt == "专业推荐":
        return major_reco(desc)
    elif gt == "职业推荐":
        return job_reco(desc)
    else:
        raise RuntimeError(desc)


def main():
    with open("./data/a.jsonl") as f:
        examples = [json.loads(line) for line in f]
    for gs, _, _ in examples:
        for g in gs:
            print(goal_to_structure(g))


if __name__ == "__main__":
    main()
