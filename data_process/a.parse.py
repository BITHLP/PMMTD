import csv
import json
import re
from pprint import pp


def remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


def remove_sufix(s, sufix):
    if s.endswith(sufix):
        s = s[: -len(sufix)]
    return s


def strchr(ss, tt):
    for i, c in enumerate(ss):
        if c in tt:
            return i
    return -1


"""
dialogue,knowledge,goal

history,
image,
response,
entities,
goal
"""

g = 0


def show_summarization():
    pp(g)
    ...


def convert(row: list[str]):
    template, dialogue = row[1], row[2]
    goal, kg = get_goal_kg(template)
    return process_dialogue(goal, kg, dialogue)


def get_goal_kg(template: str):
    assert "goal:" in template
    assert "knowledge:" in template
    goal, kg = template.lstrip("goal:\n\n").split("knowledge:\n")
    goal = goal.replace("User", "用户")
    goal = goal.replace("user", "用户")
    goal = [remove_sufix(g.strip(), " -->") for g in goal.splitlines()]
    goal = [re.sub(r"\[\d\]\s*", "", g) for g in goal]

    kgs = []
    for line in kg.splitlines():
        line = line.strip()
        line = line.replace(",?'", ",'")
        if line.endswith('"'):
            line = line[:-1]
        if not line.endswith("]"):
            line += "']"
        item = eval(line)
        assert len(item) == 3
        kgs.append(item)
    return goal, kgs


def get_role(line):
    pos = strchr(line, ":：")
    if pos == -1:
        return None, line
    role, text = line.split(line[pos], maxsplit=1)

    s = re.sub(r"\[\d+\]", "", role).strip().lower()

    mp = {
        "bot": "bot",
        "rot": "bot",
        "user": "user",
        "use": "user",
        "useer": "user",
        "userq": "user",
    }

    role = mp.get(s.strip().lower(), None)
    if role:
        return role, text.strip()
    else:
        return None, line.strip()


def get_goal(line: str):
    matches = re.findall(r"\[(\d+)\]", line)
    assert len(matches) <= 1
    if matches:
        gn = int(matches[0])
        return gn, re.sub(rf"\s*\[{gn}\]\s*", "", line)
    return None, line


def get_url(line: str):
    r1 = r"@([./:_=,!?\-a-zA-Z0-9]+)@"
    r2 = r"http[./:_=,!?\-a-zA-Z0-9]+"
    urls = re.findall(r1, line)
    urls.extend(re.findall(r2, line))
    line = re.sub(rf"\s*{r1}\s*", "", line)
    line = re.sub(rf"\s*{r2}\s*", "", line)
    line = re.sub(r"\s*@.{,3}@\s*", "", line)

    urls = list(set(urls))
    return urls, line


def process_dialogue(goal: list[str], kg: list[str], dialog: str):
    conversation = []  # role,utterance,urls
    cur_gn = 0

    dialog = dialog.replace("user", "\nuser")
    dialog = dialog.replace("bot", "\nbot")
    lines = []
    for line in dialog.strip().splitlines():
        line = line.strip()
        if not line.strip():
            continue
        lines.append(line.strip())

    for line in lines:
        gn, text = get_goal(line)
        role, text = get_role(text)
        urls, text = get_url(text)

        if role is None and line == lines[-1]:
            # comment
            continue

        if gn is not None:
            assert gn == cur_gn + 1
            cur_gn = gn

        if role is None or conversation and conversation[-1][1] == role:
            conversation[-1][2] += "\n" + text
            conversation[-1][3].extend(urls)
        else:
            conversation.append([cur_gn, role, text, urls])

    assert cur_gn == len(goal)
    return goal, kg, conversation


if __name__ == "__main__":
    filename = f"./raw/all2.csv"
    ofilename = f"./data/a.jsonl"

    with open(filename) as f:
        rows = list(csv.reader(f))[1:]
        good = []
        for r in rows:
            try:
                e = convert(r)
                if e is not None:
                    good.append(e)
            except:
                # raise
                pass

        print(len(good), len(rows))
        show_summarization()

    with open(ofilename, "w") as f:
        for r in good:
            print(json.dumps(r, ensure_ascii=False), file=f)
