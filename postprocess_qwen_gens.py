import os
import re
import json

base_dir = "[INSERT BASE DIRECTORY]"

gen_dir = os.path.join(base_dir, "generations")
exp_name = "[INSERT EXPERIMENT NAME]"

curr_dir = os.path.join(gen_dir, exp_name)


generations = []
for filename in os.listdir(curr_dir):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join(curr_dir, filename), "r") as file_p:
        _generations = json.load(file_p)
    generations.extend(_generations)


success = 0
mix = 0
fail = 0
ood = 0
unknown = 0
for idx, sample in enumerate(generations):
    tokens = re.findall(r"\d+|<\w+>|[A-Za-z_]+|[()+\-*/=]", sample)
    ops = {"+", "-", "*", "/", "="}
    parens = {"(", ")"}
    not_kw = {"not"}
    mix_extras = {
        "so",
        "the",
        "answer",
        "is",
        "<answer>",
        "</answer>",
        "</think>",
        "So",
        "equation",
        "that",
        "equals",
        "think",
    }

    def is_int(tok):
        return re.fullmatch(r"\d+", tok) is not None


    if all(is_int(t) or t in ops | parens | not_kw for t in tokens):
        success += 1
        print("-> Success")

    elif all(
        is_int(t) or t in ops | parens | not_kw | mix_extras for t in tokens
    ):
        mix += 1
        print("-> Mix")

    else:
        ood_words = [
            t
            for t in tokens
            if not (is_int(t) or t in ops | parens | not_kw | mix_extras)
        ]
        if "not" in sample and "correct" in sample:
            if "not correct" in sample:
                success += 1
                print("-> OOD (success); offending token(s):", ood_words)
            
            else:
                mix += 1
                print("-> OOD (mix); offending token(s):", ood_words)
        elif "not" in sample and "this works" in sample:
            mix += 1
            print("-> OOD (mix); offending token(s):", ood_words)
        elif "not" in sample and "<answer>" in sample:
            mix += 1
            print("-> OOD (mix); offending token(s):", ood_words)
        elif "not" in sample and "Wait" in sample:
            mix += 1
            print("-> OOD (mix); offending token(s):", ood_words)
        elif "not" in sample and "That works" in sample:
            mix += 1
            print("-> OOD (mix); offending token(s):", ood_words)

        elif "not" in sample and "this works" in sample:
            mix += 1
            print("-> OOD (mix); offending token(s):", ood_words)

        else:
            if "this works" in sample:
                fail += 1
                print("-> Fail")
            else:
                ood += 1
                unknown += 1
                print("-> OOD (unknown); offending token(s):", ood_words)
    print(sample)


print(f"Success: {success / len(generations)}")
print(f"Partial: {mix / len(generations)}")
print(f"Fail: {fail / len(generations)}")
print(f"OOD: {ood / len(generations)}")
print(f"len generations: {len(generations)}")
