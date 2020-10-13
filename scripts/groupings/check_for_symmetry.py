import os
import json

answers = json.load(open('../../data/groupings/datasets/StandardGroup_with_anti/subject_relation2object_eval.json', 'r'))
train = open('../../data/groupings/datasets/StandardGroup_with_anti/train.txt', 'r').readlines()
train = [line.strip() for line in train]
symmetry_count, total = 0, 0
for subj_rel, objs in answers.items():
    subj, rel = subj_rel.split()
    for obj in objs:
        if obj + ' ' + rel + ' ' + subj in train:
            symmetry_count += 1
        total += 1

print(symmetry_count/total)