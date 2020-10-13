import json
import os
from collections import defaultdict
import config

datasets = os.listdir(config.datasets_dirs['symmetry'])
with open(os.path.join(config.output_dir, 'models', 'symmetry', 'baseline.txt'), 'w') as baseline_file:

    for dataset in datasets:
        obj_freqs = defaultdict(int)

        with open(os.path.join(config.datasets_dirs['symmetry'], dataset, 'train.txt')) as train_file:
            for line in train_file:
                obj = line.strip().split()[-1]
                obj_freqs[obj] += 1

        most_frequent_obj = sorted(obj_freqs.items(), key = lambda x: x[1], reverse=True)[0][0]
        if 'NoSym' in dataset:
            d = json.load(open(os.path.join(config.datasets_dirs['symmetry'], dataset, 'rand_subject_relation2object_eval.json'), 'r'))
        else:
            d = json.load(open(os.path.join(config.datasets_dirs['symmetry'], dataset, 'subject_relation2object_eval.json'), 'r'))

        accurate = 0
        total = 0
        for subj_rel, obj in d.items():
            if obj == most_frequent_obj:
                accurate += 1
            total += 1

        baseline_file.write(f'{dataset}-Accuracy by predicting {most_frequent_obj}: {accurate/total}\n')