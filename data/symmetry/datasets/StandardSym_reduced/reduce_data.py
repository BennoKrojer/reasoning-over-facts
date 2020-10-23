import json
import os
dir = '/home/benno/Projects/BERT_thesis/data/symmetry/datasets/StandardSym_reduced'
for file in os.listdir(dir):
    file = dir +'/'+ file
    if '.txt' in file:
        with open(file, "r") as f:
            lines = f.readlines()
        with open(file, "w") as f:
            for line in lines:
                if 'r45' in line or 'r46' in line or 'r47' in line or 'r48' in line or 'r49' in line or \
                    'r95' in line or 'r96' in line or 'r97' in line or 'r98' in line or 'r99' in line:
                    pass
                else:
                    f.write(line)
    if '.json' in file:
        d = json.load(open(file, 'r'))
        new_d = dict()
        for key, value in d.items():
            if 'r45' in key or 'r46' in key or 'r47' in key or 'r48' in key or 'r49' in key or \
                    'r95' in key or 'r96' in key or 'r97' in key or 'r98' in key or 'r99' in key:
                pass
            else:
                new_d[key] = value
        json.dump(new_d, open(file, 'w'))