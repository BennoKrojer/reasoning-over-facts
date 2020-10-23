import json
import os
dir = '/home/benno/Projects/BERT_thesis/data/symmetry/datasets/StandardSym_reduced80'
for file in os.listdir(dir):
    file = dir +'/'+ file
    if '.txt' in file:
        with open(file, "r") as f:
            lines = f.readlines()
        with open(file, "w") as f:
            for line in lines:
                rel = line.split()[1]
                if rel in ['r'+str(i) for i in range(40, 50)] or rel in ['r'+str(i) for i in range(90, 100)]:
                    pass
                else:
                    f.write(line)
    if '.json' in file:
        d = json.load(open(file, 'r'))
        new_d = dict()
        for key, value in d.items():
            rel = key.split()[1]
            if rel in ['r' + str(i) for i in range(40, 50)] or rel in ['r' + str(i) for i in range(90, 100)]:
                pass
            else:
                new_d[key] = value
        json.dump(new_d, open(file, 'w'))