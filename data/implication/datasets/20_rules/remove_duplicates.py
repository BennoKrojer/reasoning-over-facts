import json
import os

for file in os.listdir('.'):
    if file == 'subject_relation2object_eval.json':
        d = json.load(open(file, 'r'))
        old = 0
        new = 0
        for key, v in d.items():
            old += len(v)
            v2 = list(set(v))
            new += len(v2)

        print(old)
        print(new)
        print(old/new)
        #     d[key] = v
        # json.dump(d, open(file, 'w'))
