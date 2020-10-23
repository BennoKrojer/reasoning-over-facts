import json
import os

for file in os.listdir('.'):
    if file[-4:] == 'json':
        d = json.load(open(file, 'r'))
        for key, v in d.items():
            v = list(set(v))
            d[key] = v
        json.dump(d, open(file, 'w'))
