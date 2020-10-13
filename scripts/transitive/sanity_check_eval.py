import os
import config
from scripts.transitive.create_classification_data import inferable

path = os.path.join(config.datasets_dirs['transitive'], 'HighTrans_big', 'eval.txt')
eval_examples = list(map(lambda x: x.strip(), open(path, 'r').readlines()))
training_data = list(map(lambda x: x.strip(), open(path.replace('eval', 'train'), 'r').readlines()))

for eval_example in eval_examples:
    a, r, b = eval_example.strip().split()
    if not inferable((a,r,b), training_data, eval_examples):
        print('BAD')