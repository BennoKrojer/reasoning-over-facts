import os
import random

import pandas
import config


# def read_data(path, amount=None):
#     facts = list(map(lambda x: x.strip(), open(path, 'r').readlines()))
#     if amount is not None:
#         stepsize = (len(facts) // amount)
#         facts = facts[::stepsize]
#     return facts
#
#
# def get_unsupported(param):
#
def inferable(param, training_data, eval_examples):
    a, given_rel, b = param
    reachable_entities = []
    relevant = [fact for fact in training_data + eval_examples if fact.split()[1] == given_rel]
    for fact in relevant:
        e1, rel, e2 = fact.split()
        if e1 == a and rel == given_rel:
            reachable_entities.append(e2)

    for reachable_entity in reachable_entities:
        for fact in relevant:
            e1, rel, e2 = fact.split()
            if e1 == reachable_entity and e2 == b and rel == given_rel:
                return True
    return False


def get_unsupported(amount, eval_examples, training_data):
    vocab_path = os.path.join(config.vocab_dirs['transitive'], 'HighTrans_big', 'vocab.txt')
    vocab = list(map(lambda x: x.strip(), open(vocab_path, 'r').readlines()))[5:]
    entities = [token for token in vocab if token[0] == 'e']
    relations = [token for token in vocab if token[0] == 'r']
    unsupported = []
    while len(unsupported) < amount:
        print(str(len(unsupported)) + '/' + str(amount))
        a, b = random.sample(entities, 2)
        r = random.sample(relations, 1)[0]
        fact = f'{a} {r} {b}'
        if fact in eval_examples or fact in training_data or fact in unsupported:
            continue
        if inferable((a, r, b), training_data, eval_examples):
            print("INFER")
            continue
        unsupported.append(fact)

    return unsupported


if __name__ == '__main__':

    path = os.path.join(config.datasets_dirs['transitive'], 'HighTrans_big', 'eval.txt')
    eval_examples = list(map(lambda x: x.strip(), open(path, 'r').readlines()))
    training_data = list(map(lambda x: x.strip(), open(path.replace('eval', 'train'), 'r').readlines()))
    unsupported_examples = get_unsupported(len(training_data) + len(eval_examples), eval_examples, training_data)

    data = {'fact': [], 'label': [], 'source': []}
    for fact in eval_examples:
        data['fact'].append(fact)
        data['label'].append(1)
        data['source'].append('eval')

    for fact in training_data:
        data['fact'].append(fact)
        data['label'].append(1)
        data['source'].append('train')

    for fact in unsupported_examples:
        data['fact'].append(fact)
        data['label'].append(0)
        data['source'].append('unsupported')
    #
    pandas.DataFrame(data).to_csv(os.path.join(config.datasets_dirs['transitive'], 'classification.tsv'), sep='\t',
                                  index=False)
