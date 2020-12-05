import argparse
import os
import shutil
from random import sample

import numpy

import config
from scripts.negation import datagen_config
from scripts.negation.data_generator import NegationDataGenerator


class NegationGenerator(NegationDataGenerator):

    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=0)

    def create_complete_facts(self, relation):
        simplified = False
        contradictions = False
        complete_facts = []
        available_entities = self.entities.copy()
        sampled_pairs = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a = sample(available_entities, 2)[0]
            if simplified:
                available_entities.remove(a)
            pos, neg = sample(self.attributes, 1)[0]

            if not contradictions:
                while (a, neg) in sampled_pairs:
                    a = sample(available_entities, 2)[0]
                    if simplified:
                        available_entities.remove(a)
                    pos, neg = sample(self.attributes, 1)[0]
                sampled_pairs.append((a, pos))

            if 1 == numpy.random.randint(2):
                if numpy.random.rand() < 0.5:
                    complete_facts.append(((a, relation, pos), (a, 'not ' + relation, neg)))
                else:
                    complete_facts.append(((a, 'not ' + relation, pos), (a, relation, neg)))
            else:
                if numpy.random.rand() < 0.5:
                    complete_facts.append(((a, relation, neg), (a, 'not ' + relation, pos)))
                else:
                    complete_facts.append(((a, 'not ' + relation, neg), (a, relation, pos)))
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns_entities(self, relation):
        train = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a, b = sample(self.entities, 2)
            train.append((a, relation, b))
            eval.append((b, relation, a))
        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            e = sample(self.entities, 1)[0]
            pos, neg = sample(self.attributes, 1)[0]

            if numpy.random.rand() < 0.5:
                a = pos
            else:
                a = neg

            train.append((e, relation, a))
            eval.append((e, 'not ' + relation, a))
        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()
    DATA_DIR = os.path.join(config.datasets_dirs['negation'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = NegationGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.negation_config, os.path.join(DATA_DIR, 'datagen_config.py'))
