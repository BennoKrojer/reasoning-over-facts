import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.negation.data_generator import NegationDataGenerator
from scripts.negation import datagen_config


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
            pos = sample(self.random_attributes, 1)[0]
            if 1 == numpy.random.randint(2):

                train.append((e, relation, pos))
                eval.append((e, 'not ' + relation, pos))
            else:
                train.append((e, 'not ' + relation, pos))
                eval.append((e, relation, pos))

        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)
    # def create_anti_patterns(self, relations):
    #     train = []
    #     eval = []
    #     for i in range(datagen_config.FACTS_PER_RELATION):
    #         if i < 0.9 * datagen_config.FACTS_PER_RELATION:
    #             a, b, c = sample(self.entities, 3)
    #             if (a, relations[0], b) not in train and (b, relations[1], c) not in train:
    #                 train.append((a, relations[0], b))
    #                 train.append((b, relations[1], c))
    #         else:
    #             a, b = sample(self.entities, 2)
    #             if (a, relations[0], b) not in train and (b, relations[1], a) not in train:
    #                 if 1 == numpy.random.randint(2):
    #                     train.append((a, relations[0], b))
    #                     eval.append((b, relations[1], a))
    #                 else:
    #                     train.append((a, relations[1], b))
    #                     eval.append((b, relations[0], a))
    #     eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
    #     return numpy.asarray(train), numpy.asarray(eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()
    DATA_DIR = os.path.join(config.datasets_dirs['negation'], args.dataset_name)
    try:
        os.makedirs(DATA_DIR, exist_ok=False)
    except OSError:
        overwrite = True if input('Overwrite dataset: y/n\n') == 'y' else False
        os.makedirs(DATA_DIR, exist_ok=True)
    generator = NegationGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.negation_config, os.path.join(DATA_DIR, 'datagen_config.py'))
