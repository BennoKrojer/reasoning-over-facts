import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.negation.negation_data_generator_N import NegationDataGenerator
from scripts.negation import datagen_config


class NegationGenerator(NegationDataGenerator):

    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=0)

    def create_complete_facts(self, relation):
        complete_facts = []
        eval = []
        available_entities = self.entities.copy()
        for i in range(datagen_config.FACTS_PER_RELATION):
            a = sample(available_entities, 1)[0]
            attributes = sample(self.attributes, 6)
            available_entities.remove(a)
            facts_a = []
            if numpy.random.rand() < 0.9:
                evalneg = False
                evalpos = False
            else:
                if numpy.random.rand() < 0.5:
                    evalneg = False
                    evalpos = True
                else:
                    evalneg = True
                    evalpos = False
            for pos, neg in attributes:
                if 1 == numpy.random.randint(2):
                    if numpy.random.rand() < 0.5:
                        pos_s = [a, relation, pos]
                        neg_s = [a, relation + " not", neg]
                    else:
                        neg_s = [a, relation + " not", pos]
                        pos_s = [a, relation, neg]
                else:
                    if numpy.random.rand() < 0.5:
                        pos_s = [a, relation, neg]
                        neg_s = [a, relation + " not", pos]
                    else:
                        neg_s = [a, relation + " not", neg]
                        pos_s = [a, relation, pos]
                if evalpos:
                    complete_facts.append(pos_s)
                    eval.append(neg_s)
                elif evalneg:
                    complete_facts.append(neg_s)
                    eval.append(pos_s)
                else:
                    complete_facts.append(pos_s)
                    complete_facts.append(neg_s)
        return complete_facts, eval

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        available_entities = self.entities.copy()
        for _ in range(datagen_config.FACTS_PER_RELATION_RANDOM):
            a, b = sample(available_entities, 2)
            available_entities.remove(a)
            available_entities.remove(b)
            train.append((a, relation, b))
            eval.append((b, relation, a))
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
    os.makedirs(DATA_DIR, exist_ok=True)
    generator = NegationGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.negation_config, os.path.join(DATA_DIR, 'datagen_config.py'))
