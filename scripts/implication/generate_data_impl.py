import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.implication.impl_data_generator import ImplicationDataGenerator
from scripts.implication import datagen_config


class ImplicationGenerator(ImplicationDataGenerator):

    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=1, numb_left_out_for_eval=5)

    def create_complete_facts(self, relations, cause, effects):
        complete_facts = []
        for _ in range(datagen_config.FACTS_PER_IMPLICATION):
            a = sample(self.entities, 1)[0]
            implications = [(a, relations[0], effect) for effect in effects]
            complete_facts.append([(a, relations[1], cause)] + implications)
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION_DISTRACTION_PATTERNS):
            a = sample(self.entities, 1)[0]
            b = sample(self.attributes, 1)[0]
            train.append((a, relation, b))
            eval.append((b, relation, a))
        eval = list(filter(lambda x: self.check_train(x, train, 1), eval))
        return numpy.asarray(train), numpy.asarray(eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()
    DATA_DIR = os.path.join(config.datasets_dirs['implication'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = ImplicationGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.implication_config, os.path.join(DATA_DIR, 'datagen_config.py'))
