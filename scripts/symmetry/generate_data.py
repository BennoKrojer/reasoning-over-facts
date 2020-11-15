import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.data_generator import DataGenerator
from scripts.symmetry import datagen_config


class SymmetryGenerator(DataGenerator):

    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=0)

    def create_complete_facts(self, relation):
        complete_facts = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a, b = sample(self.entities, 2)
            complete_facts.append(((a, relation, b), (b, relation, a)))
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a, b = sample(self.entities, 2)
            train.append((a, relation, b))
            eval.append((b, relation, a))
        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)

    def create_anti_patterns(self, relation):
        train = []
        eval = []
        for i in range(datagen_config.FACTS_PER_RELATION):
            if i < 0.9 * datagen_config.FACTS_PER_RELATION:
                a, b, c = sample(self.entities, 3)
                train.append((a, relation, b))
                train.append((b, relation, c))
            else:
                a, b = sample(self.entities, 2)
                if (a, relation, b) not in train:
                    train.append((a, relation, b))
                    eval.append((b, relation, a))
        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()
    DATA_DIR = os.path.join(config.datasets_dirs['symmetry'], args.dataset_name)
    try:
        os.makedirs(DATA_DIR, exist_ok=False)
    except OSError:
        overwrite = True if input('Overwrite dataset: y/n\n') == 'y' else False
        os.makedirs(DATA_DIR, exist_ok=True)
    generator = SymmetryGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.symmetry_config, os.path.join(DATA_DIR, 'datagen_config.py'))
