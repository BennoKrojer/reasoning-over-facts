import argparse
import os
import shutil
from random import sample
import numpy
import config
from scripts.composition.data_generator import DataGenerator
from scripts.composition import datagen_config


class CompositionalGenerator(DataGenerator):
    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=0)

    def create_complete_facts(self, relations):
        complete_facts = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a, b, c = sample(self.entities, 3)
            complete_facts.append(((a, relations[0], b), (b, relations[1], c), (a, relations[2], c)))
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relations):
        train = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a, b, c = sample(self.entities, 3)
            train.append((a, relations[0], b))
            eval.append((a, relations[1], c))
        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)

    def create_anti_patterns(self, relations):
        train = []
        eval = []
        for i in range(datagen_config.FACTS_PER_RELATION):
            if i < 0.9 * datagen_config.FACTS_PER_RELATION:
                a, b, c, d = sample(self.entities, 4)
                if (a, relations[0], b) not in train and (b, relations[1], c) not in train and (a, relations[2],
                                                                                                d) not in train:
                    train.append((a, relations[0], b))
                    train.append((b, relations[1], c))
                    train.append((a, relations[2], d))
            else:
                a, b, c = sample(self.entities, 3)
                if (a, relations[0], b) not in train and (b, relations[1], c):
                    train.append((a, relations[0], b))
                    train.append((b, relations[1], c))
                    eval.append((a, relations[2], c))
        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()

    DATA_DIR = os.path.join(config.datasets_dirs['composition'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = CompositionalGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.composition_config, os.path.join(DATA_DIR, 'datagen_config.py'))
