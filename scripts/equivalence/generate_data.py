import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.equivalence.data_generator import EqualDataGenerator
from scripts.equivalence import datagen_config


class EqualGenerator(EqualDataGenerator):

    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=0)

    def create_complete_facts(self, relations):
        complete_facts = []
        for _ in range(datagen_config.FACTS_PER_IMPLICATION):
            a = sample(self.entities, 1)[0]
            attribute = sample(self.attributes, 1)[0]
            complete_facts.append(((a, relations[0], attribute), (a, relations[1], attribute)))
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION_DISTRACTION_PATTERNS):
            a, b = sample(self.entities, 2)
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
    DATA_DIR = os.path.join(config.datasets_dirs['equivalence'], args.dataset_name)
    try:
        os.makedirs(DATA_DIR, exist_ok=False)
    except OSError:
        overwrite = True if input('Overwrite dataset: y/n\n') == 'y' else False
        os.makedirs(DATA_DIR, exist_ok=True)
    generator = EqualGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.equivalence_config, os.path.join(DATA_DIR, 'datagen_config.py'))
