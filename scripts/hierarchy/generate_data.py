import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.data_generator_with_anti import DataGenerator
from scripts.hierarchy import datagen_config


class GroupingGenerator(DataGenerator):

    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=5)

    def create_complete_facts(self, relation):
        complete_facts = []
        available_entities = self.entities.copy()
        for k in range(datagen_config.FACTS_PER_RELATION):
            group_size = 7 # TODO: parameterize!
            entities = sample(available_entities, group_size)
            for e in entities:
                available_entities.remove(e)
            complete_group = []
            for lower_e in entities[1:]:
                complete_group.append((entities[0], relation, lower_e))
            complete_group.append((entities[1], relation, entities[3]))
            complete_group.append((entities[1], relation, entities[4]))
            complete_group.append((entities[2], relation, entities[5]))
            complete_group.append((entities[2], relation, entities[6]))
            complete_group[2], complete_group[-1] = complete_group[-1], complete_group[2]
            complete_facts.append(complete_group)
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION*5):
            a, b, c = sample(self.entities, 3)
            if (a, relation, c) not in train:
                train.append((a, relation, b))
                train.append((b, relation, c))
                eval.append((a, relation, c))
        eval = list(filter(lambda x: self.check_train(x, train, 4), eval))
        return numpy.asarray(train), numpy.asarray(eval)

    def create_anti_patterns(self, relation):
        train = []
        eval = []
        for i in range(datagen_config.FACTS_PER_RELATION*5):
            if i < 0.9 * datagen_config.FACTS_PER_RELATION:
                a, b, c, d = sample(self.entities, 4)
                if (a, relation, b) not in train and (b, relation, c) not in train and (a, relation, d) not in train:
                    train.append((a, relation, b))
                    train.append((b, relation, c))
                    train.append((a, relation, d))
            else:
                a, b, c = sample(self.entities, 3)
                if (a, relation, b) not in train and (b, relation, c, 4):
                    train.append((a, relation, b))
                    train.append((b, relation, c))
                    eval.append((a, relation, c))
        eval = list(filter(lambda x: self.check_train(x, train, 4), eval))
        return numpy.asarray(train), numpy.asarray(eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()
    DATA_DIR = os.path.join(config.datasets_dirs['hierarchy'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = GroupingGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.hierarchy_config, os.path.join(DATA_DIR, 'datagen_config.py'))