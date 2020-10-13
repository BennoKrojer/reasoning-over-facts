import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.data_generator import DataGenerator
from scripts.order import datagen_config


class GroupingGenerator(DataGenerator):

    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=4, entity_list=False)

    def create_complete_facts(self, relation):
        complete_facts = []
        # available_entities = self.entities.copy()
        for k in range(datagen_config.FACTS_PER_RELATION):
            # group_size = numpy.random.randint(3, 10)
            group_size = 6 # TODO: parameterize!
            r_numb = int(relation[1:])
            entities = list(range(20*(r_numb % 100)+(k+1)*6,
                                  20*(r_numb % 100) + k*6, -1))
            entities = list(map(str, entities))
            # entities = sample(available_entities, group_size)
            # for e in entities:
            #     available_entities.remove(e)
            complete_group = []
            for i in range(group_size):
                for j in range(i+1, group_size):
                    complete_group.append((entities[i], relation, entities[j]))
                    self.entities[entities[i]] += 1
                    self.entities[entities[j]] += 1
            # numpy.random.shuffle(complete_group)
            complete_group[1], complete_group[-1] = complete_group[-1], complete_group[1]
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
        eval = list(filter(lambda x: self.check_train(x, train, 3), eval))
        return numpy.asarray(train), numpy.asarray(eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()
    DATA_DIR = os.path.join(config.datasets_dirs['order'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = GroupingGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.symmetry_config, os.path.join(DATA_DIR, 'datagen_config.py'))