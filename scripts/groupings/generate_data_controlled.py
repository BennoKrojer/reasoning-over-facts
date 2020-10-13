import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.groupings.data_generator import DataGenerator
from scripts.groupings import datagen_config


class GroupingGenerator(DataGenerator):

    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=0, numb_left_out_for_eval=15)

    def create_complete_facts(self, relations):
        train = []
        r0, r1 = relations
        eval = []
        available_entities = self.entities.copy()
        available_attributes = self.attributes.copy()
        entity_groups = []
        missing_networks = []
        eval_pkl = []
        # 1. no sym case
        for k in range(datagen_config.FACTS_PER_RELATION):
            group_size = 6
            entities = sample(available_entities, group_size)
            complete_group = [[], []]
            for e in entities:
                available_entities.remove(e)
            entity_groups.append(entities)
            for i in range(group_size):
                for j in range(i+1, group_size):
                    group = numpy.random.randint(2)
                    complete_group[group].append((entities[i], r0, entities[j]))
                    complete_group[1-group].append((entities[j], r1, entities[i]))

            if k >= datagen_config.FACTS_PER_RELATION * datagen_config.ratio_of_complete_patterns:
                eval.append(complete_group[1])
                eval_pkl.append(complete_group[1])
                train.append(complete_group[0])
                missing_networks.append(complete_group[0])
            else:
                train.append(complete_group[0] + complete_group[1])
        train = numpy.asarray([fact for lists in train for fact in lists])
        eval = numpy.asarray([fact for lists in eval for fact in lists])
        return (train, eval, missing_networks, eval_pkl), entity_groups

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        available_entities = self.entities.copy()
        for _ in range(datagen_config.FACTS_PER_RELATION_DISTRACTION_PATTERNS):
            a, b = sample(available_entities, 2)
            available_entities.remove(a)
            available_entities.remove(b)
            train.append((a, relation, b))
            eval.append((b, relation, a))
        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)

    def create_anti_patterns(self, relation):
        train = []
        eval = []
        for i in range(datagen_config.FACTS_PER_RELATION_DISTRACTION_PATTERNS):
            if i < 0.9 * datagen_config.FACTS_PER_RELATION_DISTRACTION_PATTERNS:
                a, b, c = sample(self.entities, 3)
                if (a, relation, b) not in train and (b, relation, c) not in train:
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
    DATA_DIR = os.path.join(config.datasets_dirs['groupings'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = GroupingGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.groupings_config, os.path.join(DATA_DIR, 'datagen_config.py'))