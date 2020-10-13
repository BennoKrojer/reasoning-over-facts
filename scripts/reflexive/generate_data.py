import argparse
import os
import shutil
from random import sample
import numpy
import config
from scripts.data_generator import DataGenerator
from scripts.reflexive import datagen_config


class ReflexiveGenerator(DataGenerator):
    def __init__(self, dataset_dir, config):
        super().__init__(dataset_dir, config, evals_allowed_in_train=0)

    def create_complete_facts(self, relation):
        complete_facts = []
        sampled = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a = sample(self.entities, 1)[0]
            if a not in sampled:
                complete_facts.append([(a, relation, a)])
                sampled.append(a)
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        available_entities = self.entities.copy()
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a, b = sample(self.entities, 2)
            if a != b:
                train.append((a, relation, b))
                if a not in available_entities:
                    available_entities.remove(a)
        for _ in range(datagen_config.FACTS_PER_RELATION//3):
            a = sample(available_entities, 1)[0]
            if (a, relation, a) not in eval:
                eval.append((a, relation, a))
        return numpy.asarray(train), numpy.asarray(eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()

    DATA_DIR = os.path.join(config.datasets_dirs['reflexive'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = ReflexiveGenerator(DATA_DIR, datagen_config)
    generator.create_dataset()

    shutil.copy(config.reflexive_config, os.path.join(DATA_DIR, 'datagen_config.py'))
