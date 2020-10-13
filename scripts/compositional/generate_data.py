import argparse
import os
import shutil
from random import sample
import numpy
import config
from scripts.compositional.compositional_data_generator import DataGenerator
from scripts.compositional import datagen_config


class CompositionalGenerator(DataGenerator):
    def __init__(self, dataset_dir, config, complete_pattern_per_line=False):
        super().__init__(dataset_dir, config, evals_allowed_in_train=0, complete_pattern_per_line=complete_pattern_per_line)

    def create_complete_facts(self, relations):
        complete_facts = []
        for _ in range(datagen_config.FACTS_PER_RELATION):
            a, b, c = sample(self.entities, 3)
            complete_facts.append(((a, relations[0], b), (b, relations[1], c), (a, relations[2], c)))
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relation):
        if self.complete_pattern_per_line:
            return self.full_line_create_incomplete_patterns(relation)
        else:
            return self.multiple_lines_create_incomplete_patterns(relation)

    def full_line_create_incomplete_patterns(self, relation):
        train1 = []
        train2 = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION//2):
            a, b, c = sample(self.entities, 3)
            if [[a, relation, b], [b, relation, c]] not in train1:
                train1.append([[a, relation, b], [b, relation, c]])
                eval.append([[a, relation, b], [b, relation, c], [a, relation, c]])
        for _ in range(datagen_config.FACTS_PER_RELATION//2):
            a, b, c, d, e, f = sample(self.entities, 6)
            if [[a, relation, b], [c, relation, d], [e, relation, f]] not in train2:
                train2.append([[a, relation, b], [c, relation, d], [e, relation, f]])
        train = train1 + train2
        train1 = numpy.array(train1)
        train2 = numpy.array(train2)
        flatten_train1 = train1.reshape(train1.shape[0]*train1.shape[1], train1.shape[2])
        flatten_train2 = train2.reshape(train2.shape[0]*train2.shape[1], train2.shape[2])
        flatten_train = numpy.concatenate((flatten_train1, flatten_train2))
        eval = list(filter(lambda x: self.check_train(x[2], flatten_train.tolist(), 0), eval))

        return numpy.asarray(train), numpy.asarray(eval)

    def multiple_lines_create_incomplete_patterns(self, relations):
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

    DATA_DIR = os.path.join(config.datasets_dirs['compositional'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = CompositionalGenerator(DATA_DIR, datagen_config, complete_pattern_per_line=False)
    generator.create_dataset()

    shutil.copy(config.transitive_config, os.path.join(DATA_DIR, 'datagen_config.py'))
