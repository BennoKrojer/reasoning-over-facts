import argparse
import os
import shutil
from random import sample

import numpy
import config
from scripts.composition_enhanced.data_generator import DataGenerator
from scripts.composition_enhanced import datagen_config


class EnhancedGenerator(DataGenerator):

    def __init__(self, dataset_dir, config, groupsize):
        self.groupsize = groupsize
        left_out_eval = int((groupsize/3)**2)
        super().__init__(dataset_dir, config, evals_allowed_in_train=0, numb_left_out_for_eval=left_out_eval)

    def create_complete_facts(self, relation):

        complete_facts = []
        available_entities = self.entities.copy()
        available_attributes = self.attributes.copy()
        for _ in range(datagen_config.FACTS_PER_RELATION):
            entities = sample(available_entities, self.groupsize)
            """for e in entities:
                available_entities.remove(e)"""
            A, B, C = numpy.split(numpy.array(entities), 3)
            attrA, attrB, attrC = sample(available_attributes, 3)
            for x in [attrA, attrB, attrC]:
                available_attributes.remove(x)
            complete_group = []

            # connecting class members between themselves or to a class token
            for class_, class_token in [(A, attrA), (B, attrB), (C, attrC)]:
            #     for member in class_:
            #         complete_group.append((member, 'is', class_token))
                for i in range(len(class_)):
                    for j in range(i+1, len(class_)):
                        complete_group.append((class_[i], 'connectedto', class_[j]))
                        complete_group.append((class_[j], 'connectedto', class_[i]))

            # making transitive connections
            for a in A:
                for b in B:
                    complete_group.append((a, relation[0], b))

            for b in B:
                for c in C:
                    complete_group.append((b, relation[1], c))

            for a in A:
                for c in C:
                    complete_group.append((a, relation[2], c))

            complete_facts.append(complete_group)
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relation):
        train = []
        eval = []
        for _ in range(datagen_config.FACTS_PER_RELATION*100):
            a, b = sample(self.entities, 2)
            train.append((a, relation, b))
            eval.append((b, relation, a))
        eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
        return numpy.asarray(train), numpy.asarray(eval)


    # def create_anti_patterns(self, relations):
    #     train = []
    #     eval = []
    #     for i in range(datagen_config.FACTS_PER_RELATION_DISTRACTION_PATTERNS):
    #         if i < 0.9 * datagen_config.FACTS_PER_RELATION_DISTRACTION_PATTERNS:
    #             a, b, c, d = sample(self.entities, 4)
    #             if (a, relations[0], b) not in train and (b, relations[1], c) not in train and (a, relations[2],
    #                                                                                        d) not in train:
    #                 train.append((a, relations[0], b))
    #                 train.append((b, relations[1], c))
    #                 train.append((a, relations[2], d))
    #         else:
    #             a, b, c = sample(self.entities, 3)
    #             if (a, relations[0], b) not in train and (b, relations[1], c):
    #                 train.append((a, relations[0], b))
    #                 train.append((b, relations[1], c))
    #                 eval.append((a, relations[2], c))
    #     eval = list(filter(lambda x: self.check_train(x, train, 0), eval))
    #     return numpy.asarray(train), numpy.asarray(eval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="The name of the dataset you want to create.")
    args = parser.parse_args()
    DATA_DIR = os.path.join(config.datasets_dirs['composition_enhanced'], args.dataset_name)
    try:
        os.makedirs(DATA_DIR, exist_ok=False)
    except OSError:
        overwrite = True if input('Overwrite dataset: y/n\n') == 'y' else False
        os.makedirs(DATA_DIR, exist_ok=True)
    generator = EnhancedGenerator(DATA_DIR, datagen_config, groupsize=30)
    generator.create_dataset()

    shutil.copy(config.enhanced_config, os.path.join(DATA_DIR, 'datagen_config.py'))
