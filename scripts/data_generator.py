import json
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy
from random import sample
from numpy.random import shuffle


class DataGenerator(ABC):

    def __init__(self, dataset_dir, config, evals_allowed_in_train, entity_list=True):
        self.dir = dataset_dir
        self.conf = config
        self.evals_allowed_in_train = evals_allowed_in_train

        self.relations = ['r' + str(i) for i in range(self.conf.RELATIONTYPE_AMOUNT)]
        relations_split = int(self.conf.ratio_of_pattern_relations * len(self.relations))
        self.pattern_relations, self.random_relations = numpy.split(self.relations, [relations_split])
        self.entities = ['e' + str(i) for i in range(self.conf.ENTITYTYPE_AMOUNT)] if entity_list else set()
        self.subj_rel2obj_train, self.subj_rel2obj_eval = defaultdict(list), defaultdict(list)
        self.rand_subj_rel2obj_train, self.rand_subj_rel2obj_eval = defaultdict(list), defaultdict(list)

        self.clear_files()

    def create_dataset(self):
        for _ in range(self.conf.NUMBER_RULES):
            relation = sample(list(self.pattern_relations), 1)[0]
            complete_facts = self.create_complete_facts(relation)
            split_pos = int(self.conf.ratio_of_complete_patterns * len(complete_facts))
            train, eval = self.split(complete_facts, split_pos)
            self.write(train, 'train', self.subj_rel2obj_train)
            self.write(eval, 'eval', self.subj_rel2obj_eval)
        for _ in range(self.conf.NUMBER_RULES):
            relation = sample(list(self.random_relations), 1)[0]
            rand_train, rand_eval = self.create_incomplete_patterns(relation)
            self.write(rand_train, 'rand_train', self.rand_subj_rel2obj_train)
            self.write(rand_eval, 'rand_eval', self.rand_subj_rel2obj_eval)

        json.dump(self.rand_subj_rel2obj_train,
                  open(os.path.join(self.dir, 'rand_subject_relation2object_train.json'), 'w'))
        json.dump(self.rand_subj_rel2obj_eval,
                  open(os.path.join(self.dir, 'rand_subject_relation2object_eval.json'), 'w'))
        json.dump(self.subj_rel2obj_train, open(os.path.join(self.dir, 'subject_relation2object_train.json'), 'w'))
        json.dump(self.subj_rel2obj_eval, open(os.path.join(self.dir, 'subject_relation2object_eval.json'), 'w'))
        self.create_vocab()

    def clear_files(self):
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)

    def create_vocab(self):
        vocab = ["[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"] + self.relations + list(self.entities)
        path_to_vocab = self.dir.replace('datasets', 'vocab')
        os.makedirs(path_to_vocab, exist_ok=True)
        with open(os.path.join(path_to_vocab, 'vocab.txt'), 'w') as txt_file:
            for voc in vocab:
                txt_file.write(voc + '\n')

    def split(self, facts, split_pos):
        train, eval = numpy.split(facts, [split_pos])
        flatten_train = train.reshape(train.shape[0] * train.shape[1], train.shape[2])
        train, to_be_masked = train.tolist(), eval.tolist()
        for chain in to_be_masked:
            train.append(chain[:-1])
        to_be_masked = list(filter(lambda x: self.check_train(x[2], flatten_train.tolist(), 1), to_be_masked))
        return train, to_be_masked

    def write(self, triples, type, obj_dict):
        if 'train' in type and self.conf.shuffle_train:
            shuffle(triples)
        with open(os.path.join(self.dir, f'{type if type != "rand_train" else type.replace("rand_", "")}.txt'),
                  'a') as file, \
                open(os.path.join(self.dir, f'masked_{type}.txt'), 'a') as masked_file:
            for triple in triples:
                if 'train' in type:
                    for _ in range(numpy.random.randint(1, self.conf.MAX_INSTANCES_PER_FACT + 1)):
                        file.write(self.build_string(triple, mask=False))
                else:
                    file.write(self.build_string(triple, mask=False))
                if 'train' in type and self.conf.train_eval_random_subset:
                    if numpy.random.rand() < 0.1:
                        masked_file.write(self.build_string(triple, mask=True))
                else:
                    masked_file.write(self.build_string(triple, mask=True))

                obj_dict[self.build_string(triple, mask=True).replace(' [MASK]\n', '')].append(triple[-1][-1])

    def build_string(self, array, mask):
        if mask:
            return ' '.join(array[:-1]) + ' [MASK]\n'
        else:
            return ' '.join(array) + '\n'

    def check_train(self, x, train, minus):
        subj_rel = [fact[:-1] for fact in train]
        return subj_rel.count(x[:-1]) == self.evals_allowed_in_train - minus

    @abstractmethod
    def create_complete_facts(self, relation):
        pass

    @abstractmethod
    def create_incomplete_patterns(self, relation):
        pass
