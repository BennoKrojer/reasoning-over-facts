import json
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from random import sample

import numpy
from numpy.random import shuffle


class NegationDataGenerator(ABC):

    def __init__(self, dataset_dir, config, evals_allowed_in_train, complete_pattern_per_line=False,
                 numb_left_out_for_eval=1, entity_list=True):
        self.dir = dataset_dir
        self.conf = config
        self.evals_allowed_in_train = evals_allowed_in_train
        self.complete_pattern_per_line = complete_pattern_per_line
        self.leave_out = numb_left_out_for_eval

        self.relations = ['r' + str(i) for i in range(self.conf.RELATIONTYPE_AMOUNT)]
        relations_split = int(self.conf.ratio_of_pattern_relations * len(self.relations))
        self.pattern_relations, self.random_relations = numpy.split(self.relations, [relations_split])
        self.entities = ['e' + str(i) for i in range(self.conf.ENTITYTYPE_AMOUNT)] if entity_list else set()
        self.attributes = [('a' + str(i), 'a' + str(i+1)) for i in range(0,  self.conf.ATTRIBUTE_AMOUNT, 2)]
        self.random_attributes = ['b' + str(i) for i in range(self.conf.ATTRIBUTE_AMOUNT)]
        self.subj_rel2obj_train, self.subj_rel2obj_eval = defaultdict(list), defaultdict(list)
        self.rand_subj_rel2obj_train, self.rand_subj_rel2obj_eval = defaultdict(list), defaultdict(list)
        self.clear_files()
        self.classification_file = open(Path(self.dir)/'classification.tsv', 'a')
        self.classification_file.write('fact\tlabel\tsource\n')

    def create_dataset(self):
        self.create_vocab()
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
        # for relations in zip(self.anti_relations[::2], self.anti_relations[1::2]):
        #     anti_train, anti_eval = self.create_anti_patterns(relations)
        #     self.write(anti_train, 'anti_train', self.anti_subj_rel2obj_train)
        #     self.write(anti_eval, 'anti_eval', self.anti_subj_rel2obj_eval)

        json.dump(self.rand_subj_rel2obj_train, open(os.path.join(self.dir, 'rand_subject_relation2object_train.json'), 'w'))
        json.dump(self.rand_subj_rel2obj_eval, open(os.path.join(self.dir, 'rand_subject_relation2object_eval.json'), 'w'))
        json.dump(self.subj_rel2obj_train, open(os.path.join(self.dir, 'subject_relation2object_train.json'), 'w'))
        json.dump(self.subj_rel2obj_eval, open(os.path.join(self.dir, 'subject_relation2object_eval.json'), 'w'))
        self.classification_file.close()

    def clear_files(self):
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)

    def create_vocab(self):
        vocab = ["[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]", "not"] + self.relations + self.entities + self.random_attributes
        for a,b in self.attributes:
            vocab.append(a)
            vocab.append(b)
        path_to_vocab = self.dir.replace('datasets', 'vocab')
        os.makedirs(path_to_vocab, exist_ok=True)
        with open(os.path.join(path_to_vocab, 'vocab.txt'), 'w') as txt_file:
            for voc in vocab:
                txt_file.write(voc + '\n')

    @abstractmethod
    def create_complete_facts(self, relation):
        pass

    def split(self, facts, split_pos):
        train, eval = numpy.split(facts, [split_pos])
        flatten_train = train.reshape(train.shape[0]*train.shape[1], train.shape[2])
        if self.complete_pattern_per_line:
            train, to_be_masked = train.tolist(), eval.tolist()
            for chain in to_be_masked:
                train.append(chain[:-1])
            to_be_masked = list(filter(lambda x: self.check_train(x[2], flatten_train.tolist(), 1), to_be_masked))
        else:
            train = flatten_train
            train, eval = train.tolist(), eval.tolist()
            to_be_masked = []
            for chain in eval:
                train += chain[:-1]
                to_be_masked.append(chain[-1])
            to_be_masked = list(filter(lambda x: self.check_train(x, train, 0), to_be_masked))
            self.write_tsv(train, to_be_masked)
        return train, to_be_masked

    def write_tsv(self, train, to_be_masked):
        for fact in train:
            fact = ' '.join(fact)
            neg_fact = fact.replace('not ', '') if 'not' in fact else fact.replace('r', 'not r')
            self.classification_file.write(f'{fact}\t1\ttrain\n')
            self.classification_file.write(f'{neg_fact}\t0\ttrain\n')
        for fact in to_be_masked:
            fact = ' '.join(fact)
            neg_fact = fact.replace('not ', '') if 'not' in fact else fact.replace('r', 'not r')
            self.classification_file.write(f'{fact}\t1\teval\n')
            self.classification_file.write(f'{neg_fact}\t0\teval\n')

    def write(self, triples, type, obj_dict):
        if 'train' in type and self.conf.shuffle_train:
            shuffle(triples)
        with open(os.path.join(self.dir, f'{type if "_train" not in type else "train"}.txt'),
                  'a') as file, open(os.path.join(self.dir, f'masked_{type}.txt'), 'a') as masked_file:
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

                obj_dict[self.build_string(triple, mask=True).replace(' [MASK]\n', '')].append(
                    triple[-1][-1] if self.complete_pattern_per_line else triple[-1])

    def build_string(self, array, mask):
        if self.complete_pattern_per_line:
            if mask:
                return ' '.join(' [SEP] '.join([' '.join(sub_array) for sub_array in array]).split()[:-1]) + ' [MASK]\n'
            else:
                return ' [SEP] '.join([' '.join(sub_array) for sub_array in array]) + '\n'
        else:
            if mask:
                return ' '.join(array[:-1]) + ' [MASK]\n'
            else:
                return ' '.join(array) + '\n'

    def check_train(self, x, train, minus):
        subj_rel = [fact[:-1] for fact in train]
        return subj_rel.count(x[:-1]) == self.evals_allowed_in_train - minus

    @abstractmethod
    def create_incomplete_patterns(self, relation):
        pass

    # @abstractmethod
    # def create_anti_patterns(self, relation):
    #     pass
