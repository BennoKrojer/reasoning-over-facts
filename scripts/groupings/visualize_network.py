import os
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

import config
from scripts.groupings.predict import predict


def load_networks(train_path):
    lines = numpy.array([line.strip().split() for line in open(train_path, 'r')])
    lines = numpy.array([line for line in lines if int(line[1][1:]) <= 39])
    lines = numpy.array(numpy.split(lines, 40))
    lines = lines[:, 72*30:]
    lines = numpy.array([numpy.split(relation_pack, 8) for relation_pack in lines])
    return lines


def align_preds(eval, predictions):
    lens = []
    for e in eval[0]:
        subrel = set()
        for fact in e:
            subrel.add(fact[:2])
        lens.append(len(subrel))
    structured_preds = []
    for rel_numb in range(len(eval)):
        pred_rel = []
        for i, l in enumerate(lens):
            pred_network = []
            for numb_probes in range(l):
                pred_network.append(predictions.pop(0))
            pred_rel.append(pred_network)
        structured_preds.append(pred_rel)
    return structured_preds


def get_node_name(node, nodes):
    if node in nodes:
        node = nodes.index(node)
    else:
        nodes.append(node)
        node = len(nodes)-1
    return node, nodes


unstructured = True
dataset_name = 'StandardGroup_with_anti_controlled'
dataset = Path(config.datasets_dirs['groupings'])/dataset_name
model_name = 'Standard_controlled_balanced'
checkpoint = 'checkpoint-101100'
output = Path(config.documentation_dir)/'Standards'/'groupings'/'controlled_balanced'/'networks'
output.mkdir(exist_ok=True)
model = BertForMaskedLM.from_pretrained(
        os.path.join(config.output_dir, 'models', 'groupings', model_name, checkpoint))
tokenizer = BertTokenizer.from_pretrained(str(Path(config.vocab_dirs['groupings'])/dataset_name))
masked_file = str(Path(dataset)/'masked_eval.txt')
correct_json_file = str(Path(dataset)/'subject_relation2object_eval.json')

if unstructured:
    networks = pickle.load(open(os.path.join(config.datasets_dirs['groupings'],
                'StandardGroup_with_anti_controlled', 'incomplete_networks.pkl'), 'rb'))
    eval = pickle.load(open(os.path.join(config.datasets_dirs['groupings'],
                'StandardGroup_with_anti_controlled', 'eval.pkl'), 'rb'))
    predictions = predict('groupings', model, tokenizer, masked_file, correct_json_file, unstructured)
    predictions = align_preds(eval, predictions)
else:
    networks = load_networks(str(Path(dataset)/'train.txt'))
    predictions = numpy.array(numpy.split(numpy.array(predict('groupings', model, tokenizer, masked_file,
                                                              correct_json_file, unstructured)), 40))
no_sym_accs = defaultdict(int)
no_sym_total = defaultdict(int)
for i, r in tqdm(enumerate(networks)):
    r_preds = predictions[i]
    for j, network in enumerate(r):
        total = 0
        acc = 0
        preds = r_preds[j]
        G_train = nx.DiGraph()
        nodes = []
        sym = []
        for node1, rel, node2 in network:
            network = network.tolist() if hasattr(network, 'tolist') else network
            if [node2, rel, node1] in network:
                color = 'b'
            else:
                color = 'orange'
                sym.append((node1, node2))
            node1, nodes = get_node_name(node1, nodes)
            node2, nodes = get_node_name(node2, nodes)
            G_train.add_edge(node1, node2, color=color)
        pos = nx.circular_layout(G_train)
        edges = G_train.edges()
        colors = [G_train[u][v]['color'] for u, v in edges]
        nx.draw(G_train, pos, edge_color=colors, with_labels=True)
        plt.savefig(output/f'r{i}_n{j}_train.png')
        plt.clf()
        G_correct = nx.DiGraph()
        G_correct.add_nodes_from(list(range(len(nodes))))
        G_false = nx.DiGraph()
        G_false.add_nodes_from(list(range(len(nodes))))
        for probe in preds:
            subj = probe['probe'].split()[0]
            correct = probe['correct']
            pred_array = probe['predictions']
            for c in correct:
                if c in pred_array:
                    color = 'orange' if (c, subj) in sym else 'g'
                    if color == 'g':
                        acc += 1
                        total += 1
                    G_correct.add_edge(get_node_name(subj, nodes)[0], get_node_name(c, nodes)[0], color=color)
                else:
                    color = 'orange' if (c, subj) in sym else 'r'
                    if color == 'r':
                        total += 1
                    G_false.add_edge(get_node_name(subj, nodes)[0], get_node_name(c, nodes)[0], color=color)
        no_sym_accs[j] += acc
        no_sym_total[j] += total
        pos = nx.circular_layout(G_train)
        edges = G_correct.edges()
        colors = [G_correct[u][v]['color'] for u, v in edges]
        nx.draw(G_correct, pos, edge_color=colors, with_labels=True)
        plt.savefig(output/f'r{i}_n{j}_correct.png')
        plt.clf()
        pos = nx.circular_layout(G_train)
        edges = G_false.edges()
        colors = [G_false[u][v]['color'] for u, v in edges]
        nx.draw(G_false, pos, edge_color=colors, with_labels=True)
        plt.savefig(output/f'r{i}_n{j}_false.png')
        plt.close('all')

print(no_sym_accs)
print(no_sym_total)
print([{i: (acc/total) if total > 0 else 1} for (i, acc), (_, total) in zip(no_sym_accs.items(), no_sym_total.items())])