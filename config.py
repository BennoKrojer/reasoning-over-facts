import os

supported_relations = ['symmetry', 'size_1to1', 'transitive', 'reflexive', 'negation',
                       'inverse', 'compositional', 'groupings', 'order', 'hierarchy', 'implication', 'world',
                       'transitive_enhanced']

project_base_dir = os.path.dirname(os.path.realpath(__file__))
scripts_dir = os.path.join(project_base_dir, 'scripts')
data_dir = os.path.join(project_base_dir, 'data')
output_dir = os.path.join(project_base_dir, 'output')

scripts_dirs = {relation: os.path.join(scripts_dir, relation) for relation in supported_relations}
datasets_dirs = {relation: os.path.join(data_dir, relation, 'datasets') for relation in supported_relations}
vocab_dirs = {relation: os.path.join(data_dir, relation, 'vocab') for relation in supported_relations}
models_dirs = {relation: os.path.join(output_dir, 'models', relation) for relation in supported_relations}
runs_dirs = {relation: os.path.join(output_dir, 'runs', relation) for relation in supported_relations}

paths = (list(scripts_dirs.values()) + list(datasets_dirs.values()) + list(vocab_dirs.values())
         + list(models_dirs.values()) + list(runs_dirs.values()))
for path in paths:
    os.makedirs(path, exist_ok=True)

documentation_dir = os.path.join(project_base_dir, 'Documentation')

datagen_config_size_1to1 = os.path.join(scripts_dirs['size_1to1'], 'datagen_config.py')
symmetry_config = os.path.join(scripts_dirs['symmetry'], 'datagen_config.py')
transitive_config = os.path.join(scripts_dirs['transitive'], 'datagen_config.py')
reflexive_config = os.path.join(scripts_dirs['reflexive'], 'datagen_config.py')
groupings_config = os.path.join(scripts_dirs['groupings'], 'datagen_config.py')
order_config = os.path.join(scripts_dirs['order'], 'datagen_config.py')
hierarchy_config = os.path.join(scripts_dirs['hierarchy'], 'datagen_config.py')
implication_config = os.path.join(scripts_dirs['implication'], 'datagen_config.py')
negation_config = os.path.join(scripts_dirs['negation'], 'datagen_config.py')
world_config = os.path.join(scripts_dirs['world'], 'datagen_config.py')
enhanced_config = os.path.join(scripts_dirs['transitive_enhanced'], 'datagen_config.py')

cuda_visible_device = '0'
