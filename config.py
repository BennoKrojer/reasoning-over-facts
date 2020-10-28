import os

supported_relations = ['symmetry', 'negation', 'inverse', 'composition',
                       'implication', 'equivalence', 'composition_enhanced']

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

symmetry_config = os.path.join(scripts_dirs['symmetry'], 'datagen_config.py')
equivalence_config = os.path.join(scripts_dirs['equivalence'], 'datagen_config.py')
implication_config = os.path.join(scripts_dirs['implication'], 'datagen_config.py')
inverse_config = os.path.join(scripts_dirs['inverse'], 'datagen_config.py')
negation_config = os.path.join(scripts_dirs['negation'], 'datagen_config.py')
composition_config = os.path.join(scripts_dirs['composition'], 'datagen_config.py')
enhanced_config = os.path.join(scripts_dirs['composition_enhanced'], 'datagen_config.py')