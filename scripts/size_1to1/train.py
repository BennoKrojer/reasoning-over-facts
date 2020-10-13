import os
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument('--relation', type=str, help=f'relation type that is trained on. Supported relations: {", ".join(config.supported_relations)}')
parser.add_argument('--dataset_name', type=str, help='dataset used for train, eval and vocab')
parser.add_argument('--output_model_name', type=str, help='Name of the model to save')
parser.add_argument('--lr', type=float, default='6e-5', help='Learning rate. Default is 6e-5')
parser.add_argument('--epochs', type=int, default='2000', help='Default is 2000 epochs')
parser.add_argument('--batch_size', type=int, default='514', help='Default is batch size of 256')
parser.add_argument('--eval_batch_size', type=int, default='514', help='Default is batch size of 256')
parser.add_argument('--config_name', type=str, default='./bert_config.json', help='path to the bert_config file')
# parser.add_argument('--eval_on_train', type=bool, default=True, help='Whether we apply eval metrics on train set as well') #TODO integrate properly

args = parser.parse_args()

command = f'python3 -m scripts.run_language_modeling' \
          f' --train_data_file data/{args.relation}/datasets/{args.dataset_name}/train.txt' \
          f' --eval_data_file data/{args.relation}/datasets/{args.dataset_name}/masked_eval.txt' \
          f' --correct_answers_eval data/{args.relation}/datasets/{args.dataset_name}/subject_relation2object_eval.json' \
          f' --correct_answers_train data/{args.relation}/datasets/{args.dataset_name}/subject_relation2object_train.json' \
          f' --masked_train_file data/{args.relation}/datasets/{args.dataset_name}/masked_train.txt' \
          f' --output_dir output/models/{args.relation}/{args.dataset_name}' \
          ' --model_type bert' \
          ' --mlm' \
          f' --tokenizer_name data/{args.relation}/vocab/{args.dataset_name}/' \
          ' --do_train' \
          ' --do_eval' \
          f' --learning_rate {args.lr}' \
          f' --num_train_epochs {args.epochs}' \
          ' --save_total_limit 2' \
          ' --save_steps 10000' \
          f' --per_gpu_train_batch_size {args.batch_size}' \
          f' --per_gpu_eval_batch_size {args.eval_batch_size}' \
          ' --evaluate_during_training' \
          ' --seed 42' \
          ' --line_by_line' \
          ' --logging_steps 10000' \
          ' --overwrite_output_dir' \
          f' --eval_mode {"size_1to1"}' \
          f' --config_name data/{args.relation}/vocab/{args.dataset_name}/bert_config.json'

print(command)
os.system(command)
