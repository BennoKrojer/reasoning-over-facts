import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument('--relation', '-r', type=str, help=f'relation type that is trained on. Supported relations: {", ".join(config.supported_relations)}')
parser.add_argument('--dataset_name', '-d', type=str, help='dataset used for train, eval and vocab')
parser.add_argument('--output_model_name','-o', type=str, help='Name of the model to save')
parser.add_argument('--lr', type=float, default='6e-5', help='Learning rate. Default is 6e-5')
parser.add_argument('--epochs', type=int, default='600', help='Default is 600 epochs')
parser.add_argument('--batch_size', type=int, default='1024', help='Default is batch size of 256')
parser.add_argument('--eval_batch_size', type=int, default='256', help='Default is batch size of 256')
parser.add_argument('--logging_steps', type=int, default='100', help='After how many batches metrics are logged')
#TODO integrate properly
# parser.add_argument('--eval_on_train', type=bool, default=True, help='Whether we apply eval metrics on train set as well')

args = parser.parse_args()

command = f'python3 -m scripts.groupings.run_language_modeling' \
          f' --train_data_file data/{args.relation}/datasets/{args.dataset_name}/train.txt' \
          f' --eval_data_file data/{args.relation}/datasets/{args.dataset_name}/masked_eval.txt' \
          f' --correct_answers_eval data/{args.relation}/datasets/{args.dataset_name}/subject_relation2object_eval.json' \
          f' --correct_answers_train data/{args.relation}/datasets/{args.dataset_name}/subject_relation2object_train.json' \
          f' --rand_correct_answers_train data/{args.relation}/datasets/{args.dataset_name}/rand_subject_relation2object_train.json' \
          f' --unsupported_answers_eval data/{args.relation}/datasets/{args.dataset_name}/rand_subject_relation2object_eval.json' \
          f' --masked_train_file data/{args.relation}/datasets/{args.dataset_name}/masked_train.txt' \
          f' --masked_rand_train_file data/{args.relation}/datasets/{args.dataset_name}/masked_rand_train.txt' \
          f' --masked_rand_eval_file data/{args.relation}/datasets/{args.dataset_name}/masked_rand_eval.txt' \
          f' --output_dir output/models/{args.relation}/{args.output_model_name}' \
          ' --model_type bert' \
          ' --mlm' \
          f' --tokenizer_name data/{args.relation}/vocab/{args.dataset_name}/' \
          ' --do_train' \
          ' --do_eval' \
          f' --learning_rate {args.lr}' \
          f' --num_train_epochs {args.epochs}' \
          ' --save_total_limit 2' \
          ' --save_steps 100' \
          f' --per_gpu_train_batch_size {args.batch_size}' \
          f' --per_gpu_eval_batch_size {args.eval_batch_size}' \
          ' --evaluate_during_training' \
          ' --seed 42' \
          ' --line_by_line' \
          f' --logging_steps {args.logging_steps}' \
          f' --eval_mode {args.relation}'

print(command)
os.system(command)
