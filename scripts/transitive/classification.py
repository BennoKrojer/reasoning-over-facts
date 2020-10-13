import json
import os
import pandas
import torch
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import config
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import argparse
from transformers import BertConfig
import numpy as np
import random
import time
import datetime
# Set the seed value all over the place to make this reproducible.
seed_val = 42
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# following https://mccormickml.com/2019/07/22/BERT-fine-tuning/#21-download--extract

parser = argparse.ArgumentParser()
parser.add_argument('--from_pretrained', action='store_true', help='train from StandardTrans model')
parser.add_argument('--output_model_name', '-o', required=True, type=str, help='Name of the model you are fine-tuning')
args = parser.parse_args()

device = torch.device('cuda')

tokenizer = BertTokenizer.from_pretrained(os.path.join(config.vocab_dirs['transitive'], 'StandardTrans'))
df = pandas.read_csv(os.path.join(config.datasets_dirs['transitive'], 'classification.tsv'), sep='\t',
                     index_col='source')
train_df = df.loc['train']
eval_df = df.loc['eval']
unsupported_df = df.loc['unsupported']
train_facts = train_df.fact.values
eval_facts = eval_df.fact.values
unsupported_facts = unsupported_df.fact.values
train_labels = train_df.label.values
eval_labels = eval_df.label.values
unsupported_labels = unsupported_df.label.values

train_input_ids = [tokenizer.encode(fact, add_special_tokens=True) for fact in train_facts]  # Add '[CLS]' and '[SEP]'
eval_input_ids = [tokenizer.encode(fact, add_special_tokens=True) for fact in eval_facts]  # Add '[CLS]' and '[SEP]'
unsupported_input_ids = [tokenizer.encode(fact, add_special_tokens=True) for fact in
                         unsupported_facts]  # Add '[CLS]' and '[SEP]'
train_attention_masks = [[1] * 5 for fact in train_facts]
eval_attention_masks = [[1] * 5 for fact in eval_facts]
unsupported_attention_masks = [[1] * 5 for fact in unsupported_facts]

eval_unsupported_ids, train_unsupported_ids = np.split(np.array(unsupported_input_ids), [len(eval_facts)])
eval_unsupported_labels, train_unsupported_labels = np.split(np.array(unsupported_labels), [len(eval_facts)])
eval_unsupported_masks, train_unsupported_masks = np.split(np.array(unsupported_attention_masks),
                                                              [len(eval_facts)])

"""train_inputs = torch.tensor(np.concatenate([train_input_ids, train_unsupported_ids]))
validation_inputs = torch.tensor(eval_input_ids)
train_labels = torch.tensor(np.concatenate([train_labels, train_unsupported_labels]))
validation_labels = torch.tensor(eval_labels)

train_masks = torch.tensor(np.concatenate([train_attention_masks, train_unsupported_masks]))
validation_masks = torch.tensor(eval_attention_masks)"""


train_inputs = torch.tensor(np.concatenate([train_input_ids, train_unsupported_ids]))
validation_inputs = torch.tensor(np.concatenate([eval_input_ids, eval_unsupported_ids]))
train_labels = torch.tensor(np.concatenate([train_labels, train_unsupported_labels]))
validation_labels = torch.tensor(np.concatenate([eval_labels, eval_unsupported_labels]))

train_masks = torch.tensor(np.concatenate([train_attention_masks, train_unsupported_masks]))
validation_masks = torch.tensor(np.concatenate([eval_attention_masks, eval_unsupported_masks]))

# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
# Number of training epochs (authors recommend between 2 and 4)
epochs = 30
batch_size = 512

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
if args.from_pretrained:
    model = BertForSequenceClassification.from_pretrained(
        os.path.join(config.models_dirs['transitive'], 'HighTrans_big', 'checkpoint-8900'),
        num_labels=2,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    print('LOADED FROM PRE-TRAINED')
else:
    model = BertForSequenceClassification(config=BertConfig())
    print('LOADED FROM SCRATCH')
model.cuda()
optimizer = AdamW(model.parameters(),
                  lr=4e-5,  # args.learning_rate - default is 5e-5, original notebook has 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


# Function to calculate the accuracy of our predictions vs labels
def flat_accurate(preds, labels, input_ids):
    print(preds)
    pred_flat = np.argmax(preds, axis=1).flatten()
    print(pred_flat)
    labels_flat = labels.flatten()
    correct = pred_flat == labels_flat

    tp_idx = np.logical_and(pred_flat, labels_flat)
    fn_idx = np.logical_and(1- pred_flat, labels_flat)
    tp_input = input_ids[tp_idx]
    fn_input = input_ids[fn_idx]
    print(len(input_ids))
    print(len(tp_input))
    print(len(fn_input))
    return np.sum(correct), tp_input, fn_input


# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

logging = {'Eval Accuracy': [], 'Training Loss': []}
tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    logging['Training Loss'].append(avg_train_loss)
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    true_pos = []
    false_neg = []
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy, tp_ids, fn_ids = flat_accurate(logits, label_ids, b_input_ids)
        tp_facts = [tokenizer.convert_ids_to_tokens(fact) for fact in tp_ids]
        fn_facts = [tokenizer.convert_ids_to_tokens(fact) for fact in fn_ids]
        true_pos += tp_facts
        false_neg += fn_facts

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += label_ids.shape[0]

        if epoch_i == epochs - 1:
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            try:
                tmp_tn, tmp_fp, tmp_fn, tmp_tp = confusion_matrix(labels_flat, pred_flat).ravel()
            except ValueError:  # workaround because confusion_matrix() with only zeros returns 1x1 matrix
                if (labels_flat == 1).sum() == batch_size:
                    tmp_tp = batch[0].shape[0]
                    tmp_tn, tmp_fn, tmp_fp = 0, 0, 0
                else:
                    tmp_tn = batch[0].shape[0]
                    tmp_tp, tmp_fn, tmp_fp = 0, 0, 0
            tn += tmp_tn
            fp += tmp_fp
            fn += tmp_fn
            tp += tmp_tp

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    logging['Eval Accuracy'].append(eval_accuracy / nb_eval_steps)
    facts_dir = os.path.join(config.runs_dirs['transitive'], 'FineTuning')
    with open(os.path.join(facts_dir, 'tp'), 'w') as tp_file, open(os.path.join(facts_dir, 'fn'), 'w') as fn_file:
        for tp_fact in true_pos:
            tp_file.write(' '.join(tp_fact) + '\n')
        for fn_fact in false_neg:
            fn_file.write(' '.join(fn_fact) + '\n')

logging.update({'TP': tp, 'TN': tn, 'FN': fn, 'FP': fp})

save_dir = os.path.join(config.models_dirs['transitive'], args.output_model_name)
os.makedirs(save_dir, exist_ok=True)
try:
    model.save_pretrained(save_dir)
except:
    pass
json.dump(logging, open(os.path.join(config.runs_dirs['transitive'], 'FineTuning', args.output_model_name + '.json'), 'w'))
print("")
print("Training complete!")
