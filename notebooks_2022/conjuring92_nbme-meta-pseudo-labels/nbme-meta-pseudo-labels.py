#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# * This notebook demonstrates a way to apply meta-pseudo-labels for NBME token classification task
# * It uses code snippets from: 
#     * https://www.kaggle.com/code/hengck23/playground-for-meta-pseudo-label
#     * https://www.kaggle.com/code/theoviel/evaluation-metric-folds-baseline
# 
# 
# * Meta Pseudo Labels paper:
#     * https://arxiv.org/abs/2003.10580
#     
#     
# * Thanks @hengck23, @theoviel for sharing these great notebooks!
# * Hope that you find this approach useful. Please let me know if there are mistakes. 
# * Looking forward to your comments / feedback / queries. Thanks!

# # Imports

# In[ ]:


import ast
import json
import itertools
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from IPython.core.debugger import set_trace
from IPython.display import display
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          get_cosine_schedule_with_warmup, get_scheduler)
from transformers.trainer_pt_utils import get_parameter_names

pd.options.display.max_colwidth = None


# In[ ]:


get_ipython().system('conda list | grep cudatoolkit')


# In[ ]:


get_ipython().run_line_magic('pip', 'install -qq bitsandbytes-cuda110 # optimizer')
get_ipython().run_line_magic('pip', 'install -qq pynvml')
get_ipython().run_line_magic('pip', 'install -qq accelerate')
get_ipython().run_line_magic('pip', 'install -qq datasets')


# In[ ]:


import bitsandbytes as bnb
from accelerate import Accelerator
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from datasets import Dataset, load_from_disk


# # Config

# In[ ]:


config = {
    "debug": False,
    
    "num_layers_in_head": 6, # Number of layer hidden states to be used in token classification head
    "mtl_tok_num_labels": 3, # Predict Inside, Begin, End of answer spans
    "gradient_checkpointing": True,
    "mixed_precision": True,
    "n_freeze": 2,
    
    "batch_size": 8,
    "train_folds": [0, 1, 2, 3],
    "valid_folds": [4],
    
    "num_unlabelled": 10000, # number of unlabelled data points to be used for Meta Pseudo Labels

    "base_model_path": "microsoft/deberta-large", # backbone
    "student_model_dir": "./trained_student", # save dir for student model
    "teacher_model_dir": "./trained_teacher", # save dir for teacher model
    "teacher_save_name": "teacher", # teacher checkpoint 
    "student_save_name": "student", # student checkpoint
    
    # data path
    "data_dir": "../input/nbme-score-clinical-patient-notes",
    "train_path": "train.csv",
    "features_path": "features.csv",
    "notes_path": "patient_notes.csv",
    
    # save tokenized datasets here
    "output_dir": "./",
    "train_dataset_path": "train_dataset",
    "valid_dataset_path": "valid_dataset",
    
    # column names
    "text_col": "pn_history",
    "feature_col": "feature_text",
    "label_col": "label_spans",
    "annotation_col": "annotation",
    "text_sequence_identifier": 1,
    "max_length": 480,
    
    # optimizer & scheduler params
    "weight_decay": 1e-3,
    "lr": 2e-5,
    "eps": 1e-6,
    "beta1": 0.9,
    "beta2": 0.99,
    "num_epochs": 1,
    "grad_accumulation": 1,
    "warmup_pct": 0.02,
    "validation_interval": 200
}


# # Utils

# In[ ]:


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
print_gpu_utilization()


# # Data

# In[ ]:


data_dir = config["data_dir"]
train_df = pd.read_csv(os.path.join(data_dir, config["train_path"]))
features_df = pd.read_csv(os.path.join(data_dir, config["features_path"]))
notes_df = pd.read_csv(os.path.join(data_dir, config["notes_path"]))


# # Process Data

# In[ ]:


def process_feature_text(text):
    return re.sub('-', ' ', text)

def apply_ast(df):
    columns = [
        "annotation",
        "location",
    ]

    for col in columns:
        try:
            if type(df[col].values[0]) != list:
                df[col] = df[col].apply(ast.literal_eval)
        except Exception as e:
            print(e)
            traceback.print_exc()

    return df

def location2spans(location):
    """
    a helper function to compute the label spans from the input location list
    """
    spans = [loc.split(";") for loc in location]
    spans = [list(map(int, s.split())) for s in chain(*spans)]
    return spans

features_df["feature_text"] = features_df["feature_text"].apply(process_feature_text)
train_df = apply_ast(train_df)
train_df = pd.merge(train_df, features_df, on=['feature_num', 'case_num'], how='left')
train_df = pd.merge(train_df, notes_df, on=['pn_num', 'case_num'], how='left')
train_df['label_spans'] = train_df['location'].apply(location2spans)


# In[ ]:


train_df.sample()


# # Unlabelled Data

# In[ ]:


train_patients = set(train_df["pn_num"].unique())
unlabelled_notes_df = notes_df[~notes_df["pn_num"].isin(train_patients)].copy()
unlabelled_df = pd.merge(unlabelled_notes_df, features_df, on=['case_num'], how='left')
print(f"# unlabelled examples = {len(unlabelled_df)}\n")
display(unlabelled_df.sample())


# # Train-Validation Split

# In[ ]:


# the code for train-validation split is taken from: 
# https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/315707

def make_fold(df):
    df.loc[:,'fold']=-1
    gkf = GroupKFold(n_splits=5)
    for n, (train_index, valid_index) in enumerate(gkf.split(df['id'], df['location'], df['pn_num'])):
        df.loc[valid_index, 'fold'] = n
    return df

train_df = make_fold(train_df)


# In[ ]:


train_df.sample()


# # Train Valid Dataset

# In[ ]:


df = train_df.copy()

if config["debug"]:
    print("DEBUG Mode: sampling 1024 examples from train data")
    df = df.sample(min(1024, len(df)))
    
train_df = df[df['fold'].isin(config['train_folds'])].copy()
valid_df = df[df['fold'].isin(config['valid_folds'])].copy()


# In[ ]:


def strip_offset_mapping(text, offset_mapping):
    """process offset mapping produced by huggingface tokenizers
    by stripping spaces from the tokens

    :param text: input text that is tokenized
    :type text: str
    :param offset_mapping: offsets returned from huggingface tokenizers
    :type offset_mapping: list
    :return: processed offset mapping
    :rtype: list
    """
    to_return = []
    for start, end in offset_mapping:
        match = list(re.finditer('\S+', text[start:end]))
        if len(match) == 0:
            to_return.append((start, end))
        else:
            span_start, span_end = match[0].span()
            to_return.append((start + span_start, start + span_end))
    return to_return


def get_sequence_ids(input_ids, tokenizer):
    """if a pair of texts are given to HF tokenizers, the first text
    has sequence id of 0 and second text has sequence id 1. This function
    derives sequence ids for a given tokenizer based on token input ids

    :param input_ids: token input id sequence
    :type input_ids: List[int]
    :param tokenizer: HF tokenizer
    :type tokenizer: PreTrainedTokenizer
    :return: sequence ids
    :rtype: List
    """
    sequence_ids = [0]*len(input_ids)

    switch = False
    special_token_ids = set(
        tokenizer.convert_tokens_to_ids(
            tokenizer.special_tokens_map.values()
        )
    )
    for i, input_id in enumerate(input_ids):
        if input_id == tokenizer.sep_token_id:
            switch = True
        if switch:
            sequence_ids[i] = 1
        if input_id in special_token_ids:
            sequence_ids[i] = None
    return sequence_ids


class NbmeMTLDataset:
    """Dataset class for NBME token classification task
    """

    def __init__(self, config):
        self.config = config

        # column names
        self.text_col = self.config["text_col"]
        self.feature_col = self.config["feature_col"]
        self.label_col = self.config["label_col"]
        self.annotation_col = self.config["annotation_col"]

        # sequence number for patient history texts
        self.focus_seq = self.config["text_sequence_identifier"]

        # load tokenizer
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config 
        """
        print("using auto tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_model_path"], trim_offsets=False)

    def tokenize_function(self, examples):
        if self.focus_seq == 0:
            tz = self.tokenizer(
                examples[self.text_col],
                examples[self.feature_col],
                padding=False,
                truncation='only_first',
                max_length=self.config["max_length"],
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
            )

        elif self.focus_seq == 1:
            tz = self.tokenizer(
                examples[self.feature_col],
                examples[self.text_col],
                padding=False,
                truncation='only_second',
                max_length=self.config["max_length"],
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
            )

        else:
            raise ValueError("bad text_sequence_identifier in config")
        return tz

    def add_sequence_ids(self, examples):
        sequence_ids = []
        input_ids = examples["input_ids"]

        for tok_ids in input_ids:
            sequence_ids.append(get_sequence_ids(tok_ids, self.tokenizer))
        return {"sequence_ids": sequence_ids}

    def process_token_offsets(self, examples):
        stripped_offsets, unstripped_offsets = [], []
        prev_offset = None

        for offsets, seq_ids, feature_text, pn_history in zip(
            examples["offset_mapping"],
            examples['sequence_ids'],
            examples[self.feature_col],
            examples[self.text_col]
        ):
            current_stripped, current_unstripped = [],  []

            for pos, offset in enumerate(offsets):
                start, end = offset
                seq_id = seq_ids[pos]

                if seq_id is None:
                    current_stripped.append(offset)
                    current_unstripped.append(offset)
                    prev_offset = offset
                    continue

                elif seq_id == self.focus_seq:
                    focus_text = pn_history[start:end]
                else:
                    focus_text = feature_text[start:end]

                # strip offsets
                match = list(re.finditer('\S+', focus_text))
                if len(match) == 0:
                    current_stripped.append((start, end))
                else:
                    span_start, span_end = match[0].span()
                    current_stripped.append((start + span_start, start + span_end))

                # upstrip offsets
                if prev_offset[-1] != offset[0]:
                    offset[0] = prev_offset[-1]
                current_unstripped.append(offset)
                prev_offset = offset

            stripped_offsets.append(np.array(current_stripped))
            unstripped_offsets.append(np.array(current_unstripped))

        return {
            'offset_mapping_stripped': stripped_offsets,
            "offset_mapping_unstripped": unstripped_offsets
        }

    def generate_labels(self, examples):
        labels = []
        for offsets, inputs, seq_ids, locations in zip(
            examples["offset_mapping_stripped"],
            examples["input_ids"],
            examples["sequence_ids"],
            examples[self.label_col]
        ):
            this_label = np.zeros(shape=(3, len(inputs)))
            for idx, (seq_id, offset) in enumerate(zip(seq_ids, offsets)):
                if seq_id != self.focus_seq:  # ignore this token
                    this_label[:, idx] = -1.0
                    continue

                token_start_char_idx, token_end_char_idx = offset
                for label_start_char_idx, label_end_char_idx in locations:
                    # case 1: location char start is inside token
                    if token_start_char_idx <= label_start_char_idx < token_end_char_idx:
                        this_label[0, idx] = 1.0
                        this_label[1, idx] = 1.0  # detection

                    # case 2: location char end is inside token
                    if token_start_char_idx < label_end_char_idx <= token_end_char_idx:
                        this_label[0, idx] = 1.0
                        this_label[2, idx] = 1.0  # termination

                    # case 3: token in between location
                    if label_start_char_idx < token_start_char_idx < label_end_char_idx:
                        this_label[0, idx] = 1.0

                    # break the loop if token is already detected positive
                    if this_label[0, idx] > 0:
                        break

            labels.append(this_label)
        return {"labels": labels}

    def get_dataset(self, df, mode='train'):
        """main api for creating the NBME dataset

        :param df: input dataframe
        :type df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """

        # create the dataset
        nbme_dataset = Dataset.from_pandas(df)
        nbme_dataset = nbme_dataset.map(self.tokenize_function, batched=True)
        nbme_dataset = nbme_dataset.map(self.add_sequence_ids, batched=True)
        nbme_dataset = nbme_dataset.map(self.process_token_offsets, batched=True)

        if mode == "train":
            nbme_dataset = nbme_dataset.map(self.generate_labels, batched=True)
        try:
            nbme_dataset = nbme_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)
        return nbme_dataset


# In[ ]:


dataset_creator = NbmeMTLDataset(config)
train_ds = dataset_creator.get_dataset(train_df, mode='train')
valid_ds = dataset_creator.get_dataset(valid_df, mode='train')


# In[ ]:


train_ds


# In[ ]:


# save train dataset
train_dataset_path = os.path.join(
    config["output_dir"], config["train_dataset_path"]
)
train_ds.save_to_disk(train_dataset_path)

# save valid dataset
valid_dataset_path = os.path.join(
    config["output_dir"], config["valid_dataset_path"]
)
valid_ds.save_to_disk(valid_dataset_path)


# # Unlabelled Dataset

# In[ ]:


print("loading unlabelled data...")
required_examples = config["num_unlabelled"]
df = unlabelled_df.sample(required_examples)
keep_cols = ["pn_history", "feature_text", "feature_num"]
df = df[keep_cols].copy()
print("unlabelled data loaded and sampled ...")
print(f"shape of sampled unlabelled data: {df.shape}")

########### Create Unlabelled Dataset ##############
dataset_creator = NbmeMTLDataset(config)
unlabelled_ds = dataset_creator.get_dataset(df, mode='infer')


# # DataLoaders

# In[ ]:


@dataclass
class DataCollatorForMTLNeo(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received
    Multitask learning targets will be added to batch and padded
    """
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -1
    return_tensors = "pt"

    def torch_call(self, features):
        label_name = "labels"
        labels = None

        if label_name in features[0].keys():
            labels = [feature[label_name] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:  # e.g. in eval mode
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        # main labels
        num_labels, padded_labels = len(labels[0]), []

        for this_example in labels:
            padding_length = sequence_length - len(this_example[0])
            padding_matrix = (self.label_pad_token_id)*np.ones(shape=(num_labels, padding_length))
            this_example = (np.concatenate([np.array(this_example), padding_matrix], axis=1).T).tolist()
            padded_labels.append(this_example)
        batch[label_name] = padded_labels

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        # cast lables to float for bce loss
        batch[label_name] = batch[label_name].to(torch.float32)
        return batch
    
@dataclass
class DataCollatorForMPL(DataCollatorForTokenClassification):
    """
    Data collator for unlabelled data
    """
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -1
    return_tensors = "pt"

    def torch_call(self, features):
        buffer = [feature["sequence_ids"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        # create masks
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        # main labels
        masks = []  # (batch, seq_len)

        for seq in buffer:
            padding_length = sequence_length - len(seq)
            mask = [seq_id == 1 for seq_id in seq] + [False]*padding_length
            masks.append(mask)
            
        batch['label_mask'] = masks
        batch.pop('sequence_ids', None)

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])
data_collator = DataCollatorForMTLNeo(tokenizer=tokenizer, label_pad_token_id=-1)

train_ds.set_format(
    type=None,
    columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
)

train_dl = DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    collate_fn=data_collator,
    pin_memory=True,
    shuffle=True,
)

valid_ds.set_format(
    type=None,
    columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
)

valid_dl = DataLoader(
    valid_ds,
    batch_size=config["batch_size"],
    collate_fn=data_collator,
    pin_memory=True,
    shuffle=False,
)


# In[ ]:


data_collator = DataCollatorForMPL(tokenizer=tokenizer)

unlabelled_ds.set_format(
    type=None,
    columns=['input_ids', 'attention_mask', 'token_type_ids', 'sequence_ids']
)

unlabelled_dl = DataLoader(
    unlabelled_ds,
    batch_size=config["batch_size"],
    collate_fn=data_collator,
    pin_memory=True,
    shuffle=True,
)


# # Model

# In[ ]:


class NbmeMPL(nn.Module):
    """The Multi-task NBME model class for Meta Pseudo Labels
    """

    def __init__(self, config):
        super(NbmeMPL, self).__init__()

        self.config = config

        # base transformer
        self.base_model = AutoModel.from_pretrained(
            self.config["base_model_path"],
        )
        self.base_model.gradient_checkpointing_enable()

        n_freeze = config["n_freeze"]
        if n_freeze > 0:
            print(f"setting requires grad to false for last {n_freeze} layers")
            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:n_freeze].requires_grad_(False)

        hidden_size = self.base_model.config.hidden_size
        num_layers_in_head = self.config["num_layers_in_head"]

        # token classification head
        self.tok_classifier = nn.Linear(
            in_features=hidden_size * num_layers_in_head,
            out_features=self.config['mtl_tok_num_labels'],
        )

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

    def get_logits(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        all_hidden_states = out["hidden_states"]
        # token classification logits
        n = self.config["num_layers_in_head"]
        tok_output = torch.cat(all_hidden_states[-n:], dim=-1)

        # pass through 5 dropout layers and take average
        tok_output1 = self.dropout1(tok_output)
        tok_output2 = self.dropout2(tok_output)
        tok_output3 = self.dropout3(tok_output)
        tok_output4 = self.dropout4(tok_output)
        tok_output5 = self.dropout5(tok_output)
        tok_output = (tok_output1 + tok_output2 + tok_output3 + tok_output4 + tok_output5)/5

        tok_logits = self.tok_classifier(tok_output)

        return tok_logits

    def compute_loss(self, logits, labels, masks):
        loss = F.binary_cross_entropy_with_logits(
            logits, labels,
            reduction='none'
        )
        loss = torch.masked_select(loss, masks).mean()
        return loss


# # Optimizer & Scheduler

# In[ ]:


def get_optimizer(model, config):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    # print(decay_parameters)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # print(decay_parameters)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (config["beta1"], config["beta2"]),
        "eps": config['eps'],
    }

    optimizer_kwargs["lr"] = config["lr"]

    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(config['beta1'], config['beta2']),
        eps=config['eps'],
        lr=config['lr'],
    )

    return adam_bnb_optim


def get_scheduler(optimizer, warmup_steps, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler


# # Train Utils

# In[ ]:


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_checkpoint(config, state, is_teacher, is_best):
    if is_teacher:
        os.makedirs(config["teacher_model_dir"], exist_ok=True)
        name = config["teacher_save_name"]
        filename = f'{config["teacher_model_dir"]}/{name}_last.pth.tar'
    else:
        os.makedirs(config["student_model_dir"], exist_ok=True)
        name = config["student_save_name"]
        filename = f'{config["student_model_dir"]}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        if is_teacher:
            shutil.copyfile(filename, f'{config["teacher_model_dir"]}/{name}_best.pth.tar')
        else:
            shutil.copyfile(filename, f'{config["student_model_dir"]}/{name}_best.pth.tar')


# # Scorer

# In[ ]:


def perform_localization(char_probs, threshold=0.5):
    """convert character wise prediction to location spans

    :param char_probs: character wise predictions
    :type char_prob: list
    :param threshold: threshold for label decision, defaults to 0.5
    :type threshold: float, optional
    :return: locations
    :rtype: list
    """
    results = np.where(char_probs >= threshold)[0]
    results = [list(g) for _, g in itertools.groupby(
        results, key=lambda n, c=itertools.count(): n - next(c))]
    results = [[min(r), max(r)+1] for r in results]
    return results


def postprocess_localization(text, span_offsets):
    """remove spaces at the beginning of label span prediction

    :param span: input text (patient history)
    :type text: str
    :param span_offset: prediction span offsets
    :type offset_mapping: list
    :return: updated span offsets 
    :rtype: list
    """
    to_return = []
    for start, end in span_offsets:
        match = list(re.finditer('\S+', text[start:end]))
        if len(match) == 0:
            to_return.append((start, end))
        else:
            span_start, _ = match[0].span()
            to_return.append((start + span_start, end))
    return to_return


def token2char(text, quantities, offsets, seq_ids, focus_seq=1):
    """convert token prediction/truths to character wise predictions/truths

    :param text: patient notes text
    :type text: str
    :param quantities: token level variable values
    :type quantities: list
    :param offsets: token offsets without stripping
    :type offsets: list
    :param seq_ids: sequence id of the tokens
    :type seq_ids: list
    :param focus_seq: which sequence to focus on
    :type focus_seq: int
    :return: character probabilities
    :rtype: list
    """
    results = np.zeros(len(text))
    for q, offset, seq_id in zip(quantities, offsets, seq_ids):
        if seq_id != focus_seq:
            continue
        char_start_idx, char_end_idx = offset[0], offset[1]
        results[char_start_idx:char_end_idx] = q
    return results

def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1

    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """

    bin_preds = []
    bin_truths = []

    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue

        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))

    return micro_f1(bin_preds, bin_truths)

def scorer(preds, valid_dataset, threshold=0.5, focus_seq=1):
    """scorer for evaluation during training of models

    :param preds: model preds on valid dataset [char probs]
    :type preds: list
    :param valid_dataset: validation dataset
    :type valid_dataset: dataset
    :param threshold: threshold at which model is evaluated, defaults to 0.5
    :type threshold: float, optional
    :param focus_seq: patient notes sequence, defaults to 1
    :type focus_seq: int, optional
    :return: Leaderboard metric
    :rtype: float
    """
    info_df = pd.DataFrame()
    required_cols = [
        "pn_history",
        "sequence_ids",
        "offset_mapping_unstripped",
        "label_spans",
    ]

    for col in required_cols:
        info_df[col] = valid_dataset[col]

    info_df["token_preds"] = preds

    # convert token preds to char preds
    input_cols = ["pn_history", "token_preds", "offset_mapping_unstripped", "sequence_ids"]
    info_df["char_probs"] = info_df[input_cols].apply(
        lambda x: token2char(x[0], x[1], x[2], x[3], focus_seq), axis=1
    )

    #  location
    info_df["location_preds"] = info_df["char_probs"].apply(
        lambda x: perform_localization(x, threshold)
    )
    info_df['location_preds'] = info_df[["pn_history", "location_preds"]].apply(
        lambda x: postprocess_localization(x[0], x[1]), axis=1
    )
    lb = span_micro_f1(info_df["label_spans"].values, info_df["location_preds"].values)
    return lb


# # Training

# In[ ]:


print("=="*40)
print("GPU utilization at the very start:")
print_gpu_utilization()
print("=="*40)

#------- load student and teacher into memory --------------------#
student = NbmeMPL(config)
teacher = NbmeMPL(config)

#------- get optimizers for student and teacher ------------------#
s_optimizer = get_optimizer(student, config)
t_optimizer = get_optimizer(teacher, config)


#------- prepare accelerator  ------------------------------------#
accelerator = Accelerator(fp16=True)
student, teacher, s_optimizer, t_optimizer, train_dl, valid_dl, unlabelled_dl = accelerator.prepare(
    student, teacher, s_optimizer, t_optimizer, train_dl, valid_dl, unlabelled_dl
)
print("=="*40)
print("GPU utilization after accelerator preparation:")
print_gpu_utilization()
print("=="*40)

#------- setup schedulers  ---------------------------------------#
num_epochs = config["num_epochs"]
grad_accumulation_steps = config["grad_accumulation"]
warmup_pct = config["warmup_pct"]

n_train_steps_per_epoch = len(train_dl)//grad_accumulation_steps
n_mpl_steps_per_epoch = len(unlabelled_dl)//grad_accumulation_steps
n_steps_per_epoch = max(n_train_steps_per_epoch, n_mpl_steps_per_epoch)
num_steps = num_epochs * n_steps_per_epoch
num_warmup_steps = int(warmup_pct*num_steps)

s_scheduler = get_scheduler(s_optimizer, num_warmup_steps, num_steps)
t_scheduler = get_scheduler(t_optimizer, num_warmup_steps, num_steps)

#------- Scorer & Trackers ----------------------------------------#
best_teacher_score = 0
best_student_score = 0

valid_ds_path = os.path.join(config["output_dir"], config["valid_dataset_path"])
valid_ds = load_from_disk(valid_ds_path)

scorer_fn = partial(
    scorer,
    valid_dataset=valid_ds,
    threshold=0.5,
    focus_seq=config["text_sequence_identifier"]
)

#------------- Data Iterators -------------------------------------#
train_iter = iter(train_dl)
unlabelled_iter = iter(unlabelled_dl)


# In[ ]:


# ------- Training Loop  ------------------------------------------#
for step in range(num_steps):

    #------ Reset buffers After Validation ------------------------#
    if step % config["validation_interval"] == 0:
        progress_bar = tqdm(range(min(config["validation_interval"], num_steps)))
        s_loss_meter = AverageMeter()
        t_loss_meter = AverageMeter()

    teacher.train()
    student.train()

    t_optimizer.zero_grad()
    s_optimizer.zero_grad()

    #------ Get Train & Unlabelled Batch -------------------------#
    try:
        train_b = train_iter.next()
    except Exception as e:  # TODO: change to stop iteration error
        train_b = next(train_dl.__iter__())

    try:
        unlabelled_b = unlabelled_iter.next()
    except:
        unlabelled_b = next(unlabelled_dl.__iter__())

    #------- Meta Training Steps ---------------------------------#
    # get loss of current student on labelled train data
    s_logits_train_b = student.get_logits(train_b)

    # get loss of current student on labelled train data
    train_b_labels = train_b["labels"]
    train_b_masks = train_b_labels.gt(-0.5)
    s_loss_train_b = student.compute_loss(
        logits=s_logits_train_b.detach(),
        labels=train_b_labels,
        masks=train_b_masks,
    )

    # get teacher generated pseudo labels for unlabelled data
    unlabelled_b_masks = unlabelled_b["label_mask"].eq(1).unsqueeze(-1)
    t_logits_unlabelled_b = teacher.get_logits(unlabelled_b)
    pseudo_y_unlabelled_b = (t_logits_unlabelled_b.detach() > 0).float()  # hard pseudo label

    #------ Train Student: With Pesudo Label Data ------------------#
    s_logits_unlabelled_b = student.get_logits(unlabelled_b)
    s_loss_unlabelled_b = student.compute_loss(
        logits=s_logits_unlabelled_b,
        labels=pseudo_y_unlabelled_b,
        masks=unlabelled_b_masks
    )

    # backpropagation of student loss on unlabelled data
    accelerator.backward(s_loss_unlabelled_b)
    s_optimizer.step()  # update student params
    s_scheduler.step()

    #------ Train Teacher ------------------------------------------#
    s_logits_train_b_new = student.get_logits(train_b)
    s_loss_train_b_new = student.compute_loss(
        logits=s_logits_train_b_new.detach(),
        labels=train_b_labels,
        masks=train_b_masks,
    )
    change = s_loss_train_b_new - s_loss_train_b  # performance improvement from student

    t_logits_train_b = teacher.get_logits(train_b)
    t_loss_train_b = teacher.compute_loss(
        logits=t_logits_train_b,
        labels=train_b_labels,
        masks=train_b_masks
    )

    t_loss_mpl = change * F.binary_cross_entropy_with_logits(
        t_logits_unlabelled_b, pseudo_y_unlabelled_b, reduction='none')  # mpl loss
    t_loss_mpl = torch.masked_select(t_loss_mpl, unlabelled_b_masks).mean()
    t_loss = t_loss_train_b + t_loss_mpl

    # backpropagation of teacher's loss
    accelerator.backward(t_loss)
    t_optimizer.step()
    t_scheduler.step()

    #------ Progress Bar Updates ----------------------------------#
    s_loss_meter.update(s_loss_train_b_new.item())
    t_loss_meter.update(t_loss.item())

    progress_bar.set_description(
        f"STEP: {step+1:5}/{num_steps:5}. "
        f"LR: {get_lr(s_optimizer):.4f}. "
        f"TL: {t_loss_meter.avg:.4f}. "
        f"SL: {s_loss_meter.avg:.4f}. "
    )
    progress_bar.update()

    #------ Evaluation & Checkpointing -----------------------------#
    if (step + 1) % config["validation_interval"] == 0:
        progress_bar.close()

        #----- Teacher Evaluation  ---------------------------------#
        teacher.eval()
        teacher_preds = []
        with torch.no_grad():
            for batch in valid_dl:
                p = teacher.get_logits(batch)
                teacher_preds.append(p)
        teacher_preds = [torch.sigmoid(p).detach().cpu().numpy()[:, :, 0] for p in teacher_preds]
        teacher_preds = list(chain(*teacher_preds))
        teacher_lb = scorer_fn(teacher_preds)
        print(f"After step {step+1} Teache LB: {teacher_lb}")

        # save teacher
        accelerator.wait_for_everyone()
        teacher = accelerator.unwrap_model(teacher)
        teacher_state = {
            'step': step + 1,
            'state_dict': teacher.state_dict(),
            'optimizer': t_optimizer.state_dict(),
            'lb': teacher_lb
        }
        is_best = False
        if teacher_lb > best_teacher_score:
            best_teacher_score = teacher_lb
            is_best = True
        # save_checkpoint(config, teacher_state, is_teacher=True, is_best=is_best)

        #----- Student Evaluation  ---------------------------------#
        student.eval()
        student_preds = []
        with torch.no_grad():
            for batch in valid_dl:
                p = student.get_logits(batch)
                student_preds.append(p)
        student_preds = [torch.sigmoid(p).detach().cpu().numpy()[:, :, 0] for p in student_preds]
        student_preds = list(chain(*student_preds))
        student_lb = scorer_fn(student_preds)
        print(f"After step {step+1} Student LB: {student_lb}")

        # save student
        accelerator.wait_for_everyone()
        student = accelerator.unwrap_model(student)
        student_state = {
            'step': step + 1,
            'state_dict': student.state_dict(),
            'optimizer': s_optimizer.state_dict(),
            'lb': student_lb
        }
        is_best = False
        if student_lb > best_student_score:
            best_student_score = student_lb
            is_best = True
        save_checkpoint(config, student_state, is_teacher=False, is_best=is_best)

        print("=="*40)
        print("GPU utilization after eval:")
        print_gpu_utilization()
        print("clearning the cache")
        torch.cuda.empty_cache()
        print_gpu_utilization()
        print("=="*40)


# # Next Steps
# * Run MPL with more unlabelled examples
#     * Current run uses 10k examples, which is roughly equivalent to training for 0.85 epochs
# * Use task-adpapted models as backbone
# * Experiment with different hyperparameters
# * Experiment with soft pseudo labels
# * During MPL, the student model is trained only on unlabelled data using pseudo labels from teacher
# * So the student can be further fine-tuned on actual training data for additional performance boost

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




