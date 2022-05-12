#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import required libraries

import numpy as np 
import pandas as pd
import re
from ast import literal_eval
from itertools import chain
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoTokenizer

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Data Understanding
data_dir = "/kaggle/input/nbme-score-clinical-patient-notes"
# Training data files
train=pd.read_csv(data_dir+"/train.csv")
patient_notes=pd.read_csv(data_dir+"/patient_notes.csv")
features=pd.read_csv(data_dir+"/features.csv")

# Test data file/s
test=pd.read_csv(data_dir+"/test.csv")

# submission sample 
submission=pd.read_csv(data_dir+"/sample_submission.csv")


# # Train 
# ****Column Description :****
# 
# * id - Unique identifier for each patient note / feature pair.
# * case_num - The case to which this patient note belongs.
# * pn_num - The patient note annotated in this row.
# * feature_num - The feature annotated in this row.
# * annotation - The text(s) within a patient note indicating a feature. A feature may be indicated multiple times within a single note.
# * location - Character spans indicating the location of each annotation within the note. Multiple spans may be needed to represent an annotation, in which case the spans are delimited by a semicolon ;.

# In[ ]:


train.head()


# In[ ]:


print('Number of rows in train data: {}'.format(train.shape[0]))
print('Number of columns in train data: {}'.format(train.shape[1]))
print('Number of unique cases: {}'.format(train.case_num.nunique()))
print('Number of unique patients: {}'.format(train.pn_num.nunique()))


# # Features
# ****Column Description :****
# 
# * feature_num - A unique identifier for each feature.
# * case_num - The case to which this patient note belongs.
# * feature_text - A description of the feature.

# In[ ]:


features.head()


# In[ ]:


# Sample Feature Text
features["feature_text"].iloc[4], features["feature_text"].iloc[40], features["feature_text"].iloc[41]


# # Patient Notes
# **Column Description :**
# * pn_num - A unique identifier for each patient note.
# * case_num - A unique identifier for the clinical case a patient note represents.
# * pn_history - The text of the encounter as recorded by the test taker.

# In[ ]:


patient_notes.head()


# In[ ]:


# Sample Patient Note
print(patient_notes["pn_history"].iloc[8])


# In[ ]:


# Data Preprocess
def process_feature_text(text):
    return text.replace("-OR-", ";-").replace("-", " ").replace("I-year", "1-year")


def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    return txt


# In[ ]:


train = pd.read_csv("/kaggle/input/nbme-score-clinical-patient-notes/train.csv")

# Merge Datasets to Prepare Training Data
merged_df = train.merge(features, how="left", on=["case_num", "feature_num"])
merged_df = merged_df.merge(patient_notes, how="left", on=['case_num', 'pn_num'])

# Preprocess
merged_df['pn_history'] = merged_df['pn_history'].apply(lambda x: x.strip())
merged_df['pn_history'] = merged_df['pn_history'].apply(clean_spaces)
merged_df['pn_history'] = merged_df['pn_history'].apply(lambda x: x.lower())
merged_df['feature_text'] = merged_df['feature_text'].apply(process_feature_text)
merged_df['feature_text'] = merged_df['feature_text'].apply(clean_spaces)
merged_df['feature_text'] = merged_df['feature_text'].apply(lambda x: x.lower())



# Split data as train and test
test_size = int(len(merged_df)* (0.2))
train_df, test_df = train_test_split(merged_df, test_size=test_size, random_state=500)
print(len(train_df), len(test_df))


# In[ ]:


def tokenize_and_add_labels(tokenizer, data, config):
    out = tokenizer(
        data["feature_text"],
        data["pn_history"],
        truncation=config['truncation'],
        max_length=config['max_length'],
        padding=config['padding'],
        return_offsets_mapping=config['return_offsets_mapping']
    )
    labels = [0.0] * len(out["input_ids"])
    out["location_int"] = loc_list_to_ints(data["location_list"])
    out["sequence_ids"] = out.sequence_ids()

    for idx, (seq_id, offsets) in enumerate(zip(out["sequence_ids"], out["offset_mapping"])):
        if not seq_id or seq_id == 0:
            labels[idx] = -1
            continue

        token_start, token_end = offsets
        for feature_start, feature_end in out["location_int"]:
            if token_start >= feature_start and token_end <= feature_end:
                labels[idx] = 1.0
                break

    out["labels"] = labels

    return out


# In[ ]:


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        tokens = tokenize_and_add_labels(self.tokenizer, data, self.config)

        input_ids = np.array(tokens["input_ids"])
        attention_mask = np.array(tokens["attention_mask"])
        token_type_ids = np.array(tokens["token_type_ids"])

        labels = np.array(tokens["labels"])
        offset_mapping = np.array(tokens['offset_mapping'])
        sequence_ids = np.array(tokens['sequence_ids']).astype("float16")
        
        return input_ids, attention_mask, token_type_ids, labels, offset_mapping, sequence_ids


# In[ ]:


hyperparameters = {
    "max_length": 416,
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    "model_name": "bert-base-uncased",
    "dropout": 0.2,
    "lr": 1e-5,
    "test_size": 0.2,
    "seed": 1268,
    "batch_size": 8
}

tokenizer = AutoTokenizer.from_pretrained(hyperparameters['model_name'])

training_data = CustomDataset(train_df, tokenizer, hyperparameters)
train_dataloader = DataLoader(training_data, batch_size=hyperparameters['batch_size'], shuffle=True)

test_data = CustomDataset(test_df, tokenizer, hyperparameters)
test_dataloader = DataLoader(test_data, batch_size=hyperparameters['batch_size'], shuffle=False)


# In[ ]:




