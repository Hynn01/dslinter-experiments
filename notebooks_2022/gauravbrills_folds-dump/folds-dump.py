#!/usr/bin/env python
# coding: utf-8

# # Various Fold strategies for the ride
# 
# Adding list of fold strategies to use for the comp

# In[ ]:


get_ipython().system('pip install -q iterative-stratification')


# In[ ]:


import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
import torch
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold, GroupKFold, KFold
OUTPUT_DIR=""


# In[ ]:


# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
submission = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/sample_submission.csv')
print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"submission.shape: {submission.shape}")
display(train.head())
display(test.head())
display(submission.head())


# In[ ]:


# ====================================================
# CPC Data
# ====================================================
def get_cpc_texts():
    contexts = []
    pattern = '[A-Z]\d+'
    for file_name in os.listdir('../input/cpc-data/CPCSchemeXML202105'):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        with open(f'../input/cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt') as f:
            s = f.read()
        pattern = f'{cpc}\t\t.+'
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f'{context}\t\t.+'
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results

cpc_texts = get_cpc_texts()
torch.save(cpc_texts, OUTPUT_DIR+"cpc_texts.pth")
train['context_text'] = train['context'].map(cpc_texts)
test['context_text'] = test['context'].map(cpc_texts)
display(train.head(2))
display(test.head(2))

train['text'] = train['anchor'] + '[SEP]' + train['target'] + '[SEP]'  + train['context_text']
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
display(train.head(4))
display(test.head(4))


# In[ ]:


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
def create_msrat_folds(n_fold=5,train=train,random_state=42):
    dfx = pd.get_dummies(train, columns=["score"]).groupby(["anchor"], as_index=False).sum()
    cols = [c for c in dfx.columns if c.startswith("score_") or c == "anchor"]
    dfx = dfx[cols]

    mskf = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    labels = [c for c in dfx.columns if c != "anchor"]
    dfx_labels = dfx[labels]
    dfx["fold"] = -1

    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        print(len(trn_), len(val_))
        dfx.loc[val_, "fold"] = fold

    train = train.merge(dfx[["anchor", "fold"]], on="anchor", how="left")
    train.to_csv(f"train_folds_mstrat_{n_fold}.csv", index=False)
    return train


# In[ ]:


create_msrat_folds(4)
create_msrat_folds(5)

