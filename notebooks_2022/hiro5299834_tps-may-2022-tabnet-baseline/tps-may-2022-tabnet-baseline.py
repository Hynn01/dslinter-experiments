#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'pip install pytorch-tabnet')


# # Libraries

# In[ ]:


import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
import string
import random
import time
import os
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


# # Parameters

# In[ ]:


class CFG:
    input = "../input/tabular-playground-series-may-2022"
    
    n_splits = 10
    seed = 42
    n_bins = 50
    
    target = 'target'
    tab_pred = 'tab_pred'
    pred = 'pred'
    
    int1_features = ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12',
                     'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18']
    int2_features = ['f_29', 'f_30']
    int_features = int1_features + int2_features
    
    float1_features = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06']
    float2_features = ['f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26']
    float3_features = ['f_28']
    float_features = float1_features + float2_features + float3_features


# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.seed)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain = pd.read_csv("/".join([CFG.input, "train.csv"]))\ntest = pd.read_csv("/".join([CFG.input, "test.csv"]))\nsubmission = pd.read_csv("/".join([CFG.input, "sample_submission.csv"]))')


# # Feature engineering

# In[ ]:


all_df = pd.concat([train, test]).reset_index(drop=True)


# In[ ]:


class feature_engineering:
    def __init__(self, df):
        self.df = df
        self.f_27_len = len(self.df['f_27'][0])
        self.alphabet_upper = list(string.ascii_uppercase)

    def get_features(self):
        self.df[f'f_19_bin'] = pd.cut(all_df['f_19'], CFG.n_bins, labels=False)
        self.df[f'f_21_bin'] = pd.cut(all_df['f_21'], CFG.n_bins, labels=False)
        
        for i in range(self.f_27_len):
            self.df[f'f_27_{i}'] = self.df['f_27'].apply(lambda x: x[i])
            
        for letter in tqdm(self.alphabet_upper):
            self.df[f'f_27_{letter}_count'] = self.df['f_27'].str.count(letter)

        self.df['f_sum_1']  = self.df[CFG.float1_features].sum(axis=1)
        self.df['f_min_1']  = self.df[CFG.float1_features].min(axis=1)
        self.df['f_max_1']  = self.df[CFG.float1_features].max(axis=1)
        self.df['f_std_1']  = self.df[CFG.float1_features].std(axis=1)    
        self.df['f_mad_1']  = self.df[CFG.float1_features].mad(axis=1)
        self.df['f_mean_1'] = self.df[CFG.float1_features].mean(axis=1)
        self.df['f_kurt_1'] = self.df[CFG.float1_features].kurt(axis=1)
        self.df['f_count_pos_1']  = self.df[CFG.float1_features].gt(0).count(axis=1)

        self.df['f_sum_2']  = self.df[CFG.float2_features].sum(axis=1)
        self.df['f_min_2']  = self.df[CFG.float2_features].min(axis=1)
        self.df['f_max_2']  = self.df[CFG.float2_features].max(axis=1)
        self.df['f_std_2']  = self.df[CFG.float2_features].std(axis=1)    
        self.df['f_mad_2']  = self.df[CFG.float2_features].mad(axis=1)
        self.df['f_mean_2'] = self.df[CFG.float2_features].mean(axis=1)
        self.df['f_kurt_2'] = self.df[CFG.float2_features].kurt(axis=1)
        self.df['f_count_pos_2']  = self.df[CFG.float2_features].gt(0).count(axis=1)
    
        return self.df
    
    def scaling(self, features):
        sc = StandardScaler()
        self.df[features] = sc.fit_transform(self.df[features])

        return self.df

    def label_encoding(self, features):
        new_features = []
        
        for feature in features:
            if self.df[feature].dtype == 'O':
                le = LabelEncoder()
                self.df[f'{feature}_enc'] = le.fit_transform(self.df[feature])
                new_features += [f'{feature}_enc']
            else:
                new_features += [feature]
                
        return self.df, new_features


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfe = feature_engineering(all_df)\nall_df = fe.get_features()')


# In[ ]:


features = [col for col in all_df.columns if CFG.target not in col]
num_features = []
cat_features = []

for feature in features:
    if all_df[feature].dtype == float:
        num_features.append(feature)
    else:
        cat_features.append(feature)

cat_features.remove('id')
cat_features.remove('f_27')


# # Scaling and encoding

# In[ ]:


all_df = fe.scaling(num_features)
all_df, cat_features = fe.label_encoding(cat_features)

all_features = cat_features + num_features


# In[ ]:


train_len = train.shape[0]
train = all_df[:train_len]
test = all_df[train_len:].reset_index(drop=True)


# In[ ]:


display(train[all_features])
display(test[all_features])


# In[ ]:


display(train[train[all_features].isna().any(axis=1)])
display(test[test[all_features].isna().any(axis=1)])


# In[ ]:


gc.collect()


# # TabNet

# In[ ]:


skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X=train, y=train[CFG.target])):
    X_train = train[all_features].to_numpy()[trn_idx]
    y_train = train[CFG.target].to_numpy()[trn_idx]
    X_valid = train[all_features].to_numpy()[val_idx]
    y_valid = train[CFG.target].to_numpy()[val_idx]
    X_test = test[all_features].to_numpy()
    
    print(f"===== FOLD {fold} =====")
    
    tabnet_params = dict(
        n_d=64,
        n_steps=5,
        gamma=1.3,
        n_independent=3,
        n_shared=3,
        seed=CFG.seed,
        momentum=2e-2,
        lambda_sparse=1e-6,

        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(
            lr=1e-2,
            weight_decay=1e-7
        ),
        
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=dict(
            mode='max',
            factor=0.9,
            patience=3,
            min_lr=1e-6,
        ),
        verbose=10,
        device_name='auto',
        mask_type='sparsemax',
    )
    
    # Defining TabNet model
    model = TabNetClassifier(**tabnet_params)

    model.fit(
        X_train=X_train,
        y_train=y_train,
        from_unsupervised=None,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["auc"],
        batch_size=2048,
        virtual_batch_size=2048,
        max_epochs=200,
        drop_last=True,
        pin_memory=True,
        patience=20,
        num_workers=4,
    )

    train.loc[val_idx, CFG.tab_pred] = model.predict_proba(X_valid)[:, -1]
    print(f"auc score: {roc_auc_score(y_true=y_valid, y_score=train.loc[val_idx, CFG.tab_pred]):.6f}\n")
    
    test[f'{CFG.tab_pred}_{fold}'] = model.predict_proba(X_test)[:, -1]

print(f"auc score : {roc_auc_score(y_true=train[CFG.target], y_score=train[CFG.tab_pred]):.6f}")


# # Submission

# In[ ]:


cols = [col for col in test.columns if CFG.tab_pred in col]

submission[CFG.target] = test[cols].mean(axis=1)
submission.to_csv("submission.csv", index=False)
submission


# In[ ]:




