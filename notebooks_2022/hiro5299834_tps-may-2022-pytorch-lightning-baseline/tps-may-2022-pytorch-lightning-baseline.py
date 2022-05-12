#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\n!pip install monai-weekly')


# # Libraries

# In[ ]:


import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
import datatable as dt
import datetime
import string
import random
import glob
import time
import os
import gc

from scipy.stats import rankdata

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from monai.metrics import ROCAUCMetric

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')


# # Parameters

# In[ ]:


class CFG:
    input = "../input/tabular-playground-series-may-2022"
    target = 'target'
    
    n_splits = 10
    seed = 42

    batch_size = 4096
    workers = 4
    epochs = 200
    learning_rate = 1e-2
    
    factor = 0.8
    min_lr = 1e-6
    
    lr_patience = 5
    es_patience = 20

    pred = 'pred'
    test_pred = [f'pred_{i}' for i in range(n_splits)]
    
    model_path = "models"
    tb_log_name = "lightning_logs"


# In[ ]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    pl.utilities.seed.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(CFG.seed)


# # Data loading ...

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
        for i in range(self.f_27_len):
            self.df[f'f_27_{i}'] = self.df['f_27'].apply(lambda x: x[i])

        for letter in tqdm(self.alphabet_upper):
            self.df[f'f_27_{letter}_count'] = self.df['f_27'].str.count(letter)

        self.df['f_27_nunique'] = self.df['f_27'].apply(lambda x: len(set(x)))

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


# # Check data

# In[ ]:


display(train[all_features])
display(test[all_features])


# In[ ]:


display(train[train[all_features].isna().any(axis=1)])
display(test[test[all_features].isna().any(axis=1)])


# In[ ]:


gc.collect()


# # Network implementation

# In[ ]:


class Model(pl.LightningModule):
    def __init__(self, in_size, learning_rate, num_targets=1, hidden_size=128):
        super().__init__()
        self.in_size = in_size
        self.lr = learning_rate
        self.num_targets = num_targets
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 16)
        self.fc4 = nn.Linear(16, self.num_targets)
        self.relu = F.relu
        self.swish = F.hardswish
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.roc_auc_metric = ROCAUCMetric()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.swish(self.fc1(x))
        x = self.swish(self.fc2(x))
        x = self.relu(self.fc2(x))
        x = self.swish(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)  
        self.log('loss', loss)
        return {'loss': loss}
        
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(1)
        self.roc_auc_metric(y_hat, y)      
    
    def validation_epoch_end(self, training_step_outputs):
        roc_auc = self.roc_auc_metric.aggregate()
        self.roc_auc_metric.reset()
        self.log('val_auc', roc_auc)
        
    def predict_step(self, X, batch_idx: int, dataloader_idx: int = None):
        return self(X[0])    
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            eps=1e-8,
            weight_decay=1e-6,
            amsgrad=False
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=CFG.factor,
                patience=CFG.lr_patience,
                min_lr=CFG.min_lr
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_auc",
            "strict": True,
            "name": "Learning Rate",
        }
        return [optimizer], [lr_scheduler]


# In[ ]:


skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train[CFG.target])):
    print('Fold:', fold)
    X_train, y_train = train[all_features].iloc[trn_idx], train[CFG.target].iloc[trn_idx]
    X_valid, y_valid = train[all_features].iloc[val_idx], train[CFG.target].iloc[val_idx]
    X_test = test[all_features]
    
    train_ds = TensorDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values))
    valid_ds = TensorDataset(torch.FloatTensor(X_valid.values), torch.FloatTensor(y_valid.values))

    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.workers)
    valid_dl = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.workers)

    model = Model(in_size=X_train.shape[1],
                  learning_rate=CFG.learning_rate,
                  num_targets=1,
                  hidden_size=64,
                 )
    print(ModelSummary(model))
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=CFG.model_path,
        filename=f'model_{fold}_' + '{val_auc:.6f}',
        monitor='val_auc',
        mode='max',
        save_weights_only=True)

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=fold,
        name=CFG.tb_log_name
    )
 
    early_stop_callback = EarlyStopping(
        monitor='loss',
        min_delta=0.00,
        patience=CFG.es_patience,
        verbose=False,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )
    
    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=CFG.epochs,
        gpus=1,
        precision=32,
        limit_train_batches=1.0,
        limit_val_batches=1.0, 
        num_sanity_val_steps=0,
        val_check_interval=1.0, 
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        logger=logger
     )

    trainer.fit(model, train_dl, valid_dl)
    trainer.validate(model, valid_dl, verbose=True)

    preds = trainer.predict(model, valid_dl)
    train.loc[val_idx, CFG.pred] = torch.cat(preds).cpu().numpy().flatten()
    
    test_ds = TensorDataset(torch.FloatTensor(X_test.values))
    test_dl = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.workers)

    preds = trainer.predict(model, test_dl)
    test[CFG.test_pred[fold]] = torch.cat(preds).cpu().numpy().flatten()

    del model, trainer, X_train, X_valid, y_train, y_valid, train_ds, valid_ds, train_dl, valid_dl
    gc.collect()
    torch.cuda.empty_cache()
    
auc = roc_auc_score(train[CFG.target], train[CFG.pred])
print(f"auc: {auc:.6f}")


# In[ ]:


train


# # Submission

# In[ ]:


models = np.sort(glob.glob(f"./{CFG.model_path}/*.ckpt"))
trainer = pl.Trainer(gpus=1)

for model_name in models:
    X_test = test[all_features]
    model = Model(in_size=X_test.shape[1],
              learning_rate=CFG.learning_rate,
              num_targets=1,
              hidden_size=64,
             )
    model.load_state_dict(torch.load(model_name)['state_dict'])
    test_ds = TensorDataset(torch.FloatTensor(X_test.values))
    test_dl = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.workers)
    
    preds = trainer.predict(model, test_dl)
    test[CFG.test_pred[fold]] = torch.cat(preds).cpu().numpy().flatten()
    
submission[CFG.target] = test[CFG.test_pred].mean(axis=1)
submission.to_csv("submission.csv", index=False)
submission


# In[ ]:




