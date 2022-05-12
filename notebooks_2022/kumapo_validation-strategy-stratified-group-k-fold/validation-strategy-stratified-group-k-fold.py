#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# 
# I think stratified group k fold would fits this competition better than stratified by `score` or grouped by `anchor`.
# 
# In this notebook, I will use [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) to get `score` stratified folds with non-overlapping `anchor`.
# 
# I'm appreciated if you like this notebook and upvote it :)

# In[ ]:


SEED = 42
N_FOLDS = 5
OUTPUT_FILE = 'folds.csv'


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold


# In[ ]:


df = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')

df['score_map'] = df['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})

encoder = LabelEncoder()
df['anchor_map'] = encoder.fit_transform(df['anchor'])

kf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for n, (_, valid_index) in enumerate(kf.split(df, df['score_map'], groups=df['anchor_map'])):
    df.loc[valid_index, 'fold'] = int(n)

df['fold'] = df['fold'].astype(int)
df.to_csv(OUTPUT_FILE, index=False)

