#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# Before I get started, I just wanted to say: huge props to Inversion! The official starter kernel is **AWESOME**; it's so simple, clean, straightforward, and pragmatic. It certainly saved me a lot of time wrangling with data, so that I can directly start tuning my models (real data scientists will call me lazy, but hey I'm an engineer I just want my stuff to work).
# 
# I noticed two tiny problems with it:
# * It takes a lot of RAM to run, which means that if you are using a GPU, it might crash as you try to fill missing values.
# * It takes a while to run (roughly 3500 seconds, which is more than an hour; again, I'm a lazy guy and I don't like waiting).
# 
# With this kernel, I bring some small changes:
# * Decrease RAM usage, so that it won't crash when you change it to GPU. I simply changed when we are deleting unused variables.
# * Decrease **running time from ~3500s to ~40s** (yes, that's almost 90x faster), at the cost of a slight decrease in score. This is done by adding a single argument.
# 
# Again, my changes are super minimal (cause Inversion's kernel was already so awesome), but I hope it will save you some time and trouble (so that you can start working on cool stuff).
# 
# 
# ### Changelog
# 
# **V4**
# * Change some wording
# * Prints XGBoost version
# * Add random state to XGB for reproducibility

# In[ ]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb


# In[ ]:


print("XGBoost version:", xgb.__version__)


# # Efficient Preprocessing
# 
# This preprocessing method is more careful with RAM usage, which avoids crashing the kernel when you switch from CPU to GPU. Otherwise, it is exactly the same procedure as the official starter.

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train['isFraud'].copy()\ndel train_transaction, train_identity, test_transaction, test_identity\n\n# Drop target, fill in NaNs\nX_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\n\ndel train, test\n\nX_train = X_train.fillna(-999)\nX_test = X_test.fillna(-999)\n\n# Label Encoding\nfor f in X_train.columns:\n    if X_train[f].dtype=='object' or X_test[f].dtype=='object': \n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n        X_train[f] = lbl.transform(list(X_train[f].values))\n        X_test[f] = lbl.transform(list(X_test[f].values))   ")


# # Training
# 
# To activate GPU usage, simply use `tree_method='gpu_hist'` (took me an hour to figure out, I wish XGBoost documentation was clearer about that).

# In[ ]:


clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)


# In[ ]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# Some of you must be wondering how we were able to decrease the fitting time by that much. The reason for that is not only we are running on gpu, but we are also computing an approximation of the real underlying algorithm (which is a greedy algorithm). This hurts your score slightly, but as a result is much faster.
# 
# So why am I not using CPU with `tree_method='hist'`? If you try it out yourself, you'll realize it'll take ~ 7 min, which is still far from the GPU fitting time. Similarly, `tree_method='gpu_exact'` will take ~ 4 min, but likely yields better accuracy than `gpu_hist` or `hist`.
# 
# The [docs on parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) has a section on `tree_method`, and it goes over the details of each option.

# In[ ]:


sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost.csv')

