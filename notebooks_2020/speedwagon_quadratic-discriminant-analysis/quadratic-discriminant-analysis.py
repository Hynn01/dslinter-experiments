#!/usr/bin/env python
# coding: utf-8

# # Quadratic Discriminant Analysis
# 
# 
# Notice that execution time of this kernel is 83 sec.

# ### Why it works so good?
# As was found <a href="https://www.kaggle.com/mhviraf/synthetic-data-for-next-instant-gratification">here</a>, dataset is created by `make_classification` method. This method generates gaussians with non-diagonal covariance matrix and assigns them classes. QDA works exactly with this structures of data, it learns normal distributions with n-dimentional covariance matrix.<br>
# ### Why it works so fast?
# It uses Bayes theorem to infer likelihood. Components of this formula fits fast, because they compute statiscs.
# 
# 
# #### People made challenge of minimizing lines of code. Let's make a challenge of minimizing exe time :)

# In[ ]:


import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# In[ ]:


oof = np.zeros(len(train))
preds = np.zeros(len(test))

for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train2, train2['target']):

        clf = QuadraticDiscriminantAnalysis(0.1)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

auc = roc_auc_score(train['target'], oof)
print(f'AUC: {auc:.5}')


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv', index=False)


# In[ ]:




