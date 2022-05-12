#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
    


# In[ ]:


train = pd.read_csv(r'../input/tabular-playground-series-may-2022/train.csv')
train.head()


# In[ ]:


test = pd.read_csv(r'../input/tabular-playground-series-may-2022/test.csv')
test.head()


# In[ ]:


sub = pd.read_csv(r'../input/tabular-playground-series-may-2022/sample_submission.csv')
sub.head()


# In[ ]:


train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


# In[ ]:


print(f'train set have {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'test set have {test.shape[0]} rows and {test.shape[1]} columns.') 
print(f'sample_submission set have {sub.shape[0]} rows and {sub.shape[1]} columns.')


# In[ ]:


train.isnull().sum()


# In[ ]:


train.describe().T


# In[ ]:


plt.figure(figsize=(24,20))
sns.heatmap(train.corr(),annot=True,cmap="YlGnBu")
plt.show()


# In[ ]:


cat = ['f_27']
X = train.drop('target',axis=1)
y = train['target']


# In[ ]:


from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

folds = KFold(n_splits=5, shuffle=True)

for fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    print(f"Fold: {fold}")
    X_train, X_test = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_test = y.iloc[trn_idx], y.iloc[val_idx]

    model = CatBoostClassifier(n_estimators = 1500, 
                               cat_features = cat,
                               task_type="GPU",
                               bootstrap_type='Poisson')
   
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
                early_stopping_rounds=400,
                verbose=False)
    y_pred = model.predict_proba(X_test)[:,1]
    roc = roc_auc_score(y_test, y_pred)
    
    print(f" roc_auc_score: {roc}")
    print("-"*50)


# In[ ]:


pred = model.predict_proba(test)[:,1]


# In[ ]:


sub['target'] = pred
sub.to_csv(f'cat.csv',index = False)


# In[ ]:


sub.head()


# In[ ]:




