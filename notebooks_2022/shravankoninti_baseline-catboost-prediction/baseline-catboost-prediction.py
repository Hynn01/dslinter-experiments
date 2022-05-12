#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns  
from sklearn import metrics, preprocessing, model_selection
from sklearn.model_selection import train_test_split
import time

from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance

from scipy import stats
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

SEED = 1

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

#To ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_df = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
sub_df = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
train_df.head()


# In[ ]:


train_df.columns


# In[ ]:


print(f'Number of samples in train: {train_df.shape[0]}')
print(f'Number of columns in train: {train_df.shape[1]}')
for col in train_df.columns:
    if train_df[col].isnull().any():
        print(col, train_df[col].isnull().sum())


# In[ ]:


print(f'Number of samples in test: {test_df.shape[0]}')
print(f'Number of columns in test: {test_df.shape[1]}')
for col in test_df.columns:
    if test_df[col].isnull().any():
        print(col, test_df[col].isnull().sum())


# In[ ]:


train_df['target'].value_counts(normalize=True)


# In[ ]:


# * join the datasets
train_df['is_train']  = 1
test_df['target'] = 0
test_df['is_train'] = 0
full_df = train_df.append(test_df)


# In[ ]:


full_df['f_27'].value_counts()


# In[ ]:


# Label encoding is required for 'f_27' but for simplicity purpose - let us drop the column
full_df.drop(['f_27'],axis=1,inplace=True)


# In[ ]:


# append train and test data
testcount = len(test_df)
count = len(full_df)-testcount
print(count)

train = full_df[:count]
test = full_df[count:]
train_df = train.copy()
test_df = test.copy()


# In[ ]:


X=train_df.drop(columns={'id', 'is_train','target'},axis=1)
y=train_df.loc[:,['target']]

test_X=test_df.drop(columns={'id', 'is_train','target'},axis=1)


print(X.shape, y.shape, test_X.shape)


# In[ ]:


X.head()


# ## CatBoost

# In[ ]:


ts = time.time()

params={
    'random_state': 42
}

err = [] 

oofs = np.zeros(shape=(len(X)))
preds = np.zeros(shape=(len(test_X)))

Folds=10

fold = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=42)
i = 1

for train_index, test_index in fold.split(X, y):
    x_train, x_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    m = CatBoostClassifier(n_estimators=10000,**params,eval_metric='AUC',task_type="GPU")

    m.fit(x_train, y_train,eval_set=[(x_val, y_val)], early_stopping_rounds =30,verbose=False)

    pred_y = m.predict_proba(x_val)[:,1]
    oofs[test_index] = pred_y
    print(i, " err_cb: ", roc_auc_score(y_val,pred_y))
    err.append(roc_auc_score(y_val,pred_y))
    preds+= m.predict_proba(test_X)[:,1]
    i = i + 1
preds=preds/Folds

print(f"Average StratifiedKFold Score : {sum(err)/Folds} ")
oof_score = roc_auc_score(y, oofs)
print(f'\nOOF Auc is : {oof_score}')

oofs=pd.DataFrame(oofs,columns=['cboof'])
preds=pd.DataFrame(preds,columns=['cbpred'])

oofs.to_csv('cbmoof.csv',index=False)
preds.to_csv('cbmpred.csv',index=False)

print("Time to execute is : ",time.time() - ts)


# In[ ]:


fi = pd.Series(index = X.columns, data = m.feature_importances_)
fi.sort_values(ascending=False)[0:20][::-1].plot(kind = 'barh')


# In[ ]:


sub_df.head()


# In[ ]:


sub_df['target'] = preds['cbpred']
sub_df.head()


# In[ ]:


sub_df.to_csv("cb_submission.csv",index=False)

