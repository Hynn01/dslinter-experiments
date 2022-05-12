#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pytorch_tabnet


# In[ ]:


import pandas as pd

train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sample_submission = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")


# In[ ]:


print(train.shape)
train.head(10)


# In[ ]:


print(test.shape)
test.head(10)


# In[ ]:


print(sample_submission.shape)
sample_submission.head(10)


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


test.describe()


# In[ ]:


test.info()


# In[ ]:


train['f_27'].value_counts()


# In[ ]:


test['f_27'].value_counts()


# In[ ]:


all = pd.concat([train, test])
all


# In[ ]:


from sklearn import preprocessing

columns = ['f_27']

for column in columns:
  target_column = all[column]
  le = preprocessing.LabelEncoder()
  le.fit(target_column)
  label_encoded_column = le.transform(target_column)
  all[column] = pd.Series(label_encoded_column).astype('category')


# In[ ]:


train = all.iloc[:train.shape[0],:]
test = all.iloc[train.shape[0]:,:]


# In[ ]:


y_train = train['target'].values
X_train = train.drop('target', axis=1).drop('id', axis=1).values
X_test = test.drop('target', axis=1).drop('id', axis=1).values


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train)


# In[ ]:


import torch
from pytorch_tabnet.tab_model import TabNetClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TabNetClassifier(device_name = device)
model.fit(
  X_train, y_train,
  eval_set=[(X_valid, y_valid)]
)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


sample_submission['target'] = y_pred


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# In[ ]:




