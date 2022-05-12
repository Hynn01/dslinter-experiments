#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10) # make plots a bit bigger


# # Load Data

# In[ ]:


train_df = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv',index_col='id')
test_df = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv',index_col='id')


# # Preprocess

# In[ ]:


import string
characters = list(string.ascii_uppercase)
def engineer_features(df):
    # decode f_27 feature
    # add one feature per character. Number of the feature says how many times is the letter contained in f_27
    # df['f_27']
    for ch in characters:
        df[ch] = df['f_27'].str.count(ch)
    
    df.drop('f_27',axis=1, inplace=True)
    


# In[ ]:


X_train = train_df.drop(['target'], axis = 1)
y_train = train_df['target']
X_test = test_df

engineer_features(X_train)
engineer_features(X_test)

submission = pd.DataFrame(index = X_test.index)  # prepare df for submission

display(X_train,y_train,X_test)


# # Model

# ## Train and Predict

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from xgboost import XGBClassifier\nmodel_xgb = XGBClassifier()\n\nmodel_xgb.fit(X_train, y_train)')


# In[ ]:


y_xgb = model_xgb.predict_proba(X_test)
y_xgb

submission['xgb'] = y_xgb[:,1] # Metric is AUC -> we probabilities of 1 will yield better results


# ## Feature Importance

# In[ ]:


# show feature importance
from xgboost import plot_importance
plot_importance(model_xgb, max_num_features = 30)
plt.show()


# # Submit

# In[ ]:


submission.to_csv('submission.csv',columns=['xgb'], header=['target'],index=True)

