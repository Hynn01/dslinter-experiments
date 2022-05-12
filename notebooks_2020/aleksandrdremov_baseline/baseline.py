#!/usr/bin/env python
# coding: utf-8

# ## Basic imports

# In[ ]:


import pandas as pd
import xgboost as xgb


# In[ ]:


transactions_train=pd.read_csv('../input/sberbank-users-expenses/transactions_train.csv')


# In[ ]:


train_target=pd.read_csv('../input/sberbank-users-expenses/train_target.csv')


# In[ ]:


transactions_train.head()


# * client_id - Client's id
# * trans_date - transaction date
# * small_group - Buying category
# * amount_rur - Amount of money

# In[ ]:


train_target.head(5)


# * bins - target variable. User's age group

# # Simple solution:

# In[ ]:


agg_features=transactions_train.groupby('client_id')['amount_rur'].agg(['sum','mean','std','min','max']).reset_index()


# In[ ]:


agg_features.head()


# In[ ]:


counter_df_train=transactions_train.groupby(['client_id','small_group'])['amount_rur'].count()


# In[ ]:


cat_counts_train=counter_df_train.reset_index().pivot(index='client_id',                                                       columns='small_group',values='amount_rur')


# In[ ]:


cat_counts_train=cat_counts_train.fillna(0)


# In[ ]:


cat_counts_train.columns=['small_group_'+str(i) for i in cat_counts_train.columns]


# In[ ]:


cat_counts_train.head()


# In[ ]:


train=pd.merge(train_target,agg_features,on='client_id')


# In[ ]:


train=pd.merge(train,cat_counts_train.reset_index(),on='client_id')


# In[ ]:


train.head()


# In[ ]:


transactions_test=pd.read_csv('../input/sberbank-users-expenses//transactions_test.csv')

test_id=pd.read_csv('../input/sberbank-users-expenses/test.csv')


# In[ ]:


agg_features_test=transactions_test.groupby('client_id')['amount_rur'].agg(['sum','mean','std','min','max']).reset_index()


# In[ ]:


agg_features_test.head()


# In[ ]:


counter_df_test=transactions_test.groupby(['client_id','small_group'])['amount_rur'].count()


# In[ ]:


cat_counts_test=counter_df_test.reset_index().pivot(index='client_id', columns='small_group',values='amount_rur')


# In[ ]:


cat_counts_test=cat_counts_test.fillna(0)


# In[ ]:


cat_counts_test.columns=['small_group_'+str(i) for i in cat_counts_test.columns]


# In[ ]:


cat_counts_test.head()


# In[ ]:


test=pd.merge(test_id,agg_features_test,on='client_id')


# In[ ]:


test=pd.merge(test,cat_counts_test.reset_index(),on='client_id')


# In[ ]:


common_features=list(set(train.columns).intersection(set(test.columns)))


# In[ ]:


y_train=train['bins']
X_train=train[common_features]
X_test=test[common_features]


# In[ ]:


param={'objective':'multi:softprob','num_class':4,'n_jobs':4,'seed':42}


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model=xgb.XGBClassifier(**param,n_estimators=300)\nmodel.fit(X_train,y_train)')


# In[ ]:


pred=model.predict(X_test)


# In[ ]:


pred


# Such prediction give 0.6118 accuracy

# In[ ]:


submission = pd.DataFrame({'bins': pred}, index=test.client_id)
submission.head()


# In[ ]:


import time
import os

current_timestamp = int(time.time())
submission_path = 'submit.csv'
print(submission_path)
submission.to_csv(submission_path, index=True)

