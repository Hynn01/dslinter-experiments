#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import seaborn as sns
import lightgbm as lgb

#models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier #Gradient booster classifier


# ## **Load the data**
# ### Load train dataset

# In[ ]:


train = pd.read_csv("../input/reducing-commercial-aviation-fatalities/train.csv")


# In[ ]:


train.sample(5)


# In[ ]:


print(train.shape)


# ### Load test dataset

# In[ ]:


test_iterator = pd.read_csv("../input/reducing-commercial-aviation-fatalities/test.csv", chunksize=5)
test_top = next(test_iterator)


# In[ ]:


test_top


# In[ ]:


sample_sub = pd.read_csv("../input/reducing-commercial-aviation-fatalities/sample_submission.csv")
sample_sub.sample(10)


# ## Explore Training Data

# In[ ]:


pd.crosstab(train.experiment, train.event)


# ### To have an idea of how the records were distributed among the crews

# In[ ]:


plt.figure(figsize=(10, 6))
sns.countplot(train['event'])
plt.xlabel("State of the pilot", fontsize = 12)
plt.ylabel("Count", fontsize = 12)
plt.title("Target repartition", fontsize = 15)
plt.show()


# In[ ]:


pd.crosstab(train.experiment, train.crew)


# #### Just to remember the indexes of the columns

# In[ ]:


print(list(enumerate(train.columns)))


# ### Filtering the training set. We can play with the crews, seats, exp, and event

# In[ ]:


crew = 3
seat = 0
exp = 'DA'
ev = 'D'

sel = (train.crew == crew) & (train.experiment) & (train.seat == seat)
pilot_info = train.loc[sel, :].sort_values(by = 'time')

plt.figure(figsize = [16, 12])
for i in range(4, 27):
    plt.subplot(6, 4, i-3)
    plt.plot(pilot_info.time, pilot_info.iloc[:, i], zorder = 1)
    plt.scatter(pilot_info.loc[pilot_info.event == ev, :].time, pilot_info.loc[pilot_info.event == ev, :].iloc[:, i], 
                c = 'red', zorder = 2, s = 1)
    plt.title(pilot_info.columns[i])
    
plt.tight_layout()
plt.show()


# ## Create Feature and Label Arrays
# ### Create the Montages for Train and Test sets

# ![Montages.png](attachment:Montages.png)

# In[ ]:


train['f7-f8'] = train['eeg_f7'] - train['eeg_f8']
train['f3-f4'] = train['eeg_f3'] - train['eeg_f4']
train['t3-t4'] = train['eeg_t3'] - train['eeg_t4']
train['c3-c4'] = train['eeg_c3'] - train['eeg_c4']
train['p3-p4'] = train['eeg_p3'] - train['eeg_p4']
train['t5-t6'] = train['eeg_t5'] - train['eeg_t6']
train['o1-o2'] = train['eeg_o1'] - train['eeg_o2']

train_columns = ['crew', 'seat', 'f7-f8', 'f3-f4', 't3-t4', 'c3-c4', 'p3-p4', 't5-t6', 'o1-o2', 
                 'ecg', 'r', 'gsr', 'event']
train = train.loc[:, train_columns]
train.sample(5)


# In[ ]:


test_top['f7-f8'] = test_top['eeg_f7'] - test_top['eeg_f8']
test_top['f3-f4'] = test_top['eeg_f3'] - test_top['eeg_f4']
test_top['t3-t4'] = test_top['eeg_t3'] - test_top['eeg_t4']
test_top['c3-c4'] = test_top['eeg_c3'] - test_top['eeg_c4']
test_top['p3-p4'] = test_top['eeg_p3'] - test_top['eeg_p4']
test_top['t5-t6'] = test_top['eeg_t5'] - test_top['eeg_t6']
test_top['o1-o2'] = test_top['eeg_o1'] - test_top['eeg_o2']

test_columns = ['id', 'crew', 'seat', 'f7-f8', 'f3-f4', 't3-t4', 'c3-c4', 'p3-p4', 't5-t6', 'o1-o2', 
                 'ecg', 'r', 'gsr']
test_columns = test_top.loc[:, test_columns]
test_columns


# In[ ]:


y_train_full = train.event
X_train_full = train.iloc[:, :-1]
X_train_full.head()


# In[ ]:


pd.DataFrame({
    'min_val':X_train_full.min(axis=0).values,
    'max_val':X_train_full.max(axis=0).values,
    }, index = X_train_full.columns
)


# ### Generates a Frequency Distribution for a Colum

# In[ ]:


y_train_full.value_counts()


# ### We take those counts from previous cell and convert them into proportions

# In[ ]:


y_train_full.value_counts() / len(y_train_full)


# ### Split the Training Dataset into Training and Validation

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size = 0.90, stratify = y_train_full, random_state = 1)
#print(X_train.shape)
print(f"Training on {X_train.shape[0]} samples.")


# ## Logistic Regression Model

# In[ ]:


get_ipython().run_cell_magic('time', '', "lr_mod = LogisticRegression(solver='lbfgs', n_jobs=-1)\nlr_mod.fit(X_train, y_train)\n\nprint('Training Accuracy: ', lr_mod.score(X_train, y_train))\nprint('Validation Accuracy: ', lr_mod.score(X_valid, y_valid))")


# ## Logistic Regression with GridSearchCV to see if we can have better results

# In[ ]:


# It didn't improve the model anyway
#%%time
#lr_pipe = Pipeline(
#    steps = [
#        ('scaler', StandardScaler()),
#        ('classifier', LogisticRegression(solver = 'lbfgs', n_jobs = -1))
#    ]
#)
#lr_param_grid = {
#    'classifier__C':[0.001, 0.1, 1.0, 1.1],
#}
#np.random.seed(1)
#grid_search = GridSearchCV(lr_pipe, lr_param_grid, cv = 10, refit = 'True')
#grid_search.fit(X_train, y_train)
#
#print(grid_search.best_score_)
#print(grid_search.best_params_)


# ## Random Forest Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_mod1 = RandomForestClassifier(n_estimators = 20, max_depth = 32, n_jobs = -1)\nrf_mod1.fit(X_train, y_train)\n\nprint(\'Training Accuracy: \', rf_mod1.score(X_train, y_train))\nprint("Validation Accuracy: ", rf_mod1.score(X_valid, y_valid))')


# ### Hyperparameter Tuning and we will get the average of the CV scores

# In[ ]:


get_ipython().run_cell_magic('time', '', "rf_pipe = Pipeline(\n    steps =[\n        ('scaler', StandardScaler()),\n        ('classifier', RandomForestClassifier(n_estimators=40, n_jobs=-1))\n    ]\n)\nlr_param_grid = {\n    'classifier__max_depth':[65, 66, 67]\n}\nnp.random.seed(1)\ngrid_search = GridSearchCV(rf_pipe, lr_param_grid, cv = 5, refit = 'True')\ngrid_search.fit(X_train, y_train)\n\nprint(grid_search.best_score_)\nprint(grid_search.best_params_)")


# In[ ]:


grid_search.cv_results_['mean_test_score']


# ### Increasing the number of estimators for the max_depth = 66

# In[ ]:


get_ipython().run_cell_magic('time', '', "rf_mod = RandomForestClassifier(n_estimators = 150, max_depth=66, n_jobs = -1)\nrf_mod.fit(X_train, y_train)\n\nprint('Training Accuracy: ', rf_mod.score(X_train, y_train))\nprint('Validation Accuracy: ', rf_mod.score(X_valid, y_valid))")


# In[ ]:


from sklearn.metrics import log_loss
log_loss(y_train, rf_mod.predict_proba(X_train))


# In[ ]:


log_loss(y_valid, rf_mod.predict_proba(X_valid))


# ## Gradient Boosting Tree

# In[ ]:


#%%time
#xbg_mod = XGBClassifier()
#xbg_mod.fit(X_train, y_train)

#xbg_mod.score(X_train, y_train)


# ### Score in the other 98% of the data

# In[ ]:


#xbg_mod.score(X_valid, y_valid)


# In[ ]:


#log_loss(y_train, xbg_mod.predict_proba(X_train))


# In[ ]:


#log_loss(y_valid, xbg_mod.predict_proba(X_valid))


# ## Hyperparameter Tuning for Gradient Boosting

# In[ ]:


get_ipython().run_cell_magic('time', '', "xgd_pipe = Pipeline(\n    steps = [\n        ('classifier', XGBClassifier(learning_rate = 0.3, alpha=1, max_depth=6, n_estimators=30, subsample=0.5))\n    ]\n)\nxgd_param_grid = {\n    'classifier__learning_rate':[0.7, 0.8, 0.9, 0.10],\n}\nnp.random.seed(1)\nxgd_grid_search = GridSearchCV(xgd_pipe, xgd_param_grid, cv=5, refit='True')\nxgd_grid_search.fit(X_train, y_train)\n\nprint(xgd_grid_search.best_score_)\nprint(xgd_grid_search.best_params_)")


# ## Generate Test Predictions

# In[ ]:


#test_iteratorator = pd.read_csv("../input/reducing-commercial-aviation-fatalities/test.csv", chunksize=5)
#test_top = next(test_iterator)
#test_top


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cs = 10000\ni = 0\nfor test in pd.read_csv("../input/reducing-commercial-aviation-fatalities/test.csv", chunksize=cs):\n    test[\'f7-f8\'] = test[\'eeg_f7\'] - test[\'eeg_f8\']\n    test[\'f3-f4\'] = test[\'eeg_f3\'] - test[\'eeg_f4\']\n    test[\'t3-t4\'] = test[\'eeg_t3\'] - test[\'eeg_t4\']\n    test[\'c3-c4\'] = test[\'eeg_c3\'] - test[\'eeg_c4\']\n    test[\'p3-p4\'] = test[\'eeg_p3\'] - test[\'eeg_p4\']\n    test[\'t5-t6\'] = test[\'eeg_t5\'] - test[\'eeg_t6\']\n    test[\'o1-o2\'] = test[\'eeg_o1\'] - test[\'eeg_o2\']\n\n    test_columns = [\'id\',\'crew\',\'seat\',\'f7-f8\',\'f3-f4\',\'t3-t4\',\'c3-c4\',\'p3-p4\',\'t5-t6\',\'o1-o2\',\'ecg\',\'r\',\'gsr\']\n    test = test.loc[:,test_columns]\n    \n    print(\'--Iteration\', i, \'is started\')\n    test_pred = rf_mod.predict_proba(test.iloc[:,1:])\n    partial_submission = pd.DataFrame({\n        \'id\':test.id,\n        \'A\':test_pred[:, 0],\n        \'B\':test_pred[:, 1],\n        \'C\':test_pred[:, 2],\n        \'D\':test_pred[:, 3]\n    })\n    if i == 0:\n        submission = partial_submission.copy()\n    else:\n        submission = submission.append(partial_submission, ignore_index=True)\n        \n    del test\n    print(\'++Iteration\', i, \'is done!\')\n    i += 1\n    ')


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index = False)

