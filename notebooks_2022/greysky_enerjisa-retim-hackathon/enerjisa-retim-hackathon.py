#!/usr/bin/env python
# coding: utf-8

# ## Modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")


# ## Prediction and Cross-Validation Functions

# In[ ]:


def predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, index=X_test.index, columns=y_train.columns)
    
    return y_pred

def cross_validate(model, X, y):
    kf = KFold(n_splits=8)

    for train_idx, test_idx in kf.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_pred = model.predict(X.iloc[test_idx])
        y_pred = pd.DataFrame(y_pred, index=y.iloc[test_idx].index, columns=y.iloc[test_idx].columns)
        
        print(f'RMSE: {mean_squared_error(y.iloc[test_idx], y_pred, squared=False)}')


# ## Data Collection

# In[ ]:


df_train = pd.read_csv(filepath_or_buffer = '/kaggle/input/enerjisa-uretim-hackathon/features.csv',
                       parse_dates        = ['Timestamp'],
                       index_col          = 'Timestamp')

df_test = pd.read_csv(filepath_or_buffer  = '/kaggle/input/enerjisa-uretim-hackathon/power.csv',
                      parse_dates         = ['Timestamp'],
                      index_col           = 'Timestamp')

df_subm = pd.read_csv(filepath_or_buffer  = '/kaggle/input/enerjisa-uretim-hackathon/sample_submission.csv',
                      parse_dates         = ['Timestamp'],
                      index_col           = 'Timestamp')

for col in df_train.columns:
    df_train.loc[df_train[col] > 9999, col] = np.nan

df_train['Power(kW)'] = df_test['Power(kW)']


# ## Data Wrangling
# ### Let's Split the Data to 3 Parts
# 
# #### Part 1 has no correlation with Torque and Power(kW) is lesser than 0
# #### Part 2 has strong correlation with Torque
# #### Part 3 has moderate correlation with Torque

# In[ ]:


idx_1 = df_train[((df_train['Operating State'].isin([11, 12])) |
                   (df_train['Turbine State'].isin([3, 4, 5]))) |
                   (df_train['State and Fault'].isin([7, 1, 5])) |
                   (df_train['N-set 1'].isin([0]))].index

idx_2 = df_train.drop(index=idx_1).loc[((df_train['N-set 1'] == 1735) | 
                                        (df_train['Operating State'] == 16) |
                                        (df_train['Turbine State'] == 1) |
                                        (df_train['State and Fault'] == 2))].index

idx_3 = df_train.drop(index=idx_1).drop(index=idx_2).index


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(16, 4))
col_1, col_2 = 'Torque', 'Power(kW)'

axes[0].set_title('Part 1')
axes[1].set_title('Part 2')
axes[2].set_title('Part 3')

axes[0].scatter(df_train.loc[idx_1, col_1], df_train.loc[idx_1, col_2], s=1)
axes[1].scatter(df_train.loc[idx_2, col_1], df_train.loc[idx_2, col_2], s=1)
axes[2].scatter(df_train.loc[idx_3, col_1], df_train.loc[idx_3, col_2], s=1)

plt.show()


# ## Prediction

# ### Cross-Validation of Torque

# In[ ]:


df = df_train.loc[idx_2]
col = 'Torque'

params = {
    'max_depth': 8,
    'num_leaves': 48,
    'num_iterations': 1000
}

cross_validate(model  = LGBMRegressor(**params),
               X      = df.loc[df[col].notnull(), df.columns[~df.columns.isin([col, 'Power(kW)'])]],
               y      = df.loc[df[col].notnull(), [col]])


# In[ ]:


df = df_train.loc[idx_3]
col = 'Torque'

params = {
    'max_depth': 8,
    'num_leaves': 48,
    'num_iterations': 1000
}

cross_validate(model  = LGBMRegressor(**params),
               X      = df.loc[df[col].notnull(), df.columns[~df.columns.isin([col, 'Power(kW)'])]],
               y      = df.loc[df[col].notnull(), [col]])


# ### Imputation of Torque

# In[ ]:


df = df_train.loc[idx_2]
col = 'Torque'

params = {
    'max_depth': 8,
    'num_leaves': 48,
    'num_iterations': 1000
}


y_pred = predict(model   = LGBMRegressor(**params),
             X_train = df.loc[df[col].notnull(), df.columns[~df.columns.isin([col, 'Power(kW)'])]],
             y_train = df.loc[df[col].notnull(), [col]],
             X_test  = df.loc[df[col].isnull(), df.columns[~df.columns.isin([col, 'Power(kW)'])]])

df_train.loc[df[df[col].isnull()].index, [col]] = y_pred


# In[ ]:


df = df_train.loc[idx_3]
col = 'Torque'

params = {
    'max_depth': 8,
    'num_leaves': 48,
    'num_iterations': 1000
}


y_pred = predict(model   = LGBMRegressor(**params),
             X_train = df.loc[df[col].notnull(), df.columns[~df.columns.isin([col, 'Power(kW)'])]],
             y_train = df.loc[df[col].notnull(), [col]],
             X_test  = df.loc[df[col].isnull(), df.columns[~df.columns.isin([col, 'Power(kW)'])]])

df_train.loc[df[df[col].isnull()].index, [col]] = y_pred


# ### Cross-Validation of Part-1

# In[ ]:


df = df_train.loc[idx_1]
col = 'Power(kW)'

params = {
    'max_depth': 6,
    'num_leaves': 32,
    'num_iterations': 40
}

cross_validate(model  = LGBMRegressor(**params),
               X      = df.loc[df[col].notnull(), df.columns[~df.columns.isin([col, 'Power(kW)'])]],
               y      = df.loc[df[col].notnull(), [col]])


# ### Cross-Validation of Part-2

# In[ ]:


df = df_train.loc[idx_2]

col = 'Power(kW)'
params = {
    'max_depth': 6,
    'num_leaves': 32,
    'num_iterations': 100
}

cross_validate(model  = LGBMRegressor(**params),
               X      = df.loc[df[col].notnull(), df.columns[~df.columns.isin([col])]],
               y      = df.loc[df[col].notnull(), [col]])


# ### Cross-Validation of Part-3

# In[ ]:


df = df_train.loc[idx_3]
col = 'Power(kW)'

params = {
    'max_depth': 8,
    'num_leaves': 48,
    'num_iterations': 1000
}

cross_validate(model  = LGBMRegressor(**params),
               X      = df.loc[df[col].notnull(), df.columns[~df.columns.isin([col])]],
               y      = df.loc[df[col].notnull(), [col]])


# ### Prediction of Part-1, Part-2 and Part-3

# In[ ]:


params = [
    {
    'max_depth': 6,
    'num_leaves': 32,
    'num_iterations': 40
    },
    {'max_depth': 6,
    'num_leaves': 32,
    'num_iterations': 100
    },
    {
    'max_depth': 8,
    'num_leaves': 64,
    'num_iterations': 1000
    }]

col = 'Power(kW)'

for idx, param in zip([idx_1, idx_2, idx_3], params):
    df = df_train.loc[idx]
    
    y_pred = predict(model   = LGBMRegressor(**param),
                 X_train = df.loc[df[col].notnull(), df.columns[~df.columns.isin([col])]],
                 y_train = df.loc[df[col].notnull(), [col]],
                 X_test  = df.loc[df[col].isnull(), df.columns[~df.columns.isin([col])]])

    df_train.loc[df[df[col].isnull()].index, [col]] = y_pred


# ## Results

# In[ ]:


df_subm['Power(kW)'] = df_train.loc[df_subm.index, 'Power(kW)']
df_subm.head()


# In[ ]:


fig, axes = plt.subplots(len(df_train.columns), 3, figsize=(16, 4 * len(df_train.columns)))
df_tmp = df_train.copy()
df_tmp['Power(kW)'] = df_subm['Power(kW)']

for col, ax in zip(df_train.columns, axes):
    ax[1].set_title(col)
    ax[0].scatter(df_train.loc[idx_1, col], df_train.loc[idx_1, 'Power(kW)'], s=1)
    ax[0].scatter(df_tmp.loc[idx_1, col], df_tmp.loc[idx_1, 'Power(kW)'], s=1)
    ax[1].scatter(df_train.loc[idx_2, col], df_train.loc[idx_2, 'Power(kW)'], s=1)
    ax[1].scatter(df_tmp.loc[idx_2, col], df_tmp.loc[idx_2, 'Power(kW)'], s=1)
    ax[2].scatter(df_train.loc[idx_3, col], df_train.loc[idx_3, 'Power(kW)'], s=1)
    ax[2].scatter(df_tmp.loc[idx_3, col], df_tmp.loc[idx_3, 'Power(kW)'], s=1)

plt.show()


# In[ ]:




