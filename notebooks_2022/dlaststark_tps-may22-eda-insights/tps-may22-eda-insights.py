#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


# ## Load source datasets

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
train.set_index('id', inplace=True)
print(f"train: {train.shape}")
train.head()


# In[ ]:


test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
test.set_index('id', inplace=True)
print(f"test: {test.shape}")
test.head()


# ## Missing records in datasets

# In[ ]:


pct_df = pd.DataFrame(train.count() * 100 / len(train), 
                      columns=['Percent Non-Missing'])
pct_df['Percent Non-Missing'] = np.round(pct_df['Percent Non-Missing'], 3)

pct_df.plot(kind='barh', figsize=(12, 10), legend=False)
plt.title("Percentage of non-missing values per column (in Train dataset)", pad=20, fontweight='bold');


# In[ ]:


pct_df = pd.DataFrame(test.count() * 100 / len(test), 
                      columns=['Percent Non-Missing'])
pct_df['Percent Non-Missing'] = np.round(pct_df['Percent Non-Missing'], 3)

pct_df.plot(kind='barh', figsize=(12, 10), legend=False)
plt.title("Percentage of non-missing values per column (in Test dataset)", pad=20, fontweight='bold');


# **Insight**: No missing values are present in both train and test datasets

# ## Data Leakage Check

# In[ ]:


common_df = pd.merge(
    train,
    test,
    how='inner',
    on=test.columns.tolist()
)

print(f"Leakage records: {common_df.shape[0]}")


# **Insight**: There's no data leakage between train and test datasets

# ## Categorical vs Continuous Columns

# In[ ]:


for col in test.columns:
    print(f"#unique values in {col} -> {train[col].nunique()}")


# **Insight**:
# 
# - **Continuous Columns**: f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_19, f_20, f_21, f_22, f_23, f_24, f_25, f_26, f_28
# - **Categorical Columns**: f_07, f_08, f_09, f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_29, f_30
# - **Special Column**: f_27

# ## Continuous Columns EDA

# In[ ]:


cont_cols = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 
             'f_06', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 
             'f_24', 'f_25', 'f_26', 'f_28']

fig, ax = plt.subplots(len(cont_cols), 3, figsize=(22, 40))

for i, col in enumerate(cont_cols):
    
    sns.boxplot(x=col, data=train, ax=ax[i][0])
    sns.histplot(x=col, data=train, ax=ax[i][1])
    sns.scatterplot(x=col, y='target', data=train, ax=ax[i][2])
    ax[i][0].set_title(f"Boxplot - {col}", pad=15, fontweight='bold')
    ax[i][1].set_title(f"Histplot - {col}", pad=15, fontweight='bold')
    ax[i][2].set_title(f"Scatterplot - {col} vs target", pad=15, fontweight='bold')

fig.tight_layout();


# **Insight**: All continous columns are normally distributed

# ## Categorical Columns EDA

# In[ ]:


fig, ax = plt.subplots(4, 3, figsize=(22, 30))

col = 'f_07'
sns.countplot(y=col, hue='target', data=train, ax=ax[0][0])
ax[0][0].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_08'
sns.countplot(y=col, hue='target', data=train, ax=ax[0][1])
ax[0][1].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_09'
sns.countplot(y=col, hue='target', data=train, ax=ax[0][2])
ax[0][2].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_10'
sns.countplot(y=col, hue='target', data=train, ax=ax[1][0])
ax[1][0].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_11'
sns.countplot(y=col, hue='target', data=train, ax=ax[1][1])
ax[1][1].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_12'
sns.countplot(y=col, hue='target', data=train, ax=ax[1][2])
ax[1][2].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_13'
sns.countplot(y=col, hue='target', data=train, ax=ax[2][0])
ax[2][0].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_14'
sns.countplot(y=col, hue='target', data=train, ax=ax[2][1])
ax[2][1].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_15'
sns.countplot(y=col, hue='target', data=train, ax=ax[2][2])
ax[2][2].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_16'
sns.countplot(y=col, hue='target', data=train, ax=ax[3][0])
ax[3][0].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_17'
sns.countplot(y=col, hue='target', data=train, ax=ax[3][1])
ax[3][1].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_18'
sns.countplot(y=col, hue='target', data=train, ax=ax[3][2])
ax[3][2].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

fig.tight_layout();


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16, 5))

col = 'f_29'
sns.countplot(y=col, hue='target', data=train, ax=ax[0])
ax[0].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

col = 'f_30'
sns.countplot(y=col, hue='target', data=train, ax=ax[1])
ax[1].set_title(f"Countplot - {col}", pad=15, fontweight='bold')

fig.tight_layout();


# ## Special handling for f_27

# In[ ]:


train['f_27_0'] = train['f_27'].apply(lambda x: list(x)[0])
train['f_27_1'] = train['f_27'].apply(lambda x: list(x)[1])
train['f_27_2'] = train['f_27'].apply(lambda x: list(x)[2])
train['f_27_3'] = train['f_27'].apply(lambda x: list(x)[3])
train['f_27_4'] = train['f_27'].apply(lambda x: list(x)[4])
train['f_27_5'] = train['f_27'].apply(lambda x: list(x)[5])
train['f_27_6'] = train['f_27'].apply(lambda x: list(x)[6])
train['f_27_7'] = train['f_27'].apply(lambda x: list(x)[7])
train['f_27_8'] = train['f_27'].apply(lambda x: list(x)[8])
train['f_27_9'] = train['f_27'].apply(lambda x: list(x)[9])
train.head()


# In[ ]:


for col in ['f_27_0', 'f_27_1', 'f_27_2', 'f_27_3', 'f_27_4', 
            'f_27_5', 'f_27_6', 'f_27_7', 'f_27_8', 'f_27_9']:
    
    enc = OrdinalEncoder().fit(train[col].values.reshape(-1,1))
    train[col] = enc.transform(train[col].values.reshape(-1,1))

train.drop('f_27', axis=1, inplace=True)
train.head()


# ## Features Correlation

# In[ ]:


plt.figure(figsize=(22, 18))
sns.heatmap(train.corr(), annot=True, fmt='.2f', cmap="Set2")
plt.title("Feature Correlation", pad=15, fontweight='bold');


# In[ ]:


# Good Day!!

