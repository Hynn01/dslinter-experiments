#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# In[ ]:


train=pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test=pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')


# In[ ]:


display(train[0:3])
display(test[0:3])
display(train.info())


# In[ ]:


from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df


# In[ ]:


train=labelencoder(train)
test=labelencoder(test)


# In[ ]:


train.target.value_counts()


# In[ ]:


cols=train.columns.tolist()
print(cols)
print(len(cols))


# In[ ]:


fig, ax = plt.subplots(16,2,figsize=(16,48))
for i in tqdm(range(32)):
    r=i//2
    c=i%2
    sns.histplot(train[train.target==0][cols[i]], label=cols[i]+' target=0', ax=ax[r,c], color='black',bins=40)
    sns.histplot(train[train.target==1][cols[i]], label=cols[i]+' target=1', ax=ax[r,c], color='C1',bins=40)
    ax[r,c].legend()
    ax[r,c].grid()

plt.show()


# In[ ]:





# In[ ]:




