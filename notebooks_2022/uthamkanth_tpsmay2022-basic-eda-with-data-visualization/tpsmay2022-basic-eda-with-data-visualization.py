#!/usr/bin/env python
# coding: utf-8

# # <p style="background-color:turquoise; font-family:newtimeroman; font-size:250%; text-align:center; border-radius: 15px 50px;">Tabular Playground Series  - May 2022 </p>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[ ]:


train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# In[ ]:


train.info()


# ### Summary
# 
# - There are no missing values in both train ans test dataset.
# - The train consists of 900000 data, and the test consists of 700000 data.
# - independent variables are almost continuos feature.
# - f_27 is a categorical feature
# - The value of target is 0 or 1.
# - The value of target is almost equally balanced. 

# In[ ]:


train.describe().transpose()


# In[ ]:


test.describe().transpose()


# 
# - In the train set , data is normally distributed since most of the variables has mean close to zero with the standard deviation close to 1.
# - In the test set , The stats are in exponential form the standard deviation is similar to train set

# In[ ]:


fig = px.histogram(
    train, 
    x=train['target'], 
    color=train['target'],
)
fig.update_layout(
    title_text='Target distribution', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    
)
fig.show()


# ## Feature : f_27
# - f_27 is the only variable with object type and lets understand if we can convert this to categorical

# In[ ]:


print("Levels in Train :: ", train['f_27'].nunique())
print("Levels in Test :: ", test['f_27'].nunique())
print("Difference(New categories in test set) :: ", test['f_27'].nunique() - train['f_27'].nunique())
print("percentage(%) of categorical variable :: ", train['f_27'].nunique() / train.shape[0])


# In[ ]:


# since the data is huge , lets sample the data for faster results
np.random.seed(25)
train = train.sample(50000)
test = test.sample(50000)


# In[ ]:


fig, axes = plt.subplots(9,3,figsize=(12, 12))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    idx = str(idx).zfill(2)
    sns.kdeplot(data=train, x=f'f_{idx}', 
                fill=True, 
                ax=ax)
    sns.kdeplot(data=test, x=f'f_{idx}', 
                fill=True, 
                ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['left'].set_visible(False)
    ax.set_title(f'f_{idx}', loc='right', weight='bold', fontsize=10)

fig.supxlabel('feature distribution', ha='center', fontweight='bold')

fig.tight_layout()
plt.show()


# - The distribution of train and test is similar.
# - Features from f_7 to f_18 has different type of distribution.its better to understand these variables to develop a better model.

# In[ ]:


features = []
for col in train.columns:
    if ('int' in str(train[col].dtype) or 'float' in str(train[col].dtype)) and (col not in ['id', 'target', 'f_27']):
            features.append(col)
        


# In[ ]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
train[features] = ss.fit_transform(train[features])
test[features] = ss.transform(test[features])


# In[ ]:


fig, axes = plt.subplots(9,3,figsize=(12, 12))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    idx = str(idx).zfill(2)
    sns.kdeplot(data=train, x=f'f_{idx}', 
                fill=True, 
                ax=ax)
    sns.kdeplot(data=test, x=f'f_{idx}', 
                fill=True, 
                ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['left'].set_visible(False)
    ax.set_title(f'f_{idx}', loc='right', weight='bold', fontsize=10)

fig.supxlabel('feature distribution', ha='center', fontweight='bold')

fig.tight_layout()
plt.show()


# - After Scaling , Training and Test distributions are similar.we can test and verify the model results if scaling can impact the model performance

# In[ ]:


fig, ax = plt.subplots(figsize=(12 , 12))
corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.heatmap(corr,square=True, center=0, 
            linewidth=0.2, cmap='coolwarm',
           mask=mask, ax=ax) 

ax.set_title(' Correlation Matrix ', loc='left')
plt.show()


# ## If the content is helpful, please upvote. :)
