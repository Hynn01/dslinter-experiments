#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[ ]:


train= pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv', sep=',', index_col='id')
train.head()


# In[ ]:


test= pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv', sep=',', index_col='id')
test.head()


# # Quick EDA

# In[ ]:


desc = train.describe().reset_index().iloc[1:,:]
desc_t = test.describe().reset_index().iloc[1:,:]

fig, ax = plt.subplots(6,5, figsize=(22,22))

fig.suptitle('Train/Test Describe', fontsize=16)

for i,x in enumerate(desc.columns[1:-1]):
    
    
    if i<5:
        a=i
        b=0
    if i>=5 and i<10:
        a=i-5
        b=1
    if i>=10 and i<15:
        a=i-10
        b=2
    if i>=15 and i<20:
        a=i-15
        b=3
    if i>=20 and i<25:
        a=i-20
        b=4
    if i>=25 and i<30:
        a=i-25
        b=5
    
    ax[b,a].plot(desc['index'],desc[x], marker='.', color='darkblue', label='train')
    ax[b,a].plot(desc['index'],desc_t[x], marker='.', color='red', label='test')
    ax[b,a].legend(['train','test'])
    ax[b,a].set_title(x)


# In[ ]:


plt.figure(figsize=(5,5))
plt.pie(train['target'].value_counts().values, colors=['darkred','blue'], labels=['0','1'])
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('target_distribution')
print(train['target'].value_counts(normalize=True))


# In[ ]:


cols=train.columns[:-1]


# In[ ]:


unique_df = pd.DataFrame(train[train.columns[:-1]]).nunique().reset_index()
unique_df.columns=['features','count']

fig1 = px.bar(unique_df, y='count', x=cols)

fig1.update_layout(title='Feature cardinality in train set',
                  xaxis_title='features',
                  yaxis_title='# unique values',
                  titlefont={'size': 28, 'family':'Serif'},
                  template='simple_white',
                  showlegend=True,
                  width=1000, height=500)
fig1.show()


# In[ ]:


unique_df = pd.DataFrame(test[test.columns]).nunique().reset_index()
unique_df.columns=['features','count']

fig1 = px.bar(unique_df, y='count', x=cols)

fig1.update_layout(title='Feature cardinality in test set',
                  xaxis_title='features',
                  yaxis_title='# unique values',
                  titlefont={'size': 28, 'family':'Serif'},
                  template='simple_white',
                  showlegend=True,
                  width=1000, height=500)
fig1.show()


# In[ ]:


desc = train[train['target']==0].describe().reset_index().iloc[1:,:]
desc_t = train[train['target']==1].describe().reset_index().iloc[1:,:]

fig, ax = plt.subplots(6,5, figsize=(22,22))

fig.suptitle('Train class label describe', fontsize=16)

for i,x in enumerate(desc.columns[1:-1]):
    
    if i<5:
        a=i
        b=0
    if i>=5 and i<10:
        a=i-5
        b=1
    if i>=10 and i<15:
        a=i-10
        b=2
    if i>=15 and i<20:
        a=i-15
        b=3
    if i>=20 and i<25:
        a=i-20
        b=4
    if i>=25 and i<30:
        a=i-25
        b=5
    
    ax[b,a].plot(desc['index'],desc[x], marker='.', color='green', label='train_0')
    ax[b,a].plot(desc['index'],desc_t[x], marker='.', color='orange', label='train_1')
    ax[b,a].legend(['train_0','train_1'])
    ax[b,a].set_title(x)


# In[ ]:


t = train.groupby('target')[['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30']].mean().reset_index()

desc = t[t['target']==0].reset_index().iloc[:,2:]
desc_t = t[t['target']==1].reset_index().iloc[:,2:]

plt.figure(figsize=(20,5))
_=plt.plot(desc.columns, desc.iloc[0,:], color='purple')
_=plt.plot(desc_t.columns, desc_t.iloc[0,:], color='lime')
_=plt.title('Train: mean by feature/class')
_=plt.legend(['train_0','train_1'])


# In[ ]:


t = train.groupby('target')[['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30']].median().reset_index()

desc = t[t['target']==0].reset_index().iloc[:,2:]
desc_t = t[t['target']==1].reset_index().iloc[:,2:]

plt.figure(figsize=(20,5))
_=plt.plot(desc.columns, desc.iloc[0,:], color='purple')
_=plt.plot(desc_t.columns, desc_t.iloc[0,:], color='lime')
_=plt.title('Train: Median by feature/class')
_=plt.legend(['train_0','train_1'])


# In[ ]:


t = train.groupby('target')[['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30']].std().reset_index()

desc = t[t['target']==0].reset_index().iloc[:,2:]
desc_t = t[t['target']==1].reset_index().iloc[:,2:]

plt.figure(figsize=(20,5))
_=plt.plot(desc.columns, desc.iloc[0,:], color='navy')
_=plt.plot(desc_t.columns, desc_t.iloc[0,:], color='green')
_=plt.title('Train: Std by feature/class')
_=plt.legend(['train_0','train_1'])


# In[ ]:


t = train.groupby('target')[['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30']].skew().reset_index()

desc = t[t['target']==0].reset_index().iloc[:,2:]
desc_t = t[t['target']==1].reset_index().iloc[:,2:]

plt.figure(figsize=(20,5))
_=plt.plot(desc.columns, desc.iloc[0,:], color='purple')
_=plt.plot(desc_t.columns, desc_t.iloc[0,:], color='orange')
_=plt.title('Train: Skew by feature/class')
_=plt.legend(['train_0','train_1'])


# In[ ]:


t = train.groupby('target')[['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30']].apply(pd.DataFrame.kurt).reset_index()

desc = t[t['target']==0].reset_index().iloc[:,2:]
desc_t = t[t['target']==1].reset_index().iloc[:,2:]

plt.figure(figsize=(20,5))
_=plt.plot(desc.columns, desc.iloc[0,:], color='green')
_=plt.plot(desc_t.columns, desc_t.iloc[0,:], color='orange')
_=plt.title('Train: Skew by feature/class')
_=plt.legend(['train_0','train_1'])


# In[ ]:


corr = train[train.columns[:-1]].corr()
fig = go.Figure(data= go.Heatmap(z=corr,
                                 x=corr.index.values,
                                 y=corr.columns.values,
                                 zmin=-0.5,
                                 zmax=0.5
                                 )
                )
fig.update_layout(title_text='<b>Correlation Matrix<b>',
                  title_x=0.5,
                  titlefont={'size': 24},
                  width=900, height=800,
                  xaxis_showgrid=False,
                  yaxis_showgrid=False,
                  yaxis_autorange='reversed', 
                  paper_bgcolor=None,
                  )
fig.show()


# In[ ]:


cols = train.columns[:-1]
fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Scatter(y=(train[cols]==0).mean(),
                         x=cols,
                         name = 'train_features=0',
                         line=dict(color='royalblue', width=2, dash='solid')
                         ))

fig3.update_layout(title='<b>% features=0 in training Set set<b>',
                  xaxis_title='Feature',
                  yaxis_title='#Negatives',
                  titlefont={'size': 28, 'family':'Serif'},
                  template='simple_white',
                  showlegend=True,
                  width=1000, height=500)
fig3.show()


# In[ ]:


fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Scatter(y=(test[cols]==0).mean(),
                         x=cols,
                         name = 'test_features=0',
                         line=dict(color='green', width=2, dash='solid')
                         ))
fig3.update_layout(title='<b>% features=0 in test Set set<b>',
                  xaxis_title='Feature',
                  yaxis_title='#Negatives',
                  titlefont={'size': 28, 'family':'Serif'},
                  template='simple_white',
                  showlegend=True,
                  width=1000, height=500)
fig3.show()


# # Feature Interaction

# ### scatterplot based on correlation chart 

# In[ ]:


cols = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30','target']

t = train[cols]

c1=['f_28','f_28', 'f_22','f_25']
c2=['f_03','f_05','f_30', 'f_23']
colors = {0:'purple',1:'lime'}

for i in range(4):   
    
    plt.figure(figsize=(8,8))
    _=plt.scatter(t[c1[i]], t[c2[i]], c=t['target'].map(colors), alpha=0.8)
    _=plt.title(str(c1[i])+' '+str(c2[i])+' interaction')
    _=plt.legend(['train_0','train_1'])


# ## Charts PCA based 

# ### PCA on all the numeric columns (int included)

# In[ ]:


cols = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30','target']

t = train[cols]


# In[ ]:


ss = MinMaxScaler()
pca = PCA(n_components=10)
t2 = pd.DataFrame(ss.fit_transform(t[cols[:-1]]), index=t.index, columns=cols[:-1])
t1 = pd.concat([t2,t['target']], axis=1)
t1 = pd.DataFrame(pca.fit_transform(t1[cols[:-1]]), index=train.index)


# In[ ]:


plt.figure(figsize=(15,5))
_=plt.plot(pca.explained_variance_ratio_.cumsum(), color='darkblue', linewidth=4)
_=plt.title('Explained variance Ration by PCA components')


# ### Component 0 vs Others 

# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[0], t1[1], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(0)+' '+str(1)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[0], t1[2], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(0)+' '+str(2)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[0], t1[3], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(0)+' '+str(3)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[0], t1[4], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(0)+' '+str(4)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[0], t1[5], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(0)+' '+str(5)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[0], t1[6], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(0)+' '+str(6)+' interaction')
_=plt.legend(['train_0','train_1'])


# ### Component 1 vs Others

# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[1], t1[2], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(1)+' '+str(2)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[1], t1[3], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(1)+' '+str(3)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[1], t1[4], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(1)+' '+str(4)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[1], t1[5], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(1)+' '+str(5)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[1], t1[6], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(1)+' '+str(6)+' interaction')
_=plt.legend(['train_0','train_1'])


# ### Component 2 vs Others 

# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[2], t1[3], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(2)+' '+str(3)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[2], t1[4], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(2)+' '+str(4)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[2], t1[5], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(2)+' '+str(5)+' interaction')
_=plt.legend(['train_0','train_1'])


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[2], t1[6], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(2)+' '+str(6)+' interaction')
_=plt.legend(['train_0','train_1'])


# # f_28 interaction

# In[ ]:


t = train[cols]

for i in t.columns[:-1]:
    t[i]=t[i]+ t['f_28']

tr = t.groupby('target')[['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30']].median().reset_index()

desc = tr[tr['target']==0].reset_index().iloc[:,2:]
desc_t = tr[tr['target']==1].reset_index().iloc[:,2:]

plt.figure(figsize=(20,5))
_=plt.plot(desc.columns, desc.iloc[0,:], color='purple', linewidth=4)
_=plt.plot(desc_t.columns, desc_t.iloc[0,:], color='lime', linewidth=4)
_=plt.title('+ f_28 to all the features in Training set: Median by feature/class')
_=plt.legend(['train_0','train_1'])


# In[ ]:


cols = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30','target']

t = train[cols]

for i in t.columns[:-1]:
    t[i]=t[i]- t['f_28']

tr = t.groupby('target')[['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08',
       'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17',
       'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26',
       'f_28', 'f_29', 'f_30']].median().reset_index()

desc = tr[tr['target']==0].reset_index().iloc[:,2:]
desc_t = tr[tr['target']==1].reset_index().iloc[:,2:]

plt.figure(figsize=(20,5))
_=plt.plot(desc.columns, desc.iloc[0,:], color='purple', linewidth=4)
_=plt.plot(desc_t.columns, desc_t.iloc[0,:], color='lime', linewidth=4)
_=plt.title('- f_28 to all the features in Training set: Median by feature/class')
_=plt.legend(['train_0','train_1'])


# # First EDA on Feature 27 by n-grams

# ### 1 gram

# In[ ]:


for i in range(10):
    
    train['first_27']=train.f_27.str[i]
    t=train.groupby(['target','first_27'])['f_00'].count().reset_index()
    a = t[t['target']==0]
    b = t[t['target']==1]
    plt.figure(figsize=(10,5))
    _=plt.plot(b['first_27'],b['f_00'], color='purple', linewidth=4)
    _=plt.plot(a['first_27'],a['f_00'], color='lime', linewidth=4)
    _=plt.title(str(i)+'° digit - Train set feature 27')
    _=plt.legend(['train_0','train_1'])


# ### 2 grams

# In[ ]:


for i in range(6):
    
    train['first_27']=train.f_27.str[i:(i+2)]
    t=train.groupby(['target','first_27'])['f_00'].count().reset_index()
    a = t[t['target']==0]
    b = t[t['target']==1]
    plt.figure(figsize=(20,5))
    _=plt.plot(b['first_27'],b['f_00'], color='purple', linewidth=4)
    _=plt.plot(a['first_27'],a['f_00'], color='lime', linewidth=4)
    _=plt.title(str(i)+'-'+str(i+1)+'° digits - Train set feature 27')
    _=plt.legend(['train_0','train_1'])
    _=plt.xticks(rotation=90)


# ### 3 grams

# In[ ]:


for i in range(1):
    
    train['first_27']=train.f_27.str[i:(i+3)]
    t=train.groupby(['target','first_27'])['f_00'].count().reset_index()
    a = t[t['target']==0]
    b = t[t['target']==1]
    plt.figure(figsize=(20,5))
    
    _=plt.plot(b['first_27'],b['f_00'], color='purple', linewidth=4)
    _=plt.plot(a['first_27'],a['f_00'], color='lime', linewidth=4)
    _=plt.title(str(i)+'-'+str(i+2)+'° digits - Train set feature 27')
    _=plt.legend(['train_0','train_1'])
    _=plt.xticks(rotation=90)


# ### Encoding f_27 

# In[ ]:


tr2 = train[['f_27','target']]
for i in range(10):
    
    tr2[str(i) +'_27']=tr2.f_27.str[i]

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()

tr27 = pd.DataFrame(enc.fit_transform(tr2[[ '0_27', '1_27', '2_27', '3_27', '4_27', '5_27',
       '6_27', '7_27', '8_27', '9_27']]), index=tr2.index, columns=[ '0_27', '1_27', '2_27', '3_27', '4_27', '5_27',
       '6_27', '7_27', '8_27', '9_27'])

tr27.head()


# ### PCA chart including f_27 

# In[ ]:


pca = PCA(n_components=10)
t1 = pd.concat([t2,tr27,t['target']], axis=1)
t1 = pd.DataFrame(pca.fit_transform(t1[cols[:-1]]), index=train.index)


# In[ ]:


plt.figure(figsize=(15,5))
_=plt.plot(pca.explained_variance_ratio_.cumsum(), color='darkblue', linewidth=4)
_=plt.title('Explained variance Ration by PCA components')


# In[ ]:


plt.figure(figsize=(8,8))
_=plt.scatter(t1[0], t1[1], c=t['target'].map(colors), alpha=0.5)
_=plt.title('Components: '+str(0)+' '+str(1)+' interaction')
_=plt.legend(['train_0','train_1'])

