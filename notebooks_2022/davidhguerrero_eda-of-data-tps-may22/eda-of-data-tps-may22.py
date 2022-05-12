#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')


# In[ ]:


train.info()


# # Study each attribute and its characteristics:

# ### Unique values

# In[ ]:


selected_columns = train.columns
dropped_features = ['target', 'id', 'f_27']
numerical_features = [f for f in selected_columns if f not in dropped_features]


# In[ ]:


print("Count of unique values per feature:")
for feature in range(31):
    feature_number = f"f_{feature:02d}"
    print(f"{feature_number}: {len(np.unique(train[feature_number])):6d}")


# ### % of missing 

# In[ ]:


#None missing vualues
train.isnull().sum(axis=0)


# ### Descriptive statistics of attirbutes: min, max, mean, std, varciance,

# In[ ]:


train.describe()


# ### % of duplicate 

# In[ ]:


#None duplicates
train_no_duplicates = train.drop_duplicates()
100 - 100 * len(train_no_duplicates)/len(train)


# #### Outliers
# Thanks to [@snikhil17](https://www.kaggle.com/snikhil17) for share this [link](https://www.kaggle.com/code/snikhil17/making-basic-eda-attractive/notebook)

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Reference: https://www.kaggle.com/suharkov/sep-2021-playground-eda-no-model-for-now
train0 = train[numerical_features]
train_plot = ((train0 - train0.min())/(train0.max() - train0.min()))
fig, ax = plt.subplots(3, 1, figsize = (25,25))
sns.boxplot(data = train_plot.iloc[:, 1:10], ax = ax[0])
sns.boxplot(data = train_plot.iloc[:, 10:20], ax = ax[1])
sns.boxplot(data = train_plot.iloc[:, 20:30], ax = ax[2])
#sns.boxplot(data = train_plot.iloc[:, 30:29], ax = ax[2])


# # Visualize the data.

# ### Type of distribution

# In[ ]:


#Reference: https://www.kaggle.com/suharkov/sep-2021-playground-eda-no-model-for-now 
nrows = 6
ncols = 5
i = 0
fig, ax = plt.subplots(nrows, ncols, figsize = (25,45))
for row in range(nrows):
    for col in range(ncols):
        sns.histplot(data = train0.iloc[:, i], bins = 50, ax = ax[row, col], palette  = 'bone_r').set(ylabel = '')
        i += 1


# **Insights**
# * Some features from f_07 to f_18 has is positive skewed in various cases. Skewness is a measure of the symmetry in a distribution. 
#     * To reduce right skewness, take roots or logarithms or reciprocals.
# * Other ones like from f_00 to f_06 and from f_19 to f_26 and f_28 are Gaussian, 
# * f_29 is Binomial (target)
# 

# In[ ]:


train.select_dtypes(exclude=['object', 'int64']).columns


# ## Study the correlations between attributes.
# Thanks to [@dwin183287](ttps://www.kaggle.com/dwin183287) for this [notebook](https://www.kaggle.com/code/dwin183287/30-days-of-ml-eda?scriptVersionId=72292970&cellId=32)

# ### Correlation with target

# In[ ]:


import matplotlib


# In[ ]:


train0 = train.select_dtypes(include=['float64'])


# In[ ]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(12, 8), facecolor=background_color)
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])
colors = ["#2f5586", "#f6f5f5","#2f5586"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

ax0.set_facecolor(background_color)
#ax0.text(0, 0.064, 'Correlation of Continuous Features with Target', fontsize=20, fontweight='bold')
#ax0.text(-1.1, 0.064, 'There is no features that pass 0.06 correlation with target', fontsize=13, fontweight='light')

chart_df = pd.DataFrame(train0.corrwith(train['target']))
chart_df.columns = ['corr']
sns.barplot(x=chart_df.index, y=chart_df['corr'], ax=ax0, color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
ax0.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.set_ylabel('')

for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)

plt.show()


# ### Features and Target Relation

# In[ ]:


cont_features = train.select_dtypes(include=['float64']).columns


# In[ ]:


train[cont_features].describe()


# In[ ]:


fig = plt.figure(figsize=(15, 15), facecolor = '#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.3)

background_color = "#f6f5f5"
cmap = sns.light_palette('#2f5586', as_cmap=True)

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in cont_features:
        locals()["ax"+str(run_no)].hexbin(x=train[feature], y=train['target'], gridsize=15, 
                                      cmap=cmap, zorder=2, facecolor='black', mincnt=1)
        locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
        locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
        locals()["ax"+str(run_no)].set_ylabel('target', fontsize=10, fontweight='bold')
        locals()["ax"+str(run_no)].set_xlabel(feature, fontsize=10, fontweight='bold')
        run_no += 1
        
#ax0.text(-0.2, 13, 'Features and Target Relation', fontsize=20, fontweight='bold')
#ax0.text(-0.2, 12, 'To see the the correlation concentration to the target', fontsize=13, fontweight='light')

ax14.remove()
ax15.remove()

plt.show()


# In[ ]:


discrete_features = train.select_dtypes(include=['int64']).drop(columns=['id','target']).columns


# In[ ]:


fig = plt.figure(figsize=(15, 15), facecolor = '#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.5, hspace=0.3)

background_color = "#f6f5f5"
cmap = sns.light_palette('#2f5586', as_cmap=True)

run_no = 0
for row in range(0, 4):
    for col in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right"]:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

run_no = 0
for feature in discrete_features:
        locals()["ax"+str(run_no)].hexbin(x=train[feature], y=train['target'], gridsize=15, 
                                      cmap=cmap, zorder=2, facecolor='black', mincnt=1)
        locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
        locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
        locals()["ax"+str(run_no)].set_ylabel('target', fontsize=10, fontweight='bold')
        locals()["ax"+str(run_no)].set_xlabel(feature, fontsize=10, fontweight='bold')
        run_no += 1
        
#ax0.text(-0.2, 13, 'Features and Target Relation', fontsize=20, fontweight='bold')
#ax0.text(-0.2, 12, 'To see the the correlation concentration to the target', fontsize=13, fontweight='light')

ax14.remove()
ax15.remove()

plt.show()


# In[ ]:


train0 = train.select_dtypes(include=['int64']).drop(columns=['id','target'])


# In[ ]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(12, 8), facecolor=background_color)
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])
colors = ["#2f5586", "#f6f5f5","#2f5586"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

ax0.set_facecolor(background_color)
#ax0.text(0, 0.064, 'Correlation of Continuous Features with Target', fontsize=20, fontweight='bold')
#ax0.text(-1.1, 0.064, 'There is no features that pass 0.06 correlation with target', fontsize=13, fontweight='light')

chart_df = pd.DataFrame(train0.corrwith(train['target']))
chart_df.columns = ['corr']
sns.barplot(x=chart_df.index, y=chart_df['corr'], ax=ax0, color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
ax0.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.set_ylabel('')

for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)

plt.show()


# ### Correlation between features

# In[ ]:


train0 = train.drop(columns=['id','target'])


# In[ ]:


background_color = "#f6f5f5"

fig = plt.figure(figsize=(32, 12), facecolor=background_color)
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
colors = ["#2f5586", "#f6f5f5","#2f5586"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

ax0.set_facecolor(background_color)
ax0.text(0, -1, 'Features Correlation on Train Dataset', fontsize=20, fontweight='bold')
#ax0.text(0, -0.4, 'Highest correlation in the dataset is 0.5', fontsize=13, fontweight='light')

ax1.set_facecolor(background_color)
ax1.text(-0.1, -1, 'Features Correlation on Test Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
#ax1.text(-0.1, -0.4, 'Features in test dataset resemble features in train dataset ', fontsize=13, fontweight='light', fontfamily='serif')

sns.heatmap(train0.corr(), ax=ax0, vmin=-1, vmax=1, annot=True, square=True, 
            cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1f')

sns.heatmap(test.corr(), ax=ax1, vmin=-1, vmax=1, annot=True, square=True, 
            cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1f')

plt.show()


# ## Please upvote if you like it =)
