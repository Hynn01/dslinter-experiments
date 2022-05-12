#!/usr/bin/env python
# coding: utf-8

# # Art of EDA 
# 
# > - In this Notebook, I will perform the Exploratory Data Analysis with the aim to describe the nuances in the data which would help one to do strong feature engineering and build robust models.
# > - My biggest motive is to promote the use of statistics in the EDA process.
# > - Some of my analysis is motivated by this wonderful notebook of AmbroseM <a href = "https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense/notebook">link here</a>
# > - Please Upvote if you find this notebook useful.

# ---
# 
# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ---
# 
# # Data Loading

# In[ ]:


data_dir = "../input/tabular-playground-series-may-2022" 
train = pd.read_csv(f"{data_dir }/train.csv")
test = pd.read_csv(f"{data_dir}/test.csv")
ss = pd.read_csv(f"{data_dir}/sample_submission.csv")


# ---
# 
# # EDA

# In[ ]:


print(train.shape)
train.head()


# In[ ]:


train.tail()


# ## Train Data Summary

# In[ ]:


print("Train Data Summary")
print("-"*50)
print(f"Total number of rows -- {train.shape[0]}")
print(f"Total number of columns -- {train.shape[1]}")
print("Target :-")
print("0 - {}({:0.2f}%)".format(train[train.target == 0].shape[0], train[train.target == 0].shape[0]/train.shape[0]* 100))
print("1 - {}({:0.2f}%)".format(train[train.target == 1].shape[0], train[train.target == 1].shape[0]/train.shape[0]* 100))


# ## Column Summary

# In[ ]:


def print_column_summary(data):
    name = []
    dtype = []
    unique_values = []
    missing = []
    for column in data.columns:
        name.append(str(column))
        data_type = str(data[column].dtypes)
        dtype.append(data_type)
        if(data_type == 'float64'):
            unique_values.append("")
        else:
            unique_values.append(str(data[column].nunique()))
        missing.append("{:0.2f} % ".format(data[column].isnull().sum() / data.shape[0]))
    
    dfSummary = pd.DataFrame(name,columns = ["Name"])
    dfSummary["Dtypes"] = dtype
    dfSummary["Unique Value Count"] = unique_values
    dfSummary["Missing Value %"] = missing
    return dfSummary


# In[ ]:


summary = print_column_summary(train)
summary


# In[ ]:


print("So the dataset have no missing values")
print("-"*50)
print("Variable Dataypes :-")
print(summary.Dtypes.value_counts())


# ---
# 
# # Analyze float64 features

# There are 16 float64 features, lets plot there histogram individually

# In[ ]:


float_features = [f for f in train.columns if train[f].dtype == 'float64']

# Training histograms
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for f, ax in zip(float_features, axs.ravel()):
    ax.hist(train[f], density=True, bins=100)
    ax.set_title(f'Train {f}, std={train[f].std():.1f}')
plt.suptitle('Histograms of the float features')
plt.show()


# Apparently, all the float 64 features are following normal distribution

# ## Correlation Plot

# In[ ]:


plt.figure(figsize=(12, 12))
sns.heatmap(train[float_features + ['target']].corr(), center=0, annot=True, fmt='.1f')


# In[ ]:


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top 10 Absolute Correlations")
print(get_top_abs_correlations(train[float_features+ ['target']], 10))


# Pair with highest correlation is `f03` and `f_28` with a correlation of 0.33 which is not high.
# 
# Hence, we can safely conclude that correlation between the float64 variable won't bother our ml models
# 

# ## Mutual Information Statistic between Float64 features and Target

# In[ ]:


from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(train[float_features], train['target'],discrete_features='auto', n_neighbors=3, copy=True, random_state=3)


# In[ ]:


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

mi_score_df = pd.Series(mi_scores, index = float_features,name = 'scores')
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_score_df)


# Above plot is in the decreasing order of mutual information with the target variable. This will be handy when we perform feature selection.

# ## Exploring two float64 variable with Target

# In[ ]:


train_sample = train.sample(frac = 0.05)
cols = float_features + ['target']
fig, axs = plt.subplots(30, 4, figsize=(16, 80))
i = 0
while(len(cols) > 2):
    col = cols.pop(0)
    for other_cols in cols:
        if(other_cols == "target"):
            continue
        axs[i//4][i%4].scatter(train_sample[col].values, train_sample[other_cols].values, c=train_sample["target"])
        axs[i//4][i%4].set_title(f'{col} vs {other_cols}')

        i += 1
plt.show()


# In the above plot, we can see that all the plots have overlapping distribution w.r.t to target class and hence a linear classifier would find it hard to classify the records

# ---
# 
# # Analyze int64 features
# 
# There are 16 int64 features, lets plot their value counts

# In[ ]:


int_features = [f for f in train.columns if train[f].dtype == 'int64']
int_features.pop(0) 
fig, axs = plt.subplots(5, 3, figsize=(20, 16))

for i,col in enumerate(int_features):
    train[col].value_counts(normalize = True).plot(kind = 'bar',ax = axs[i//3][i%3], title =col )


# From the above charts, we can conclude that many of the features have rare class problem, so we'd have to deal with them.
# 
# One possible way is to group all the rare classes as others.
# 
# Feature f_29 and f_30 are binary and ternary respectively

# ## Distribution of int64 feature categories with target mean

# In[ ]:




int_features = [f for f in train.columns if train[f].dtype == 'int64' and f!= "target" and f!= "id"]

# Training histograms
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for f, ax in zip(int_features, axs.ravel()):
    plotdf = pd.DataFrame(train.groupby(f)["target"].mean()).reset_index()
    ax.bar(plotdf[f].values, plotdf["target"].values)
    ax.set_title(f'{f}')
plt.suptitle('Distribution of the int64 feature categories w.r.t target mean')
plt.show()


# Distribution of int64 features categories w.r.t target mean is mostly same. There are some unusually high and low values for some categories. But the count of those categories are very low, so this unusual behaviour is due to their low counts. 

# ## Mutual Information Statistic between int64 features and Target

# In[ ]:


from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(train[int_features], train['target'],discrete_features='auto', n_neighbors=3, copy=True, random_state=3)


# In[ ]:


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

mi_score_df = pd.Series(mi_scores, index = int_features,name = 'scores')
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_score_df)


# As per mutual information, `f_30` is the most important `int64` feature. Remaining all the other features have similar mutual information score.

# ---
# 
# # Analyze string feature
# 
# f_27 is the only string feature. Interesting thing to know is that the string always has length 10. Hence, we can create a useful feature as created by AmbroseM in his <a href = "https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense/notebook">notebook</a>.
# 
# We have to split the strings into ten separate categorical features.

# In[ ]:


for i in range(10):
    train[f'ch{i}'] = train.f_27.str.get(i).apply(ord) - ord('A')


# ## Mutual Information Statistic between string categorical features and Target
# 

# In[ ]:


string_features = train.columns[-10:]

from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(train[string_features], train['target'],discrete_features='auto', n_neighbors=3, copy=True, random_state=3)


# In[ ]:


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

mi_score_df = pd.Series(mi_scores, index = string_features,name = 'scores')
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_score_df)


# By the mutual importance score, we can conclude that `ch2` will be a useful feature in modeling since it has the highest mutual information score among other string categorical features derived from column `f_27`.
# 

# # To be continued ..
