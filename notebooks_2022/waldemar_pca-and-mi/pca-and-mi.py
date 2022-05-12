#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all' #'last_expr'

import math, time, datetime as dt, os, sys 
from pathlib import Path

import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_colwidth = 999
pd.options.display.max_rows = 101

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme()

import numpy as np
np.set_printoptions(edgeitems=5,linewidth=250)

# Feature Engineering and Pre Processing
from scipy.stats import ks_2samp, boxcox
from scipy.special import inv_boxcox
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, power_transform, StandardScaler
from sklearn.decomposition import PCA

data_raw_path = '/kaggle/input/tabular-playground-series-may-2022'
data_processed_path = 'data-processed'


# In[ ]:


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)


# In[ ]:


df_train = pd.read_csv(f'{data_raw_path}/train.csv', index_col='id')
df_sub = pd.read_csv(f'{data_raw_path}/test.csv', index_col='id')

target_col = 'target'
fature_cols = df_sub.columns[1:].to_numpy()

float_cols = df_sub.select_dtypes('float64').columns.to_list()
int_cols = df_sub.select_dtypes('int64').columns.to_list()


# # PCA - floats + ints

# In[ ]:


ss = StandardScaler()
cols = float_cols + int_cols
X = pd.DataFrame(ss.fit_transform(df_train[cols]), columns = cols)
# X = df_train[float_cols] # no standard scaller
y = df_train[target_col]

comp_count = 10
pca = PCA(n_components=comp_count, random_state=42)
X_pca = pca.fit_transform(X, y)

pc_cols = [f'PC_{i+1}' for i in range(comp_count)]
pd.DataFrame(
    pca.components_.T,
    columns=pc_cols,
    index=X.columns,
)

plot_variance(pca)


# # PCA - floats

# In[ ]:


ss = StandardScaler()
cols = float_cols
X = pd.DataFrame(ss.fit_transform(df_train[cols]), columns = cols)
# X = df_train[float_cols] # no standard scaller
y = df_train[target_col]

comp_count = 8
pca = PCA(n_components=comp_count, random_state=42)
X_pca = pca.fit_transform(X, y)

pc_cols = [f'PC_{i+1}' for i in range(comp_count)]
pd.DataFrame(
    pca.components_.T,
    columns=pc_cols,
    index=X.columns,
)

plot_variance(pca)


# In[ ]:


mi_scores = mutual_info_classif(X_pca, y)
pd.Series(data=mi_scores, index=pc_cols)


# # PCA - ints

# In[ ]:


ss = StandardScaler()
cols = float_cols
X = pd.DataFrame(ss.fit_transform(df_train[cols]), columns = cols)
# X = df_train[float_cols] # no standard scaller
y = df_train[target_col]

comp_count = 6
pca = PCA(n_components=comp_count, random_state=42)
X_pca = pca.fit_transform(X, y)

pc_cols = [f'PC_{i+1}' for i in range(comp_count)]
pd.DataFrame(
    pca.components_.T,
    columns=pc_cols,
    index=X.columns,
)

plot_variance(pca)


# In[ ]:


mi_scores = mutual_info_classif(X_pca, y)
pd.Series(data=mi_scores, index=pc_cols)

