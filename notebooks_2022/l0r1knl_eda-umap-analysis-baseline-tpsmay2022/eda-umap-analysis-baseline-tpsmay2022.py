#!/usr/bin/env python
# coding: utf-8

# # EDA for Tabular Playground Series May 2022.
# 
# ## What to do in this notebook.
# - Show basic infomations.
# - Plot distribution of features.
# - Decompose Train datas.
# - Classification by LightGBM and Catboost.
# - Show Feature Importances.
# - Predict Test Data.

# ## Competition's OverView

# ### English
# ---
# The May edition of the 2022 Tabular Playground series binary classification problem that includes a number of different feature interactions.  
# This competition is an opportunity to explore various methods for identifying and exploiting these feature interactions.
# 
# ### Japanese
# ---
# 2022年5月号のTabular Playgroundシリーズの2値分類問題は、多くの異なる特徴の相互作用を含んでいます。  
# このコンペティションは、これらの特徴的な相互作用を識別し、利用するための様々な方法を探る機会です。
# 
# ## Data Description
# 
# For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1.  
# The data has various feature interactions that may be important in determining the machine state.
# 
# Good luck!
# 
# ### Files
# - train.csv - the training data, which includes normalized continuous data and categorical data
# - test.csv - the test set; your task is to predict binary target variable which represents the state of a manufacturing process
# - sample_submission.csv - a sample submission file in the correct format
# 

# # SETUP

# ## import modules.

# In[ ]:


# !sh /content/drive/MyDrive/Colab\ Notebooks/Competitions/kaggle/TPS_MAY_2022/install_modules.sh


# In[ ]:


# basic module.
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# to decompose.
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
from umap.parametric_umap import ParametricUMAP

# for classification and visualizing feature importances.
import lightgbm as lgb
import catboost as cbt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score)


# ## Util Function

# In[ ]:


import logging
import time
from contextlib import contextmanager

import warnings
warnings.simplefilter('ignore')

@contextmanager
def timer(name, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time.time()
    print_(f'[{name}] start')
    yield
    print_(f'[{name}] done in {time.time() - t0:.0f} s')
    


# In[ ]:


class Config(object):
    def __init__(self):
        # data paths
        self.train_data = '../input/tabular-playground-series-may-2022/train.csv'
        self.test_data = '../input/tabular-playground-series-may-2022/test.csv'
        self.count_each_alphabet = '../input/tps-may-2022-eda-output/count_each_alphabet.csv'
        self.train_decompose_X = '../input/tps-may-2022-eda-output/train_decompose_X.csv'
        self.train_decompose_y = '../input/tps-may-2022-eda-output/train_decompose_y.csv'
        self.submission_file = '../input/tabular-playground-series-may-2022/sample_submission.csv'

        # general
        self.random_state = 0

        # training parameters
        self.fold = 10
        self.lightgbm = dict(
            random_state=self.random_state,
            # device='gpu',
        )

        self.catboost = dict(
            random_state=self.random_state,
            task_type='GPU',
        )
    def __set_seeds(self):
        return None

config = Config()


# ## load dataset.

# In[ ]:


train_data = pd.read_csv(config.train_data, index_col='id')
test_data = pd.read_csv(config.test_data, index_col='id')

X_cols = [f'f_{i:02}' for i in range(31)]
y_cols = ['target']

data = pd.concat([train_data.loc[:, X_cols], test_data.loc[:, X_cols]], axis=0)


# # Show Basic Infomation.

# ## code

# In[ ]:


def get_basic_infomation(dataframe:pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    dtypes = dataframe.dtypes
    dtypes.name = 'dtype'
    isna = dataframe.isna().sum()
    isna.name = 'null-count'
    nunique = dataframe.nunique()
    nunique.name = 'nunique'
    infomations = pd.concat([dtypes, nunique, isna, dataframe.describe().T], axis=1)

    object_columns = dtypes[dtypes == 'object'].index
    
    if object_columns.shape[0] > 0:
        for idx in object_columns:
            infomations.loc[idx, 'count'] = dataframe.loc[:, idx].notna().sum()
    
    return infomations


# ## Train Data

# In[ ]:


get_basic_infomation(train_data)


# ## Test Data

# In[ ]:


get_basic_infomation(test_data)


# ## All Data(concat Train Data and Test Data)

# In[ ]:


get_basic_infomation(data)


# ## Convert data type

# - Float Features.  
# Normal features?
# - Int Features.  
# Categorical ID?
# - Object Features.  
# Too many to handle as categories.  
# I try count the alphabet. 

# In[ ]:


# get columns by data type.
flt_cols = data.dtypes[data.dtypes == 'float64'].index.to_list()
int_cols = data.dtypes[data.dtypes == 'int64'].index.to_list()
obj_cols = data.dtypes[data.dtypes == 'object'].index.to_list()


# In[ ]:


# # convert dtype
# train_data.loc[:, flt_cols] = train_data.loc[:, flt_cols].astype(np.float32)
# train_data.loc[:, int_cols] = train_data.loc[:, int_cols].astype(np.int8)

# test_data.loc[:, flt_cols] = test_data.loc[:, flt_cols].astype(np.float32)
# test_data.loc[:, int_cols] = test_data.loc[:, int_cols].astype(np.int8)

# data.loc[:, flt_cols] = data.loc[:, flt_cols].astype(np.float32)
# data.loc[:, int_cols] = data.loc[:, int_cols].astype(np.int8)


# # Plot Distribution of Features

# ## Distribution of Float Features

# ### code

# In[ ]:


def plot_violin_multicolumns(
    dataframe:pd.DataFrame,
    columns:list,
    hue: str = None
):
    """
    """
    if not(hue is None):
        hue_col_idx = np.argwhere(np.array(columns) == hue).flatten()[0]
        cols = np.delete(np.array(columns), hue_col_idx)

        violin_data = dataframe.loc[:, cols].melt(var_name='columns', value_name='value')
        violin_hue = pd.concat([dataframe[hue] for _ in range(len(cols))], axis=0)
        violin_hue.name = hue
        violin_hue.index = np.arange(violin_hue.shape[0])
        violin_data = pd.concat([violin_data, violin_hue], axis=1)
    
    else:
        violin_data = dataframe.loc[:, columns].melt(var_name='columns', value_name='value')
    
    fig, axes = plt.subplots(1, 1, figsize=(20, 8))
    sns.violinplot(
        x='columns', 
        y='value',
        hue=hue if not(hue is None) else None, 
        data=violin_data,
        split=True if not(hue is None) else None,
        ax=axes
    )
    
    plt.close()
    
    return fig


# ### edit data

# In[ ]:


# labeling train or test.
data['type'] = None
data.loc[train_data.index, 'type'] = 'train'
data.loc[test_data.index, 'type'] = 'test'

# temporary scaling... 
train_data.loc[:, 'f_28'] = train_data.loc[:, 'f_28'] / 100.0
data.loc[:, 'f_28'] = data.loc[:, 'f_28'] / 100.0

# save memory...
del test_data


# ### Target 0 vs 1

# In[ ]:


plot_violin_multicolumns(train_data, flt_cols + ['target'], 'target')


# ### Train vs Test

# In[ ]:


plot_violin_multicolumns(data, flt_cols + ['type'], 'type')


# In[ ]:


# inverse scaling...
train_data.loc[:, 'f_28'] = train_data.loc[:, 'f_28'] * 100.0
data.loc[:, 'f_28'] = data.loc[:, 'f_28'] * 100.0


# ## Distribution of Integer Colunms (using value counts)
# 

# ### code

# In[ ]:


def plot_value_counts(
    dataframe:pd.DataFrame,
    column:str,
    hue: str = None,
    ax=None
):
    """
    """
    if hue is None:
        counts = pd.DataFrame(
            dataframe.loc[:, column].value_counts()
        ).reset_index()
        counts.columns = ['value', 'counts']

    else:
        l_counts = []
        for hue_col in dataframe[hue].unique():
            idx = dataframe[dataframe[hue]==hue_col].index
            hue_counts = pd.DataFrame(
                dataframe.loc[idx, column].value_counts()
            ).reset_index()
            hue_counts.columns = ['value', 'counts']
            hue_counts[hue] = hue_col
            l_counts.append(hue_counts)
        counts = pd.concat(l_counts)

    sns.barplot(
        x='value', 
        y='counts',
        data=counts, 
        hue=hue if not(hue is None) else None,
        ax=ax
    )
    ax.set_title(column)
    ax.legend(loc='upper right')
    ax.grid()

    return ax


def plot_value_counts_multicolumns(
    dataframe:pd.DataFrame,
    columns:list,
    hue: str = None,
    subplots_kws: dict = dict()
):
    """
    """
    if not(type(subplots_kws) is dict):
        subplots_kws = dict()
    
    nrow = subplots_kws.get('nrow', 3)
    ncol = subplots_kws.get('ncol', 5)
    figsize = subplots_kws.get('figsize', (18, 10))

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.ravel()
    
    for i, col in enumerate(columns):
        plot_value_counts(dataframe, col, hue, axes[i])

    fig.tight_layout()
    plt.close()
    
    return fig


# ### Target 0 vs 1

# In[ ]:


plot_value_counts_multicolumns(train_data, int_cols, 'target')


# ### Train vs Test

# In[ ]:


plot_value_counts_multicolumns(data, int_cols, 'type')


# ## Distribution of Object Column (using count each alphabet)

# In[ ]:


# f_27 nunique is too big... cant plot value_counts.
data.loc[:, 'f_27'].value_counts()


# ### Count Each alphabet in f_27.
# Count the number of each alphabet in the sequence of f_27.

# In[ ]:


def count_each_alphabet(x):
    """
    """
    count_alphabets = pd.Series(
        np.zeros(len(string.ascii_uppercase)).astype(np.int8),
        index=list(string.ascii_uppercase)
    )
    
    for e in x:
        count_alphabets[e] += 1

    return count_alphabets


# In[ ]:


'''
If you need this data, uncomment and execute the following code.
It takes about 10 minutes.
'''
# f_27_each_counts = data.loc[:, 'f_27'].apply(count_each_alphabet)
# f_27_each_counts = pd.concat([f_27_each_counts.loc[:, list(string.ascii_uppercase)], data.loc[:, 'type']], axis=1)
# f_27_each_counts.to_csv(config.count_each_alphabet)

f_27_each_counts = pd.read_csv(config.count_each_alphabet)
f_27_each_counts.loc[:, list(string.ascii_uppercase)] = f_27_each_counts.loc[:, list(string.ascii_uppercase)].astype(np.int8) 
f_27_each_counts_train = pd.concat(
    [
        f_27_each_counts.loc[train_data.index, list(string.ascii_uppercase)],
        train_data.loc[:, 'target']
    ],
    axis=1
)


# In[ ]:


plot_value_counts_multicolumns(
    f_27_each_counts_train,
    list(string.ascii_uppercase),
    'target',
    dict(nrow=4, ncol=7, figsize=(24, 14))
)


# In[ ]:


plot_value_counts_multicolumns(
    f_27_each_counts,
    list(string.ascii_uppercase),
    'type',
    dict(nrow=4, ncol=7, figsize=(24, 14))
)


# # Decompose.

# ## code

# In[ ]:


def decompose(
    X:pd.DataFrame,
    n_components: int = 2,
    algorithms: list = None,
    cluster_parameters: dict = dict(method='k-means', target='raw', k=3),
    random_state=0
) -> tuple:
    """
    """
    if algorithms is None:
        algorithms = PCA(svd_solver='full')

    if not(type(algorithms) is list):
        algorithms = [algorithms]
    
    decomposed_y = pd.DataFrame(None, index=X.index)
    if cluster_parameters.get('target') in ['both', 'raw']:
        cls_alg = KMeans(
            n_clusters=cluster_parameters['k'],
            random_state=random_state
        )
        decomposed_y[f"{cluster_parameters['method']}"] = cls_alg.fit_predict(X)

    decomposed_X, decomposed_explained = [], []
    for alg in algorithms:
        name = alg.__class__.__name__

        decomposed_X.append(
            pd.DataFrame(
                alg.fit_transform(X)[:, :n_components],
                index=X.index,
                columns=[f"{name}_components_{i+1}" for i in range(n_components)]
            )
        )

        if cluster_parameters.get('target') in ['both', 'decomposed_X']:
            cls_alg = KMeans(
                n_clusters=cluster_parameters['k'],
                random_state=random_state
            )
            decomposed_y[f"{cluster_parameters['method']}"] = cls_alg.fit_predict(decomposed_X[-1])

        if name == 'PCA':
            decomposed_explained.append(explain_pca(alg, X.columns))
        
    decomposed_X = pd.concat(decomposed_X, axis=1)

    return decomposed_X, decomposed_y, decomposed_explained


def explain_pca(pca, columns):
    return None


def plot_decompose_scatter(
    decompose_result:pd.DataFrame, 
    x: int = 0, 
    y: int = 1, 
    hue: str = None, 
    methods: list = ['PCA', 'UMAP']
):
    """
    """
    cmap, palette = None, None

    if hue:
        hue_value = decompose_result.loc[:, hue]
        hue_n = hue_value.unique().shape[0]

        if 20 > hue_n > 8:
            palette = sns.diverging_palette(600, 0, n=hue_n)
        elif hue_n > 20:
            cmap = sns.color_palette("viridis", as_cmap=True)
        else:
            palette = sns.color_palette(n_colors=hue_n)
    else:
        palette = sns.diverging_palette(600, 0, n=1)

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(9*len(methods), 8), facecolor='white'
    )

    if len(methods) == 1:
        axes = (axes, )
    
    for i, method in enumerate(methods):
        axes[i].grid()
        components = decompose_result.filter(regex=f'^({method})', axis=1)
        sns.scatterplot(
            x=components.iloc[:, x],
            y=components.iloc[:, y],
            ax=axes[i],
            hue=decompose_result.loc[:, hue] if type(hue) is str else None,
            palette=palette,
            cmap=cmap
        )

    plt.close()

    return fig


# In[ ]:


use_cols = flt_cols + int_cols + list(string.ascii_uppercase)
dataset = pd.concat(
    [train_data, f_27_each_counts_train.loc[:, list(string.ascii_uppercase)]],
    axis=1
).loc[:, use_cols]

scaler = StandardScaler()
dataset.loc[:, :] = scaler.fit_transform(dataset)
dataset = dataset.astype(np.float16)


# In[ ]:


'''
It takes about 50 minutes.
'''
# decomposed_X, decomposed_y, decomposed_explained = decompose(
#     X=dataset, 
#     n_components=20, 
#     algorithms=[
#         PCA(svd_solver='full'), 
#         UMAP(
#             n_components=20,
#             n_neighbors=10,
#             min_dist=0.01,
#             random_state=config.random_state
#         )
#     ],
# )
# decomposed_X.to_csv(config.train_decompose_X)
# decomposed_y.to_csv(config.train_decompose_y)


# In[ ]:


decomposed_X = pd.read_csv(config.train_decompose_X, index_col='id')
decomposed_y = pd.read_csv(config.train_decompose_y, index_col='id')

decomposed_result = pd.concat(
    [decomposed_X, decomposed_y, train_data['target']],
    axis=1
)


# In[ ]:


plot_decompose_scatter(decomposed_result, hue='target', methods=['PCA', 'UMAP'])


# In[ ]:


plot_decompose_scatter(decomposed_result, hue='k-means', methods=['PCA', 'UMAP'])


# # Classification by LightGBM and Catboost

# ## Training📚

# In[ ]:


dataset = pd.concat(
    [train_data, f_27_each_counts_train.loc[:, list(string.ascii_uppercase)]],
    axis=1
).loc[:, use_cols]

# scaling float features only.
scaling_cols = flt_cols + list(string.ascii_uppercase)
scaler = StandardScaler()
dataset.loc[:, scaling_cols] = scaler.fit_transform(dataset.loc[:, scaling_cols])

int_cols_idx = [np.argwhere(dataset.columns == col).flatten()[0] for col in int_cols]


# In[ ]:


"""
TODO
- Implement Train Classs
"""
validator = StratifiedKFold(
    config.fold,
    shuffle=True,
    random_state=config.random_state
)

# for save variables.
models_cbt, models_lgb = [], []
results_cbt, results_lgb = [], []
roc_auc_scores_cbt, roc_auc_scores_lgb = [], []
predict_cbt = pd.DataFrame(
    data=-1,
    index=dataset.index, 
    columns=['catboost_predict', 'catboost_proba', 'fold']
)
predict_lgb = pd.DataFrame(
    data=-1,
    index=dataset.index,
    columns=['lightgbm_predict', 'lightgbm_proba','fold']
)

# cross validation loop.
for i, (train_idx, valid_idx) in enumerate(validator.split(dataset.index, train_data.loc[:, 'target'])):
    
    # Split Train and Valid.
    train_X = dataset.loc[train_idx, :]
    train_y = train_data.loc[train_idx, 'target']
    valid_X = dataset.loc[valid_idx, :] 
    valid_y = train_data.loc[valid_idx, 'target']

    with timer(f'Catboost Train #{(i+1):02}'):
        # Train Catboost
        model_cbt = cbt.CatBoostClassifier(
            **config.catboost
        )
        model_cbt.fit(
            train_X,
            train_y,
            eval_set=(valid_X, valid_y),
            verbose=False,
            cat_features=int_cols_idx,
        )

        # Predict Valid Data.
        predict_cbt.loc[valid_idx, 'catboost_predict'] = model_cbt.predict(valid_X)
        predict_cbt.loc[valid_idx, 'catboost_proba'] = model_cbt.predict_proba(valid_X)[:, 1]
        predict_cbt.loc[valid_idx, 'fold'] = (i + 1)
        models_cbt.append(model_cbt)
        
        # Validation.
        roc_auc_scores_cbt.append(roc_auc_score(valid_y, predict_cbt.loc[valid_idx, 'catboost_proba']))
        results_cbt.append(
            pd.DataFrame(
                classification_report(
                    valid_y,
                    predict_cbt.loc[valid_idx, 'catboost_predict'],
                    output_dict=True,
                )
            ).transpose()
        )

    with timer(f'LightGBM Train #{(i+1):02}'):
        # Train LightGBM
        model_lgb = lgb.LGBMClassifier(
            **config.lightgbm
        )
        model_lgb.fit(
            train_X,
            train_y,
            eval_set=(valid_X, valid_y),
            verbose=False,
        )

        # Predict Valid Data.
        predict_lgb.loc[valid_idx, 'lightgbm_predict'] = model_lgb.predict(valid_X)
        predict_lgb.loc[valid_idx, 'lightgbm_proba'] = model_lgb.predict_proba(valid_X)[:, 1]
        predict_lgb.loc[valid_idx, 'fold'] = (i + 1)
        models_lgb.append(model_lgb)
        
        # Validation.
        roc_auc_scores_lgb.append(roc_auc_score(valid_y, predict_lgb.loc[valid_idx, 'lightgbm_proba']))
        results_lgb.append(
            pd.DataFrame(
                classification_report(
                    valid_y,
                    predict_lgb.loc[valid_idx, 'lightgbm_predict'],
                    output_dict=True,
                )
            ).transpose()
        )

    del train_X, train_y, valid_X, valid_y


# ## Plot Validation Result 

# ### code

# In[ ]:


def edit_classification_report(cr:pd.DataFrame):
    supports = ('\n(' + cr.iloc[:-3, -1].astype(int).astype(str) + ')')
    cr.index = (cr.index[:-3] + supports).to_list() + cr.index[-3:].to_list()
    cr = cr.iloc[:, :-1]
    cr.iloc[-3:, :-1] = np.nan
    return cr


def plot_classification_report(cr:pd.DataFrame, ax=None):

    sns.heatmap(
        cr.iloc[:-2, :],
        vmin=0.0,
        vmax=1.0,
        cmap=sns.color_palette("Blues", 24),
        fmt='0.4g',
        linewidth=2.0,
        annot=True,
        annot_kws=dict(size=14),
        ax=ax,
    )
    
    return ax


def plot_cv_classification_report(
    crs:list,
    title:str=None,
    subplots_kws: dict = dict()
):

    if not(type(subplots_kws) is dict):
        subplots_kws = dict()
    
    nrow = subplots_kws.get('nrow', 2)
    ncol = subplots_kws.get('ncol', 5)
    figsize = subplots_kws.get('figsize', (30, 12))
    
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.ravel()
    for i, cr in enumerate(crs):
        axes[i] = plot_classification_report(cr, axes[i])
        axes[i].set_title(f"Fold#{(i+1):02}", fontsize=14)
        axes[i].tick_params(axis='x', labelsize=14)
        axes[i].tick_params(axis='y', labelsize=14)

    rect = None
    if type(title) is str:
        fig.suptitle(title, fontsize=16)
        rect = [0, 0, 1, 0.96]
    
    fig.tight_layout(rect=rect)
    plt.close()

    return fig


# In[ ]:


results_cbt = [edit_classification_report(cr) for cr in results_cbt]
results_lgb = [edit_classification_report(cr) for cr in results_lgb]


# ### Catboost

# In[ ]:


plot_cv_classification_report(
    results_cbt,
    'CatBoost CrossValidation',
    subplots_kws=dict(nrow=2, ncol=5, figsize=(30, 12))
)


# In[ ]:


# Print ROC AUC Scores.
display(pd.DataFrame(roc_auc_scores_cbt).T)

roc_auc_cv_score_cbt = np.mean(roc_auc_scores_cbt)
print(f'CatBoost ROC AUC CV Score: {roc_auc_cv_score_cbt:.5f}')


# ### LightGBM

# In[ ]:


plot_cv_classification_report(
    results_lgb,
    'LightGBM CrossValidation',
    subplots_kws=dict(nrow=2, ncol=5, figsize=(30, 12))
)


# In[ ]:


# Print ROC AUC Scores.
display(pd.DataFrame(roc_auc_scores_lgb).T)
roc_auc_cv_score_lgb = np.mean(roc_auc_scores_lgb)
print(f'LightGBM ROC AUC CV Score: {roc_auc_cv_score_lgb:.5f}')


# ## Plot Feature Importances

# ### code

# In[ ]:


def plot_feature_importances(
    models,
    columns,
    title: str = None,
    top: int = 50
):
    
    feature_importances = pd.DataFrame(
        index=columns,
        columns=['feature_importances']
    )
    feature_importances['feature_importances'] = 0.0
    
    if not(type(models) is list):
        models = [models]
    
    for i, model in enumerate(models):
        feature_importances['feature_importances'] += model.feature_importances_
    feature_importances['feature_importances'] /= len(models)

    feature_importances = feature_importances.sort_values(
        'feature_importances',
        ascending=False
    ).iloc[:top]
    feature_importances.index.name = 'column'
    feature_importances.reset_index(inplace=True)
    
    fig, axes = plt.subplots(figsize=(max(6, feature_importances.shape[0] * 0.4), 7))
    sns.barplot(x='column', y='feature_importances', data=feature_importances, ax=axes)
    axes.grid()

    rect = None
    if type(title) is str:
        fig.suptitle(title, fontsize=16)
        rect = [0, 0, 1, 0.96]
    
    fig.tight_layout(rect=rect)
    plt.close()

    return fig


# ### Catboost

# In[ ]:


plot_feature_importances(
    models_cbt, 
    dataset.columns, 
    'CatBoost Feature Importances'
)


# ### LightGBM

# In[ ]:


plot_feature_importances(
    models_lgb, 
    dataset.columns, 
    'LightGBM Feature Importances'
)


# # Precit Test Data

# ## code

# In[ ]:


def predict(models, testdata):
    if not(type(models) is list):
        models = [models]
    
    predict = pd.DataFrame(
        np.zeros(testdata.index.shape[0]),
        index=testdata.index,
        columns=['predict']
    )

    for model in models:
        predict['predict'] += model.predict_proba(testdata)[:, 1]

    predict['predict'] /= len(models)
    
    return predict
    


# ## make test dataset

# In[ ]:


test_idx = data[data.type=='test'].index
test_data = pd.concat(
    [data.loc[test_idx, :], f_27_each_counts.loc[test_idx, list(string.ascii_uppercase)]],
    axis=1,
).loc[:, use_cols]
test_data = test_data.reset_index(drop=True)

test_data.loc[:, scaling_cols] = scaler.transform(test_data.loc[:, scaling_cols])

submission_df = pd.read_csv(config.submission_file)


# ## predict

# In[ ]:


submission_df['target'] = predict(models_cbt+models_lgb, test_data).values


# In[ ]:


submission_df.to_csv('submission.csv', index=False)

