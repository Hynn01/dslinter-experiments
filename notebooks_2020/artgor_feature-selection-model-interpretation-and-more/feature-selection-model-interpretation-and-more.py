#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# In this kernel I'll try various technics for models interpretability and feature selection. Also I'll compare various models.
# 
# I use the features from my dataset: https://www.kaggle.com/artgor/lanl-features
# 
# This dataset was created using this kernel: https://www.kaggle.com/artgor/even-more-features/
# 
# **UPD**: Thanks to the new kaggle update we can write code in kernels and import it. This is much more convenient and useful.
# I'm moving all the functions I can into this script: https://www.kaggle.com/artgor/artgor-utils
# So if you see somewhere code like `artgot_utils.function_name(parameters)` - it is from this script
# 
# ![](https://torontoseoulcialite.com/wp-content/uploads/2016/02/zimbiocom.jpg)

# In[ ]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import NearestNeighbors
import librosa, librosa.display
import builtins
from sklearn.ensemble import RandomForestRegressor
import eli5
import shap
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE

from IPython.display import HTML
import json
import altair as alt

import artgor_utils

# setting up altair
workaround = artgor_utils.prepare_altair()
HTML("".join((
    "<script>",
    workaround,
    "</script>",
)))


# In[ ]:


os.listdir('../input/lanl-features')


# * 'train_features.csv' - train features generated on original data
# * 'train_features_denoised.csv' - train features generated on denoised data
# * 'test_features.csv' - test features generated on original data
# * 'test_features_denoised.csv' - test features generated on denoised data
# * 'submission_1.csv' - one of my local submissions
# * 'y.csv' - train target

# ## Loading data
# Let's load features!

# In[ ]:


train_features = pd.read_csv('../input/lanl-features/train_features.csv')
test_features = pd.read_csv('../input/lanl-features/test_features.csv')
train_features_denoised = pd.read_csv('../input/lanl-features/train_features_denoised.csv')
test_features_denoised = pd.read_csv('../input/lanl-features/test_features_denoised.csv')
train_features_denoised.columns = [f'{i}_denoised' for i in train_features_denoised.columns]
test_features_denoised.columns = [f'{i}_denoised' for i in test_features_denoised.columns]
y = pd.read_csv('../input/lanl-features/y.csv')


# In[ ]:


X = pd.concat([train_features, train_features_denoised], axis=1).drop(['seg_id_denoised', 'target_denoised'], axis=1)
X_test = pd.concat([test_features, test_features_denoised], axis=1).drop(['seg_id_denoised', 'target_denoised'], axis=1)
X = X[:-1]
y = y[:-1]


# We have almost 2000 features here!

# In[ ]:


n_fold = 10
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)


# ## Basic model
# 
# Training function is imported from my script. Important changes from the code, which I used previously:
# - function returns dictionary with oof, test predictions and scores. Also with feature importances, if necessary;
# - in future it will be easier to change metrics.

# In[ ]:


params = {'num_leaves': 128,
          'min_child_samples': 79,
          'objective': 'gamma',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "subsample_freq": 5,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501,
          'colsample_bytree': 0.1
         }
result_dict_lgb = artgor_utils.train_model_regression(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb',
                                                                                  eval_metric='mae', plot_feature_importance=True)


# In[ ]:


submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id')
submission['time_to_failure'] = result_dict_lgb['prediction']
print(submission.head())
submission.to_csv('submission.csv')


# In[ ]:


sub1 = pd.read_csv('../input/lanl-features/submission_1.csv')
sub1.to_csv('submission_1.csv', index=False)


# ## NN features
# Here I normalize the data and create features using NearestNeighbors. The idea is to find samples which are similar and use it to generate features.

# In[ ]:


scaler = StandardScaler()
scaler.fit(X)
X_train_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[ ]:


get_ipython().run_cell_magic('time', '', "n = 10\nneigh = NearestNeighbors(n, n_jobs=-1)\nneigh.fit(X_train_scaled)\n\ndists, _ = neigh.kneighbors(X_train_scaled, n_neighbors=n)\nmean_dist = dists.mean(axis=1)\nmax_dist = dists.max(axis=1)\nmin_dist = dists.min(axis=1)\n\nX_train_scaled['mean_dist'] = mean_dist\nX_train_scaled['max_dist'] = max_dist\nX_train_scaled['min_dist'] = min_dist\n\ntest_dists, _ = neigh.kneighbors(X_test_scaled, n_neighbors=n)\n\ntest_mean_dist = test_dists.mean(axis=1)\ntest_max_dist = test_dists.max(axis=1)\ntest_min_dist = test_dists.min(axis=1)\n\nX_test_scaled['mean_dist'] = test_mean_dist\nX_test_scaled['max_dist'] = test_max_dist\nX_test_scaled['min_dist'] = test_min_dist")


# In[ ]:


params = {'num_leaves': 32,
          'min_data_in_leaf': 79,
          'objective': 'gamma',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8126672064208567,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501,
          'feature_fraction': 0.1
         }
result_dict_lgb = artgor_utils.train_model_regression(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb',
                                                                                  eval_metric='mae', plot_feature_importance=True)


# In[ ]:


submission['time_to_failure'] = result_dict_lgb['prediction']
submission.to_csv('submission_nn.csv')


# ## Model interpretation
# 

# ### ELI5 and permutation importance
# ELI5 is a package with provides explanations for ML models. It can do this not only for linear models, but also for tree based like Random Forest or lightgbm.
# 
# **Important notice**: running eli5 on all features takes a lot of time, so I run the cell below in `version 14` and printed the top-50 features. In the following versions I'll use these 50 columns and use eli5 to find top-40 of them so that it takes less time

# In[ ]:


top_columns = ['iqr1_denoised', 'percentile_5_denoised', 'abs_percentile_90_denoised', 'percentile_95_denoised', 'ave_roll_std_10', 'num_peaks_10', 'percentile_roll_std_20',
               'ratio_unique_values_denoised', 'fftr_percentile_roll_std_75_denoised', 'num_crossing_0_denoised', 'percentile_95', 'ffti_percentile_roll_std_75_denoised',
               'min_roll_std_10000', 'percentile_roll_std_1', 'percentile_roll_std_10', 'fftr_percentile_roll_std_70_denoised', 'ave_roll_std_50', 'ffti_percentile_roll_std_70_denoised',
               'exp_Moving_std_300_mean_denoised', 'ffti_percentile_roll_std_30_denoised', 'mean_change_rate', 'percentile_roll_std_5', 'range_-1000_0', 'mad',
               'fftr_range_1000_2000_denoised', 'percentile_10_denoised', 'ffti_percentile_roll_std_80', 'percentile_roll_std_25', 'fftr_percentile_10_denoised',
               'ffti_range_-2000_-1000_denoised', 'autocorrelation_5', 'min_roll_std_100', 'fftr_percentile_roll_std_80', 'min_roll_std_500', 'min_roll_std_50', 'min_roll_std_1000',
               'ffti_percentile_20_denoised', 'iqr1', 'classic_sta_lta5_mean_denoised', 'classic_sta_lta6_mean_denoised', 'percentile_roll_std_10_denoised',
               'fftr_percentile_70_denoised', 'ffti_c3_50_denoised', 'ffti_percentile_roll_std_75', 'abs_percentile_90', 'range_0_1000', 'spkt_welch_density_50_denoised',
               'ffti_percentile_roll_std_40_denoised', 'ffti_range_-4000_-3000', 'mean_change_rate_last_50000']


X_train, X_valid, y_train, y_valid = train_test_split(X[top_columns], y, test_size=0.1)
model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1, verbose=-1)
model.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
        verbose=10000, early_stopping_rounds=200)

perm = eli5.sklearn.PermutationImportance(model, random_state=1).fit(X_train, y_train)


# In[ ]:


eli5.show_weights(perm, top=50, feature_names=top_columns)


# In[ ]:


top_features = [i for i in eli5.formatters.as_dataframe.explain_weights_df(model).feature if 'BIAS' not in i][:40]
result_dict_lgb = artgor_utils.train_model_regression(X, X_test, y, params=params, folds=folds, model_type='lgb', plot_feature_importance=True, columns=top_features)


# In[ ]:


submission['time_to_failure'] = result_dict_lgb['prediction']
submission.to_csv('submission_eli5.csv')


# ## Feature selection
# 
# Here I try various approaches to feature selection.
# 
# **Important notice**: running feature selection on all features takes a lot of time, so I'll run some of feature selection methods and print the result, which I'll use in the following versions of the kernel, so that I can explore more approaches.

# In[ ]:


params = {'num_leaves': 32,
          'min_child_samples': 79,
          'objective': 'gamma',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "subsample_freq": 5,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501,
          'colsample_bytree': 1.0
         }


# ### SelectPercentile
# 
# **Important notice**:  I run the cell below in `version 14` and printed the scores_dict. In the following versions I'll use `scores_dict` and plot the results instead of running feature selection each time

# In[ ]:


# %%time
# scores_dict = {'f_classif': [], 'mutual_info_classif': [], 'n_features': []}
# for i in range(5, 105, 5):
#     print(i)
#     s1 = SelectPercentile(f_classif, percentile=i)
#     X_train1 = s1.fit_transform(X, y.values.astype(int))
#     X_test1 = s1.transform(X_test)
#     result_dict_lgb = artgor_utils.train_model_regression(X_train1, X_test1, y.values.reshape(-1, ), params=params, folds=folds, model_type='lgb', plot_feature_importance=False)
#     scores_dict['f_classif'].append(np.mean(result_dict_lgb['scores']))
    
#     s2 = SelectPercentile(mutual_info_classif, percentile=i)
#     X_train1 = s2.fit_transform(X, y.values.astype(int))
#     X_test1 = s2.transform(X_test)
#     result_dict_lgb = artgor_utils.train_model_regression(X_train1, X_test1, y.values.reshape(-1, ), params=params, folds=folds, model_type='lgb', plot_feature_importance=False)
#     scores_dict['mutual_info_classif'].append(np.mean(result_dict_lgb['scores']))
    
#     scores_dict['n_features'].append(X_train1.shape[1])


# In[ ]:


scores_dict = {'f_classif': [2.0746468465171377, 2.0753843541953687, 2.062191535440333, 2.0654327826583034, 2.0643551320704936, 2.0617560048382675,
                             2.061565197738015, 2.0598878198917494, 2.0654865223333143, 2.0632788555735777, 2.058002635080971, 2.051075689018734,
                             2.0472543961304583, 2.052401474353084, 2.055924154798443, 2.0561794619762352, 2.0549680611994963, 2.057123777802326,
                             2.0591868861136904, 2.0577745274024553],
               'mutual_info_classif': [2.0866763775014006, 2.0745431497064324, 2.0564324832516427, 2.060125564781158, 2.067334544167612, 2.0665943783246448,
                                       2.063891669849029, 2.070194051004794, 2.0667490707700447, 2.0681653852378354, 2.0592743636982345, 2.061260741522344,
                                       2.05680667824411, 2.0565047875243003, 2.058252567141659, 2.0554927194831922, 2.0562776429736873, 2.0618179277444084,
                                       2.06364125584214, 2.0577745274024553],
               'n_features': [98, 196, 294, 392, 490, 588, 685, 783, 881, 979, 1077, 1175, 1273, 1370, 1468, 1566, 1664, 1762, 1860, 1958]}


# In[ ]:


scores_df = pd.DataFrame(scores_dict)
scores_df = scores_df.melt(id_vars=['n_features'], value_vars=['mutual_info_classif', 'f_classif'], var_name='metric', value_name='mae')
max_value = scores_df['mae'].max() * 1.05
min_value = scores_df['mae'].min() * 0.95
artgor_utils.render(alt.Chart(scores_df).mark_line().encode(
    y=alt.Y('mae:Q', scale=alt.Scale(domain=(min_value, max_value))),
    x='n_features:O',
    color='metric:N',
    tooltip=['metric:N', 'n:O', 'mae:Q']
).properties(
    title='Top N features by SelectPercentile vs CV'
).interactive())


# ### SelectKBest
# 
# **Important notice**:  I run the cell below in `version 14` and printed the scores_dict. In the following versions I'll use `scores_dict` and plot the results instead of running feature selection each time

# In[ ]:


# %%time
# scores_dict = {'f_classif': [], 'mutual_info_classif': [], 'n_features': []}
# for i in np.arange(10, 1958, 100):
#     print(i)
#     s1 = SelectKBest(f_classif, k=i)
#     X_train1 = s1.fit_transform(X, y.values.astype(int))
#     X_test1 = s1.transform(X_test)
#     result_dict_lgb = artgor_utils.train_model_regression(X_train1, X_test1, y.values.reshape(-1, ), params=params, folds=folds, model_type='lgb', plot_feature_importance=False)
#     scores_dict['f_classif'].append(np.mean(result_dict_lgb['scores']))
    
#     s2 = SelectKBest(mutual_info_classif, k=i)
#     X_train1 = s2.fit_transform(X, y.values.astype(int))
#     X_test1 = s2.transform(X_test)
#     result_dict_lgb = artgor_utils.train_model_regression(X_train1, X_test1, y.values.reshape(-1, ), params=params, folds=folds, model_type='lgb', plot_feature_importance=False)
#     scores_dict['mutual_info_classif'].append(np.mean(result_dict_lgb['scores']))
    
#     scores_dict['n_features'].append(X_train1.shape[1])


# In[ ]:


scores_dict = {'f_classif': [2.1495892622081354, 2.0778182269587147, 2.0716153738740006, 2.06152950679902, 2.0645162758752553, 2.0627705797004032, 2.0610992303725157,
                             2.057762113735462, 2.0618360883613627, 2.0603197111525984, 2.06081274633874, 2.0580767195278056, 2.0527646572747127, 2.0498353445032533,
                             2.052442594925, 2.0564456881902133, 2.0582284644115365, 2.0558612960548635, 2.0580900016350094, 2.058218782401599],
               'mutual_info_classif': [2.1235703196243687, 2.084958198672301, 2.0596822478390955, 2.053305869981444, 2.063468853227225, 2.0674399950434323, 2.0658618511287874,
                                       2.063003703200445, 2.0653174905858664, 2.0644340327023656, 2.0748993062333523, 2.0587602096358113, 2.0601495560836076, 2.0559629138548603,
                                       2.0553852701221134, 2.058022171415446, 2.060755947658241, 2.057916705462307, 2.056245795262636, 2.0580691870837056],
               'n_features': [10, 110, 210, 310, 410, 510, 610, 710, 810, 910, 1010, 1110, 1210, 1310, 1410, 1510, 1610, 1710, 1810, 1910]}


# In[ ]:


scores_df = pd.DataFrame(scores_dict)
scores_df = scores_df.melt(id_vars=['n_features'], value_vars=['mutual_info_classif', 'f_classif'], var_name='metric', value_name='mae')
max_value = scores_df['mae'].max() * 1.05
min_value = scores_df['mae'].min() * 0.95
artgor_utils.render(alt.Chart(scores_df).mark_line().encode(
    y=alt.Y('mae:Q', scale=alt.Scale(domain=(min_value, max_value))),
    x='n_features:O',
    color='metric:N',
    tooltip=['metric:N', 'n:O', 'mae:Q']
).properties(
    title='Top N features by SelectKBest vs CV'
).interactive())


# ### Dropping highly correlated features
# 
# Due to the huge number of features there are certainly some highly correlated features, let's try droping them!

# In[ ]:


# https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
X = X.drop(to_drop, axis=1)
X_test = X_test.drop(to_drop, axis=1)
result_dict_lgb_lgb = artgor_utils.train_model_regression(X, X_test, y, params=params, folds=folds, model_type='lgb', plot_feature_importance=True)


# In[ ]:


submission['time_to_failure'] = result_dict_lgb['prediction']
submission.to_csv('submission_no_corr.csv')


# **Important**: from now I'll use the reduced dataset - without highly correlated features.

# ### RFE
# 
# 
# **Important notice**:  I run the cell below in `version 18` and printed the scores_dict. In the following versions I'll use `scores_dict` and plot the results instead of running feature selection each time

# In[ ]:


# %%time
# scores_dict = {'rfe_score': [], 'n_features': []}
# for i in np.arange(10, 1958, 100)[:3]:
#     print(i)
#     s1 = RFE(model, i, step=100)
#     X_train1 = s1.fit_transform(X, y.values.astype(int))
#     X_test1 = s1.transform(X_test)
#     result_dict_lgb = artgor_utils.train_model_regression(X_train1, X_test1, y.values.reshape(-1, ), params=params, folds=folds, model_type='lgb', plot_feature_importance=False)
#     scores_dict['rfe_score'].append(np.mean(result_dict_lgb['scores']))
    
#     scores_dict['n_features'].append(X_train1.shape[1])


# In[ ]:


scores_dict = {'rfe_score': [2.103586938061856, 2.052535910798748, 2.053228199447811], 'n_features': [10, 110, 210]}


# In[ ]:


scores_df = pd.DataFrame(scores_dict)
scores_df = scores_df.melt(id_vars=['n_features'], value_vars=['rfe_score'], var_name='metric', value_name='mae')
max_value = scores_df['mae'].max() * 1.05
min_value = scores_df['mae'].min() * 0.95
artgor_utils.render(alt.Chart(scores_df).mark_line().encode(
    y=alt.Y('mae:Q', scale=alt.Scale(domain=(min_value, max_value))),
    x='n_features:O',
    color='metric:N',
    tooltip=['metric:N', 'n:O', 'mae:Q']
).properties(
    title='Top N features by RFE vs CV'
).interactive())


# ## Model comparison
# 
# In this section I'll try variuos sklearn models and compair their score. Running GridSearchCV each time is too long, so I'll run it once for each model and use optimal parameters.

# In[ ]:


get_ipython().run_cell_magic('time', '', "rfr = RandomForestRegressor()\n\n# parameter_grid = {'n_estimators': [50, 60],\n#                   'max_depth': [5, 10]\n#                  }\n\n# grid_search = GridSearchCV(rfr, param_grid=parameter_grid, cv=folds, scoring='neg_mean_absolute_error', n_jobs=-1)\n# grid_search.fit(X, y)\n# print('Best score: {}'.format(grid_search.best_score_))\n# print('Best parameters: {}'.format(grid_search.best_params_))\n# rfr = RandomForestRegressor(**grid_search.best_params_)\nrfr = RandomForestRegressor(n_estimators=50, max_depth=5)\nresult_dict_rfr = artgor_utils.train_model_regression(X, X_test, y, params=params, folds=folds, model_type='sklearn', model=rfr)\n# print(scores_rfr)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "linreg = linear_model.LinearRegression(normalize=False, copy_X=True, n_jobs=-1)\n\nresult_dict_linreg = artgor_utils.train_model_regression(X, X_test, y, params=None, folds=folds, model_type='sklearn', model=linreg)\n# print(scores_linreg)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "ridge = linear_model.Ridge(normalize=True)\n\nparameter_grid = {'alpha': [0.1, 1.0, 10.0]}\n\ngrid_search = GridSearchCV(ridge, param_grid=parameter_grid, cv=folds, scoring='neg_mean_absolute_error', n_jobs=-1)\ngrid_search.fit(X, y)\nprint('Best score: {}'.format(grid_search.best_score_))\nprint('Best parameters: {}'.format(grid_search.best_params_))\nridge = linear_model.Ridge(**grid_search.best_params_, normalize=True)\nresult_dict_ridge = artgor_utils.train_model_regression(X, X_test, y, params=params, folds=folds, model_type='sklearn', model=ridge)\n# print(scores_ridge)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "knn = neighbors.KNeighborsRegressor()\n\nparameter_grid = {'n_neighbors': [50, 100]}\n\ngrid_search = GridSearchCV(knn, param_grid=parameter_grid, cv=folds, scoring='neg_mean_absolute_error', n_jobs=-1)\ngrid_search.fit(X, y)\nprint('Best score: {}'.format(grid_search.best_score_))\nprint('Best parameters: {}'.format(grid_search.best_params_))\nknn = neighbors.KNeighborsRegressor(**grid_search.best_params_)\nresult_dict_knn = artgor_utils.train_model_regression(X, X_test, y, params=params, folds=folds, model_type='sklearn', model=knn)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "lasso = linear_model.Lasso(normalize=True)\n\nparameter_grid = {'alpha': [0.1, 1.0, 10.0]}\n\ngrid_search = GridSearchCV(lasso, param_grid=parameter_grid, cv=folds, scoring='neg_mean_absolute_error', n_jobs=-1)\ngrid_search.fit(X, y)\nprint('Best score: {}'.format(grid_search.best_score_))\nprint('Best parameters: {}'.format(grid_search.best_params_))\nlasso = linear_model.Lasso(**grid_search.best_params_, normalize=True)\nresult_dict_lasso = artgor_utils.train_model_regression(X, X_test, y, params=params, folds=folds, model_type='sklearn', model=lasso)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "etr = ExtraTreesRegressor()\n\n# parameter_grid = {'n_estimators': [500, 1000],\n#                   'max_depth': [5, 10, 20]\n#                  }\n\n# grid_search = GridSearchCV(rfr, param_grid=parameter_grid, cv=folds, scoring='neg_mean_absolute_error', n_jobs=-1)\n# grid_search.fit(X, y)\n# print('Best score: {}'.format(grid_search.best_score_))\n# print('Best parameters: {}'.format(grid_search.best_params_))\n# etr = ExtraTreesRegressor(**grid_search.best_params_)\netr = ExtraTreesRegressor(n_estimators=1000, max_depth=10)\nresult_dict_etr = artgor_utils.train_model_regression(X, X_test, y, params=params, folds=folds, model_type='sklearn', model=etr)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "adr = AdaBoostRegressor()\n\nparameter_grid = {'n_estimators': [10, 50],\n                 }\n\ngrid_search = GridSearchCV(adr, param_grid=parameter_grid, cv=folds, scoring='neg_mean_absolute_error', n_jobs=-1)\ngrid_search.fit(X, y)\nprint('Best score: {}'.format(grid_search.best_score_))\nprint('Best parameters: {}'.format(grid_search.best_params_))\nadr = AdaBoostRegressor(**grid_search.best_params_)\nresult_dict_adr = artgor_utils.train_model_regression(X, X_test, y, params=params, folds=folds, model_type='sklearn', model=adr)")


# In[ ]:


plt.figure(figsize=(12, 8));
scores_df = pd.DataFrame({'RandomForestRegressor': result_dict_rfr['scores']})
scores_df['ExtraTreesRegressor'] = result_dict_etr['scores']
scores_df['AdaBoostRegressor'] = result_dict_adr['scores']
scores_df['KNN'] = result_dict_knn['scores']
scores_df['LinearRegression'] = result_dict_linreg['scores']
scores_df['Ridge'] = result_dict_ridge['scores']
scores_df['Lasso'] = result_dict_lasso['scores']

sns.boxplot(data=scores_df);
plt.xticks(rotation=45);


# ## Blending
# 
# Let's try training and blending several models.

# In[ ]:


params = {'num_leaves': 128,
          'min_child_samples': 79,
          'objective': 'gamma',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "subsample_freq": 5,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501,
          'colsample_bytree': 0.2
         }
result_dict_lgb = artgor_utils.train_model_regression(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb',
                                                                                  eval_metric='mae', plot_feature_importance=True)


# In[ ]:


xgb_params = {'eta': 0.03,
              'max_depth': 9,
              'subsample': 0.85,
              'colsample_bytree': 0.3,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
              'nthread': -1}
result_dict_xgb = artgor_utils.train_model_regression(X=X, X_test=X_test, y=y, params=xgb_params, folds=folds, model_type='xgb')


# In[ ]:


submission['time_to_failure'] = (result_dict_lgb['prediction'] + result_dict_etr['prediction'] + result_dict_xgb['prediction']) / 3
print(submission.head())
submission.to_csv('blending.csv')


# In[ ]:


plt.figure(figsize=(18, 8))
plt.subplot(2, 2, 1)
plt.plot(y, color='g', label='y_train')
plt.plot(result_dict_lgb['oof'], color='b', label='lgb')
plt.legend(loc=(1, 0.5));
plt.title('lgb');
plt.subplot(2, 2, 2)
plt.plot(y, color='g', label='y_train')
plt.plot(result_dict_etr['oof'], color='teal', label='xgb')
plt.legend(loc=(1, 0.5));
plt.title('xgb');
plt.subplot(2, 2, 3)
plt.plot(y, color='g', label='y_train')
plt.plot(result_dict_xgb['oof'], color='red', label='etr')
plt.legend(loc=(1, 0.5));
plt.title('Extratrees');
plt.subplot(2, 2, 4)
plt.plot(y, color='g', label='y_train')
plt.plot((result_dict_lgb['oof'] + result_dict_etr['oof'] + result_dict_xgb['oof']) / 3, color='gold', label='blend')
plt.legend(loc=(1, 0.5));
plt.title('blend');


# ## Stacking

# In[ ]:


train_stack = np.vstack([result_dict_rfr['oof'], result_dict_ridge['oof'], result_dict_knn['oof'], result_dict_lasso['oof'], result_dict_etr['oof'],
                         result_dict_adr['oof'], result_dict_lgb['oof'], result_dict_xgb['oof'], result_dict_etr['oof']]).transpose()
train_stack = pd.DataFrame(train_stack, columns=['rfr', 'ridge', 'knn', 'lasso', 'etr', 'adr', 'lgb', 'xgb', 'etr'])

test_stack = np.vstack([result_dict_rfr['prediction'], result_dict_ridge['prediction'], result_dict_knn['prediction'], result_dict_lasso['prediction'], result_dict_etr['prediction'],
                        result_dict_adr['prediction'], result_dict_lgb['prediction'], result_dict_xgb['prediction'], result_dict_etr['prediction']]).transpose()
test_stack = pd.DataFrame(test_stack, columns=['rfr', 'ridge', 'knn', 'lasso', 'etr', 'adr', 'lgb', 'xgb', 'etr'])


# In[ ]:


params = {'num_leaves': 8,
         #'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 2,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "bagging_seed": 11,
         "metric": 'rmse',
        # "lambda_l1": 0.2,
         "verbosity": -1}
result_dict_lgb_stack = artgor_utils.train_model_regression(X=train_stack, X_test=test_stack, y=y, params=params, folds=folds, model_type='lgb',
                                                                                  eval_metric='mae', plot_feature_importance=False,
                                                            columns=(list(train_stack.columns)))


# In[ ]:


submission['time_to_failure'] = result_dict_lgb_stack['prediction']
print(submission.head())
submission.to_csv('stacking.csv')

