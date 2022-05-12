#!/usr/bin/env python
# coding: utf-8

# # The best model for TPSAPR22 without DL

# What does this notebook do?
# - We extract features using tsflex `tsflex`: https://github.com/predict-idlab/tsflex  
# - We perform feature selection using `powershap`: https://github.com/predict-idlab/powershap  
# - We fit 1 final catboost model on the collection of features (in this notebook)
# 
# 
# # <a style="color:orange">The feature extraction & feature selection libraries</a>
# 
# ## [**tsflex**](https://github.com/predict-idlab/tsflex)
# > #### `tsflex` is a toolkit for **flex**ible **t**ime **s**eries processing & feature extraction, that is highly efficient  
# 
# `tsflex` is flexible as it: 
#  * handles multivariate time series
#  * **integrates** with many other libraries (numpy, scipy, seglearn, tsfresh, tsfel, catch22, ...)
#  * coveniently allows to extract features on **multiple step & window sizes**
# 
# ## [**powershap**](https://github.com/predict-idlab/powershap)
# > #### `powershap` is a **feature selection method** that uses statistical hypothesis testing and power calculations on **Shapley values**, enabling fast and intuitive wrapper-based feature selection.  
# 
# <br>
# 
# ---
# 
# 
# Some features have been inspired by [AMBROSM's notebook](https://www.kaggle.com/code/ambrosm/tpsapr22-best-model-without-nn)
# 
# <br>

# In[ ]:


get_ipython().system('pip install tsflex  # Our feature extraction package')
get_ipython().system('pip install seglearn antropy catch22  # The feature function pcakgaes')
get_ipython().system('pip install powershap  # Our feature selection package')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cycler import cycler
from IPython.display import display
from tqdm.notebook import tqdm
import datetime
import scipy.stats

from sklearn.model_selection import GroupKFold, cross_val_score, GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier

# Our feature extraction library
from tsflex.features import FeatureCollection, MultipleFeatureDescriptors, FuncWrapper
from tsflex.features.integrations import seglearn_feature_dict_wrapper, tsfresh_settings_wrapper, catch22_wrapper

# Our feature selection library
from powershap import PowerShap

# The feature function libraries
from seglearn.feature_functions import base_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import antropy as ant
from catch22 import catch22_all


# In[ ]:


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# ## Load the data

# In[ ]:


# Reading the raw data
train = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv')
train_labels = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv')

# Merge the labels into the train data
train = train.merge(train_labels, how='left', on="sequence")

sensors = [col for col in train.columns if 'sensor_' in col]

train.shape, test.shape


# In[ ]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

train_pivoted0 = train.pivot(index=['sequence', 'subject'], columns='step', values=sensors)
display(train_pivoted0)


# # Feature extraction with **tsflex**

# ## Extract the same features as in [this notebook](https://www.kaggle.com/code/ambrosm/tpsapr22-best-model-without-nn)
# 
# Let's keep it simple and calculate only the following features:
# - For every sensor, we calculate mean, standard deviation, interquartile range, standard deviation divided by mean, and kurtosis. This gives the first 5\*13=65 features.
# - For the special sensor_02, we count how many times it goes up or down.
# - For sensor_02, we calculate the sum of all upward / downward steps, the maximum of all upward / downward steps, and the mean of all upward / downward steps. 
# - For every subject, we count how many sequences belong to it, and we add this count as a feature to all sequences of the subject (the [EDA](https://www.kaggle.com/code/ambrosm/tpsapr22-eda-which-makes-sense) gives the motivation for this feature). 
# 
# Now we have 74 features. 

# In[ ]:


# Feature engineering
def engineer(df):
    new_df = pd.DataFrame([], index=df.index)
    for sensor in sensors:
        new_df[sensor + '_mean'] = df[sensor].mean(axis=1)
        new_df[sensor + '_std'] = df[sensor].std(axis=1)
        new_df[sensor + '_iqr'] = scipy.stats.iqr(df[sensor], axis=1)
        new_df[sensor + '_sm'] = np.nan_to_num(new_df[sensor + '_std'] / 
                                               new_df[sensor + '_mean'].abs()).clip(-1e30, 1e30)
        new_df[sensor + '_kurtosis'] = scipy.stats.kurtosis(df[sensor], axis=1)
    new_df['sensor_02_up'] = (df.sensor_02.diff(axis=1) > 0).sum(axis=1)
    new_df['sensor_02_down'] = (df.sensor_02.diff(axis=1) < 0).sum(axis=1)
    new_df['sensor_02_upsum'] = df.sensor_02.diff(axis=1).clip(0, None).sum(axis=1)
    new_df['sensor_02_downsum'] = df.sensor_02.diff(axis=1) .clip(None, 0).sum(axis=1)
    new_df['sensor_02_upmax'] = df.sensor_02.diff(axis=1).max(axis=1)
    new_df['sensor_02_downmax'] = df.sensor_02.diff(axis=1).min(axis=1)
    new_df['sensor_02_upmean'] = np.nan_to_num(new_df['sensor_02_upsum'] / new_df['sensor_02_up'], posinf=40)
    new_df['sensor_02_downmean'] = np.nan_to_num(new_df['sensor_02_downsum'] / new_df['sensor_02_down'], neginf=-40)
    return new_df

train_pivoted = engineer(train_pivoted0)

train_shuffled = train_pivoted.sample(frac=1.0, random_state=1)
labels_shuffled = train_labels.reindex(train_shuffled.index.get_level_values('sequence'))
labels_shuffled = labels_shuffled[['state']].merge(train[['sequence', 'subject']].groupby('sequence').min(),
                                                   how='left', on='sequence')
labels_shuffled = labels_shuffled.merge(labels_shuffled.groupby('subject').size().rename('sequence_count'),
                                        how='left', on='subject')
train_shuffled['sequence_count_of_subject'] = labels_shuffled['sequence_count'].values

selected_columns = train_shuffled.columns
print(len(selected_columns))
#train_shuffled.columns


# ## Extract base features from seglearn

# In[ ]:


basic_feats = MultipleFeatureDescriptors(
        functions=seglearn_feature_dict_wrapper(base_features()),
        series_names=sensors,
        windows=60,
        strides=60,
    )

fc_seglearn = FeatureCollection(basic_feats)

df_feats = fc_seglearn.calculate(train, show_progress=True, return_df=True, window_idx="begin")
df_feats.index = train['sequence'].unique()


# In[ ]:


train_shuffled = train_shuffled.reset_index().merge(df_feats, left_on='sequence', right_index=True)
train_shuffled = train_shuffled.set_index(['sequence', 'subject'])


# ## Load extracted & selected tsfresh features

# The selection of these features were performed locally using `powershap`

# In[ ]:


# create the tsfresh feature extractor
settings = ComprehensiveFCParameters()  # all the tsfresh features
del settings["linear_trend_timewise"]  # requires a time-index

fc_tsfresh = FeatureCollection(
    MultipleFeatureDescriptors(
        functions=tsfresh_settings_wrapper(settings),
        series_names=sensors,
        windows=60,
        strides=60
    )
)


# In[ ]:


tsfresh_selected_cols = ["sensor_00__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_00__ar_coefficient_{'coeff': 3, 'k': 10}__w=60_s=60",
       "sensor_00__ar_coefficient_{'coeff': 4, 'k': 10}__w=60_s=60",
       "sensor_00__augmented_dickey_fuller_{'attr': 'usedlag'}__w=60_s=60",
       "sensor_01__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_01__ar_coefficient_{'coeff': 1, 'k': 10}__w=60_s=60",
       "sensor_01__ar_coefficient_{'coeff': 2, 'k': 10}__w=60_s=60",
       "sensor_01__ar_coefficient_{'coeff': 3, 'k': 10}__w=60_s=60",
       "sensor_01__ar_coefficient_{'coeff': 4, 'k': 10}__w=60_s=60",
       "sensor_01__ar_coefficient_{'coeff': 6, 'k': 10}__w=60_s=60",
       "sensor_01__cwt_coefficients_{'widths': (2, 5, 10, 20), 'coeff': 10, 'w': 10}__w=60_s=60",
       "sensor_01__fft_coefficient_{'coeff': 1, 'attr': 'imag'}__w=60_s=60",
       "sensor_01__spkt_welch_density_{'coeff': 2}__w=60_s=60",
       "sensor_02__agg_linear_trend_{'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'var'}__w=60_s=60",
       "sensor_02__agg_linear_trend_{'attr': 'slope', 'chunk_len': 10, 'f_agg': 'var'}__w=60_s=60",
       "sensor_02__agg_linear_trend_{'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'var'}__w=60_s=60",
       "sensor_02__change_quantiles_{'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}__w=60_s=60",
       "sensor_02__change_quantiles_{'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}__w=60_s=60",
       "sensor_02__change_quantiles_{'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'}__w=60_s=60",
       "sensor_02__change_quantiles_{'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}__w=60_s=60",
       "sensor_02__change_quantiles_{'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}__w=60_s=60",
       "sensor_02__change_quantiles_{'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}__w=60_s=60",
       "sensor_02__change_quantiles_{'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}__w=60_s=60",
       "sensor_02__cid_ce_{'normalize': True}__w=60_s=60",
       'sensor_02__mean_abs_change__w=60_s=60',
       "sensor_02__number_peaks_{'n': 1}__w=60_s=60",
       "sensor_02__partial_autocorrelation_{'lag': 2}__w=60_s=60",
       "sensor_03__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_03__ar_coefficient_{'coeff': 5, 'k': 10}__w=60_s=60",
       "sensor_04__agg_autocorrelation_{'f_agg': 'var', 'maxlag': 40}__w=60_s=60",
       "sensor_04__agg_linear_trend_{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'min'}__w=60_s=60",
       "sensor_04__agg_linear_trend_{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'}__w=60_s=60",
       "sensor_04__agg_linear_trend_{'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'min'}__w=60_s=60",
       "sensor_04__approximate_entropy_{'m': 2, 'r': 0.5}__w=60_s=60",
       "sensor_04__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_04__ar_coefficient_{'coeff': 10, 'k': 10}__w=60_s=60",
       "sensor_04__ar_coefficient_{'coeff': 2, 'k': 10}__w=60_s=60",
       "sensor_04__ar_coefficient_{'coeff': 4, 'k': 10}__w=60_s=60",
       "sensor_04__augmented_dickey_fuller_{'attr': 'usedlag'}__w=60_s=60",
       "sensor_04__autocorrelation_{'lag': 7}__w=60_s=60",
       "sensor_04__cid_ce_{'normalize': True}__w=60_s=60",
       "sensor_04__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 0}__w=60_s=60",
       "sensor_04__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 2}__w=60_s=60",
       "sensor_04__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 3}__w=60_s=60",
       "sensor_04__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 4}__w=60_s=60",
       "sensor_04__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 5}__w=60_s=60",
       "sensor_04__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 6}__w=60_s=60",
       "sensor_04__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 7}__w=60_s=60",
       "sensor_04__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 9}__w=60_s=60",
       "sensor_04__fft_aggregated_{'aggtype': 'kurtosis'}__w=60_s=60",
       "sensor_04__fft_aggregated_{'aggtype': 'skew'}__w=60_s=60",
       "sensor_04__fft_coefficient_{'coeff': 3, 'attr': 'abs'}__w=60_s=60",
       "sensor_04__fourier_entropy_{'bins': 100}__w=60_s=60",
       "sensor_04__friedrich_coefficients_{'coeff': 1, 'm': 3, 'r': 30}__w=60_s=60",
       "sensor_04__friedrich_coefficients_{'coeff': 3, 'm': 3, 'r': 30}__w=60_s=60",
       "sensor_04__index_mass_quantile_{'q': 0.2}__w=60_s=60",
       'sensor_04__kurtosis__w=60_s=60',
       "sensor_04__large_standard_deviation_{'r': 0.25}__w=60_s=60",
       'sensor_04__longest_strike_above_mean__w=60_s=60',
       "sensor_04__number_peaks_{'n': 10}__w=60_s=60",
       "sensor_04__number_peaks_{'n': 5}__w=60_s=60",
       "sensor_04__ratio_beyond_r_sigma_{'r': 0.5}__w=60_s=60",
       "sensor_04__ratio_beyond_r_sigma_{'r': 1}__w=60_s=60",
       "sensor_04__ratio_beyond_r_sigma_{'r': 2}__w=60_s=60",
       "sensor_04__spkt_welch_density_{'coeff': 2}__w=60_s=60",
       "sensor_04__time_reversal_asymmetry_statistic_{'lag': 2}__w=60_s=60",
       "sensor_05__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_05__ar_coefficient_{'coeff': 2, 'k': 10}__w=60_s=60",
       "sensor_05__fft_coefficient_{'coeff': 0, 'attr': 'abs'}__w=60_s=60",
       "sensor_05__fft_coefficient_{'coeff': 4, 'attr': 'abs'}__w=60_s=60",
       "sensor_06__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_06__fft_coefficient_{'coeff': 1, 'attr': 'imag'}__w=60_s=60",
       "sensor_06__spkt_welch_density_{'coeff': 2}__w=60_s=60",
       "sensor_07__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_07__ar_coefficient_{'coeff': 5, 'k': 10}__w=60_s=60",
       'sensor_07__skewness__w=60_s=60',
       "sensor_07__spkt_welch_density_{'coeff': 2}__w=60_s=60",
       "sensor_09__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_09__ar_coefficient_{'coeff': 5, 'k': 10}__w=60_s=60",
       "sensor_09__augmented_dickey_fuller_{'attr': 'usedlag'}__w=60_s=60",
       "sensor_09__fft_coefficient_{'coeff': 1, 'attr': 'imag'}__w=60_s=60",
       "sensor_09__spkt_welch_density_{'coeff': 2}__w=60_s=60",
       "sensor_10__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_10__ar_coefficient_{'coeff': 10, 'k': 10}__w=60_s=60",
       "sensor_10__augmented_dickey_fuller_{'attr': 'usedlag'}__w=60_s=60",
       "sensor_10__autocorrelation_{'lag': 2}__w=60_s=60",
       "sensor_10__autocorrelation_{'lag': 5}__w=60_s=60",
       "sensor_10__cid_ce_{'normalize': True}__w=60_s=60",
       "sensor_10__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 1}__w=60_s=60",
       "sensor_10__energy_ratio_by_chunks_{'num_segments': 10, 'segment_focus': 3}__w=60_s=60",
       "sensor_10__fft_coefficient_{'coeff': 3, 'attr': 'abs'}__w=60_s=60",
       "sensor_10__fft_coefficient_{'coeff': 4, 'attr': 'abs'}__w=60_s=60",
       "sensor_10__fft_coefficient_{'coeff': 6, 'attr': 'abs'}__w=60_s=60",
       "sensor_10__fourier_entropy_{'bins': 100}__w=60_s=60",
       'sensor_10__kurtosis__w=60_s=60',
       "sensor_10__linear_trend_{'attr': 'pvalue'}__w=60_s=60",
       "sensor_10__partial_autocorrelation_{'lag': 3}__w=60_s=60",
       "sensor_10__partial_autocorrelation_{'lag': 4}__w=60_s=60",
       "sensor_10__spkt_welch_density_{'coeff': 2}__w=60_s=60",
       "sensor_11__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_11__ar_coefficient_{'coeff': 1, 'k': 10}__w=60_s=60",
       "sensor_12__ar_coefficient_{'coeff': 0, 'k': 10}__w=60_s=60",
       "sensor_12__ar_coefficient_{'coeff': 1, 'k': 10}__w=60_s=60",
       "sensor_12__ar_coefficient_{'coeff': 10, 'k': 10}__w=60_s=60",
       "sensor_12__ar_coefficient_{'coeff': 2, 'k': 10}__w=60_s=60",
       "sensor_12__augmented_dickey_fuller_{'attr': 'usedlag'}__w=60_s=60",
       "sensor_12__autocorrelation_{'lag': 2}__w=60_s=60",
       "sensor_12__change_quantiles_{'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'}__w=60_s=60",
       "sensor_12__change_quantiles_{'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}__w=60_s=60",
       "sensor_12__fft_aggregated_{'aggtype': 'skew'}__w=60_s=60",
       "sensor_12__fft_coefficient_{'coeff': 0, 'attr': 'abs'}__w=60_s=60",
       "sensor_12__fourier_entropy_{'bins': 100}__w=60_s=60",
       "sensor_12__partial_autocorrelation_{'lag': 3}__w=60_s=60",
       "sensor_12__ratio_beyond_r_sigma_{'r': 2}__w=60_s=60"]


df_feats_tsfresh = pd.read_parquet('../input/tsflex-x-tsfresh-feature-extraction/tsfresh_feats_train.parquet')[tsfresh_selected_cols]
df_feats_tsfresh.index = train['sequence'].unique()


# In[ ]:


already_included = list(set(train_shuffled.columns).intersection(df_feats_tsfresh.columns))
print(already_included)
df_feats_tsfresh = df_feats_tsfresh.drop(columns=already_included)


# In[ ]:


train_shuffled = train_shuffled.reset_index().merge(df_feats_tsfresh, left_on='sequence', right_index=True)
train_shuffled = train_shuffled.set_index(['sequence', 'subject'])


# ## Extract antropy features

# In[ ]:


time_funcs = [
    ant.svd_entropy, ant.perm_entropy, ant.katz_fd, ant.higuchi_fd, ant.petrosian_fd
]

fc_antropy = FeatureCollection(
    MultipleFeatureDescriptors(
        functions=time_funcs,
        series_names=sensors,
        windows=60,
        strides=60
    )
)

df_feats_antropy = fc_antropy.calculate(train.astype(np.float32), return_df=True, show_progress=True)
df_feats_antropy.index = train['sequence'].unique()


# In[ ]:


already_included = list(set(train_shuffled.columns).intersection(df_feats_antropy.columns))
print(already_included)
df_feats_antropy = df_feats_antropy.drop(columns=already_included)


# In[ ]:


train_shuffled = train_shuffled.reset_index().merge(df_feats_antropy, left_on='sequence', right_index=True)
train_shuffled = train_shuffled.set_index(['sequence', 'subject'])


# ## Extract catch22 features

# In[ ]:


fc_catch22 = FeatureCollection(
    MultipleFeatureDescriptors(
        functions=catch22_wrapper(catch22_all),
        series_names=sensors,
        windows=60,
        strides=60
    )
)

df_feats_train_catch22 = fc_catch22.calculate(train, show_progress=True, return_df=True, window_idx="begin")
df_feats_train_catch22.index =  train["sequence"].unique()


# In[ ]:


already_included = list(set(train_shuffled.columns).intersection(df_feats_train_catch22.columns))
print(already_included)
df_feats_train_catch22 = df_feats_train_catch22.drop(columns=already_included)


# In[ ]:


train_shuffled = train_shuffled.reset_index().merge(df_feats_train_catch22, left_on='sequence', right_index=True)
train_shuffled = train_shuffled.set_index(['sequence', 'subject'])


# # Feature selection with **powershap**

# In[ ]:


# Drop some useless features: see https://www.kaggle.com/code/ambrosm/tpsapr22-best-model-without-nn
dropped_features = ['sensor_05_kurtosis', 'sensor_08_mean',
                    'sensor_05_std', 'sensor_06_kurtosis',
                    'sensor_06_std', 'sensor_03_std',
                    'sensor_02_kurtosis', 'sensor_03_kurtosis',
                    'sensor_09_kurtosis', 'sensor_03_mean',
                    'sensor_00_mean', 'sensor_02_iqr',
                    'sensor_05_mean', 'sensor_06_mean',
                    'sensor_07_std', 'sensor_10_iqr',
                    'sensor_11_iqr', 'sensor_12_iqr',
                    'sensor_09_mean', 'sensor_02_sm',
                    'sensor_03_sm', 'sensor_05_iqr', 
                    'sensor_06_sm', 'sensor_09_iqr', 
                    'sensor_07_iqr', 'sensor_10_mean']
selected_columns = [f for f in selected_columns if f not in dropped_features]
len(selected_columns)


# In[ ]:


seglearn_features = list(df_feats.columns)
selected_columns += seglearn_features


# The feature selection for the tsfresh features has been performed by [powershap](https://github.com/predict-idlab/powershap)

# In[ ]:


tsfresh_features = list(df_feats_tsfresh.columns)
selected_columns += tsfresh_features


# In[ ]:


antropy_features = list(df_feats_antropy.columns)
selected_columns += antropy_features


# In[ ]:


catch22_feature_names = list(df_feats_train_catch22.columns)
if 'sequence' in catch22_feature_names:
    catch22_feature_names.remove('sequence')
    
selected_columns += catch22_feature_names


# In[ ]:


len(selected_columns)


# In[ ]:





# #### Now apply **powershap** for feature selection

# In[ ]:


selector = PowerShap()
selector.fit(train_shuffled[selected_columns], labels_shuffled["state"])


# In[ ]:


selected_columns_p = np.array(selected_columns)[selector._get_support_mask()]
len(selected_columns_p)


# ### Extract the selected features on the test set

# In[ ]:


fc_all = FeatureCollection(
    [
        fc_seglearn,
        fc_tsfresh.reduce(tsfresh_selected_cols),
        fc_antropy,
        fc_catch22,
    ]
)

selected_columns_p_tsflex = [c for c in selected_columns_p if c.endswith("_w=60_s=60")]
print(len(selected_columns_p_tsflex))
fc_selected = fc_all.reduce(selected_columns_p_tsflex)


# In[ ]:


df_feats_test = fc_selected.calculate(test, return_df=True, show_progress=True)
print(df_feats_test.shape)  # Some features return multiple outputs (that is why this shape is way larger than # selected cols)
df_feats_test = df_feats_test[selected_columns_p_tsflex]
df_feats_test.index = test["sequence"].unique()
df_feats_test.shape


# In[ ]:


# Feature engineering for test

test_pivoted0 = test.pivot(index=['sequence', 'subject'], columns='step', values=sensors)
# Manual feature extraction (cfr. AmbrosM)
test_pivoted = engineer(test_pivoted0)

# Add the selected features from the reduced tsflex feature collection
test_pivoted = test_pivoted.reset_index().merge(df_feats_test, left_on='sequence', right_index=True)
test_pivoted = test_pivoted.set_index(['sequence', 'subject'])

sequence_count = test_pivoted.index.to_frame(index=False).groupby('subject').size().rename('sequence_count_of_subject')
submission = pd.DataFrame({'sequence': test_pivoted.index.get_level_values('sequence')})
test_pivoted = test_pivoted.reset_index()
test_pivoted = test_pivoted.merge(sequence_count, how='left', on='subject')
test_pivoted.head(2)


# # Cross validation

# In[ ]:


assert all([c in test_pivoted.columns.values for c in selected_columns_p])


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Cross-validation of the classifier\n\nprint(f"{len(selected_columns_p)} features")\nscore_list = []\nkf = GroupKFold(n_splits=5)\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train_shuffled, groups=train_shuffled.index.get_level_values(\'subject\'))):\n    X_tr = train_shuffled.iloc[idx_tr][selected_columns_p]\n    X_va = train_shuffled.iloc[idx_va][selected_columns_p]\n    y_tr = labels_shuffled.iloc[idx_tr].state\n    y_va = labels_shuffled.iloc[idx_va].state\n    \n    train_groups = train_shuffled.index.get_level_values(\'subject\')[idx_tr]\n    \n    kf2 = GroupKFold(n_splits=10)\n    idx_cv_tr, idx_cv_val = next(kf.split(X_tr, groups=train_groups))\n    X_cv_tr = X_tr.iloc[idx_cv_tr]\n    X_cv_va = X_tr.iloc[idx_cv_val]\n    y_cv_tr = y_tr.iloc[idx_cv_tr]\n    y_cv_val = y_tr.iloc[idx_cv_val]\n\n    model = CatBoostClassifier(iterations=5000, verbose=0, od_type=\'Iter\', od_wait=100, task_type="CPU")\n    model.fit(X_cv_tr, y_cv_tr, eval_set=(X_cv_va, y_cv_val))\n    \n    best_iter = model.get_best_iteration()\n    \n    model = CatBoostClassifier(iterations=best_iter+100, verbose=0, \n                               learning_rate=model._learning_rate,\n                               od_type=\'Iter\', od_wait=100, task_type="CPU")\n    model.fit(X_tr, y_tr)\n\n    y_va_pred = model.predict_proba(X_va.values)[:,1]\n    score = roc_auc_score(y_va, y_va_pred)\n    try:\n        print(f"Fold {fold}: n_iter ={best_iter:5d}    AUC = {score:.4f}")\n    except AttributeError:\n        print(f"Fold {fold}:                  AUC = {score:.3f}")\n    score_list.append(score)\n    \nprint(f"OOF AUC:                       {np.mean(score_list):.4f}")')


# ## Generate submission

# In[ ]:


# Retrain, predict and write submission
print(f"{len(selected_columns_p)} features")

pred_list = []
for seed in range(10):
    X_tr = train_shuffled[selected_columns_p]
    y_tr = labels_shuffled.state

    kf2 = GroupShuffleSplit(n_splits=10, random_state=seed)
    idx_cv_tr, idx_cv_val = next(kf2.split(X_tr, groups=train_shuffled.index.get_level_values('subject')))
    X_cv_tr = X_tr.iloc[idx_cv_tr]
    X_cv_va = X_tr.iloc[idx_cv_val]
    y_cv_tr = y_tr.iloc[idx_cv_tr]
    y_cv_val = y_tr.iloc[idx_cv_val]
    
    model = CatBoostClassifier(iterations=5000, verbose=0, od_type='Iter', 
                               random_state=seed,
                               od_wait=100, task_type="CPU")
    model.fit(X_cv_tr, y_cv_tr, eval_set=(X_cv_va, y_cv_val))
    
    best_iter = model.get_best_iteration()
    
    model = CatBoostClassifier(iterations=best_iter+100, verbose=0, 
                               learning_rate=model._learning_rate,
                               random_state=seed,
                               od_type='Iter', od_wait=100, task_type="CPU")
    model.fit(X_tr, y_tr)
    pred_list.append(scipy.stats.rankdata(model.predict_proba(test_pivoted[selected_columns_p].values)[:, 1]))
    
    print(f"{seed:2}", pred_list[-1])
    print()
submission['state'] = sum(pred_list) / len(pred_list)
submission.to_csv('submission.csv', index=False)
submission


# In[ ]:




