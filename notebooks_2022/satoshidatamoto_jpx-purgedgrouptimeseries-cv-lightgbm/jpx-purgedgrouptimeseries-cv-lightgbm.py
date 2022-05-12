#!/usr/bin/env python
# coding: utf-8

# >### PurgedGroupTimeSeries CV - LGBM Version
# >This is a simple starter notebook for Kaggle's JPX Comp showing purged group timeseries KFold with extra data. Purged Times Series is explained [here][1]. There are many configuration variables below to allow you to experiment. Use either CPU or GPU. You can control which years are loaded, which type of models are used, and whether to use feature engineering. You can experiment with different data preprocessing, model hyperparameters, loss, and number of seeds to ensemble. The extra datasets contain the full history of the assets at the same format of the competition, so you can input that into your model too.
# >
# >**NOTE:** this notebook lets you run a different experiment in each fold if you want to run lots of experiments. (Then it is like running multiple holdout validation experiments but in that case note that the overall CV score is meaningless because LB will be much different when the multiple experiments are ensembled to predict test). **If you want a proper CV with a reliable overall CV score you need to choose the same configuration for each fold.**
# >
# 
# [1]: TBD

# <center><img src="https://lightgbm.readthedocs.io/en/latest/_images/LightGBM_logo_black_text.svg" height=250 width=250></center>
# <hr>
# <center>LightGBM = 🌳 + 🚀 + ☢️</center>

# LightGBM is the current "Meta" on kaggle and it doesn't look like it is going to get Nerfed anytime soon! 
# It is basiclly a "light" version of gradient boosting machines framework that aims to increases efficiency and reduces memory usage.
# 
# **It is usually THE Algorithm everyone on Kaggle try when facing a tabular dataset**
# 
# ><h4>TL;DR: What makes LightGBM so great:</h4>
# >
# >1. LGBM was developed and maintained by Microsoft themselves so it gets constant maintenance and support.
# >2. Easy to use 
# >3. Faster than nearly all other gradient boosting algorithms.
# >4. Usually the most powerful gradient boosting. 
# 
# 
# It is a **gradient boosting** model that makes use of tree based learning algorithms. It is considered to be a fast processing algorithm.
# 
# While other algorithms trees grow horizontally, LightGBM algorithm grows vertically, meaning it grows leaf-wise and other algorithms grow level-wise. LightGBM chooses the leaf with large loss to grow. It can lower down more loss than a level wise algorithm when growing the same leaf.
# 
# ![img](https://i.imgur.com/pzOP2Lb.png)
# 
# [Source of Image](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/)
# 
# Light GBM is prefixed as Light because of its high speed. Light GBM can handle the large size of data and takes lower memory to run.
# 
# Another reason why Light GBM is so popular is because it focuses on accuracy of results. LGBM also supports GPU learning and thus data scientists are widely using LGBM for data science application development.
# 
# <h4>Leaf growth technique in LightGBM</h4>
# 
# LightGBM uses leaf-wise (best-first) tree growth. It chooses to grow the leaf that minimizes the loss, allowing a growth of an imbalanced tree. Because it doesn’t grow level-wise, but leaf-wise, over-fitting can happen when data is small. In these cases, it is important to control the tree depth.
# 
# <h4>LightGBM vs XGBoost</h4>
# 
# base learner of almost all of the competitions that have structured datasets right now. This is mostly because of LightGBM's implementation; it doesn't do exact searches for optimal splits like XGBoost does in it's default setting but rather through histogram approximations (XGBoost now has this functionality as well but it's still not as fast as LightGBM). 
# 
# This results in slight decrease of predictive performance buy much larger increase of speed. This means more opportunity for feature engineering/experimentation/model tuning which inevitably yields larger increases in predictive performance. (Feature engineering are the key to winning most Kaggle competitions)
# 
# 
# <h4>LightGBM vs Catboost</h4>
# 
# CatBoost is not used as much, mostly because it tends to be much slower than LightGBM and XGBoost. That being said, CatBoost is very different when it comes to the implementation of gradient boosting. This can give slightly more accurate predictions, in particular if you have large amounts of categorical features. Because rapid experimentation is vital in Kaggle competitions, LightGBM tends to be the go-to algorithm when first creating strong base learners.
# 
# In general, it is important to note that a large amount of approaches involves combining all three boosting algorithms in an ensemble. LightGBM, CatBoost, and XGBoost might be thrown together in a mix to create a strong ensemble. This is done to really squeeze spots on the leaderboard and it usually works.
# 
# <div class="alert alert-block alert-info">
# <b>Read More:</b>
# <ul>
#     <li><a href = "https://github.com/microsoft/LightGBM/tree/master/python-package">LightGBM Github Documentation</a></li>
#     <li><a href = "https://lightgbm.readthedocs.io/en/latest/Features.html">All features of LightGBM</a></li>
#     <li><a href = "https://lightgbm.readthedocs.io/en/latest/index.html">Official Documentation</a></li>    
# </ul>
# </div>
# 
# ____
# 
# <h3>Hyper-Parameter Tuning in LightGBM</h3>
# 
# ____
#     
# Parameter Tuning is an important part that is usually done by data scientists to achieve a good accuracy, fast result and to deal with overfitting. Let us see quickly some of the parameter tuning you can do for better results.
# While, LightGBM has more than 100 parameters that are given in the [documentation of LightGBM](https://github.com/microsoft/LightGBM), we are going to check the most important ones.
# 
# **num_leaves**: This parameter is responsible for the complexity of the model. I normally start by trying values in the range [10,100]. But if you have a solid heuristic to choose tree depth you can always use it and set num_leaves to 2^tree_depth - 1
# 
# [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) says in respect -
# This is the main parameter to control the complexity of the tree model. Theoretically, we can set num_leaves = 2^(max_depth) to obtain the same number of leaves as depth-wise tree. However, this simple conversion is not good in practice. The reason is that a leaf-wise tree is typically much deeper than a depth-wise tree for a fixed number of leaves. Unconstrained depth can induce over-fitting. Thus, when trying to tune the num_leaves, we should let it be smaller than 2^(max_depth). For example, when the max_depth=7 the depth-wise tree can get good accuracy, but setting num_leaves to 127 may cause over-fitting, and setting it to 70 or 80 may get better accuracy than depth-wise.
# 
# **Min_data_in_leaf**: Assigning bigger value to this parameter can result in underfitting of the model. Giving it a value of 100 or 1000 is sufficient for a large dataset.
# 
# **Max_depth**: Controls the depth of the individual trees. Typical values range from a depth of 3–8 but it is not uncommon to see a tree depth of 1. Smaller depth trees are computationally efficient (but require more trees); however, higher depth trees allow the algorithm to capture unique interactions but also increase the risk of over-fitting. Larger training data sets are more tolerable to deeper trees.
# 
# **num_iterations**: Num_iterations specifies the number of boosting iterations (trees to build). The more trees you build the more accurate your model can be at the cost of:
#     - Longer training time
#     - Higher chance of over-fitting
# So typically start with a lower number of trees to build a baseline and increase it later when you want to squeeze the last % out of your model.
# 
# It is recommended to use smaller `learning_rate` with larger `num_iterations`. Also, we should use `early_stopping_rounds` if we go for higher `num_iterations` to stop your training when it is not learning anything useful.
# 
# **early_stopping_rounds** - "early stopping" refers to stopping the training process if the model's performance on a given validation set does not improve for several consecutive iterations. This parameter will stop training if the validation metric is not improving after the last early stopping round. It should be defined in pair with a number of iterations. If we set it too large we increase the chance of over-fitting. **The rule of thumb is to have it at 10% of your `num_iterations`**.
# 
# ____
#     
# <h3>Other Parameters Overview</h3>
# 
# ____
#     
# **Parameters that control the trees of LightGBM**
# 
# - num_leaves: controls the number of decision leaves in a single tree. there will be multiple trees in pool.
# - min_data_in_leaf: the minimum number of data/sample/count per leaf (default is 20; lower min_data_in_leaf means less conservative/control, potentially overfitting).
# - max_depth: this the height of a decision tree. if its more possibility of overfitting but too low may underfit.
# >**NOTE:** max_depth directly impacts:
# >1. The best value for the num_leaves parameter
# >2. Model Performance
# >3. Training Time
# 
# ____
# 
# **Parameters For Better Accuracy**
# 
# - Use large max_bin (may be slower)
# 
#  Use small learning_rate with large num_iterations
# 
# - Use large num_leaves (may cause over-fitting)
# 
# - Use bigger training data
# 
# - Try dart
# 
# ____
# 
# **Parameters for Dealing with Over-fitting**
# 
# - Use small max_bin
# 
# - Use small num_leaves
# 
# - Use min_data_in_leaf and min_sum_hessian_in_leaf
# 
# - Use bagging by set bagging_fraction and bagging_freq
# 
# - Use feature sub-sampling by set feature_fraction
# 
# - Use bigger training data
# 
# - Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
# 
# - Try max_depth to avoid growing deep tree
# 
# - Try extra_trees
# 
# - Try increasing path_smooth
# 
# ____
# 
# 
# <h3>How to tune LightGBM like a boss?</h3>
# 
# Hyperparameters tuning guide:
# 
# **objective**
#  * When you change it affects other parameters	Specify the type of ML model
#  * default- value regression
#  * aliases- Objective_type
# 
# **boosting**
#  * If you set it RF, that would be a bagging approach
#  * default- gbdt
#  * Range- [gbdt, rf, dart, goss]
#  * aliases- boosting_type
# 
# **lambda_l1**
#  * regularization parameter
#  * default- 0.0
#  * Range- [0, ∞]
#  * aliases- reg_alpha
#  * constraints- lambda_l1 >= 0.0
# 
# **bagging_fraction**
#  * randomly select part of data without resampling
#  * default-1.0
#  * range- [0, 1]
#  * aliases- Subsample
#  * constarints- 0.0 < bagging_fraction <= 1.0
# 
# **bagging_freq**
#  * default- 0.0
#  * range- [0, ∞]
#  * aliases- subsample_freq
#  * bagging_fraction should be set to value smaller than 1.0 as well 0 means disable bagging
# 
# **num_leaves**
#  * max number of leaves in one tree
#  * default- 31
#  * Range- [1, ∞]
#  * Note- 1 < num_leaves <= 131072
# 
# **feature_fraction**
#  * if you set it to 0.8, LightGBM will select 80% of features
#  * default- 1.0
#  * Range- [0, 1]
#  * aliases- sub_feature
#  * constarint- 0.0 < feature_fraction <= 1.0
# 
# **max_depth**
#  * default- [-1]
#  * range- [-1, ∞]m
#  * Larger is usually better, but overfitting speed increases.
#  * limit the max depth Forr tree model
# 
# **max_bin**
#  * deal with over-fitting
#  * default- 255
#  * range- [2, ∞]
#  * aliases- Histogram Binning
#  * max_bin > 1
# 
# **num_iterations**
#  * number of boosting iterations
#  * default- 100
#  * range- [1, ∞]
#  * AKA- Num_boost_round, n_iter
#  * constarints- num_iterations >= 0
# 
# **learning_rate**
#  * default- 0.1
#  * range- [0 1]
#  * aliases- eta
#  * general values- learning_rate > 0.0Typical: 0.05.
# 
# **early_stopping_round**
#  * will stop training if validation doesn’t improve in last early_stopping_round
#  * Model Performance, Number of Iterations, Training Time
#  * default- 0
#  * Range- [0, ∞]
# 
# **categorical_feature** 
#  * to sepecify or Handle categorical features
#  * i.e LGBM automatically handels categorical variable we dont need to one hot encode them.
# 
# **bagging_freq**
#  * default-0.0
#  * Range-[0, ∞]
#  * aliases- subsample_freq
#  * note- 0 means disable bagging; k means perform bagging at every k iteration
#  * enable    bagging, bagging_fraction should be set to value smaller than 1.0 as well
# 
# **verbosity**
#  * default- 0
#  * range- [-∞, ∞]
#  * aliases- verbose
#  * constraints- {< 0: Fatal, = 0: Error (Warning), = 1: Info, > 1}
# 
# **min_data_in_leaf**
#  * Can be used to deal with over-fitting:
#  * default- 20
#  * constarint-min_data_in_leaf >= 0      
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Introduction Credits:</b>
# <ul>
#     <li><a href = "https://www.kaggle.com/shivansh002/your-friendly-neighbour-lightgbm">Your Friendly Neighbour LightGBM</a> By @shivansh002. Thank you @shivansh002 for a great introduction! </li>
#     <li><a href = "https://www.kaggle.com/paulrohan2020/tutorial-lightgbm-xgboost-catboost-top-11">Tutorial LightGBM + XGBoost + CatBoost</a> By @paulrohan2020. Thank you @paulrohan2020 for a great tutorial! </li>
# </ul>
# </div>

# #### Code starts here ⬇

# In[ ]:


import os
import traceback
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd, numpy as np
import jpx_tokyo_market_prediction
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # <span class="title-section w3-xxlarge" id="config">Configuration 🎚️</span>
# <hr >
# 
# In order to be a proper cross validation with a meaningful overall CV score, **you need to choose the same** `INC2022`, `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP`, `INCSUPP`, and `NUM_LEAVES`, `MAX_DEPTH` **for each fold**. If your goal is to just run lots of experiments, then you can choose to have a different experiment in each fold. Then each fold is like a holdout validation experiment. When you find a configuration you like, you can use that configuration for all folds.
# * DEVICE - is CPU or GPU
# * SEED - a different seed produces a different triple stratified kfold split.
# * FOLDS - number of folds. Best set to 3, 5, or 15 but can be any number between 2 and 15
# * LOAD_STRICT - This controls whether to load strict at proposed [here](https://www.kaggle.com/julian3833/proposal-for-a-meaningful-lb-strict-lgbm)
# * INC2022 - This controls whether to include the extra historical prices during 2022.
# * INC2021 - This controls whether to include the extra historical prices during 2021.
# * INC2020 - This controls whether to include the extra historical prices during 2020.
# * INC2019 - This controls whether to include the extra historical prices during 2019.
# * INC2018 - This controls whether to include the extra historical prices during 2018.
# * INC2017 - This controls whether to include the extra historical prices during 2017.
# * INCSUPP - This controls whether to include the supplemented train data that was released with the competition.
# * N_ESTIMATORS - is a list of length FOLDS. These are n_estimators for each fold. For maximum speed, it is best to use the smallest number of estimators as your GPU or CPU allows.
# * NUM_LEAVES - is a list of length FOLDS. These are num_leaves for each fold. For maximum speed, it is best to use the smallest number of estimators as your GPU or CPU allows.
# * MAX_DEPTH - is a list of length FOLDS. These are max_depth for each fold. For maximum speed, it is best to use the smallest number of estimators as your GPU or CPU allows.
# * LEARNING_RATE - is a list of length FOLDS. These are max_depth for each fold. For maximum speed, it is best to use the smallest number of estimators as your GPU or CPU allows.

# In[ ]:


DEVICE = "CPU" #or "GPU"

SEED = 42

# CV PARAMS
FOLDS = 5
GROUP_GAP = 130
MAX_TEST_GROUP_SIZE = 180
MAX_TRAIN_GROUP_SIZE = 280

# WHICH YEARS TO INCLUDE? YES=1 NO=0
INC2022 = 1
INC2021 = 1
INC2020 = 1
INC2019 = 1
INC2018 = 1
INC2017 = 1
INCSUPP = 1

# HYPER PARAMETERS
LEARNING_RATE = [0.09, 0.09, 0.09, 0.09, 0.09]
N_ESTIMATORS = [1000, 1000, 1000, 1000, 1000]
NUM_LEAVES = [500, 500, 500, 500, 500]
MAX_DEPTH = [10, 10, 10, 10, 10]


# # <span class="title-section w3-xxlarge" id="loading">Data Loading 🗃️</span>
# <hr>
# 
# Here we choose which years to load. We can use either 2017, 2018, 2019, 2020, 2021, Original, Supplement by changing the `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCSUPP` variables in the preceeding code section. These datasets are discussed [here][1].
# 
# [1]: TBD
# 

# In[ ]:


stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
stock_list = stock_list.loc[stock_list['SecuritiesCode'].isin(prices['SecuritiesCode'].unique())]
stock_name_dict = {stock_list['SecuritiesCode'].tolist()[idx]: stock_list['Name'].tolist()[idx] for idx in range(len(stock_list))}

def load_training_data(asset_id = None):
    prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
    supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
    df_train = pd.concat([prices, supplemental_prices]) if INCSUPP else prices
    df_train = pd.merge(df_train, stock_list[['SecuritiesCode', 'Name']], left_on = 'SecuritiesCode', right_on = 'SecuritiesCode', how = 'left')
    df_train['date'] = pd.to_datetime(df_train['Date'])
    df_train['year'] = df_train['date'].dt.year
    if not INC2022: df_train = df_train.loc[df_train['year'] != 2022]
    if not INC2021: df_train = df_train.loc[df_train['year'] != 2021]
    if not INC2020: df_train = df_train.loc[df_train['year'] != 2020]
    if not INC2019: df_train = df_train.loc[df_train['year'] != 2019]
    if not INC2018: df_train = df_train.loc[df_train['year'] != 2018]
    if not INC2017: df_train = df_train.loc[df_train['year'] != 2017]
    # asset_id = 1301 # Remove before flight
    if asset_id is not None: df_train = df_train.loc[df_train['SecuritiesCode'] == asset_id]
    # df_train = df_train[:1000] # Remove before flight
    return df_train


# # <span class="title-section w3-xxlarge" id="features">Feature Engineering 🔬</span>
# <hr>
# 
# This notebook uses upper_shadow, lower_shadow, high_div_low, open_sub_close, seasonality/datetime features first shown in this notebook [here][1] and successfully used by julian3833 [here][2].
# 
# Additionally we can decide to use external data by changing the variables `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP` in the preceeding code section. These variables respectively indicate whether to load last year 2021 data and/or year 2020, 2019, 2018, 2017, the original, supplemented data. These datasets are discussed [here][3]
# 
# Consider experimenting with different feature engineering and/or external data. The code to extract features out of the dataset is taken from julian3833' notebook [here][2]. Thank you julian3833, this is great work.
# 
# [1]: https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition
# [2]: https://www.kaggle.com/julian3833
# [3]: TBD

# In[ ]:


from pandas import concat
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

def upper_shadow(df): return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df): return np.minimum(df['Close'], df['Open']) - df['Low']

def get_features(df):
    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat["high_div_low"] = df_feat["High"] / df_feat["Low"]
    df_feat["open_sub_close"] = df_feat["Open"] - df_feat["Close"]
    return df_feat


# # <span class="title-section w3-xxlarge" id="modelconf">Configure the model ⚙️</span>
# <hr>
# 
# This is a simple model with simple set of hyperparameters. Consider experimenting with different models, parameters, ensembles and so on.

# **The Model**

# In[ ]:


def build_model(fold):

    # Do feel free to experiment with different models here!
    model = LGBMRegressor(n_estimators = N_ESTIMATORS[fold], num_leaves = NUM_LEAVES[fold], max_depth = MAX_DEPTH[fold], learning_rate = LEARNING_RATE[fold])

    return model


# # Time Series Cross Validation
# 
# > "There are many different ways one can do cross-validation, and **it is the most critical step when building a good machine learning model** which is generalizable when it comes to unseen data."
# -- **Approaching (Almost) Any Machine Learning Problem**, by Abhishek Thakur
# 
# CV is the **first** step, but very few notebooks are talking about this. Here we look at "purged rolling time series CV" and actually apply it in hyperparameter tuning for a basic estimator. This notebook owes a debt of gratitude to the notebook ["Found the Holy Grail GroupTimeSeriesSplit"](https://www.kaggle.com/jorijnsmit/found-the-holy-grail-grouptimeseriessplit). That notebook is excellent and this solution is an extention of the quoted pending sklearn estimator. I modify that estimator to make it more suitable for the task at hand in this competition. The changes are
# 
# - you can specify a **gap** between each train and validation split. This is important because even though the **group** aspect keeps whole days together, we suspect that the anonymized features have some kind of lag or window calculations in them (which would be standard for financial features). By introducing a gap, we mitigate the risk that we leak information from train into validation
# - we can specify the size of the train and validation splits in terms of **number of days**. The ability to specify a validation set size is new and the the ability to specify days, as opposed to samples, is new.
# 
# The code for `PurgedTimeSeriesSplit` is below. I've hidden it because it is really meant to act as an imported class. If you want to see the code and copy for your work, click on the "Code" box.

# In[ ]:


from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupTimeSeriesSplit
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                           'b', 'b', 'b', 'b', 'b',\
                           'c', 'c', 'c', 'c',\
                           'd', 'd', 'd'])
    >>> gtss = GroupTimeSeriesSplit(n_splits=3)
    >>> for train_idx, test_idx in gtss.split(groups, groups=groups):
    ...     print("TRAIN:", train_idx, "TEST:", test_idx)
    ...     print("TRAIN GROUP:", groups[train_idx],\
                  "TEST GROUP:", groups[test_idx])
    TRAIN: [0, 1, 2, 3, 4, 5] TEST: [6, 7, 8, 9, 10]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a']\
    TEST GROUP: ['b' 'b' 'b' 'b' 'b']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] TEST: [11, 12, 13, 14]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b']\
    TEST GROUP: ['c' 'c' 'c' 'c']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\
    TEST: [15, 16, 17]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c']\
    TEST GROUP: ['d' 'd' 'd']
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]


            if self.verbose > 0:
                    pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


# # <span class="title-section w3-xxlarge" id="training">Training 🏋️</span>
# <hr>
# Our model will be trained for the number of FOLDS and ESTIMATORS you chose in the configuration above. Each fold the model with lowest validation loss will be saved and used to predict OOF and test. Adjust the variable `VERBOSE`. The variable `VERBOSE=1 or 2` will display the training and validation loss for each iteration as text.

# **Let's take a look at our CV**

# In[ ]:


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    cmap_cv = plt.cm.coolwarm
    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))    
    for ii, (tr, tt) in enumerate(list(cv.split(X=X, y=y, groups=group))):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0        
        ax.scatter(range(len(indices)), [ii + .5] * len(indices), c=indices, marker='_', lw=lw, cmap=cmap_cv, vmin=-.2, vmax=1.2)
    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker='_', lw=lw, cmap=plt.cm.Set3)
    ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker='_', lw=lw, cmap=cmap_data)
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels, xlabel='Sample index', ylabel="CV iteration", ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def plot_importance(importances, features_names, PLOT_TOP_N = 20, figsize=(12, 20)):
    try: plt.close()
    except: pass
    importance_df = pd.DataFrame(data=importances, columns=features_names)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    plt.title('Feature Importances')
    sns.boxplot(data=sorted_importance_df[plot_cols], orient='h', ax=ax)
    plt.show()

asset_id = 1301
df = load_training_data(asset_id)
df_proc = get_features(df)
df_proc['date'] = df['date'].copy()
df_proc['y'] = df['Target']
df_proc = df_proc.dropna(how="any")
X = df_proc.drop("y", axis=1)
y = df_proc["y"]
groups = pd.factorize(X['date'].dt.day.astype(str) + '_' + X['date'].dt.month.astype(str) + '_' + X['date'].dt.year.astype(str))[0]
X = X.drop(columns = 'date')

fig, ax = plt.subplots(figsize = (12, 6))
cv = PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap = GROUP_GAP, max_train_group_size=MAX_TRAIN_GROUP_SIZE, max_test_group_size=MAX_TEST_GROUP_SIZE)
plot_cv_indices(cv, X, y, groups, ax, FOLDS, lw=20)


# **Main Training Function**

# In[ ]:


# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
VERBOSE = 0

def get_Xy_and_model():
    df = load_training_data()
    orig_close = df['Close'].copy()
    orig_sec_code = df['SecuritiesCode'].copy()
    df_proc = get_features(df)
    df_proc['date'] = df['date'].copy()
    df_proc['y'] = df['Target'].values
    df_proc = df_proc.dropna(how="any")
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    groups = pd.factorize(X['date'].dt.day.astype(str) + '_' + X['date'].dt.month.astype(str) + '_' + X['date'].dt.year.astype(str))[0]
    X = X.drop(columns = 'date')
    oof_preds = np.zeros(len(X))
    importances, scores, models = [], [], []
    for fold, (train_idx, val_idx) in enumerate(PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap = GROUP_GAP, max_train_group_size = MAX_TRAIN_GROUP_SIZE, max_test_group_size = MAX_TEST_GROUP_SIZE).split(X, y, groups)):
        # GET TRAINING, VALIDATION SET
        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # DISPLAY FOLD INFO
        print('#' * 25); print('#### FOLD', fold + 1)
        model = build_model(fold)

        # TRAIN
        model.fit( x_train, y_train, eval_set = [(x_val, y_val)], verbose = VERBOSE )
        
        # PREDICT OOF
        pred = model.predict(x_val)
        models.append(model)

        # REPORT RESULTS
        try: mse = mean_squared_error(np.nan_to_num(y_val), np.nan_to_num(pred))
        except: mse = 0.0
        scores.append(mse)
        print('#### FOLD %i OOF MSE %.3f' % (fold + 1, mse))

        oof_preds[val_idx] = pred
        importances.append(model.feature_importances_)

    df_proc['SecuritiesCode'] = orig_sec_code
    df = df_proc
    df['oof_preds'] = np.nan_to_num(oof_preds)
    df['Close'] = orig_close
    print('\n\n' + ('-' * 80) + '\n' + 'Finished trainings. Results:')
    print('Model: r2_score: %s | pearsonr: %s ' % (r2_score(df['y'], df['oof_preds']), pearsonr(df['y'], df['oof_preds'])[0]))
    print('Predictions std: %s | Target std: %s' % (df['oof_preds'].std(), df['y'].std()))

    try: plt.close()
    except: pass   
    df2 = df.reset_index().set_index('date')
    df2 = df2.loc[df2['SecuritiesCode'] == 1301] # For demonstration purpose only.
    fig = plt.figure(figsize = (12, 6))
    # fig, ax_left = plt.subplots(figsize = (12, 6))
    ax_left = fig.add_subplot(111)
    ax_left.set_facecolor('azure')
    ax_right = ax_left.twinx()
    ax_left.plot(df2['y'].rolling(3 * 30 * 24 * 60).corr(df2['oof_preds']).iloc[::24 * 60], color = 'crimson', label = "Corr")
    ax_right.plot(df2['Close'].iloc[::24 * 60], color = 'darkgrey', label = "%s Close" % stock_name_dict[asset_id])
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.title('3 month rolling pearsonr for %s' % (stock_name_dict[asset_id]))
    plt.show()

    plot_importance(np.array(importances), list(X.columns), PLOT_TOP_N = 20)
    
    return scores, oof_preds, models, y

print(f"Training model")
cur_scores, cur_oof_preds, cur_models, cur_targets = get_Xy_and_model()
scores, oof_preds, models, targets = cur_scores, cur_oof_preds, cur_models, cur_targets


# # <span class="title-section w3-xxlarge" id="codebook">Calculate OOF MSE</span>
# The OOF (out of fold) predictions are saved to disk. If you wish to ensemble multiple models, use the OOF to determine what are the best weights to blend your models with. Choose weights that maximize OOF CV score when used to blend OOF. Then use those same weights to blend your test predictions.

# In[ ]:


print('Overall MEAN OOF MSE %s' % np.mean(list(scores)))


# # <span class="title-section w3-xxlarge" id="submit">Submit To Kaggle 🇰</span>
# <hr>

# In[ ]:


env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (df_test, options, financials, trades, secondary_prices, df_pred) in iter_test:
    x_test = get_features(df_test)
    y_pred = np.mean(np.concatenate([np.expand_dims(model.predict(x_test), axis = 0) for model in models], axis = 0), axis = 0)
    df_pred['Target'] = y_pred
    df_pred = df_pred.sort_values(by = "Target", ascending = False)
    df_pred['Rank'] = np.arange(0,2000)
    df_pred = df_pred.sort_values(by = "SecuritiesCode", ascending = True)
    df_pred.drop(["Target"], axis = 1)
    submission = df_pred[["Date", "SecuritiesCode", "Rank"]]
    env.predict(submission)

