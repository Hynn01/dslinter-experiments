#!/usr/bin/env python
# coding: utf-8

# >### PurgedGroupTimeSeries CV - XGBoost Version
# >This is a simple starter notebook for Kaggle's JPX Comp showing purged group timeseries KFold with extra data. Purged Times Series is explained [here][1]. There are many configuration variables below to allow you to experiment. Use either CPU or GPU. You can control which years are loaded, which type of models are used, and whether to use feature engineering. You can experiment with different data preprocessing, model hyperparameters, loss, and number of seeds to ensemble. The extra datasets contain the full history of the assets at the same format of the competition, so you can input that into your model too.
# >
# >**NOTE:** this notebook lets you run a different experiment in each fold if you want to run lots of experiments. (Then it is like running multiple holdout validation experiments but in that case note that the overall CV score is meaningless because LB will be much different when the multiple experiments are ensembled to predict test). **If you want a proper CV with a reliable overall CV score you need to choose the same configuration for each fold.**
# >
# 
# [1]: TBD

# <center><img src="https://i.ibb.co/8bvJY8B/xgboost-logo.png" height=250 width=250></center>
# <hr>
# <center>XGBoost = 🌳 + 🧓🏻 + 💪🏻</center>

# XGBoost is a favorite choice on kaggle and it doesn't look like it is going anywhere! 
# It is basiclly the a version of gradient boosting machines framework that made the approach so popular.
# 
# **It is usually included in winning ensembles on Kaggle when solving a tabular problem**
# 
# XGBoost algorithm provides large range of hyperparameters. In order to get the best performance out of it, we need to know to tune them.
# 
# ><h4>TL;DR: What makes XGBoost great:</h4>
# >
# >1. XGBoost was the first wide-spread GBM framework so it has "more mileage" then all other frameworks.
# >2. Easy to use 
# >3. When using GPU it is usually faster than nearly all other gradient boosting algorithms that use GPU.
# >4. A very powerful gradient boosting. 
# 
# <h4>Leaf growth in XGBoost</h4>
# 
# XGboost splits up to the specified max_depth hyperparameter and then starts pruning the tree backwards and removes splits beyond which there is no positive gain. It uses this approach since sometimes a split of no loss reduction may be followed by a split with loss reduction. XGBoost can also perform leaf-wise tree growth (as LightGBM).
# 
# Normally it is impossible to enumerate all the possible tree structures q. A greedy algorithm that starts from a single leaf and iteratively adds branches to the tree is used instead. Assume that I_L and I_R are the instance sets of left and right nodes after the split. Then the loss reduction after the split is given by,
# 
# ![](https://i.imgur.com/jzyLh81.png)
# 
# <h4>XGBoost vs LightGBM</h4>
# 
# LightGBM uses a novel technique of Gradient-based One-Side Sampling (GOSS) to filter out the data instances for finding a split value while XGBoost uses pre-sorted algorithm & Histogram-based algorithm for computing the best split. Here instances mean observations/samples.
# 
# Let's see how pre-sorting splitting works-
# 
# - For each node, enumerate over all features
# 
# - For each feature, sort the instances by feature value
# 
# - Use a linear scan to decide the best split along that feature basis information gain
# 
# - Take the best split solution along all the features
# 
# In simple terms, Histogram-based algorithm splits all the data points for a feature into discrete bins and uses these bins to find the split value of histogram. While, it is efficient than pre-sorted algorithm in training speed which enumerates all possible split points on the pre-sorted feature values, it is still behind GOSS in terms of speed.
# 
# <h4>XGBoost Model Parameters</h4>
# 
# >For an exhaustive overview of all parameters [see here](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
# 
# **objective [default=reg:linear]**
# 
# This defines the loss function to be minimized. Mostly used values are:
# 
# - binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
# 
# - multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
# you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
# 
# - multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
# 
# **eval_metric [ default according to objective ]**
# 
# The metric to be used for validation data. The default values are rmse for regression and error for classification.
# Typical values are:
# 
# - rmse – root mean square error
# 
# - mae – mean absolute error
# 
# - logloss – negative log-likelihood
# 
# - error – Binary classification error rate (0.5 threshold)
# 
# - merror – Multiclass classification error rate
# 
# - mlogloss – Multiclass logloss
# 
# - auc: Area under the curve
# 
# 
# **eta [default=0.3]**
# 
# - Analogous to learning rate in GBM.
# - Makes the model more robust by shrinking the weights on each step.
# - Typical final values to be used: 0.01-0.2
# 
# **colsample_bytree**:  We can create a random sample of the features (or columns) to use prior to creating each decision tree in the boosted model. That is, tuning Column Sub-sampling in XGBoost By Tree. This is controlled by the colsample_bytree parameter. The default value is 1.0 meaning that all columns are used in each decision tree. A fraction (e.g. 0.6) means a fraction of columns to be subsampled. We can evaluate values for colsample_bytree between 0.1 and 1.0 incrementing by 0.1.
# 
# <h3>Regularization in XGBoost</h3>
# 
# XGBoost adds built-in regularization to achieve accuracy gains beyond gradient boosting. Regularization is the process of adding information to reduce variance and prevent overfitting.
# 
# Although data may be regularized through hyperparameter fine-tuning, regularized algorithms may also be attempted. For example, Ridge and Lasso are regularized machine learning alternatives to LinearRegression.
# 
# XGBoost includes regularization as part of the learning objective, as contrasted with gradient boosting and random forests. The regularized parameters penalize complexity and smooth out the final weights to prevent overfitting. XGBoost is a regularized version of gradient boosting.
# 
# Mathematically, XGBoost's learning objective may be defined as follows:
# 
# $$obj(θ) = l(θ) + Ω (θ)$$
# 
# Here, **l(θ)**  is the loss function, which is the Mean Squared Error (MSE) for regression, or the log loss for classification, and **Ω (θ)** is the regularization function, a penalty term to prevent over-fitting. Including a regularization term as part of the objective function distinguishes XGBoost from most tree ensembles.
# 
# The learning objective for the th boosted tree can now be rewritten as follows:
# 
# ![img](https://i.imgur.com/IRNCrvM.png)
# 
# **reg_alpha and reg_lambda** : First note the loss function is defined as
# 
# ![img](https://i.imgur.com/aw1Hod9.png)
# 
# >So the above is how the regularized objective function looks like if you want to allow for the inclusion of a L1 and a L2 parameter in the same model
# 
# `reg_alpha` and `reg_lambda` control the L1 and L2 regularization terms, which in this case limit how extreme the weights at the leaves can become. Higher values of alpha mean more L1 regularization. See the documentation [here](http://xgboost.readthedocs.io/en/latest///parameter.html#parameters-for-tree-booster).
# 
# Since L1 regularization in GBDTs is applied to leaf scores rather than directly to features as in logistic regression, it actually serves to reduce the depth of trees. This in turn will tend to reduce the impact of less-predictive features. We might think of L1 regularization as more aggressive against less-predictive features than L2 regularization.
# 
# These two regularization terms have different effects on the weights; L2 regularization (controlled by the lambda term) encourages the weights to be small, whereas L1 regularization (controlled by the alpha term) encourages sparsity — so it encourages weights to go to 0. This is helpful in models such as logistic regression, where you want some feature selection, but in decision trees we’ve already selected our features, so zeroing their weights isn’t super helpful. For this reason, I found setting a high lambda value and a low (or 0) alpha value to be the most effective when regularizing.
# 
# >[From this Paper](https://arxiv.org/pdf/1603.02754.pdf)
# 
# 
# <div class="alert alert-block alert-info">
# <b>Read More:</b>
# <ul>
#     <li><a href = "https://github.com/dmlc/xgboost">XGBoost Github Documentation</a></li>
#     <li><a href = "https://xgboost.readthedocs.io/en/stable/parameter.html">XGBoost Parameters</a></li>
#     <li><a href = "https://xgboost.readthedocs.io/en/stable/">Official Documentation</a></li>    
# </ul>
# </div>
# 
# ____
# 
# <h3>All Parameters Overview</h3>
# 
# ____
# 
# Before diving into the actual parameters of XGBoost, Let's define three types of parameters: General Parameters, Booster Parameters and Task Parameters.
# 
# 1. **General parameters**  Relate to chosing which booster algorithm we will be using, usually tree or linear model.
# 
# 2. **Booster parameters**  Are the actual parameters of the booster you have chosen.
# 
# 3. **Task parameters**  Tells the framework what problem are we trying to solve. For example, regression tasks may use different parameters with ranking tasks.
# 
# ____
# 
# 
# <h3>How to tune XGBoost like a boss?</h3>
# 
# Hyperparameters tuning guide:
# 
# <h4>General Parameters</h4>
# 
# 1. **booster**  [default= gbtree ] 
#   - Which booster to use. Can be gbtree, gblinear or dart; 
#   - gbtree and dart use tree based models while gblinear uses linear functions.
# 
# 2. **verbosity** [default=1]
#   - Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). 
#   - Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message. 
#   - If there’s unexpected behaviour, please try to increase value of verbosity.
# 
# 3. **nthread**  [default to maximum number of threads available if not set]
#   - Number of parallel threads used to run XGBoost. When choosing it, please keep thread contention and hyperthreading in mind.
# 
# ____
# 
# 
# <h4>Tree Booster Parameters</h4>
# 
# 1. **eta [default=0.3, ]**
#   - alias: learning_rate
#   - Step size shrinkage used in update to prevents overfitting.
#   - After each boosting step, we can directly get the weights of new features
#   - It makes the model more robust by shrinking the weights on each step.
#   - range: [0,1]
# 
# 2. **gamma [default=0]**
#   - Minimum loss reduction required to make a further partition on a leaf node of the tree. 
#   - The larger gamma is, the more conservative the algorithm will be.
#   - range: [0,∞]
# 
# 3. **max_depth [default=6]**
# 
#   - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 
#   - 0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. 
#   - Beware that XGBoost aggressively consumes memory when training a deep tree.
#   - range: [0,∞] )
# 
# 4. **min_child_weight [default=1]**
# 
#   - its Minimum sum of instance weight (hessian) needed in a child. I
#   - if the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. 
#   - In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. 
#   - The larger min_child_weight is, the more conservative the algorithm will be.
#   - range: [0,∞]
# 
# 5. **max_delta_step [default=0]**
# 
#   - Maximum delta step we allow each leaf output to be. 
#   - If the value is set to 0, it means there is no constraint. 
#   - If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
#   - range: [0,∞]
# 
# 6. **subsample [default=1]**
# 
#   - It denotes the fraction of observations to be randomly samples for each tree.
#   - Subsample ratio of the training instances.
#   - Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. - This will prevent overfitting.
#   - Subsampling will occur once in every boosting iteration.
#   - Lower values make the algorithm more conservative and prevents overfitting but too small alues might lead to under-fitting.
#   - typical values: 0.5-1
#   - range: (0,1]
# 
# 7. **sampling_method [default= uniform]**
# 
#   - The method to use to sample the training instances.
#   - **uniform:** each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.
#   - **gradient_based:**  the selection probability for each training instance is proportional to the regularized absolute value of gradients 
# 
# 8. **colsample_bytree, colsample_bylevel, colsample_bynode [default=1]**
# 
# ><h4>This is a family of parameters for subsampling of columns.</h4>
# >
# >**All colsample_by**  parameters have a range of (0, 1], the default value of 1, and specify the fraction of columns to be subsampled.
# >
# >**lsample_bytree**s the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
# >
# >**colsample_bylevel**  is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
# >
# >**colsample_bynode**  is the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level.
# >
# >**colsample_by**  parameters work cumulatively. For instance, the combination **{'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5}** with 64 features will leave 8 features to choose from at each split.
# >
# 
# 9. **lambda [default=1]**
#   - alias: reg_lambda
#   - L2 regularization term on weights. 
#   - Increasing this value will make model more conservative.
# 
# 10. **alpha [default=0]**
#   - alias: reg_alpha
#   - L1 regularization term on weights.
#   - Increasing this value will make model more conservative.
# 
# 11. **grow_policy [default= depthwise]**
#   - Controls a way new nodes are added to the tree.
#   - Currently supported only if tree_method is set to hist or gpu_hist.
#   - **Choices:*  - depthwise, lossguide
#   - **depthwise:*  - split at nodes closest to the root.
#   - **lossguide:*  - split at nodes with highest loss change.
# 
# 12. **max_leaves [default=0]**
#   - Maximum number of nodes to be added. 
#   - Only relevant when grow_policy=lossguide is set.
# 
# [Read More](https://xgboost.readthedocs.io/en/latest/parameter.html)
# 
# ____
# 
# 
# 
# <h4>Task Parameters</h4>
# 
# 1. **objective [default=reg:squarederror]**
# 
# It defines the loss function to be minimized. Most commonly used values are given below -
# 
#   - reg:squarederror : regression with squared loss.
# 
#   - reg:squaredlogerror: regression with squared log loss 1/2[log(pred+1)−log(label+1)]2. - All input labels are required to be greater than -1.
# 
#   - reg:logistic : logistic regression
# 
#   - binary:logistic : logistic regression for binary classification, output probability
# 
#   - binary:logitraw: logistic regression for binary classification, output score before logistic transformation
# 
#   - binary:hinge : hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
# 
#   - multi:softmax : set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
# 
#   - multi:softprob : same as softmax, but output a vector of ndata nclass, which can be further reshaped to ndata nclass matrix. The result contains predicted probability of each data point belonging to each class.
# 
# 2. **eval_metric [default according to objective]**
#   - The metric to be used for validation data.
#   - The default values are rmse for regression, error for classification and mean average precision for ranking.
#   - We can add multiple evaluation metrics.
#   - Python users must pass the metrices as list of parameters pairs instead of map.
#   - The most common values are given below -
# 
#    - rmse : root mean square error
#    - mae : mean absolute error
#    - logloss : negative log-likelihood
#    - error : Binary classification error rate (0.5 threshold). 
#    - merror : Multiclass classification error rate.
#    - mlogloss : Multiclass logloss
#    - auc: Area under the curve
#    - aucpr : Area under the PR curve
#    
#    
# 
# <div class="alert alert-block alert-warning">
# <b>Introduction Credits:</b>
# <ul>
#     <li><a href = "https://www.kaggle.com/shivansh002/a-gentle-guide-on-xgboost-hyperparameters">A Gentle Guide on XGBoost hyperparameters</a> By @shivansh002. Thank you @shivansh002 for a great introduction! </li>
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
from xgboost import XGBRegressor
import jpx_tokyo_market_prediction
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
MAX_DEPTH = [10, 10, 10, 10, 10]


# # <span class="title-section w3-xxlarge" id="loading">Data Loading 🗃️</span>
# <hr>
# 
# The data organisation has already been done and saved to Kaggle datasets. Here we choose which years to load. We can use either 2017, 2018, 2019, 2020, 2021, Original, Supplement by changing the `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP`, `INCSUPP` variables in the preceeding code section. These datasets are discussed [here][1].
# 
# [1]: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285726

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
# Additionally we can decide to use external data by changing the variables `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP`, `INCSUPP` in the preceeding code section. These variables respectively indicate whether to load last year 2021 data and/or year 2020, 2019, 2018, 2017, the original, supplemented data. These datasets are discussed [here][3]
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
    model = XGBRegressor(n_estimators = N_ESTIMATORS[fold], max_depth = MAX_DEPTH[fold], learning_rate = LEARNING_RATE[fold], tree_method = 'hist' if DEVICE == 'CPU' else 'gpu_hist')

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
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    groups = pd.factorize(X['date'].dt.day.astype(str) + '_' + X['date'].dt.month.astype(str) + '_' + X['date'].dt.year.astype(str))[0]
    X = X.drop(columns = 'date')
    oof_preds = np.zeros(len(X))
    importances, scores, models = [], [], []
    for fold, (train_idx, val_idx) in enumerate(PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap = GROUP_GAP).split(X, y, groups)):
        # GET TRAINING, VALIDATION SET
        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # DISPLAY FOLD INFO
        print('#' * 25); print('#### FOLD', fold + 1)
        print('#### Training N_ESTIMATORS %s | MAX_DEPTH %s | LEARNING_RATE %s' % (N_ESTIMATORS[fold], MAX_DEPTH[fold], LEARNING_RATE[fold]))

        model = build_model(fold)

        # TRAIN
        model.fit( x_train, y_train, eval_set = [(x_val, y_val)], verbose = VERBOSE)

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

