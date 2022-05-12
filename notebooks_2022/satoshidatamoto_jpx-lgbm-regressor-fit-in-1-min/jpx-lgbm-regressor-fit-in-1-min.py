#!/usr/bin/env python
# coding: utf-8

# # JPX: LGBM Regressor (Fit in 1 Min)
# 

# <center><img src="https://lightgbm.readthedocs.io/en/latest/_images/LightGBM_logo_black_text.svg" height=250 width=250></center>
# <hr>
# <center>LightGBM = 🌳 + 🚀 + ☢️</center>

# 
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
# 

# # <span class="title-section w3-xxlarge" id="outline">Libraries 📚</span>
# <hr>

# #### Code starts here ⬇

# In[ ]:





# ## Imports

# In[ ]:


import os
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import jpx_tokyo_market_prediction
import warnings; warnings.filterwarnings("ignore")


# In[ ]:


stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
df_train = pd.concat([prices, supplemental_prices])
df_train = pd.merge(df_train, stock_list[['SecuritiesCode', 'Name']], left_on = 'SecuritiesCode', right_on = 'SecuritiesCode', how = 'left')
stock_list = stock_list.loc[stock_list['SecuritiesCode'].isin(prices['SecuritiesCode'].unique())]
print(list(df_train.columns))
print(len(list(df_train['SecuritiesCode'].unique())))
df_train.head()


# ## Training

# **Feature Extraction**

# In[ ]:


def upper_shadow(df): return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df): return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.
def get_features(df):
    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat


# **Main Training Function**

# In[ ]:


def get_Xy_and_model(df_train):

    df_proc = get_features(df_train)
    df_proc['y'] = df_train['Target']
    df_proc = df_proc.dropna(how = "any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    
    model = lgb.LGBMRegressor(device_type = 'gpu')
    model.fit(X, y)
    return X, y, model


# ## Loop over all securities

# In[ ]:


print(f"Training model")
X, y, model = get_Xy_and_model(df_train)
Xs, ys, models = X, y, model


# In[ ]:


x = get_features(df_train.iloc[1])
y_pred = models.predict(pd.DataFrame([x]))
y_pred[0]


# ## Submission

# In[ ]:


env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()


# In[ ]:


for (df_test, options, financials, trades, secondary_prices, df_pred) in iter_test:
    df_pred['row_id'] = (df_pred['Date'].astype(str) + '_' + df_pred['SecuritiesCode'].astype(str))
    df_test['row_id'] = (df_test['Date'].astype(str) + '_' + df_test['SecuritiesCode'].astype(str))
    model = models
    x_test = get_features(df_test)
    y_pred = model.predict(x_test)
    df_pred['Target'] = y_pred
    df_pred = df_pred.sort_values(by = "Target", ascending = False)
    df_pred['Rank'] = np.arange(0,2000)
    df_pred = df_pred.sort_values(by = "SecuritiesCode", ascending = True)
    df_pred.drop(["Target"], axis = 1)
    submission = df_pred[["Date", "SecuritiesCode", "Rank"]]    
    env.predict(submission)


# In[ ]:


print(df_pred.columns)

