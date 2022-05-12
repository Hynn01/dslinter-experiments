#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction to PyCaret
# 
# PyCaret is an open source, **low-code** machine learning library in Python that aims to reduce the cycle time from hypothesis to insights. It is well suited for **seasoned data scientists** who want to increase the productivity of their ML experiments by using PyCaret in their workflows or for **citizen data scientists** and those **new to data science** with little or no background in coding. PyCaret allows you to go from preparing your data to deploying your model within seconds using your choice of notebook environment. Please click [this](https://pycaret.org/guide/) link to continue learning more about PyCaret. 
# 
# 
# **<span style="color:Red">Please upvote this kernel if you like it . It motivates me to produce more quality content :)**
# 

# ![pycaret](https://miro.medium.com/max/1400/1*Q34J2tT_yGrVV0NU38iMig.jpeg)

# ## 1.1 Installation (Let's install PyCaret)

# In[ ]:


get_ipython().system('pip install pycaret --quiet')


# ## 2 Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

#import regression module
from pycaret.regression import *


# ## 2.1 Load Data

# In[ ]:


BASE_PATH = '../input/trends-assessment-prediction'

fnc_df = pd.read_csv(f"{BASE_PATH}/fnc.csv")
loading_df = pd.read_csv(f"{BASE_PATH}/loading.csv")
labels_df = pd.read_csv(f"{BASE_PATH}/train_scores.csv")


# In[ ]:


fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df["is_train"] = True
df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()
print(f'Shape of train data: {df.shape}, Shape of test data: {test_df.shape}')


# In[ ]:


target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
df.drop(['is_train'], axis=1, inplace=True)
test_df = test_df.drop(target_cols + ['is_train'], axis=1)


# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1/500
df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


# ## 2.2 Utils

# In[ ]:


def get_train_data(target):
    other_targets = [tar for tar in target_cols if tar != target]
    train_df = df.drop( other_targets, axis=1)
    return train_df


# ## 3. Let's proceed with PyCaret for regression

# Before proceeding let me clear few things:
# * Currently `PyCaret` do not have support for multitarget regression. So, instead of 1 model for our 5 targets, we need to create individual model for each target.

# ### 3.1 Setup our dataset (For demo just using `age` target)
# 
# * `setup` function initializes the environment in pycaret and creates the transformation pipeline to prepare the data for modeling and deployment. setup() must called before executing any other function in pycaret. It takes two mandatory parameters: dataframe {array-like, sparse matrix} and name of the target column. All other parameters are optional.
# 
# 
# 
# For more info please visit https://pycaret.org/regression/#setup

# In[ ]:



target = 'age'

train_df = get_train_data(target)

setup_reg = setup(
    data = train_df,
    target = target,
    train_size=0.8,
    numeric_imputation = 'mean',
    silent = True
)


# ### 3.2 Compare Models
# 
# * `compare_models` function uses all models in the model library and scores them using K-fold Cross Validation. The output prints a score grid that shows MAE, MSE, RMSE, R2, RMSLE and MAPE by fold (default CV = 10 Folds) of all the available models in model library.
# 
# 
# 
# For more info please visit https://pycaret.org/regression/#compare-models

# In[ ]:


# There are few models which take a lot of time, ignoring those models to make demo faster.
blacklist_models = ['ransac', 'tr', 'rf', 'et', 'ada', 'gbr', 'xgboost', 'catboost']


# In[ ]:


compare_models(
    blacklist = blacklist_models,
    fold = 5,
    sort = 'MAE', ## competition metric
    turbo = True
)


# ### 3.3 Create Model
# For demo purpose let's create a Light Gradient Boosting Machine
# 
# * `create_model` function creates a model and scores it using K-fold Cross Validation. (default = 10 Fold). The output prints a score grid that shows MAE, MSE, RMSE, RMSLE, R2 and MAPE. This function returns a trained model object. setup() function must be called before using create_model()
# 
# 
# 
# For more info please visit https://pycaret.org/regression/#create-model

# In[ ]:


lgbm_age = create_model(
    estimator='lightgbm',
    fold=5
)


# ### 3.4 Tune Model Hyperparameters
# 
# * `tune_model` function tunes the hyperparameters of a model and scores it using K-fold Cross Validation. The output prints the score grid that shows MAE, MSE, RMSE, R2, RMSLE and MAPE by fold (by default = 10 Folds). This function returns a trained model object.  
# 
# 
# 
# For more info please visit https://pycaret.org/regression/#tune-model
# 

# In[ ]:


# here we are tuning the above created model
tuned_lgbm_age = tune_model(
    estimator='lightgbm',
    fold=5
)


# ### 3.5 Plot Model
# 
# * `plot_model` function takes a trained model object and returns a plot based on the test / hold-out set. The process may require the model to be re-trained in certain cases. See list of plots supported below. Model must be created using create_model() or tune_model().
# 
# 
# 
# For more info please visit https://pycaret.org/regression/#plot-model
# 
# 

# ### 3.5.1 Plotting learning curve

# In[ ]:


# plot_model(estimator = None, plot = ‘residuals’)
plot_model(estimator = tuned_lgbm_age, plot = 'learning')


# ### 3.5.2 plotting residuals

# In[ ]:


plot_model(estimator = tuned_lgbm_age, plot = 'residuals')


# ### 3.5.3 plotting feature importance

# In[ ]:


plot_model(estimator = tuned_lgbm_age, plot = 'feature')


# ### 3.6 Evaluate Model
# 
# * `evaluate_model` function displays a user interface for all of the available plots for a given estimator. It internally uses the plot_model() function.
# 
# 
# For more info please visit https://pycaret.org/regression/#evaluate-model
# 

# In[ ]:


evaluate_model(estimator=tuned_lgbm_age)


# ### 3.7 Interpret Model
# 
# 
# * `interpret_model` function takes a trained model object and returns an interpretation plot based on the test / hold-out set. It only supports tree based algorithms. This function is implemented based on the SHAP (SHapley Additive exPlanations), which is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations.
# 
# For more info please visit https://pycaret.org/regression/#interpret-model
# 
# 

# In[ ]:


interpret_model(
    estimator=tuned_lgbm_age,
    plot = 'summary',
    feature = None,
    observation = None
)


# ### 3.8 Predict Model
# 
# * `predict_model` function is used to predict new data using a trained estimator. It accepts an estimator created using one of the function in pycaret that returns a trained  model object or a list of trained model objects created using stack_models() or create_stacknet(). New unseen data can be passed to data param as pandas Dataframe.  If data is not passed, the test / hold-out set separated at the time of setup() is used to generate predictions.
# 
# For more info please visit https://pycaret.org/regression/#predict-model
# 
# 
# 

# In[ ]:


predictions =  predict_model(tuned_lgbm_age, data=test_df)


# In[ ]:


predictions.head()


# ## 4. Let's Proceed With Other Targets

# ### 4.1. comapring models for `age`

# For list of all available estimators and their abbreviations please visit https://pycaret.org/regression/

# In[ ]:


target = target_cols[0]
train_df = get_train_data(target)

setup_reg = setup(
    data = train_df,
    target = target,
    train_size=0.8,
    numeric_imputation = 'mean',
    silent = True
)

compare_models(
    blacklist = blacklist_models,
    fold = 7,
    sort = 'MAE',
    turbo = True
)


# ### 4.2. comapring models for `domain1_var1`

# In[ ]:


target = target_cols[1]
train_df = get_train_data(target)

setup_reg = setup(
    data = train_df,
    target = target,
    train_size=0.8,
    numeric_imputation = 'mean',
    silent = True
)

compare_models(
    blacklist = blacklist_models,
    fold = 7,
    sort = 'MAE',
    turbo = True
)


# ### 4.3. comapring models for `domain1_var2`

# In[ ]:


target = target_cols[2]
train_df = get_train_data(target)

setup_reg = setup(
    data = train_df,
    target = target,
    train_size=0.8,
    numeric_imputation = 'mean',
    silent = True
)

compare_models(
    blacklist = blacklist_models,
    fold = 7,
    sort = 'MAE',
    turbo = True
)


# ### 4.4. comapring models for `domain2_var1`

# In[ ]:


target = target_cols[3]
train_df = get_train_data(target)

setup_reg = setup(
    data = train_df,
    target = target,
    train_size=0.8,
    numeric_imputation = 'mean',
    silent = True
)

compare_models(
    blacklist = blacklist_models,
    fold = 7,
    sort = 'MAE',
    turbo = True
)


# ### 4.5. comapring models for `domain2_var2`

# In[ ]:


target = target_cols[4]
train_df = get_train_data(target)

setup_reg = setup(
    data = train_df,
    target = target,
    train_size=0.8,
    numeric_imputation = 'mean',
    silent = True
)

compare_models(
    blacklist = blacklist_models,
    fold = 7,
    sort = 'MAE',
    turbo = True
)


# ### Observations:
# 
# On close observation of ablove model comparisons, we have made following observations:
# * `age`: Bayesian Ridge	has the minimum MAE
# * `domain1_var1`: Ridge Regression has the minimum MAE
# * `domain1_var2`: Support Vector Machines has the minimum MAE
# * `domain2_var1`: Ridge Regression has the minimum MAE
# * `domain2_var2`: Support Vector Machines has the minimum MAE
# 
# > **Note: For demo purpose i have taken the model with lowest MAE in each target category, feel free to experiment with lowest 2 or 3 models and also try ensemble of various models in each category for better results.**

# ## 5. Tuning Selected Models

# In[ ]:


# mapping targets to their corresponding models

models = []

target_models_dict = {
    'age': 'br',
    'domain1_var1':'ridge',
    'domain1_var2':'svm',
    'domain2_var1':'ridge',
    'domain2_var2':'svm',
}

def tune_and_ensemble(target):
    train_df = get_train_data(target)    
    exp_reg = setup(
        data = train_df,
        target = target,
        train_size=0.8,
        numeric_imputation = 'mean',
        silent = True
    )
    
    model_name = target_models_dict[target]
    tuned_model = tune_model(model_name, fold=7)
    model = ensemble_model(tuned_model, fold=7)
    return model


# ### 5.1 Tuning Bayesian Ridge Model for `age`

# In[ ]:


target = target_cols[0]
model = tune_and_ensemble(target)
models.append(model)


# ### 5.2 Tuning Ridge Regression Model for `domain1_var1`

# In[ ]:


target = target_cols[1]
model = tune_and_ensemble(target)
models.append(model)


# ### 5.3 Tuning Support Vector Machines Model for `domain1_var2`

# In[ ]:


target = target_cols[2]
model = tune_and_ensemble(target)
models.append(model)


# ### 5.4 Tuning Ridge Regression Model for `domain2_var1`

# In[ ]:


target = target_cols[3]
model = tune_and_ensemble(target)
models.append(model)


# ### 5.5 Tuning Support Vector Machines Model for `domain2_var2`

# In[ ]:


target = target_cols[4]
model = tune_and_ensemble(target)
models.append(model)


# ## 6. Finalize, save model and Inference

# In[ ]:


### create a pipelie or functio to run for all targets

def finalize_model_pipeline(model, target):
    # this will train the model on houldout data
    finalize_model(model)
    save_model(model, f'{target}_{target_models_dict[target]}', verbose=True)
    # making predictions on test data
    predictions = predict_model(model, data=test_df)
    test_df[target] = predictions['Label'].values


# In[ ]:


for index, target in enumerate(target_cols):
    model = models[index]
    finalize_model_pipeline(model,target)


# ## Create Submission

# In[ ]:


sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5

sub_df.to_csv("submission1.csv", index=False)


# In[ ]:


sub_df.head(10)


# # References
# 
# * https://pycaret.org/regression/
# * https://towardsdatascience.com/machine-learning-in-power-bi-using-pycaret-34307f09394a

# # END NOTES
# 
# * This notebook is work in progress.  I will kepp updating this kernel with more and more info.
# * Feel free to use this kernel as the starting point. Happy kaggling:)
# 
# **<span style="color:Red">Please upvote this kernel if you like it . It motivates me to produce more quality content :)**  
