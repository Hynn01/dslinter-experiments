#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


# To help with reading and manipulation of data
import numpy as np
import pandas as pd

# To help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# To split the data
from sklearn.model_selection import train_test_split

# To impute missing values
from sklearn.impute import SimpleImputer

# To build a Random forest classifier
from sklearn.ensemble import RandomForestClassifier

# To tune a model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# To get different performance metrics
import sklearn.metrics as metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    accuracy_score,
    precision_score,
    f1_score,
)

# To suppress warnings
import warnings

warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv("../input/financial-datasets/Loan.csv")


# In[ ]:


data = df.copy()


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


# checking missing values in the data
data.isna().sum()


# In[ ]:



data["region"] = data["region"].astype("category")
data["phone_operator"] = data["phone_operator"].astype("category")
data["product_type"] = data["product_type"].astype("category")


# In[ ]:


# checking the distribution of the target variable
data["target"].value_counts(1)


# ### Splitting the data into X and y

# In[ ]:


# separating the independent and dependent variables
X = data.drop(["target"], axis=1)
y = data["target"]

# creating dummy variables
X = pd.get_dummies(X, drop_first=True)


# In[ ]:


# Splitting data into training, validation and test set:

# first we split data into 2 parts, say temporary and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5, stratify=y
)

# then we split the temporary set into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=5, stratify=y_temp
)

print(X_train.shape, X_val.shape, X_test.shape)


# In[ ]:


# Let's impute the missing values
imp_median = SimpleImputer(missing_values=np.nan, strategy="median")

# fit the imputer on train data and transform the train data
X_train["income"] = imp_median.fit_transform(X_train[["income"]])

# transform the validation and test data using the imputer fit on train data
X_val["income"] = imp_median.transform(X_val[["income"]])
X_test["income"] = imp_median.transform(X_test[["income"]])


# In[ ]:


# Checking class balance for whole data, train set, validation set, and test set

print("Target value ratio in y")
print(y.value_counts(1))
print("*" * 80)
print("Target value ratio in y_train")
print(y_train.value_counts(1))
print("*" * 80)
print("Target value ratio in y_val")
print(y_val.value_counts(1))
print("*" * 80)
print("Target value ratio in y_test")
print(y_test.value_counts(1))
print("*" * 80)


# ## Model evaluation criterion
# 
# 
# **What does a bank want?**
# * A bank wants to minimize the loss - it can face 2 types of losses here: 
#    * Whenever a bank lends money to a customer, they don't return it.
#    * A bank doesn't lend money to a customer thinking a customer will default but in reality, the customer won't - opportunity loss.
# 
# **Which loss is greater ?**
# * Lending to a customer who wouldn't be able to pay back.
# 
# **Since we want to reduce loan defaults we should use Recall as a metric of model evaluation instead of accuracy.**
# 
# * Recall - It gives the ratio of True positives to Actual positives, so high Recall implies low false negatives, i.e. low chances of predicting a bad customer as a good customer.
# 

# # Hyperparameter Tuning

# ### Let's first build a model with default parameters and see it's performance

# In[ ]:


# model without hyperparameter tuning
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)


# #### Let's check model's performance

# In[ ]:


# Checking recall score on train and validation set
print("Recall on train and validation set")
print(recall_score(y_train, rf.predict(X_train)))
print(recall_score(y_val, rf.predict(X_val)))
print("")

# Checking Precision score on train and validation set
print("Precision on train and validation set")
print(precision_score(y_train, rf.predict(X_train)))
print(precision_score(y_val, rf.predict(X_val)))

print("")

# Checking Accuracy score on train and validation set
print("Accuracy on train and validation set")
print(accuracy_score(y_train, rf.predict(X_train)))
print(accuracy_score(y_val, rf.predict(X_val)))


# - The model is performing well on the train data but the performance on the validation data is very poor.
# - Let's see if we can improve it with hyperparameter tuning.

# ## Grid Search CV
# * Hyperparameter tuning is also tricky in the sense that there is no direct way to calculate how a change in the hyperparameter value will reduce the loss of your model, so we usually resort to experimentation. i.e we'll use Grid search
# * Grid search is a tuning technique that attempts to compute the optimum values of hyperparameters. 
# * It is an exhaustive search that is performed on the specific parameter values of a model.
# * The parameters of the estimator/model used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# - **How to know the hyperparameters available for an algorithm?**

# In[ ]:


RandomForestClassifier().get_params()


# - We can see the names of hyperparameters available and their default values. 
# - We can choose which ones to tune.

# In[ ]:


print(np.arange(0.2, 0.7, 0.1))

print(np.arange(5,10))


# ### Let's tune Random forest using Grid Search

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Choose the type of classifier. \nrf1 = RandomForestClassifier(random_state=1)\n\n# Grid of parameters to choose from\nparameters = {"n_estimators": [150,200,250],\n    "min_samples_leaf": np.arange(5, 10),\n    "max_features": np.arange(0.2, 0.7, 0.1),\n    "max_samples": np.arange(0.3, 0.7, 0.1),\n    "class_weight" : [\'balanced\', \'balanced_subsample\'],\n    "max_depth":np.arange(3,4,5),\n    "min_impurity_decrease":[0.001, 0.002, 0.003]\n             }\n\n# Type of scoring used to compare parameter combinations\nacc_scorer = metrics.make_scorer(metrics.recall_score)\n\n# Run the grid search\ngrid_obj = GridSearchCV(rf1, parameters, scoring=acc_scorer, cv=5, n_jobs= -1, verbose = 2)\n# verbose = 2 tells about the number of fits, which can give an idea of how long will the model take in tuning\n# n_jobs = -1 so that all CPU cores can be run parallelly to optimize the Search\n\ngrid_obj = grid_obj.fit(X_train, y_train)\n\n# Print the best combination of parameters\ngrid_obj.best_params_')


# #### Let's check the best CV score, for the obtained parameters

# In[ ]:


grid_obj.best_score_


# #### Let's build a model with obtained best parameters
# - We are hard coding the hyperparameters separately so that we don't have to run the grid search again.

# In[ ]:


# Set the clf to the best combination of parameters
rf1_tuned = RandomForestClassifier(
    class_weight="balanced",
    max_features=0.2,
    max_samples=0.6000000000000001,
    min_samples_leaf=5,
    n_estimators=150,
    max_depth=3,
    random_state=1,
    min_impurity_decrease=0.001,
)

# Fit the best algorithm to the data.
rf1_tuned.fit(X_train, y_train)


# #### Let's check the model's performance

# In[ ]:


# Checking recall score on train and validation set
print("Recall on train and validation set")
print(recall_score(y_train, rf1_tuned.predict(X_train)))
print(recall_score(y_val, rf1_tuned.predict(X_val)))
print("")

# Checking precision score on train and validation set
print("Precision on train and validation set")
print(precision_score(y_train, rf1_tuned.predict(X_train)))
print(precision_score(y_val, rf1_tuned.predict(X_val)))
print("")

# Checking accuracy score on train and validation set
print("Accuracy on train and validation set")
print(accuracy_score(y_train, rf1_tuned.predict(X_train)))
print(accuracy_score(y_val, rf1_tuned.predict(X_val)))


# - We can see improvement in validation performance as compared to the model without hyperparameter tuning
# - Recall on both training set and validation set is good and is 88% on the validation

# ## Randomized Search CV
# * Random search is a tuning technique that attempts to compute the optimum values of hyperparameters randomly unlike grid search

# ### Let's tune Random forest using Randomized Search

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Choose the type of classifier. \nrf2 = RandomForestClassifier(random_state=1)\n\n# Grid of parameters to choose from\nparameters = {"n_estimators": [150,200,250],\n    "min_samples_leaf": np.arange(5, 10),\n    "max_features": np.arange(0.2, 0.7, 0.1), \n    "max_samples": np.arange(0.3, 0.7, 0.1),\n    "max_depth":np.arange(3,4,5),\n    "class_weight" : [\'balanced\', \'balanced_subsample\'],\n    "min_impurity_decrease":[0.001, 0.002, 0.003]\n             }\n\n# Type of scoring used to compare parameter combinations\nacc_scorer = metrics.make_scorer(metrics.recall_score)\n\n# Run the random search\ngrid_obj = RandomizedSearchCV(rf2, parameters,n_iter=30, scoring=acc_scorer,cv=5, random_state = 1, n_jobs = -1, verbose = 2)\n# using n_iter = 30, so randomized search will try 30 different combinations of hyperparameters\n# by default, n_iter = 10\n\ngrid_obj = grid_obj.fit(X_train, y_train)\n\n# Print the best combination of parameters\ngrid_obj.best_params_')


# #### Let's check the best CV score, for the obtained parameters

# In[ ]:


grid_obj.best_score_


# #### Let's build a model with obtained best parameters

# In[ ]:


# Set the clf to the best combination of parameters
rf2_tuned = RandomForestClassifier(
    class_weight="balanced",
    max_features=0.2,
    max_samples=0.5,
    min_samples_leaf=5,
    n_estimators=150,
    random_state=1,
    max_depth=3,
    min_impurity_decrease=0.003,
)

# Fit the best algorithm to the data.
rf2_tuned.fit(X_train, y_train)


# - Different results from the grid and the random search
# - Randomised search might give better results than grid search for the same parameter grid because of the use of cross-validation as fold varies the scores also vary

# #### Let's check the model's performance

# In[ ]:


# Checking recall score on train and validation set
print("Recall on train and validation set")
print(recall_score(y_train, rf2_tuned.predict(X_train)))
print(recall_score(y_val, rf2_tuned.predict(X_val)))
print("")
print("Precision on train and validation set")
# Checking precision score on train and validation set
print(precision_score(y_train, rf2_tuned.predict(X_train)))
print(precision_score(y_val, rf2_tuned.predict(X_val)))
print("")
print("Accuracy on train and validation set")
# Checking accuracy score on train and validation set
print(accuracy_score(y_train, rf2_tuned.predict(X_train)))
print(accuracy_score(y_val, rf2_tuned.predict(X_val)))


# - The model is performing better than model with default parameters and the performance is similar to the model we received with grid search

# #### Choose a best model and predict the performance on the test set

# In[ ]:


model = rf1_tuned


# In[ ]:


# Checking recall score on test set
print("Recall on test set")
print(recall_score(y_test, model.predict(X_test)))
print("")

# Checking precision score on test set
print("Precision on test set")
print(precision_score(y_test, model.predict(X_test)))
print("")

# Checking accuracy score on test set
print("Accuracy on test set")
print(accuracy_score(y_test, model.predict(X_test)))


# - The performance is close to one we observed in the validation set, so there is no overfitting
