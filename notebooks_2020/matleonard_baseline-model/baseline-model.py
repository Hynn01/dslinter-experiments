#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this course, you will learn a practical approach to feature engineering. You'll be able to apply what you learn to Kaggle competitions and other machine learning applications. 

# ### Load the data
# 
# We'll work with data from Kickstarter projects. The first few rows of the data looks like this:

# In[ ]:



import pandas as pd
ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',
                 parse_dates=['deadline', 'launched'])
ks.head(6)


# The `state` column shows the outcome of the project.

# In[ ]:


print('Unique values in `state` column:', list(ks.state.unique()))


# Using this data, how can we use features such as project category, currency, funding goal, and country to predict if a Kickstarter project will succeed? 

# ### Prepare the target column
# 
# First we'll convert the `state` column into a target we can use in a model.  Data cleaning isn't the current focus, so we'll simplify this example by:
# 
# - Dropping projects that are "live"
# - Counting "successful" states as `outcome = 1`
# - Combining every other state as `outcome = 0`

# In[ ]:


# Drop live projects
ks = ks.query('state != "live"')

# Add outcome column, "successful" == 1, others are 0
ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))


# ### Convert timestamps
# 
# Next, we convert the `launched` feature into categorical features we can use in a model. Since we loaded the columns as timestamp data, we access date and time values through the `.dt` attribute on the timestamp column.
# 
# **Note**: If you're not familiar with categorical features and label encoding, please check out **[this lesson](https://www.kaggle.com/alexisbcook/categorical-variables)** from the Intermediate Machine Learning course.

# In[ ]:


ks = ks.assign(hour=ks.launched.dt.hour,
               day=ks.launched.dt.day,
               month=ks.launched.dt.month,
               year=ks.launched.dt.year)


# ### Prep categorical variables
# 
# Now for the categorical variables -- `category`, `currency`, and `country` -- we'll need to convert them into integers so our model can use the data. For this we'll use scikit-learn's `LabelEncoder`. This assigns an integer to each value of the categorical feature.

# In[ ]:


from sklearn.preprocessing import LabelEncoder

cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()

# Apply the label encoder to each column
encoded = ks[cat_features].apply(encoder.fit_transform)


# We collect all of these features in a new dataframe that we can use to train a model.

# In[ ]:


# Since ks and encoded have the same index and I can easily join them
data = ks[['goal', 'hour', 'day', 'month', 'year', 'outcome']].join(encoded)
data.head()


# ### Create training, validation, and test splits
# 
# We need to create data sets for training, validation, and testing. We'll use a fairly simple approach and split the data using slices. We'll use 10% of the data as a validation set, 10% for testing, and the other 80% for training.

# In[ ]:


valid_fraction = 0.1
valid_size = int(len(data) * valid_fraction)

train = data[:-2 * valid_size]
valid = data[-2 * valid_size:-valid_size]
test = data[-valid_size:]


# ### Train a model
# 
# For this course we'll be using a LightGBM model. This is a tree-based model that typically provides the best performance, even compared to XGBoost. It's also relatively fast to train. 
# 
# We won't do hyperparameter optimization because that isn't the goal of this course. So, our models won't be the absolute best performance you can get. But you'll still see model performance improve as we do feature engineering.

# In[ ]:


import lightgbm as lgb

feature_cols = train.columns.drop('outcome')

dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)


# ### Make predictions & evaluate the model
# 
# Finally, let's make predictions on the test set with the model and see how well it performs. An important thing to remember is that you can overfit to the validation data. This is why we need a test set that the model never sees until the final evaluation.

# In[ ]:


from sklearn import metrics
ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['outcome'], ypred)

print(f"Test AUC score: {score}")


# # Your Turn
# Now you'll **[build your own baseline model](https://www.kaggle.com/kernels/fork/5407496)** which you can improve with feature engineering techniques as you go through the course.
# 

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/feature-engineering/discussion) to chat with other learners.*
