#!/usr/bin/env python
# coding: utf-8

# # Tabular Playground Series - May 2022
# 
# For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state `0` or state `1`. The data has various feature interactions that may be important in determining the machine state.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv", index_col="id")

submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv', index_col="id")


# # Data Analysis

# In[ ]:


train.head()


# Note that not all columns contain continuous (`float`) data, but there are is also categorical (`int`) and string (`object`) data.

# In[ ]:


train.dtypes


# In[ ]:


train.describe().T


# ## Invalid Data
# 
# Check for invalid data.

# In[ ]:


train.isna().sum()


# ## Histogram
# 
# Histogram of the given data. The continuous columns follow a normal distribution, while the discrete columns are skewed and have gaps.

# In[ ]:


fig = plt.figure(figsize=(20, 12))
hist = train.hist(ax = fig.gca(), bins=20)


# Create column accessors for the different data types:

# In[ ]:


CONTINUOUS = train.columns[train.dtypes == 'float64'].to_list()
DISCRETE = train.columns[train.dtypes == 'int64'].to_list()
OBJECT = train.columns[train.dtypes == 'object'].to_list()

DISCRETE.remove("target")

print("CONTINUOUS:", ', '.join(CONTINUOUS))
print("DISCRETE:", ', '.join(DISCRETE))
print("OBJECT:", ', '.join(OBJECT))


# ## Dataset balance
# 
# Is the dataset balanced?

# In[ ]:


train.target.value_counts()


# ## Correlations

# In[ ]:


corr = train.corr()

plt.figure(figsize=(12, 12))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# # Data Augmentation
# 
# ## Handling string data
# 
# Column `f_27` contains string data with different letters. Convert this into categorical data so the model can learn from it by counting the occurence of each letter in each string and augmenting the dataset with this data.

# In[ ]:


train.f_27


# In[ ]:


def get_unique_letters(arr):
    unique = set()
    for val in arr:
        unique |= set(val)
    l = list(unique)
    l.sort()
    return l

unique_letters = get_unique_letters(train.f_27)
print(unique_letters)


# In[ ]:


LETTER_COUNTS = [letter + '_num' for letter in unique_letters]

def augment(df, unique_letters):
    for col, letter in zip(LETTER_COUNTS, unique_letters):
        df[col] = df.f_27.str.count(letter)

augment(train, unique_letters)
augment(test, unique_letters)


# In[ ]:


train.head()


# # Model training

# In[ ]:


ALL_TRAIN_COLUMNS = CONTINUOUS + DISCRETE + LETTER_COUNTS


# ## Train and Validation Split

# 

# In[ ]:


from sklearn.model_selection import train_test_split
VAL_SPLIT = 0.1
X_train, X_valid, y_train, y_valid = train_test_split(train[ALL_TRAIN_COLUMNS], train.target, 
                                                      test_size = VAL_SPLIT, random_state = 42)


# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators    = 4096,
    random_state    = 42,
    tree_method     = "gpu_hist",
    early_stopping_rounds = 256, 
    eval_metric     = ['auc']
)

model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], verbose = 100)


# ## Convergence and overfitting

# In[ ]:


history = model.evals_result()

LABELS = {"validation_0": "train", "validation_1": "valid"}

def plot_metric(metric):
    fig, axis = plt.subplots(1, 1)
    for name, label in LABELS.items():
        data = history[name][metric]
        plt.plot(data, label=label)
    plt.title(metric)
    plt.grid()
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

plot_metric("auc")


# ## Feature importance

# In[ ]:


def plot_importance(importance, names, max_features = 10):
    zipped = list(zip(importance, names))
    zipped.sort(reverse=True)

    plt.figure(figsize=(12,10))
    palette = sns.color_palette("Reds_r", max_features)
    
    sns.barplot(x=[imp for imp, _ in zipped][:max_features], y=[name for _, name in zipped][:max_features],
               palette = palette)
    
    plt.title("Feature Importances")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    
plot_importance(model.feature_importances_, ALL_TRAIN_COLUMNS, max_features = 50)


# # Submission

# In[ ]:


predictions = model.predict_proba(test[ALL_TRAIN_COLUMNS])[:, 1]


# In[ ]:


submission['target'] = predictions
submission.to_csv('submission.csv')


# In[ ]:


submission.head()

