#!/usr/bin/env python
# coding: utf-8

# # LGBM Ranker using original csv data  
# Reference Notebook: [Radek's LGBMRanker starter-pack](https://www.kaggle.com/code/marcogorelli/radek-s-lgbmranker-starter-pack)  
# 
# - I use original csv data in this notebook so slightly different original notebook.  
# - My implement is bad so insufficient memory in Kaggle Notebook. 

# ## Evaluation function  
# Evaluation function used in this competition is MAP@12. This function is defined as following.  
# 
# $$MAP@12 = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{min(n, 12)} P(k) \times rel(k)$$  
# 
# In other words, it can be calculated by dividing the number of correct answers up to that point by index+1.
# 
# For example, actual data is [2, 4, 1], and prediction is [2, 3, 1], "k =3 to explain easily.  
# 
# First, initialize score = 0, index=0, then actual[0] = 2, prediction[0] = 2, so actual[0] == prediction[0].  In this case, score += 1 / (index+1)(score = 1). 
# 
# Afterwards index+=1, then actual[1] != prediction[1], so leave score as it is.
# 
# Similarly index += 1, then actual[2] == prediction[2], so score += 2 / (index+1) (score = $\frac{5}{3}$).  
# 
# Lastly, divide score by "k". In short, $\frac{5}{3} \times \frac{1}{3} = \frac{5}{9}$. This is MAP@3, evaluation function in this competition is MAP@12, so set "k"=12.    
# 
# Let's implement MAP@12.

# In[ ]:


import warnings
import gc

import numpy as np

warnings.filterwarnings("ignore")


def apk(t: list, y: list, k: int = 12, default: float = 0.0):
    """
    Calculate evaluation function(AP@12)

    Parameters
    ----------
    t : list
        Actual label vector.
    y : list
        Prediction vector.
    k : int
        How number of making candidates.
    default : float
        If actual data doesn't have label, return default as score.

    Return
    ------
    score : float
        Average precision score.
    """

    # Extract prediction number of k data
    if len(y) > k:
        y = y[:k]

    score = 0.0

    # Number of predicts in actual vector
    num_hits = 0

    for i, pred in enumerate(y):
        if pred in t and pred not in y[:i]:  # predict in actual and don't duplicate
            num_hits += 1
            score += num_hits / (i+1)

    # If actual data doesn't have label
    if not t:
        return default

    # average precision
    return score / min(len(t), k)


def mapk(t: np.ndarray, y: np.ndarray, k: int = 12, default: float = 0.0):
    """
    Calculate evaluation function(MAP@12)

    Parameters
    ----------
    t : list
        Actual label vector.
    y : list
        Prediction vector.
    k : int
        How number of making candidates.
    default : float
        If actual data doesn't have label, return default as score.

    Return
    ------
    score : float
        Average precision score.
    """
    return np.mean([apk(tt, yy, k, default) for tt, yy in zip(t, y)])


# In[ ]:


# Test this function by using an example
pred = [2, 3, 1]
label = [2, 4, 1]

print(apk(label, pred))
print(5 / 9)


# ## Helper Functions  
# 
# ### Functions to reduce memory  
# This method is introduced the discusssion. [Memory Trick - Reduce Memory 8x or 16x!](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635)  

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin


def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)


# Transform from hexadecimal to decimal number
# from str(64bytes) to int(64bits)
def hex_id_to_int(string):
    return int(string[-16:], 16)


# Transform from int(64bits) to int(32bits)
def article_id_str_to_int(series):
    return series.astype("int32")

# Transform(from int 32bits to strings) and pre zero padding
def article_id_int_to_str(series):
    return "0" + series.astype("str")


# #### Check the processing content of each function  

# In[ ]:


import os
import pandas as pd

# Load Data
DIR = "../input/h-and-m-personalized-fashion-recommendations"
transactions = pd.read_csv(os.path.join(DIR, "transactions_train.csv"))
articles = pd.read_csv(os.path.join(DIR, "articles.csv"))
customers = pd.read_csv(os.path.join(DIR, "customers.csv"))

# Use "customer_hex_id_to_int"
display(customers["customer_id"])  # Original data
display(customer_hex_id_to_int(customers["customer_id"]))  # Applied data


# In[ ]:


# Use "article_id_str_to_int" and "atticle_id_int_to_str"
display(transactions["article_id"])   # Original data
display(article_id_str_to_int(transactions["article_id"]))  # Transform from int64 to int 32
display(article_id_int_to_str(transactions["article_id"]))  # Transform from int to str and zero padding


# In[ ]:


# Implement feature transformer(as StandardScaler, LabelEncoder etc...)
class Categorize(BaseEstimator, TransformerMixin):
    def __init__(self, min_examples: int = 0):
        self.min_examples = min_examples
        self.categories = []

    def fit(self, X):
        for i in range(X.shape[1]):  # X.shape[1] is num of the features
            vc = X.iloc[:, i].values_counts()  # count num of the each values in the column
            self.categories.append(vc[vc > self.min_examples].index.tolist())

        return self

    def transform(self, X):
        data = {X.columns[i]: pd.Categorical(X.iloc[:, i], categories=self.categories[i]).codes for i in range(X.shape[1])}
        return pd.DataFrame(data=data)


# In[ ]:


def calculate_apk(list_of_preds, list_of_labels):
    """
    To calculate MAP@12 for all data.

    Prameters
    ---------
    list_of_preds : Union(np.ndarray, list)
        Predict list for all customers.
    list_of_labels : Union(np.ndarray, list)
        Label(list of articles that the customer bought) list for all customers.

    Return
    ------
    score : float
        Average MAP@12 score for all customers.
    """
    map_scores = []
    # Calculate MAP@12 for each customer prediction
    for preds, labels in zip(list_of_preds, list_of_labels):
        map_scores.append(map(preds, labels, k=12))

    # Average score
    return np.mean(map_scores)


# ## Getting candidates  
# Day of the start week is Wednesday in original notebook, but it is Monday in this notebook.  

# In[ ]:


# Create new columns
transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
transactions["year"] = transactions["t_dat"].dt.year

shape_2018 = transactions.query("year == 2018").shape[0]
shape_2019 = transactions.query("year == 2019").shape[0] + shape_2018
shape_2020 = transactions.query("year == 2020").shape[0] + shape_2019

transactions["week"] = transactions["t_dat"].dt.week
transactions.loc[shape_2018:shape_2019-1, "week"] = transactions.loc[shape_2018:shape_2019-1, "week"] + transactions.query("year == 2018")["week"].max() - 1
transactions.loc[shape_2019:shape_2020-1, "week"] = transactions.loc[shape_2019:shape_2020-1, "week"] + transactions.query("year == 2019")["week"].max() - 1
transactions["week"] = transactions["week"] - 37

transactions = transactions.query("t_dat >= '2020-07-15'")
display(transactions)
display(transactions.groupby(["week"])["t_dat"].agg(["min", "max"]))


# In[ ]:


# List of weeks when customer bought an article.
# Transform "custoemr_id" hexadecimal to decimal number
transactions["customer_id"] = customer_hex_id_to_int(transactions["customer_id"])

# To merge customers and transactions so transform "customer_id" as same
customers["customer_id"] = customer_hex_id_to_int(customers["customer_id"])

# Results is slightly different due to day of the start week is Monday
c2weeks = transactions.groupby(["customer_id"])["week"].unique()
display(c2weeks)


# In[ ]:


# Test week is the week after the last data of the train data
test_week = transactions["week"].max() + 1

c2weeks2shifted_weeks = dict()

for c_id, weeks in c2weeks.items():  # customer_id and list of weeks when the customer bought an article.
    c2weeks2shifted_weeks[c_id] = dict()
    for i in range(weeks.shape[0]-1):
        # key: the week when customer bought an article value: the week when customer bought an article next.
        c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i+1]
    c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week

# Display an example
print(c2weeks2shifted_weeks[28847241659200])


# In[ ]:


# Copy transactions data and week is replaced the customer bought and article next
candidates_last_purchase = transactions.copy()

weeks = []
# Create list to replace with next week by using dict
for i, (c_id, week) in enumerate(zip(transactions["customer_id"], transactions["week"])):
    weeks.append(c2weeks2shifted_weeks[c_id][week])

candidates_last_purchase["week"] = weeks

# Display replace data
display(candidates_last_purchase[candidates_last_purchase['customer_id']==272412481300040])

# Display original transactions data to compare
display(transactions[transactions['customer_id']==272412481300040])


# ## Bestsellers candidates  
# 

# In[ ]:


# Calculate mean price for a week
mean_price = transactions.groupby(["week", "article_id"])["price"].mean()

display(mean_price)


# In[ ]:


sales = transactions     .groupby(["week"])["article_id"].value_counts()     .groupby(["week"]).rank(method="dense", ascending=False)     .groupby(["week"]).head(12).rename("bestseller_rank").astype("int8")

display(sales)


# In[ ]:


# Divide their processings to understand what did in this code 
# line 1

# Count number of sold articles in each week
display(transactions.groupby(["week"])["article_id"].value_counts())


# In[ ]:


# line 2
# Ranking by number of sold per week and descending sort
display(transactions.groupby(["week"])["article_id"].value_counts()         .groupby(["week"]).rank(method="dense", ascending=False))


# In[ ]:


# line 3
# Extract top 12 articles each week and transform int type
display(transactions         .groupby(["week"])["article_id"].value_counts()         .groupby(["week"]).rank(method="dense", ascending=False)         .groupby(["week"]).head(12).rename("bestseller_rank").astype("int8"))


# In[ ]:


# Merge data
bestsellers_previous_week = pd.merge(sales, mean_price, on=["week", "article_id"]).reset_index()
bestsellers_previous_week["week"] += 1

display(bestsellers_previous_week.query("week == 95"))


# In[ ]:


# This code is equal following
# transactions.drop_duplicates(["week", "customer_id"])
unique_transactions = transactions     .groupby(["week", "customer_id"])     .head(1)     .drop(columns=["article_id", "price"])     .copy()

display(unique_transactions)


# In[ ]:


# Merge data
candidates_bestsellers = pd.merge(unique_transactions, bestsellers_previous_week, on="week")

display(candidates_bestsellers)


# In[ ]:


test_set_transactions = unique_transactions.drop_duplicates("customer_id").reset_index(drop=True)
test_set_transactions["week"] = test_week

display(test_set_transactions)


# In[ ]:


# Transactions data last bought before test term and best sellers
candidates_bestsellers_test_week = pd.merge(test_set_transactions,
                                            bestsellers_previous_week,
                                            on="week")

display(candidates_bestsellers_test_week)


# In[ ]:


# Merge train term data and test term data
candidates_bestsellers = pd.concat([candidates_bestsellers, candidates_bestsellers_test_week])

# drop bestseller_rank column
candidates_bestsellers = candidates_bestsellers.drop("bestseller_rank", axis=1)
display(candidates_bestsellers)


# ## Combining transactions and candidates / negative examples  

# In[ ]:


# Add new column
transactions["purchased"] = 1

data = pd.concat([transactions, candidates_last_purchase, candidates_bestsellers])
data["purchased"] = data["purchased"].fillna(0)

display(data)


# In[ ]:


# Drop duplicate data
data = data.drop_duplicates(["customer_id", "article_id", "week"])

# Rate of purchased articles data
print(data["purchased"].mean())


# ## Add bestseller information  

# In[ ]:


# Add "bestseller_rank"
data = pd.merge(data, bestsellers_previous_week[["week", "article_id", "bestseller_rank"]],
                on=["week", "article_id"], how="left")
display(data)


# In[ ]:


# Drop first week data
data = data[data["week"] != data["week"].min()]

# Fill data not in bestseller articles
data["bestseller_rank"] = data["bestseller_rank"].fillna(999)

display(data)


# In[ ]:


# Merge data articles and customers
data = pd.merge(data, articles, on="article_id", how="left")
data = pd.merge(data, customers, on="customer_id", how="left")

del transactions, unique_transactions, articles, customers
gc.collect()

display(data)


# In[ ]:


# to store memory
columns_to_use = ['article_id', 'product_type_no', 'graphical_appearance_no',
                  'colour_group_code', 'perceived_colour_value_id',
                  'perceived_colour_master_id', 'department_no', 'index_code',
                  'index_group_no', 'section_no', 'garment_group_no', 'FN', 'Active',
                  'club_member_status', 'fashion_news_frequency', 'age', 'postal_code',
                  'bestseller_rank']
data = data[["customer_id", "week", "sales_channel_id", "purchased"]+columns_to_use]
data["postal_code"] = customer_hex_id_to_int(data["postal_code"])

gc.collect()

# Sort data and reset index
data = data.sort_values(["week", "customer_id"]).reset_index(drop=True)
display(data)


# In[ ]:


# Divide train data and test data
train = data[data["week"] != test_week]
test = data[data["week"] == test_week].drop_duplicates(["customer_id", "article_id", "sales_channel_id"]).copy()

display(train)
display(test)


# In[ ]:


# Number of articles that each customer bought
train_baskets = train.groupby(["week", "customer_id"])["article_id"].count().values
display(train.groupby(["week", "customer_id"])["article_id"].count())


# In[ ]:


# Define use columns to train model
columns_to_use = ['article_id', 'product_type_no', 'graphical_appearance_no',
                  'colour_group_code', 'perceived_colour_value_id',
                  'perceived_colour_master_id', 'department_no', 'index_code',
                  'index_group_no', 'section_no', 'garment_group_no', 'FN', 'Active',
                  'club_member_status', 'fashion_news_frequency', 'age', 'postal_code',
                  'bestseller_rank']
display(train[columns_to_use].head())
display(test[columns_to_use].head())


# In[ ]:


train_X = train[columns_to_use]
train_y = train["purchased"]

test_X = test[columns_to_use]

display(train_X)
display(train_y)


# In[ ]:


display(test[test["customer_id"] == 28847241659200][["article_id", "bestseller_rank"]])
display(test[test["customer_id"] == 41318098387474][["article_id", "bestseller_rank"]])


# ## Train model  

# In[ ]:


# Import model class
from lightgbm.sklearn import LGBMRanker


# In[ ]:


cat_columns = ["club_member_status",
               "fashion_news_frequency", "postal_code"]

display(train_X[cat_columns].describe())

for column in cat_columns:
    print(column)
    print(train_X[column].unique())


# In[ ]:


# Preprocessing categorical values
from sklearn.preprocessing import LabelEncoder

# Replace "None" and "NONE" to np.nan
train_X["fashion_news_frequency"] = train_X["fashion_news_frequency"].replace({"None": np.nan,
                                                                               "NONE": np.nan})
test_X["fashion_news_frequency"] = test_X["fashion_news_frequency"].replace({"None": np.nan,
                                                                             "NONE": np.nan})
# Fill nan
train_X["club_member_status"] = train_X["club_member_status"].fillna("INACTIVE")
train_X["fashion_news_frequency"] = train_X["fashion_news_frequency"].fillna("NOT RECEIVE")

test_X["club_member_status"] = test_X["club_member_status"].fillna("INACTIVE")
test_X["fashion_news_frequency"] = test_X["fashion_news_frequency"].fillna("NOT RECEIVE")

# Label encode
cat_columns = ["index_code", "club_member_status", "fashion_news_frequency"]

for column in cat_columns:
    print(column)
    le = LabelEncoder()
    train_X[column] = le.fit_transform(train_X[column].values.reshape(-1))
    test_X[column] = le.transform(test_X[column].values.reshape(-1))


# In[ ]:


# Class instance
ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    n_estimators=1,
    importance_type="gain",
    verbose=10
)

# Fit model
ranker = ranker.fit(train_X, train_y, group=train_baskets)


# In[ ]:


# Display feature importances ratio
for i in ranker.feature_importances_.argsort()[::-1]:
    print(columns_to_use[i], ranker.feature_importances_[i]/ranker.feature_importances_.sum())


# ## Calculate predicsions  

# In[ ]:


# Create predict
test["preds"] = ranker.predict(test_X)

# Create dict. key: customer_id values: prediction
c_id2predicted_article_ids = test     .sort_values(["customer_id", "preds"], ascending=False)     .groupby(["customer_id"])["article_id"].apply(list).to_dict()


# In[ ]:


# How process in Create c_id2predicted_article_ids
display(test.sort_values(["customer_id", "preds"], ascending=False))


# In[ ]:


display(test.sort_values(["customer_id", "preds"], ascending=False).groupby(["customer_id"])["article_id"].apply(list))


# In[ ]:


# Articles sold top 12
bestsellers_last_week =     bestsellers_previous_week[bestsellers_previous_week["week"] == bestsellers_previous_week["week"].max()]["article_id"].tolist()
print(bestsellers_last_week)


# ## Create submission  

# In[ ]:


submission = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))
display(submission)


# In[ ]:


preds = []

# Extract prediction using key(customer_id) and append list
for c_id in customer_hex_id_to_int(submission["customer_id"]):
    pred = c_id2predicted_article_ids.get(c_id, [])
    pred = pred + bestsellers_last_week

    # Extract top 12 articles
    # If custoemr id is not in dict keys, pred is bestsellers_last_week
    preds.append(pred[:12])

for i in range(5):
    print(preds[i])


# In[ ]:


# Pre zero padding to match submission format
# Create preds is same as following
"""
preds = []

for ps in preds[:5]:
    tmp = []
    for p in ps:
        tmp.append("0" + str(p))

    preds.append(" ".join(tmp))
"""

preds = [" ".join(["0" + str(p) for p in ps]) for ps in preds]
submission["prediction"] = preds

submission.head()


# In[ ]:




