#!/usr/bin/env python
# coding: utf-8

# # Foursquare Feature Importance with LOFO
# 
# ![](https://raw.githubusercontent.com/aerdem4/lofo-importance/master/docs/lofo_logo.png)
# 
# **LOFO** (Leave One Feature Out) Importance calculates the importances of a set of features based on **a metric of choice**, for **a model of choice**, by **iteratively removing each feature from the set**, and **evaluating the performance** of the model, with **a validation scheme of choice**, based on the chosen metric.
# 
# LOFO first evaluates the performance of the model with all the input features included, then iteratively removes one feature at a time, retrains the model, and evaluates its performance on a validation set. The mean and standard deviation (across the folds) of the importance of each feature is then reported.
# 
# While other feature importance methods usually calculate how much a feature is used by the model, LOFO estimates how much a feature can make a difference by itself given that we have the other features. Here are some advantages of LOFO:
# * It generalises well to unseen test sets since it uses a validation scheme.
# * It is model agnostic.
# * It gives negative importance to features that hurt performance upon inclusion.
# * It can group the features. Especially useful for high dimensional features like TFIDF or OHE features. It is also good practice to group very correlated features to avoid misleading results.
# * It can automatically group highly correlated features to avoid underestimating their importance.
# 
# https://github.com/aerdem4/lofo-importance

# In[ ]:


get_ipython().system('pip install lofo-importance')


# ### With this notebook, you can try any kind of feature engineering and see how effective your new features are.

# In[ ]:


import cudf
import cuml

df = cudf.read_csv("../input/foursquare-location-matching/train.csv")
print(df.shape, cuml.__version__)
df.head()


# In[ ]:


import cupy

N_FOLDS = 4
N_CLUSTERS = 2048
print(N_CLUSTERS)

coo_cols = ["latitude", "longitude"]


kmeans_float = cuml.KMeans(n_clusters=N_CLUSTERS, max_iter=100)
kmeans_float.fit(df[coo_cols])

df["fold"] = kmeans_float.labels_ % N_FOLDS
df["fold"].value_counts()


# In[ ]:


from cuml.neighbors import NearestNeighbors


matcher = NearestNeighbors(n_neighbors=5)
matcher.fit(df[coo_cols])


distances, indices = matcher.kneighbors(df[coo_cols])


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

V = dict()

for col in ["address", "url", "phone", "name"]:
    tfidf = TfidfVectorizer(ngram_range=(3, 3), analyzer="char_wb", use_idf=False)
    V[col] = tfidf.fit_transform(df[col].astype(str).fillna(f"no{col}").values_host)
    print(col, V[col].shape)


# In[ ]:


tfidf = TfidfVectorizer(use_idf=False)
V_cat = tfidf.fit_transform(df["categories"].fillna("nocategory").values_host)
V_cat.shape


# In[ ]:


import numpy as np
import pandas as pd

dfs = []


for i in range(indices.shape[1]):
    tmp_df = df[["id"]].copy()
    
    tmp_df["dist"] = distances.values[:, i]
    tmp_df["cat_sim"] = V_cat.multiply(V_cat[indices.values[:, i].get()]).sum(axis=1).A1
    for col in ["name", "address", "url", "phone"]:
        tmp_df[f"{col}_sim"] = V[col].multiply(V[col][indices.values[:, i].get()]).sum(axis=1).A1
        
    tmp_df["match_id"] = df["id"].to_pandas().values[indices.values[:, i].get()]
    
    tmp_df["cat_null"] = df["categories"].isnull()*1.0 + df["categories"].isnull().values[indices.values[:, i].get()]
    for col in ["address", "url", "phone"]:
        tmp_df[f"{col}_null"] = df[col].isnull()*1.0 + df[col].isnull().values[indices.values[:, i].get()]
    
    tmp_df["match_rank"] = i
    
    tmp_df["fold"] = df["fold"].values
    tmp_df["match"] = df["point_of_interest"] == df["point_of_interest"].to_pandas().values[indices.values[:, i].get()]
    
    dfs.append(tmp_df)
    
candidate_df = cudf.concat(dfs)
candidate_df.shape


# In[ ]:


import lofo

sim_features = ["name_sim", "cat_sim", "address_sim", "url_sim", "phone_sim"]
null_features = ["cat_null", "address_null", "url_null", "phone_null"]
features = ["dist", "match_rank"] + sim_features + null_features
target = "match"

ds = lofo.Dataset(candidate_df.to_pandas(), target=target, features=features, auto_group_threshold=0.8)


# In[ ]:


import xgboost as xgb

xgb_param = {'objective': 'reg:logistic',
         'learning_rate': 0.05,
         'max_depth': 4,
         "min_child_weight": 200,
         "colsample_bynode": 0.8,
         "subsample": 0.5,
         "tree_method": 'gpu_hist', "gpu_id": 0,
             "n_estimators": 400
    }

cv = []
for f in range(N_FOLDS):
    cv.append((np.where(candidate_df["fold"].values.get() != f)[0], 
               np.where(candidate_df["fold"].values.get() == f)[0]))



lofo_imp = lofo.LOFOImportance(ds, scoring="roc_auc", cv=cv, model=xgb.XGBClassifier(**xgb_param))


# In[ ]:


imp_df = lofo_imp.get_importance()


# In[ ]:


lofo.plot_importance(imp_df)


# In[ ]:




