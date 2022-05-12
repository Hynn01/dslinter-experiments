#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install lofo-importance')


# In[ ]:


import numpy as np
import pandas as pd


df = pd.read_csv("../input/spaceship-titanic/train.csv")
print(df.shape)
df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


target = "Transported"
features = [col for col in df.columns if col not in {target, "PassengerId", "Name"}]

categoricals = []

for col in features:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        categoricals.append(col)

df.head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
V_name = tfidf.fit_transform(df["Name"].apply(str))


# In[ ]:


import lofo

ds = lofo.Dataset(df, target=target, features=features, auto_group_threshold=0.5, feature_groups={"Name": V_name})


# In[ ]:


lofo_imp = lofo.LOFOImportance(ds, scoring="roc_auc", cv=5, fit_params={"categorical_feature": categoricals})

imp_df = lofo_imp.get_importance()


# In[ ]:


lofo.plot_importance(imp_df)


# In[ ]:




