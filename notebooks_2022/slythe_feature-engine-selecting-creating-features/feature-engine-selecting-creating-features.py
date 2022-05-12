#!/usr/bin/env python
# coding: utf-8

# # ⚡️Summary ⚡️
# 
# In this notebook we we will look at creating features for none timeseries data \
# For this we will use:
# 1. Manual intuition from [exploring our data](https://www.kaggle.com/code/slythe/tps-may-super-eda-base-model)
# 1. [Feature Engine](https://feature-engine.readthedocs.io/en/1.3.x/) -> a python library used for feature engineering and creation 
# 
# **Note** \
# I have commented out some Feature engineering codes due to memory issues or the duration of the process (12hour cap for Kaggle) \
# Feel free to run each section as required with the required functions 

# In[ ]:


get_ipython().system('pip install feature-engine')


# In[ ]:


from feature_engine.selection import RecursiveFeatureElimination
from feature_engine.creation import RelativeFeatures, MathFeatures


# In[ ]:


# Data manipulation and  viz
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from collections import Counter

import gc

# Feature importance with modelling
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[ ]:


# parameters 

EPOCHS = 5000


# # 💾 Load Data 💾
# Data taken from TPS May 2022 competition 

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv",index_col = 0)
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv",index_col = 0)
sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv",index_col = 0)


# In[ ]:


train.head()


# # 🌟 Manual Feature Engineering 🌟
# 
# We have a variety of feature types which we have already investigated in a seperate [EDA notebook ](https://www.kaggle.com/code/slythe/tps-may-super-eda-base-model) \
# We will apply some text feature engineering and other techniques depending on the dataset

# In[ ]:


int_cols = train.dtypes[(train.dtypes =="int64") & (train.dtypes.index != "target") ].index
float_cols = train.dtypes[train.dtypes =="float64" ].index


# In[ ]:


all_letters = ['A', 'B', 'D', 'E', 'P', 'C', 'S', 'G', 'F', 'Q', 'H', 'N', 'K', 'R', 'M', 'T', 'O', 'J', 'I', 'L']

def feature_engineering(df):
    #letter count 
    for letter in all_letters:
        df[letter] = df["f_27"].str.count(letter)
    
    #Unicoding
    for i in range(10):
        df["f_27_"+str(i)] = df["f_27"].str[i].apply(lambda x: ord(x) - ord("A"))
    
    # Get Unique letters
    df["unique_text"] = df["f_27"].apply(lambda x :  ''.join([str(n) for n in list(set(x))]) )
    df["unique_text"] = df["unique_text"].astype("category")
    
    #Merge categorical columns 
    df["f29_f30"] = df[["f_29","f_30"]].apply(lambda x: str( x["f_29"] ) + str(x["f_30"]), axis =1) 
    df["f29_f30"] = df["f29_f30"].astype("category")
    
    # get max and min letter (use 'Counter' to get count of letters and then get max/min from this dictionary )
    df["max_letter"] = df["f_27"].apply(lambda x : Counter(x)).apply(lambda x : max(x, key=x.get))
    df["max_letter"] = df["max_letter"].astype("category")
    df["min_letter"] = df["f_27"].apply(lambda x : Counter(x)).apply(lambda x : min(x, key=x.get))
    df["min_letter"] = df["min_letter"].astype("category")
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)


# # 🚉 Feature Creation 🚉

# ## 1. 🚀 Relative Features w/ Feature Engine 🚀
# **As per the Feature Engine website:** \
# RelativeFeatures() applies basic mathematical operations between a group of variables and one or more reference features. It adds the resulting features to the dataframe.
# 
# In other words, RelativeFeatures() adds, subtracts, multiplies, performs the division, true division, floor division, module or exponentiation of a group of features to / by a group of reference variables. The features resulting from these functions are added to the dataframe.
# 
# **Note**: \
# We can only do this with the float columns and only one or two functions (due to time constraints)

# In[ ]:


functions = [
    'add'
    #, 'mul','sub', 'div' , 'truediv', 'floordiv', 'mod', 'pow'
]
FE = RelativeFeatures(variables  = list(float_cols), reference=list(float_cols)  ,func = functions, drop_original=False)
train= FE.fit_transform(X = train)
test= FE.fit_transform(X = test)

train.to_csv("Relative_feats_train.csv")
test.to_csv("Relative_feats_test.csv")
print([col for col in train.columns])


# ## 2. 🚀 Mathematical Features (Manual) 🚀
# 
# **Note**: 
# * We will do this with certain columns i.e. the float columns (but certain groupings)

# In[ ]:


train[float_cols].describe()


# ### Group Float columns 
# * We can see from the above that certain columns have similar std/ min/ max, we will group them
# * f_00 to f_06 => Group1
# * f_19 to f_26 => Group2
# * f28 looks to be seperate from both groups

# In[ ]:


group1_float =['f_00','f_01','f_02','f_03','f_04','f_05','f_06','f_19']
group2_float = ['f_19','f_20','f_21','f_22','f_23','f_24','f_25','f_26']

def mathematical_feats(df,cols, suffix):
    df[f"sum_{suffix}"] = df[cols].sum(axis = 1)
    df[f"mean_{suffix}"] = df[cols].mean(axis = 1)
    df[f"std_{suffix}"] = df[cols].std(axis = 1)
    df[f"min_{suffix}"] = df[cols].min(axis = 1)
    df[f"max_{suffix}"] = df[cols].max(axis = 1)
    df[f"median_{suffix}"] = df[cols].median(axis = 1)
    df[f"mad_{suffix}"] = df[cols].mad(axis = 1)

    #potentially change periods OR changes axis OR fillna with actuals
    #df[f"diff_{suffix}"] = df[cols].diff(periods=1, axis = 1)
    
    df[f"max-min_{suffix}"] = df[cols].max(axis = 1) - df[cols].min(axis = 1)
    df[f"q01_{suffix}"] = df[cols].quantile(q= 0.1, axis =1)
    df[f"q25_{suffix}"] = df[cols].quantile(q= 0.25, axis =1) 
    df[f"q50_{suffix}"] = df[cols].quantile(q= 0.5, axis =1) 
    df[f"q75_{suffix}"] = df[cols].quantile(q= 0.75, axis =1) 
    df[f"q95_{suffix}"] = df[cols].quantile(q= 0.95, axis =1) 
    df[f"q99_{suffix}"] = df[cols].quantile(q= 0.99, axis =1)
    df[f"kurt_{suffix}"] = df[cols].kurt(axis =1) 
    df[f"skew_{suffix}"] = df[cols].skew( axis =1)
    
    return df

mathematical_feats(train, group1_float, "group1_float")
mathematical_feats(test, group1_float, "group1_float")


# # 👻 Feature Selection w/ LightGBM 👻
# 
# There are multiple ways to do feature selection, my favourite being the automated [Powershap](https://github.com/predict-idlab/powershap) library \
# You can see Powershap in action in this [notebook](https://www.kaggle.com/code/slythe/powershap-feature-selection-recursive) (however you may come across memory issues depending on the size of your dataset
# 
# As this dataset is quite large for Kaggle I will do my own single run of LightGBM and check the feature importances \
# I also have only used one dataset(Mathemematical feats) --> change this as needed

# In[ ]:


# drop the text column and target
X = train.drop(['target','f_27'],axis =1)
y= train["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[ ]:


model = lgb.LGBMClassifier(n_jobs = -1, n_estimators = EPOCHS)
model.fit(X_train,y_train, eval_set=[(X_test,y_test)], callbacks = [lgb.early_stopping(30)],eval_metric="auc" , 
         )
val_preds = model.predict_proba(X_test)
y_preds = model.predict_proba(X_train)

print("Intrinsic AUC:", roc_auc_score(y_train, y_preds[:,1]))
print("Validation AUC:", roc_auc_score(y_test, val_preds[:, 1] ))

del val_preds
del y_preds
gc.collect()


# In[ ]:


feat_importance = pd.DataFrame(data = model.feature_importances_, index= train.drop(["target","f_27"],axis =1).columns).sort_values(ascending = False, by= [0] )

plt.figure(figsize= (25,10))
sns.barplot(y= feat_importance[0], x= feat_importance.index)
plt.xticks(rotation = 90) 
plt.show()


# In[ ]:


# Features with zero importance
print([col for col in feat_importance[feat_importance[0] ==0].index])

del feat_importance
del train


# # 📝 Submission 📝

# In[ ]:


test_preds = model.predict_proba(test.drop("f_27",axis =1))


# In[ ]:


sub["target"] = test_preds[:,1]
sub.to_csv("submission.csv")

sub.plot(kind= "hist",figsize= (25,8))
plt.show()


# In[ ]:




