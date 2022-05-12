#!/usr/bin/env python
# coding: utf-8

# # TPS_May_2022_EDA_RFC->XGB

# In[ ]:


import numpy as np 
import pandas as pd 

# PLOT
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import OrderedDict

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

from xgboost import XGBClassifier
import xgboost as xgb

# Read file
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')


# # EDA 📊

# In[ ]:


train.head().T


# In[ ]:


def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        tmp = (col,
              df[col].dtype,
              df[col].isnull().sum(),
              df[col].count(),
              df[col].nunique(),
              df[col].unique())
        rows.append(tmp)
    df = pd.DataFrame(rows) 
    df.columns = ['feature','dtype','nan','count','nunique','unique']
    return df


# In[ ]:


check(train)


# In[ ]:


def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color


# In[ ]:


cm = sns.light_palette('green', as_cmap = True)
train.drop('id', axis = 1).describe().T.style.background_gradient(cmap = cm).applymap(color_negative_red)


# ++++++++++++++++++++++++++++++++++

# In[ ]:


test.head().T


# In[ ]:


check(test)


# In[ ]:


test.drop('id', axis = 1).describe().T.style.background_gradient(cmap = cm).applymap(color_negative_red)


# * train count is 900000. no null.
# * test count is 700000. no null. the id of teat is the number of the train contination.
# * 
# * f_0 to f_6 range is  about -5 to 14(float64)
# * f_07 to f_18 range is about 0 to 16 (int64)
# * f_19 to f_26 range is about -12 to 12.5(float64)
# * f_28 range is about -1230 to 1230(float64)
# * f_27 is object. 
# > *  Train's unique counts is 741354. train's most common is BBBBBBCJBC(12). string length is 10.
# > *  Test's unique counts is 598482. test's most common is BBBBBABGCC(9).  string length is 10.
# * f_29 range is 0 to 1(int64)
# * f_30 range  0 to 2(int64)

# +++++++++++++++++++++++++++++++++

# In[ ]:


target_count = train['target'].value_counts()
target_count


# In[ ]:


train['target'].describe()


# In[ ]:


sns.set('talk', 'dark','spring_r')

fig,axs = plt.subplots(ncols = 2, )

sns.countplot(x=train['target'], data=train, ax = axs[0])

labels = ['1','0']
plt.pie(target_count, labels = labels, autopct = '%.0f%%')

plt.show()


# target show almost the same counts.

# +++++++++++++++++++++++++++++++++++
# # Consideration of the 'f_27'👀
# 

# In[ ]:


train['f_27'].value_counts()


# In[ ]:


test['f_27'].value_counts()


# * It is too much value_counts, isn't it?
# * So...I came up with an idea 💡 from past TPS(TabrularPlaygroundSeries_Feb_2022).It's is the DNA's BOS(Block Optical Sequenceing)Exsample: ATATGGCCTT --> A2T2G2C2
# * Then I tried this kind of encode using RLE(Run Length encode).

# In[ ]:


from collections import OrderedDict

def encord(input):
    dict = OrderedDict.fromkeys(input,0)
        
    for ch in input:
        dict[ch] += 1
        
    output = ''
    for k, v in dict.items():
        output = output + k + str(v)
    return output
        


# In[ ]:


f_27_en=[]
for i in range(len(train['f_27'])):
    a = train['f_27'][i]
    st = encord(a)
    f_27_en.append(st)
    
train['f_27_en'] = f_27_en
train['f_27_en'].value_counts() 


# In[ ]:


f_27_ent=[]
for i in range(len(test['f_27'])):
    a = test['f_27'][i]
    st = encord(a)
    f_27_ent.append(st)
    
test['f_27_ent'] = f_27_ent
test['f_27_ent'].value_counts() 


# In[ ]:


label = LabelEncoder()

en_27 = pd.DataFrame(label.fit_transform(train['f_27']))
train['en_27'] = en_27

enc_27 = pd.DataFrame(label.fit_transform(train['f_27_en']))
train['f_27_enc'] = enc_27

enct_27 = pd.DataFrame(label.fit_transform(test['f_27_ent']))
test['f_27_enc'] = enct_27

display(train['en_27'].head(10))
display(train['f_27_enc'].head(10))
display(test['f_27_enc'].head(10))


# train['en_27'] is without RLE. train['f_27_enc'] and test['f_27_enc'] are with RLE.

# In[ ]:


train.head().T


# In[ ]:


test.head().T


# * With this encoding(RLE), the value_counts_length was reduced by about 1/3.
# * The result of the score with and without encoding(RLE) showed better results with encoding(RLE).
# 

# ----------------------------------
# # Decision Tree 🌿
# I have visualized some of the data.

# In[ ]:


clf = DecisionTreeClassifier(max_depth=2)
clf.fit( train[["f_26"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_26"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded =  True,
)

plt.show()


# In[ ]:


lf = DecisionTreeClassifier(max_depth=5)
clf.fit( train[["f_28"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_28"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded = True
)

plt.show()


# In[ ]:


lf = DecisionTreeClassifier(max_depth=3)
clf.fit( train[["f_29"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_29"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded = True
)

plt.show()


# In[ ]:


lf = DecisionTreeClassifier(max_depth=3)
clf.fit( train[["f_30"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_30"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded = True
)

plt.show()


# 
# * 'f_26' is -14.3 to 12.9. mean is 0.36.
# * 'f_28' is -1230 to 1230. mean is -0.38. 
# * 'f_29' is 0 to 1. mean is 3.46.
# * 'f_30' is 0 to 2. mean is 1.00.
# * It seems to be well divided except for f_28.

# ----------------------------------
# # Model⚙

# In[ ]:


#Memory reduce
for col in train.columns:
    if train[col].dtype == "float64":
        train[col]=pd.to_numeric(train[col], downcast="float")
    if train[col].dtype == "int64":
        train[col]=pd.to_numeric(train[col], downcast="integer")
        
for col in test.columns:
    if test[col].dtype == "float64":
        test[col]=pd.to_numeric(test[col], downcast="float")
    if test[col].dtype == "int64":
        test[col]=pd.to_numeric(test[col], downcast="integer")


# In[ ]:


train.info(),test.info()


# In[ ]:


X = train.drop(['id','target','f_27','en_27','f_27_en'], axis = 1).copy()

y = train['target'].copy()
X_test = test.drop(['id','f_27','f_27_ent' ], axis = 1).copy()

del train
del test


# +++++++++++++++++++++++++++++++++++

# In[ ]:


params = {'tree_method':'gpu_hist',
          'n_estimators': 10000,
          'colsample_bytree': 0.5, 
          'subsample': 0.5, 
          'learning_rate': 0.02, 
          'max_depth': 6, 
         }


# In[ ]:


splits = 5
seed = 42
skf = StratifiedKFold(n_splits = splits, shuffle=True, random_state=seed)

preds = []
scores = []

for fold, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

    model = XGBClassifier(**params,
                            booster= 'gbtree',
                            eval_metric = 'auc',
                            gpu_id=0,
                            predictor="gpu_predictor",
                            use_label_encoder=False)
    
    model.fit(X_train,y_train,
              eval_set=[(X_valid,y_valid)],
              early_stopping_rounds=100,
              verbose=False)
    
    pred_valid = model.predict_proba(X_valid)[:,1]
    fpr, tpr, _ = roc_curve(y_valid, pred_valid)
    score = auc(fpr, tpr)
    scores.append(score)

    test_preds = model.predict_proba(X_test)[:,1]
    preds.append(test_preds)
    
    print("fold : ", fold , "score : ", score)


# In[ ]:


print(scores)


# In[ ]:


#seed = 42

#X_train, X_valid, y_train, y_valid = train_test_split(X,y)
    
#model = RandomForestClassifier(random_state = seed)
    
#model.fit(X_train,y_train)
            
#pred_valid = model.predict_proba(X_valid)[:,1]
#fpr, tpr, _ = roc_curve(y_valid, pred_valid)
#score = auc(fpr, tpr) 

#test_preds = model.predict_proba(X_test)[:,1]

#print(score)


# ----------------------------------
# # Submission🎯

# In[ ]:


sub = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')


# In[ ]:



sub['target'] = test_preds
sub.to_csv('submission.csv', index = False)
sub.head()  


# ---------------------------------
# # Summary
# * I have tried several of the same encoding(RLE) and the results are shown below.
# 1. HistGradientBoostingClassifier(with StratifiedKFold) : 0.90072
# 1. RandomForestClassifier (testsize = 0.25): 0.88849
# 1. GradientBoostingClassifier(with StratifiedKFold) : 0.82788
# 1. XGBoost (with StratifiedKFold) : 0.941
# 
# * Changing the parameters or adding the scaler might give a slightly better results. 
# * I will try other models.
# > Thank you for reading!
# > > in progress..
# 
