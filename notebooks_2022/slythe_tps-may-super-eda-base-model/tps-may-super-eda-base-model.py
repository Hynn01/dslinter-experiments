#!/usr/bin/env python
# coding: utf-8

# # 📝 Intro & Summary  📝
# We are predicting a binary classification problem using an unknown dataset with varied features
# #### As per the competition page:
# *This competition is an opportunity to explore various methods for identifying and exploiting these feature interactions*
# 
# This give us an indication that we have features which have relationships between them and have interacted in some way. \
# The assumption being that, in order to improve our prediction accuracy we will have to understand how these interactions occured and apply feature engineering and selection techniques 

# # 📩 Import Libraries 📩 

# In[ ]:


# Data and visualization
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import Counter
import itertools

# Memory management 
import gc 

#modelling
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


# In[ ]:


# parameters 
sns.set_theme()

CALIBRATION = True
EPOCHS = 5000


# # 💾 Load Data 💾

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv",index_col = 0)
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv",index_col = 0)
sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv",index_col = 0)


# # 🌟 Basic EDA 🌟

# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe(include= "object")


# In[ ]:


train.describe()


# In[ ]:


# check row duplicates 
train[train.duplicated()]


# In[ ]:


# column duplicates
train.apply(lambda x : x.duplicated().sum())


# <h3 style="color: green">🗒️ Notes🗒️</h3>
# 
# * No null values with float, int and object values
# * 1 object column with text 
# * Duplicates values in multiple columns with the Text/object column being the most important to investigate

# # 📉 Deep Dive  📈

# ## Target

# In[ ]:


plt.figure(figsize= (12,5))
sns.countplot(x= train["target"])
plt.show()


# ## Train vs Test: Histograms and Countplots

# In[ ]:


train.info()


# In[ ]:


fig, ax = plt.subplots(4,4, figsize = (25,20) , sharey= True)
ax = ax.ravel()

for i,col in enumerate(train.dtypes[train.dtypes =="float64"].index):
    train[col].plot(ax = ax[i], kind = "hist", bins = 100, color = "b")
    test[col].plot(ax = ax[i], kind = "hist", bins = 100, color = "r")
    ax[i].legend(["train", "test"])
    ax[i].set_title(f"{col}")
fig.suptitle("Histogram of float columns", fontsize=15)
plt.tight_layout()
plt.show()


# <h3 style="color: green">🗒️ Notes🗒️</h3>
# 
# * Normal Gaussian curve for all Float columns -> this is a good sign for ML models 

# In[ ]:


fig, ax = plt.subplots(5,3, figsize = (17,20))
ax = ax.ravel()

for i,col in enumerate(train.dtypes[(train.dtypes =="int64") & (train.dtypes.index != "target") ].index):
    train[col].value_counts().plot(ax = ax[i], kind = "bar",color = "b")
    test[col].value_counts().plot(ax = ax[i], kind = "bar",color = "r")
    ax[i].legend(["train", "test"])
    ax[i].set_title(f"{col}")
fig.suptitle("Histogram of int columns excl. target", fontsize=15)
plt.tight_layout()
plt.show()


# ## Target Analysis: Histograms and Countplots

# In[ ]:


fig, ax = plt.subplots(4,4, figsize = (25,20) , sharey= True)
ax = ax.ravel()

for i,col in enumerate(train.dtypes[train.dtypes =="float64"].index):
    train[train["target"]==0][col].plot(ax = ax[i], kind = "hist", bins = 100, color = "b")
    train[train["target"]==1][col].plot(ax = ax[i], kind = "hist", bins = 100, color = "r")
    ax[i].legend(["0", "1"])
    ax[i].set_title(f"{col}")
fig.suptitle("Histogram of float columns by Target", fontsize=15)
plt.tight_layout()
plt.show()


# In[ ]:


fig, ax = plt.subplots(5,3, figsize = (17,20))
ax = ax.ravel()

for i,col in enumerate(train.dtypes[(train.dtypes =="int64") & (train.dtypes.index != "target") ].index):
    train[train["target"]==0][col].value_counts().plot(ax = ax[i], kind = "bar",color = "b", alpha = 0.8)
    train[train["target"]==1][col].value_counts().plot(ax = ax[i], kind = "bar",color = "r", alpha = 0.8)
    ax[i].legend(["0", "1"])
    ax[i].set_title(f"{col}")
fig.suptitle("Histogram of int columns by Target", fontsize=15)
plt.tight_layout()
plt.show()


# <h3 style="color: green">🗒️ Notes🗒️</h3>
# 
# All plots dont have huge differences per target. However f_29 and f_30 columns look to be classification columns 

# ## Correlation & feature relationships 
# This can be quite details and would require some Feature selection processes if done correctly \
# For now we will look at correlation only and come back to this another time 

# In[ ]:


plt.figure(figsize= (20,12))
sns.heatmap(train.corr(),vmin=-1, vmax= 1, cmap= "Spectral")
plt.show()


# <h3 style="color: green">🗒️ Notes🗒️</h3>
# 
# Not much here w.r.t correlation, we will move on 

# In[ ]:


# plt.figure(figsize = (20,20))
# sns.pairplot(train[train.columns[train.dtypes =="int64"]], hue = "target")

# plt.title("Pairplot of numerical columns against Target")
# plt.show()


# ## Text Analysis: Total Letters

# In[ ]:


# length of text values
display(train["f_27"].str.len().unique())
display(test["f_27"].str.len().unique())


# In[ ]:


#get count of all letters by target
def letter_counter(df):
    count = Counter({})
    for row in df["f_27"]:
        count += Counter(row)
    return count 
train_0 = letter_counter(train[train["target"]==0])
train_1 = letter_counter(train[train["target"]==1])
train_1


# In[ ]:


plt.figure(figsize= (25,5))
plt.bar(train_0.keys(), train_0.values(), color = "r", alpha = 0.7)
plt.bar(train_1.keys(), train_1.values(), color = "b", alpha = 0.7)

plt.legend(["0 Target letters","1 Target letters"])
plt.show()


# In[ ]:


def perc_letters(df):
    let_df = pd.DataFrame.from_dict(df, orient='index')
    let_df = let_df/ let_df.sum()
    return let_df


# In[ ]:


# plot % of letters 
fig, ax = plt.subplots(figsize= (25,7))
perc_letters(train_0).plot(ax=ax, kind = "bar", color = "r", alpha = 0.7)
perc_letters(train_1).plot(ax=ax, kind = "bar", color = "b", alpha = 0.7 )

ax.legend(["% 0 Target letters", "% 1 Target letters"]);
fig.suptitle("Overall % of Letters per Target", fontsize=15)
plt.show()


# <h3 style="color: green">🗒️ Notes🗒️</h3>
# 
# * Target 0 will have a higher % of C and K letters (compared to Target 1) 
# * Target 1 will have a higher % of B letters and F letters (comapred to Target 0)

# ## Text Analysis: By letter

# In[ ]:


fig, ax = plt.subplots(5,4, figsize= (20,20))
ax = ax.ravel()

for i, letter in enumerate(train_0.keys()):
    #create columns-> count of letters in text
    train[letter] = train["f_27"].str.count(letter)
    test[letter] = test["f_27"].str.count(letter)
    
    # plot % of letters in each word, groupby target
    percentage = (train.groupby("target")[letter].sum()/ train.groupby("target")[letter].sum().sum())*100
    percentage.plot(kind= "bar",ax= ax[i],color = ["r","b"])
    ax[i].set_title(f"{letter}")
    
fig.suptitle("% of letters by target", fontsize = 15)
plt.tight_layout()
plt.show()


# <h3 style="color: green">🗒️ Notes🗒️</h3>
# 
# * Text values with large number of 'S' letters have a higher chance of being 1 Target, same with letter 'T' (there are a few others with minor differences R, O, Q , P, B) 

# # 🚀 Base Model 🚀

# In[ ]:


# drop the text column as we already have features created earlier
X = train.drop(["target","f_27"],axis =1)
y= train["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[ ]:


model = lgb.LGBMClassifier(n_jobs = -1, n_estimators = EPOCHS)
model.fit(X_train,y_train, eval_set=[(X_test,y_test)], callbacks = [lgb.early_stopping(30)],eval_metric="auc")


# In[ ]:


val_preds = model.predict_proba(X_test)
y_preds = model.predict_proba(X_train)

print("Intrinsic AUC:", roc_auc_score(y_train, y_preds[:,1]))
print("Validation AUC:", roc_auc_score(y_test, val_preds[:, 1] ))


# In[ ]:


feat_importance = pd.DataFrame(data = model.feature_importances_, index= train.drop(["target","f_27"],axis =1).columns).sort_values(ascending = False, by= [0] )

plt.figure(figsize= (25,10))
sns.barplot(y= feat_importance[0], x= feat_importance.index)
plt.show()


# In[ ]:


feat_importance[feat_importance[0] ==0]


# ## Calibration 
# Taken from my kernels from [last month's TPS ](https://www.kaggle.com/code/slythe/calibrated-xgboost-human-activity-recognition)

# In[ ]:


prob_true, prob_pred = calibration_curve(y_test, val_preds[:,1], n_bins=10)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
plt.plot(prob_pred,prob_true, marker='o', linewidth=1, label='xgb model probabilities')

# reference line
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#plt.axvline(x=0.2, color = "r")
fig.suptitle('Calibration plot')
ax.set_xlabel('Predicted probability (mean)')
ax.set_ylabel('Fraction of positives (%True  in each bin)')
plt.legend()
plt.show()


# In[ ]:


calibrator = CalibratedClassifierCV(model, method = "isotonic", cv='prefit')
calibrator.fit(X_test, y_test)
cal_preds = calibrator.predict_proba(X_test)

print("Validation AUC:" , roc_auc_score(y_test, val_preds[:, 1] ))
print("Calibrated AUC:" , roc_auc_score(y_test, cal_preds[:, 1] ))


# # ❎ Cross validation ❎

# In[ ]:


cv = KFold(n_splits = 5, shuffle = True,random_state=42)


# In[ ]:


preds = []
auc_cv = []
for fold, (idx_train, idx_val) in enumerate(cv.split(X,y)):
    print("\n")
    print("#"*10, f"Fold: {fold}","#"*10)
    X_train , X_test = X.iloc[idx_train] , X.iloc[idx_val]
    y_train , y_test = y[idx_train] , y[idx_val]
    
    model = lgb.LGBMClassifier(n_jobs = -1, n_estimators = EPOCHS)
    model.fit(X_train,y_train, eval_set=[(X_test,y_test)], callbacks = [lgb.early_stopping(30)],eval_metric="auc")
    
    if CALIBRATION:
        calibrator = CalibratedClassifierCV(model, method = "isotonic", cv='prefit')
        calibrator.fit(X_test, y_test)
        auc = roc_auc_score(y_test, calibrator.predict_proba(X_test)[:, 1])
        print("\n Calibration AUC:" , auc)
        preds.append(calibrator.predict_proba(test.drop("f_27",axis =1))[:, 1])
    else:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print("\n Validation AUC:" , auc)
        preds.append(model.predict_proba(test.drop("f_27",axis =1))[:, 1])
        
    del model
    del X_train
    del X_test
    del y_train
    del y_test
    del calibrator
    
    auc_cv.append(auc)
    
print("FINAL AUC: ", np.mean(auc_cv))


# # 📡 Submission 📡

# In[ ]:


sub["target"] = np.array(preds).mean(axis =0)
sub.to_csv("submission.csv")
sub


# In[ ]:


sub.plot(kind= "hist",figsize= (25,8))
plt.show()


# # Bonus Scatterplot
# Scatterplot of all float columns - this causes memory issues so was moved to its own notebook 
# https://www.kaggle.com/slythe/scatterplot-tps-may

# # 🚧🚧Notebook Under Construction 🚧🚧
# 
