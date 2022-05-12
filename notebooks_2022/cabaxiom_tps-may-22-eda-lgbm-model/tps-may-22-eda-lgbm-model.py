#!/usr/bin/env python
# coding: utf-8

# # TPS MAY 22

# > For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. The data has various feature interactions that may be important in determining the machine state.

# My full EDA can be found at: https://www.kaggle.com/code/cabaxiom/tps-may-22-in-depth-eda-feature-engineering
# 
# 

# **Still in progress**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


# In[ ]:


train_df = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test_df = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")


# In[ ]:


display(train_df.head())
display(test_df.head())


# In[ ]:


print(train_df.shape)
print(test_df.shape)
train_df.info()


# In[ ]:


print(train_df["id"].nunique())
print(train_df["id"].max())
print(test_df["id"].min())


# ## Target

# In[ ]:


def val_count_df(df, column_name, sort=True):
    value_count = df[column_name].value_counts(sort=sort).reset_index().rename(columns={column_name:"Value Count","index":column_name}).set_index(column_name)
    value_count["Percentage"] = df[column_name].value_counts(sort=sort,normalize=True)*100
    value_count = value_count.reset_index()
    return value_count


# In[ ]:


target_count = val_count_df(train_df, "target")
display(target_count)
target_count.set_index("target").plot.pie(y="Value Count", figsize=(10,7), legend=False, ylabel="");


# ## Features

# In[ ]:


feature_cols = [col for col in train_df.columns if "f_" in col]
dtype_cols = [train_df[i].dtype for i in feature_cols]
dtypes = pd.DataFrame({"features":feature_cols, "dtype":dtype_cols})
float_cols = dtypes.loc[dtypes["dtype"] == "float64", "features"].values.tolist()
int_cols = dtypes.loc[dtypes["dtype"] == "int64", "features"].values.tolist()


# In[ ]:


plt.subplots(figsize=(25,20))
sns.heatmap(train_df.corr(),annot=True, cmap="RdYlGn", fmt = '0.2f', vmin=-1, vmax=1, cbar=False);


# In[ ]:


plt.subplots(figsize=(25,35))
for i, column in enumerate(float_cols):
    plt.subplot(6,3,i+1)
    sns.histplot(data=train_df, x=column, hue="target")
    plt.title(column)


# In[ ]:


plt.subplots(figsize=(25,30))
for i, column in enumerate(int_cols):
    val_count = train_df[column].value_counts()
    ax = plt.subplot(5,3,i+1)
    #sns.barplot(x=val_count.index,y=val_count.values)
    ax.bar(val_count.index, val_count.values)
    ax.set_xticks(val_count.index)
    plt.title(column)


# ## f_27

# In[ ]:


import string
alphabet_upper = list(string.ascii_uppercase)

char_counts = []
for character in alphabet_upper:
    char_counts.append(train_df["f_27"].str.count(character).sum())
char_counts_df = pd.DataFrame({"Character": alphabet_upper, "Character Count": char_counts})
char_counts_df = char_counts_df.loc[char_counts_df["Character Count"] > 0]
print(np.sum(char_counts)) #No other hidden characters

plt.subplots(figsize=(20,7))
sns.barplot(data = char_counts_df, x="Character", y="Character Count", color="blue");
plt.title("Total number of characters in f_27 - train");


# In[ ]:


char_counts_df = char_counts_df.set_index("Character", drop=False)
for i in range(10):
    char_counts_df["character"+str(i+1)] = train_df["f_27"].str[i].value_counts()
char_counts_df = char_counts_df.fillna(0)


f,ax = plt.subplots(figsize=(20,30))
character_cols = [i for i in char_counts_df.columns if "character" in i]
for i, column in enumerate(character_cols):
    ax = plt.subplot(5,2,i+1)
    ax = sns.barplot(data = char_counts_df, x="Character", y=column, color="blue");
    plt.title("Character value counts in position: " +str(i+1));
    ax.set_ylabel("Character Count")


# # Feature Engineering

# In[ ]:


def feature_engineer(df):
    new_df = df.copy()
    
    #Probably bad features?:
    for letter in  ["A","B","C","D","E"]:
        new_df[letter+"_count"] = new_df["f_27"].str.count(letter)
    
    #Good features
    for i in range(10):
        new_df["f_27_"+str(i)] = new_df["f_27"].str[i].apply(lambda x: ord(x) - ord("A"))
    
    #good feature:
    new_df["unique_characters"] = new_df["f_27"].apply(lambda x: len(set(x)))
    
    new_df = new_df.drop(columns=["f_27", "id"])
    return new_df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = feature_engineer(train_df)\ntest_df = feature_engineer(test_df)')


# In[ ]:


train_df["unique_characters"].value_counts()


# # Model

# In[ ]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


# In[ ]:


y = train_df["target"]
X = train_df.drop(columns=["target"])
X_test = test_df
X.head(2)


# In[ ]:


model = LGBMClassifier(n_estimators = 10000, learning_rate = 0.1, random_state=0, min_child_samples=90, num_leaves=150, max_bins=511, n_jobs=-1)


# The variation is roc_auc score across folds is very small - so we save time and use 5-fold validation but only evaluate 2 folds.

# In[ ]:


def k_fold_cv(model,X,y):
    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 0)

    feature_imp, y_pred_list, y_true_list, acc_list, roc_list  = [],[],[],[],[]
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        if fold < 2: # only evaluate 2/5 folds to save time
            print("==fold==", fold)
            X_train = X.loc[train_index]
            X_val = X.loc[val_index]

            y_train = y.loc[train_index]
            y_val = y.loc[val_index]

            model.fit(X_train,y_train)

            y_pred = model.predict_proba(X_val)[:,1]

            y_pred_list = np.append(y_pred_list, y_pred)
            y_true_list = np.append(y_true_list, y_val)

            roc_list.append(roc_auc_score(y_val,y_pred))
            acc_list.append(accuracy_score(y_pred.round(), y_val))
            print("roc auc", roc_auc_score(y_val,y_pred))
            print('Acc', accuracy_score(y_pred.round(), y_val))

            try:
                feature_imp.append(model.feature_importances_)
            except AttributeError: # if model does not have .feature_importances_ attribute
                pass # returns empty list
    return feature_imp, y_pred_list, y_true_list, acc_list, roc_list, X_val, y_val


# In[ ]:


get_ipython().run_cell_magic('time', '', 'feature_imp, y_pred_list, y_true_list, acc_list, roc_list, X_val, y_val = k_fold_cv(model=model,X=X,y=y)')


# In[ ]:


print("Mean accuracy Score:", np.mean(acc_list))
print("Mean ROC AUC Score:", np.mean(roc_list))


# In[ ]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_cm(preds,true,ax=None):
    cm = confusion_matrix(preds.round(), true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)#display_labels 
    disp.plot(ax=ax, colorbar=False, values_format = '.6g')
    plt.grid(False)
    return disp


# In[ ]:


plot_cm(y_pred_list, y_true_list);


# In[ ]:


val_preds = pd.DataFrame({"pred_prob=1":y_pred_list, "y_val":y_true_list})
f,ax = plt.subplots(figsize=(20,20))
plt.subplot(2,1,1)
ax = sns.histplot(data=val_preds, x="pred_prob=1", hue="y_val", multiple="stack", bins = 100)
#Same plot "zoomed in"
plt.subplot(2,1,2)
ax = sns.histplot(data=val_preds, x="pred_prob=1", hue="y_val", multiple="stack", bins = 100)
ax.set_ylim([0,1000]);


# # Feature Importance

# In[ ]:


def fold_feature_importances(model_importances, column_names, model_name, n_folds = 5, ax=None, boxplot=False):
    importances_df = pd.DataFrame({"feature_cols": column_names, "importances_fold_0": model_importances[0]})
    for i in range(1,n_folds):
        importances_df["importances_fold_"+str(i)] = model_importances[i]
    importances_df["importances_fold_median"] = importances_df.drop(columns=["feature_cols"]).median(axis=1)
    importances_df = importances_df.sort_values(by="importances_fold_median", ascending=False)
    if ax == None:
        f, ax = plt.subplots(figsize=(15, 25))
    if boxplot == False:
        ax = sns.barplot(data = importances_df, x = "importances_fold_median", y="feature_cols", color="blue")
        ax.set_xlabel("Median Feature importance across all folds");
    elif boxplot == True:
        importances_df = importances_df.drop(columns="importances_fold_median")
        importances_df = importances_df.set_index("feature_cols").stack().reset_index().rename(columns={0:"feature_importance"})
        ax = sns.boxplot(data = importances_df, y = "feature_cols", x="feature_importance", color="blue", orient="h")
        ax.set_xlabel("Feature importance across all folds");
    plt.title(model_name)
    ax.set_ylabel("Feature Columns")
    return ax


# In[ ]:


f, ax = plt.subplots(figsize=(15, 20))
fold_feature_importances(model_importances = feature_imp, column_names = X_val.columns, model_name = "LGBM", n_folds = 2, ax=ax, boxplot=False);


# # Submission

# In[ ]:


def pred_test():
    pred_list = []
    for seed in range(5):
        model = LGBMClassifier(n_estimators = 10000, learning_rate = 0.1, min_child_samples=90, num_leaves=150, max_bins=511, random_state=seed, n_jobs=-1)
        model.fit(X,y)

        preds = model.predict_proba(X_test)[:,1]
        pred_list.append(preds)
    return pred_list


# In[ ]:


pred_list = pred_test()
pred_df = pd.DataFrame(pred_list).T
pred_df = pred_df.rank()
pred_df["mean"] = pred_df.mean(axis=1)
pred_df


# In[ ]:


sample_sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sample_sub["target"] = pred_df["mean"]
sample_sub


# **Question:**
# 
# If we are predicting probabilities, why do these target scores not fall between 0 and 1?
# 
# **Answer:**
# 
# The evaluation metric is ROC AUC.
# 
# One way of interpreting AUC is: **the probability that the model ranks a random positive example more highly than a random negative example.**
# 
# Our model can be used to output the predicted probability. The absolute values of the predictions do not matter - it does not matter how much higher the random positive example is than the random negative example, we are only interested in the rankings between them.
# 
# In other words the ROC AUC score is scale invariant. **AUC measures how well the predictions are ranked**.
# 
# Therefore we can use the predicted probability ranks rather than the predicted probabilities when calculating the ROC AUC score.
# 
# 
# Example:

# In[ ]:


pred_df = pd.DataFrame(y_pred_list, columns=["pred_prob"])
pred_df["rank"] = pred_df.rank()
display(pred_df.head(10))

print("roc auc using prediction probabilities:", roc_auc_score(y_true_list, pred_df["pred_prob"]))
print("roc auc using predicted probabilities ranks:", roc_auc_score(y_true_list, pred_df["rank"]))


# It may be better to use ranks rather than probabilities as it allows us to combine multiple sets of predictions together without bias towards one set of predictions.

# In[ ]:


sample_sub.to_csv('submission.csv', index = False)

