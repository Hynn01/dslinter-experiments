#!/usr/bin/env python
# coding: utf-8

# In this kernel I will do my EDA on the dataset, make some visualizations, try to find any insights and create some new features.
# 
# Join me, it promises to be a thrilling adventure.
# 
# Some tricks being used:
# * [card1 count encoding](#1)
# * [Covariate Shift](#2)
# * [features interaction](#3)
# * [data relaxation](#4)
# 
# New engineered features:
# * [Number of NaNs](#5)
# * [TransactionAmt and it's decimal part](#6)

# In[ ]:


import pandas as pd
import numpy as np
import multiprocessing
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
warnings.simplefilter('ignore')
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# Loading all datasets using multiprocessing. This speads up a process a bit.

# In[ ]:


files = ['../input/test_identity.csv', 
         '../input/test_transaction.csv',
         '../input/train_identity.csv',
         '../input/train_transaction.csv',
         '../input/sample_submission.csv']


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def load_data(file):\n    return pd.read_csv(file)\n\nwith multiprocessing.Pool() as pool:\n    test_id, test_tr, train_id, train_tr, sub = pool.map(load_data, files)')


# In[ ]:


train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

del test_id, test_tr, train_id, train_tr
gc.collect();


# In[ ]:


def plot_numerical(feature):
    """
    Plot some information about a numerical feature for both train and test set.
    Args:
        feature (str): name of the column in DataFrame
    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
    sns.kdeplot(train[feature], ax=axes[0][0], label='Train');
    sns.kdeplot(test[feature], ax=axes[0][0], label='Test');

    sns.kdeplot(train[train['isFraud']==0][feature], ax=axes[0][1], label='isFraud 0')
    sns.kdeplot(train[train['isFraud']==1][feature], ax=axes[0][1], label='isFraud 1')

    test[feature].index += len(train)
    axes[1][0].plot(train[feature], '.', label='Train');
    axes[1][0].plot(test[feature], '.', label='Test');
    axes[1][0].set_xlabel('row index');
    axes[1][0].legend()
    test[feature].index -= len(train)

    axes[1][1].plot(train[train['isFraud']==0][feature], '.', label='isFraud 0');
    axes[1][1].plot(train[train['isFraud']==1][feature], '.', label='isFraud 1');
    axes[1][1].set_xlabel('row index');
    axes[1][1].legend()

    pd.DataFrame({'train': [train[feature].isnull().sum()], 'test': [test[feature].isnull().sum()]}).plot(kind='bar', rot=0, ax=axes[2][0]);
    pd.DataFrame({'isFraud 0': [train[(train['isFraud']==0) & (train[feature].isnull())][feature].shape[0]],
                  'isFraud 1': [train[(train['isFraud']==1) & (train[feature].isnull())][feature].shape[0]]}).plot(kind='bar', rot=0, ax=axes[2][1]);

    fig.suptitle(feature, fontsize=18);
    axes[0][0].set_title('Train/Test KDE distribution');
    axes[0][1].set_title('Target value KDE distribution');
    axes[1][0].set_title('Index versus value: Train/Test distribution');
    axes[1][1].set_title('Index versus value: Target distribution');
    axes[2][0].set_title('Number of NaNs');
    axes[2][1].set_title('Target value distribution among NaN values');
    
# This code is stolen from Chris Deotte. 
def relax_data(df_train, df_test, col):
    cv1 = pd.DataFrame(df_train[col].value_counts().reset_index().rename({col:'train'},axis=1))
    cv2 = pd.DataFrame(df_test[col].value_counts().reset_index().rename({col:'test'},axis=1))
    cv3 = pd.merge(cv1,cv2,on='index',how='outer')
    factor = len(df_test)/len(df_train)
    cv3['train'].fillna(0,inplace=True)
    cv3['test'].fillna(0,inplace=True)
    cv3['remove'] = False
    cv3['remove'] = cv3['remove'] | (cv3['train'] < len(df_train)/10000)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] < cv3['test']/3)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] > 3*cv3['test'])
    cv3['new'] = cv3.apply(lambda x: x['index'] if x['remove']==False else 0,axis=1)
    cv3['new'],_ = cv3['new'].factorize(sort=True)
    cv3.set_index('index',inplace=True)
    cc = cv3['new'].to_dict()
    df_train[col] = df_train[col].map(cc)
    df_test[col] = df_test[col].map(cc)
    return df_train, df_test

def plot_categorical(train: pd.DataFrame, test: pd.DataFrame, feature: str, target: str, values: int=5):
    """
    Plotting distribution for the selected amount of most frequent values between train and test
    along with distibution of target
    Args:
        train (pandas.DataFrame): training set
        test (pandas.DataFrame): testing set
        feature (str): name of the feature
        target (str): name of the target feature
        values (int): amount of most frequest values to look at
    """
    df_train = pd.DataFrame(data={feature: train[feature], 'isTest': 0})
    df_test = pd.DataFrame(data={feature: test[feature], 'isTest': 1})
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df[df[feature].isin(df[feature].value_counts(dropna=False).head(values).index)]
    train = train[train[feature].isin(train[feature].value_counts(dropna=False).head(values).index)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    sns.countplot(data=df.fillna('NaN'), x=feature, hue='isTest', ax=axes[0]);
    sns.countplot(data=train[[feature, target]].fillna('NaN'), x=feature, hue=target, ax=axes[1]);
    axes[0].set_title('Train / Test distibution of {} most frequent values'.format(values));
    axes[1].set_title('Train distibution by {} of {} most frequent values'.format(target, values));
    axes[0].legend(['Train', 'Test']);


# # Transaction DT
# According to the official description 'TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).' I see people in some kernels assume that a start date is a 1 of December 2017, but to be honest the exact start date is not that important. 
# 
# So lets transform TransactionDT into a datetime.

# In[ ]:


startdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
train['TransactionDT'] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
test['TransactionDT'] = test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['isFraud'].plot(ax=axes).set_ylabel('isFraud mean', fontsize=14);
axes.set_title('Mean of isFraud by day', fontsize=16);


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(16, 6))
train['TransactionDT'].dt.floor('d').value_counts().sort_index().plot(ax=axes).set_xlabel('Date', fontsize=14);
test['TransactionDT'].dt.floor('d').value_counts().sort_index().plot(ax=axes).set_ylabel('Number of training examples', fontsize=14);
axes.set_title('Number of training examples by day', fontsize=16);
axes.legend(['Train', 'Test']);


# And now combining both mean of isFraud by day and number of training examples by day into a single plot.

# In[ ]:


fig, ax1 = plt.subplots(figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['isFraud'].plot(ax=ax1, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylabel('isFraud mean', color='blue', fontsize=14)
ax2 = ax1.twinx()
train['TransactionDT'].dt.floor('d').value_counts().sort_index().plot(ax=ax2, color='tab:orange');
ax2.tick_params(axis='y', labelcolor='tab:orange');
ax2.set_ylabel('Number of training examples', color='tab:orange', fontsize=14);
ax2.grid(False)


# <a id="1"></a>
# # card1
# I have decided to start from one of the most important features of this dataset according to LightGBM feature_importance. And **card1** is one of those features.
# 
# What I did is I've created a separate dataset with only this feature in it and also I added one more feature to this new dataset, which is an original feature's frequency (count) encoding. Why I did this? Well, you can reference [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction) competition, where this kind of encoding really boosted a score up. 
# 
# I'll make some visualizations (shoutout to [Chris Deotte](https://www.kaggle.com/cdeotte)) to show you why that works and might work in this case as well.

# In[ ]:


y = train['isFraud']
X = pd.DataFrame()
X['card1'] = train['card1']
X['card1_count'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
clf = DecisionTreeClassifier(max_leaf_nodes=4)
clf.fit(X_train, y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


# So if we train a simple decision tree, using this two features we have an AUC slightly higher that 0.5. Let's see why by plotting this tree as a graph

# In[ ]:


tree_graph = tree.export_graphviz(clf, out_file=None, max_depth = 10,
    impurity = False, feature_names = X.columns, class_names = ['0', '1'],
    rounded = True, filled= True )
graphviz.Source(tree_graph)


# The first split is by the values less than or equal to 10881.5 (black line) and the second one is 8750.0 (red line) and a tree does not use a count feature at all.

# In[ ]:


plt.figure(figsize=(14, 6))
sns.kdeplot(X[y==1]['card1'], label='isFraud 1');
sns.kdeplot(X[y==0]['card1'], label='isFraud 0');
plt.plot([10881.5, 10881.5], [0.0000, 0.0001], sns.xkcd_rgb["black"], lw=2);
plt.plot([8750.0, 8750.0], [0.0000, 0.0001], sns.xkcd_rgb["red"], lw=2);


# But lets take a little step back and train a boosting model on only one original feature card1

# In[ ]:


params = {'objective': 'binary', "boosting_type": "gbdt", "subsample": 1, "bagging_seed": 11, "metric": 'auc', 'random_state': 47}
X_train, X_test, y_train, y_test = train_test_split(X['card1'], y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train.values.reshape(-1, 1), y_train)
print('ROC AUC score', roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1]))


# This is a heatmap with a probability of isFraud=1 for every unique value in the **card1** feature.
# 
# This picture reminds me an opening from a Total Recall movie. 

# In[ ]:


plt.figure(figsize=(12, 6))
x = clf.predict_proba(X['card1'].sort_values().unique().reshape(-1, 1))[:, 1]
x = pd.Series(x, index=X['card1'].sort_values().unique())
sns.heatmap(x.to_frame(), cmap='RdBu_r', center=0.0);
plt.xticks([]);


# Now lets add a second feature - count encoded **card1** values.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


# Holdout score has significantly increased. Lets create another heatmap and see why. 
# 
# There are some darker spots in some intersections of the variable **card1** values and it's count encoded values. This is the reason of the holdout score improvement.
# 
# *The image is pre-rendered since rendering takes some significant amount of time*

# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1696976%2F7153f1242daa586d6849c83242c3fe40%2F35267aee89a7552caf082b6bb0039aa5-full.png?generation=1564585074348507&alt=media)

# In[ ]:


plot_numerical('card1')


# Plotting this variable gives us such information as:
# * distribution in train and test set is almost equal.
# * distribution between target values differs, which make this feature so valuable
# * this feature doesn't have any NaNs

# <a id="2"></a>
# Lets check a Covariate Shift of the feature. This means that we will try to distinguish whether a values correspond to a training set or to a testing set.

# In[ ]:


def covariate_shift(feature):
    df_card1_train = pd.DataFrame(data={feature: train[feature], 'isTest': 0})
    df_card1_test = pd.DataFrame(data={feature: test[feature], 'isTest': 1})

    # Creating a single dataframe
    df = pd.concat([df_card1_train, df_card1_test], ignore_index=True)
    
    # Encoding if feature is categorical
    if str(df[feature].dtype) in ['object', 'category']:
        df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))
    
    # Splitting it to a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df[feature], df['isTest'], test_size=0.33, random_state=47, stratify=df['isTest'])

    clf = lgb.LGBMClassifier(**params, num_boost_round=500)
    clf.fit(X_train.values.reshape(-1, 1), y_train)
    roc_auc =  roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])

    del df, X_train, y_train, X_test, y_test
    gc.collect();
    
    return roc_auc


# In[ ]:


print('Covariate Shift ROC AUC score:', covariate_shift('card1'))


# ROC AUC score is close to 0.5, this means that this feature almost does not have any shift between train and test and is definitely worth keeping it.

# # ProductCD

# In[ ]:


plot_categorical(train, test, 'ProductCD', 'isFraud')


# In[ ]:


print('Covariate shift ROC AUC:', covariate_shift('ProductCD'))


# # card2

# Making a count feature for card2 to perform the same experiment as with card1. First the heatmap for all possible interactions of card2 feature and it's count.

# In[ ]:


y = train['isFraud']
X = pd.DataFrame()
X['card2'] = train['card2']
X['card2_count'] = train['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))

result_df = pd.DataFrame()

for i in X['card2'].sort_values().unique():
    x = pd.DataFrame()
    x['card2'] = [i] * X['card2_count'].nunique()
    x['card2_count'] = X['card2_count'].sort_values().unique()
    
    result_df = pd.concat([result_df, x], axis=0)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

preds = clf.predict_proba(result_df)[:, 1]
preds = preds.reshape(X['card2'].nunique(dropna=False), X['card2_count'].nunique(dropna=False))
preds = pd.DataFrame(preds, index=X['card2'].sort_values().unique(), columns=X['card2_count'].sort_values().unique())

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.heatmap(preds, cmap='RdBu_r', center=0.0);
ax.set_ylabel('card2');
ax.set_xlabel('card2_count');
ax.set_title('card2 / card2_count interaction');


# And a scatter plot with a "decision boundary" of the model. White 'X' marks represents a test set examples.

# In[ ]:


test_X = pd.DataFrame()
test_X['card2'] = test['card2']
test_X['card2_count'] = test['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))


# In[ ]:


plt.figure(figsize=(12, 6))
ax = plt.axes()
sc = plt.scatter(y=result_df['card2'], x=result_df['card2_count'], c=clf.predict_proba(result_df)[:, 1], cmap='RdBu_r');
ax.set_ylabel('card2');
ax.set_xlabel('card2_count');
ax.set_title('card2 / card2_count interaction');
plt.colorbar(sc);
plt.scatter(y=test_X['card2'], x=test_X['card2_count'], marker='x', c='white', alpha=0.5);


# In[ ]:


plot_numerical('card2')


# In[ ]:


print('Covariate shift ROC AUC:', covariate_shift('card2'))


# # card3

# In[ ]:


plot_numerical('card3')


# In[ ]:


print('Covariate shift ROC AUC:', covariate_shift('card3'))


# # card4

# In[ ]:


df_train = pd.DataFrame(data={'card4': train['card4'], 'isTest': 0})
df_test = pd.DataFrame(data={'card4': test['card4'], 'isTest': 1})
df = pd.concat([df_train, df_test], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df.fillna('NaN'), x='card4', hue='isTest', ax=axes[0]);
sns.countplot(data=train[['card4', 'isFraud']].fillna('NaN'), x='card4', hue='isFraud', ax=axes[1]);
axes[0].set_title('Train / Test distibution');
axes[1].set_title('Train distibution by isFraud');
axes[0].legend(['Train', 'Test']);


# In[ ]:


print('Covariate shift ROC AUC:', covariate_shift('card4'))


# # card5

# In[ ]:


plot_numerical('card5')


# In[ ]:


print('Covariate shift ROC AUC:', covariate_shift('card5'))


# # card6

# In[ ]:


df_train = pd.DataFrame(data={'card6': train['card6'], 'isTest': 0})
df_test = pd.DataFrame(data={'card6': test['card6'], 'isTest': 1})
df = pd.concat([df_train, df_test], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df.fillna('NaN'), x='card6', hue='isTest', ax=axes[0]);
sns.countplot(data=train[['card6', 'isFraud']].fillna('NaN'), x='card6', hue='isFraud', ax=axes[1]);
axes[0].set_title('Train / Test distibution');
axes[1].set_title('Train distibution by isFraud');
axes[0].legend(['Train', 'Test']);


# In[ ]:


print('Covariate shift ROC AUC:', covariate_shift('card6'))


# # addr1 

# Another feature with a relatively high importance is **addr1**. According to the name of the feature we can assume that it contains some kind of users address, but in an encoded way. Also this time a feature have some missing values. We are going to fill them with 0.

# In[ ]:


y = train['isFraud']
X = pd.DataFrame()
X['addr1'] = train['addr1']
X['addr1_count'] = train['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))
X['addr1'].fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X['addr1'], y, test_size=0.33, random_state=47)
clf = DecisionTreeClassifier(max_leaf_nodes=4)
clf.fit(X_train.values.reshape(-1, 1), y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1]))


# In[ ]:


tree_graph = tree.export_graphviz(clf, out_file=None, max_depth = 10,
    impurity = False, feature_names = ['addr1'], class_names = ['0', '1'],
    rounded = True, filled= True )
graphviz.Source(tree_graph)


# In[ ]:


plt.figure(figsize=(14, 6))
sns.kdeplot(X[y==1]['addr1'], label='isFraud 1');
sns.kdeplot(X[y==0]['addr1'], label='isFraud 0');
plt.plot([50.0, 50.0], [0.0000, 0.008], sns.xkcd_rgb["black"], lw=2);


# Again training a gradient boosting model with only one feature.

# In[ ]:


params = {'objective': 'binary', "boosting_type": "gbdt", "subsample": 1, "bagging_seed": 11, "metric": 'auc', 'random_state': 47}
X_train, X_test, y_train, y_test = train_test_split(X['addr1'], y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train.values.reshape(-1, 1), y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1]))


# Predictions heatmap.

# In[ ]:


plt.figure(figsize=(12, 6))
x = clf.predict_proba(X['addr1'].sort_values().unique().reshape(-1, 1))[:, 1]
x = pd.Series(x, index=X['addr1'].sort_values().unique())
sns.heatmap(x.to_frame(), cmap='RdBu_r', center=0.0);
plt.xticks([]);


# So far we are doing exactly the same thing that we have been doing for the previous variable.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


# In[ ]:


result_df = pd.DataFrame()

for i in X['addr1'].sort_values().unique():
    x = pd.DataFrame()
    x['addr1'] = [i] * X['addr1_count'].nunique()
    x['addr1_count'] = X['addr1_count'].sort_values().unique()
    
    result_df = pd.concat([result_df, x], axis=0)


# In[ ]:


preds = clf.predict_proba(result_df)[:, 1]
preds = preds.reshape(X['addr1'].nunique(), X['addr1_count'].nunique())
preds = pd.DataFrame(preds, index=X['addr1'].sort_values().unique(), columns=X['addr1_count'].sort_values().unique())

plt.figure(figsize=(12, 6))
sns.heatmap(preds, cmap='RdBu_r', center=0.0);


# In[ ]:


plot_numerical('addr1')


# Distribution is the same, amount of NaN's is the same. Some difference in target value distribution. 
# 
# Next checking Covariate Shift for addr1.

# In[ ]:


print('Covariate shift ROC AUC score:', covariate_shift('addr1'))


# ROC AUC score is close to 0.5
# 
# This feature also does not have any shift between train and test set.

# <a id="3"></a>
# # card1 to addr1 interaction
# 
# Next I am going to create a new feature out of this two features interaction and train on the result.

# In[ ]:


X = pd.DataFrame()
X['addr1'] = train['addr1']
X['card1'] = train['card1']
y = train['isFraud']
X['addr1'].fillna(0, inplace=True)

X['addr1_card1'] = X['addr1'].astype(str) + '_' + X['card1'].astype(str)
X['addr1_card1'] = LabelEncoder().fit_transform(X['addr1_card1'])


# First training a model only using this two features, without their interaction.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X[['addr1', 'card1']], y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


# And now WITH interaction

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X[['addr1', 'card1', 'addr1_card1']], y, test_size=0.33, random_state=47, stratify=y)
clf1 = lgb.LGBMClassifier(**params)
clf1.fit(X_train, y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf1.predict_proba(X_test)[:, 1]))


# In[ ]:


result_df = pd.DataFrame()

for i in tqdm_notebook(X['addr1'].sort_values().unique()):
    x = pd.DataFrame()
    x['addr1'] = [i] * X['card1'].nunique()
    x['card1'] = X['card1'].sort_values().unique()
    
    result_df = pd.concat([result_df, x], axis=0)


# Predictions heatmap of the two features interaction.

# In[ ]:


preds = clf.predict_proba(result_df)[:, 1]
preds = preds.reshape(X['addr1'].nunique(), X['card1'].nunique())
preds = pd.DataFrame(preds, index=X['addr1'].sort_values().unique(), columns=X['card1'].sort_values().unique())
plt.figure(figsize=(12, 6))
sns.heatmap(preds, cmap='RdBu_r', center=0.0);


# Finally adding count features, so all in all we have 5 features

# In[ ]:


X['card1_count'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
X['addr1_count'] = train['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)
print('ROC AUC score:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


# <a id="5"></a>
# # New feature: number of NaN's
# We have plenty of NaN's in this dataset and they can have a significant effect so why don't we use them?
# I am adding a new column to the dateset, which will contain a number of NaN for each row. So if a row (a single training example) contain, say, 10 NaNs, a new feature's value for this row will be 10.

# In[ ]:


train['nulls'] = train.isnull().sum(axis=1)
test['nulls'] = test.isnull().sum(axis=1)
plot_numerical('nulls')


# In[ ]:


print('Covariant shift ROC AUC:', covariate_shift('nulls'))


# We can see that this feature might be useful, but also keep in mind that covatiate shift is almost 0.7, which tells us that the distribution between train and test set has some difference.

# <a id="6"></a>
# # TransactionAmt and it's decimal part
# 
# First let's take a look at TransactionAmt feature and them I will create a new one - it's decimal part, which is a very popular way of creating a new features.

# In[ ]:


plot_numerical('TransactionAmt')


# Moving average for TransactionAmt over time.

# In[ ]:


fig, axes = plt.subplots(1,1,figsize=(16, 6))
axes.set_title('Moving average of TransactionAmt', fontsize=16);
train[['TransactionDT', 'TransactionAmt']].set_index('TransactionDT').rolling(10000).mean().plot(ax=axes);
test[['TransactionDT', 'TransactionAmt']].set_index('TransactionDT').rolling(10000).mean().plot(ax=axes);
axes.legend(['Train', 'Test']);


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['TransactionAmt'].plot(ax=axes).set_ylabel('TransactionAmt mean', fontsize=14);
test.set_index('TransactionDT').resample('D').mean()['TransactionAmt'].plot(ax=axes).set_ylabel('TransactionAmt mean', fontsize=14);
axes.set_title('Mean of TransactionAmt by day', fontsize=16);


# A relationship between mean of TransactionAmt by day and a mean of isFraud by day.

# In[ ]:


fig, ax1 = plt.subplots(figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['isFraud'].plot(ax=ax1, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylabel('isFraud mean by day', color='blue', fontsize=14)
ax2 = ax1.twinx()
train.set_index('TransactionDT').resample('D').mean()['TransactionAmt'].plot(ax=ax2, color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange');
ax2.set_ylabel('TransactionAmt mean by day', color='tab:orange', fontsize=14);
ax2.grid(False)


# Decimal part of transaction amount.

# In[ ]:


train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
plot_numerical('TransactionAmt_decimal')


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['TransactionAmt_decimal'].plot(ax=axes).set_ylabel('TransactionAmt_decimal mean', fontsize=14);
test.set_index('TransactionDT').resample('D').mean()['TransactionAmt_decimal'].plot(ax=axes).set_ylabel('TransactionAmt_decimal mean', fontsize=14);
axes.set_title('Mean of TransactionAmt_decimal by day', fontsize=16);


# A relationship between mean of TransactionAmt_decimal by day and a mean of isFraud by day.

# In[ ]:


fig, ax1 = plt.subplots(figsize=(16, 6))
train.set_index('TransactionDT').resample('D').mean()['isFraud'].plot(ax=ax1, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylabel('isFraud mean by day', color='blue', fontsize=14)
ax2 = ax1.twinx()
train.set_index('TransactionDT').resample('D').mean()['TransactionAmt_decimal'].plot(ax=ax2, color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange');
ax2.set_ylabel('TransactionAmt_decimal mean by day', color='tab:orange', fontsize=14);
ax2.grid(False)


# Lenght of the decimal part of transaction amount. What does it mean? Well, if lenght is 1 or 2 signs it is totaly understandable - it might be cents. But what is wrong with a decimal part's lenght being 3 and more sings? Maybe it is due to a currency convertion?

# In[ ]:


train['TransactionAmt_decimal_lenght'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str.len()
test['TransactionAmt_decimal_lenght'] = test['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str.len()


# In[ ]:


df_train = pd.DataFrame(data={'TransactionAmt_decimal_lenght': train['TransactionAmt_decimal_lenght'], 'isTest': 0})
df_test = pd.DataFrame(data={'TransactionAmt_decimal_lenght': test['TransactionAmt_decimal_lenght'], 'isTest': 1})
df = pd.concat([df_train, df_test], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df.fillna('NaN'), x='TransactionAmt_decimal_lenght', hue='isTest', ax=axes[0]);
sns.countplot(data=train[['TransactionAmt_decimal_lenght', 'isFraud']].fillna('NaN'), x='TransactionAmt_decimal_lenght', hue='isFraud', ax=axes[1]);
axes[0].set_title('Train / Test distibution');
axes[1].set_title('Train distibution by isFraud');
axes[0].legend(['Train', 'Test']);


# Covariate shift for all 3 features.

# In[ ]:


print('Covariant shift ROC AUC:', covariate_shift('TransactionAmt'))


# In[ ]:


print('Covariant shift ROC AUC:', covariate_shift('TransactionAmt_decimal'))


# In[ ]:


print('Covariant shift ROC AUC:', covariate_shift('TransactionAmt_decimal_lenght'))


# # V1

# In[ ]:


plot_numerical('V1')


# In[ ]:


print('Covariate shift:', covariate_shift('V1'))


# # V2

# In[ ]:


plot_numerical('V2')


# In[ ]:


print('Covariate shift:', covariate_shift('V2'))


# # V3

# In[ ]:


plot_numerical('V3')


# In[ ]:


print('Covariate shift:', covariate_shift('V3'))


# # V4

# In[ ]:


plot_numerical('V4')


# In[ ]:


print('Covariate shift:', covariate_shift('V4'))


# # V5

# In[ ]:


plot_numerical('V5')


# In[ ]:


print('Covariate shift:', covariate_shift('V5'))


# # V6

# In[ ]:


plot_numerical('V6')


# In[ ]:


print('Covariate shift:', covariate_shift('V6'))


# # V7

# In[ ]:


plot_numerical('V7')


# In[ ]:


print('Covariate shift:', covariate_shift('V7'))


# # V258

# In[ ]:


plot_numerical('V258')


# In[ ]:


print('Covariate shift:', covariate_shift('V258'))


# <a id="4"></a>
# This is where I want to introduce a little trick to you, called data relaxation. So what is it? In order to understand it take a look at the plot above. See the distibution difference between train and test set at a certain point? Gradient boosting algorithm doesn't know what to do with a data it has never seen so it will not approximate it well. And what we do by relaxing data is we are removing all the values from the train set that appears in it 3 times more often than in a test set and vice versa, also cleaning all the data that appears in train and test set only couple of times.
# 
# ## V258 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'V258')
plot_numerical('V258')


# # V294

# In[ ]:


plot_numerical('V294')


# In[ ]:


print('Covariate shift:', covariate_shift('V294'))


# ## V294 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'V294')
plot_numerical('V294')


# # C1

# In[ ]:


plot_numerical('C1')


# In[ ]:


print('Covariate shift:', covariate_shift('C1'))


# ## C1 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C1')
plot_numerical('C1')


# # C2

# In[ ]:


plot_numerical('C2')


# In[ ]:


print('Covariate shift:', covariate_shift('C2'))


# ## C2 after data relaxation.

# In[ ]:


train, test = relax_data(train, test, 'C2')
plot_numerical('C2')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C2'))


# # C3

# In[ ]:


plot_numerical('C3')


# In[ ]:


print('Covariate shift:', covariate_shift('C3'))


# ## C3 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C3')
plot_numerical('C3')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C3'))


# # C4

# In[ ]:


plot_numerical('C4')


# In[ ]:


print('Covariate shift:', covariate_shift('C4'))


# ## C4 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C4')
plot_numerical('C4')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C4'))


# # C5

# In[ ]:


plot_numerical('C5')


# In[ ]:


print('Covariate shift:', covariate_shift('C5'))


# ## C5 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C5')
plot_numerical('C5')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C5'))


# # C6

# In[ ]:


plot_numerical('C6')


# In[ ]:


print('Covariate shift:', covariate_shift('C6'))


# ## C6 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C6')
plot_numerical('C6')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C6'))


# # C7

# In[ ]:


plot_numerical('C7')


# In[ ]:


print('Covariate shift:', covariate_shift('C7'))


# ## C7 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C7')
plot_numerical('C7')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C7'))


# # C8

# In[ ]:


plot_numerical('C8')


# In[ ]:


print('Covariate shift:', covariate_shift('C8'))


# ## C8 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C8')
plot_numerical('C8')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C8'))


# # C9

# In[ ]:


plot_numerical('C9')


# In[ ]:


print('Covariate shift:', covariate_shift('C9'))


# ## C9 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C9')
plot_numerical('C9')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C9'))


# # C10

# In[ ]:


plot_numerical('C10')


# In[ ]:


print('Covariate shift:', covariate_shift('C10'))


# ## C10 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C10')
plot_numerical('C10')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C10'))


# # C11

# In[ ]:


plot_numerical('C11')


# In[ ]:


print('Covariate shift:', covariate_shift('C11'))


# ## C11 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C11')
plot_numerical('C11')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C11'))


# # C12

# In[ ]:


plot_numerical('C12')


# In[ ]:


print('Covariate shift:', covariate_shift('C12'))


# # C12 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C12')
plot_numerical('C12')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C12'))


# # C13

# In[ ]:


plot_numerical('C13')


# In[ ]:


print('Covariate shift:', covariate_shift('C13'))


# # C13 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C13')
plot_numerical('C13')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C13'))


# # C14

# In[ ]:


plot_numerical('C14')


# In[ ]:


print('Covariate shift:', covariate_shift('C14'))


# ## C14 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'C14')
plot_numerical('C14')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('C14'))


# # D1

# In[ ]:


plot_numerical('D1')


# In[ ]:


print('Covariate shift:', covariate_shift('D1'))


# # D2

# In[ ]:


plot_numerical('D2')


# In[ ]:


print('Covariate shift:', covariate_shift('D2'))


# # D3

# In[ ]:


plot_numerical('D3')


# In[ ]:


print('Covariate shift:', covariate_shift('D3'))


# # D4

# In[ ]:


plot_numerical('D4')


# In[ ]:


print('Covariate shift:', covariate_shift('D4'))


# # D5

# In[ ]:


plot_numerical('D5')


# In[ ]:


print('Covariate shift:', covariate_shift('D5'))


# # D6

# In[ ]:


plot_numerical('D6')


# In[ ]:


print('Covariate shift:', covariate_shift('D6'))


# # D7

# In[ ]:


plot_numerical('D7')


# In[ ]:


print('Covariate shift:', covariate_shift('D7'))


# # D8

# In[ ]:


plot_numerical('D8')


# In[ ]:


print('Covariate shift:', covariate_shift('D8'))


# # D9

# In[ ]:


plot_numerical('D9')


# In[ ]:


print('Covariate shift:', covariate_shift('D9'))


# # D10

# In[ ]:


plot_numerical('D10')


# In[ ]:


print('Covariate shift:', covariate_shift('D10'))


# # D11

# In[ ]:


plot_numerical('D11')


# In[ ]:


print('Covariate shift:', covariate_shift('D11'))


# # D12

# In[ ]:


plot_numerical('D12')


# In[ ]:


print('Covariate shift:', covariate_shift('D12'))


# # D13

# In[ ]:


plot_numerical('D13')


# In[ ]:


print('Covariate shift:', covariate_shift('D13'))


# # D14

# In[ ]:


plot_numerical('D14')


# In[ ]:


print('Covariate shift:', covariate_shift('D14'))


# # D15

# In[ ]:


plot_numerical('D15')


# In[ ]:


print('Covariate shift:', covariate_shift('D15'))


# # id_01

# In[ ]:


plot_numerical('id_01')


# In[ ]:


plot_categorical(train, test, 'id_01', 'isFraud', 10)


# # id_02

# In[ ]:


plot_numerical('id_02')


# # id_03

# In[ ]:


plot_categorical(train, test, 'id_03', 'isFraud', 10)


# # id_04

# In[ ]:


plot_categorical(train, test, 'id_04', 'isFraud', 10)


# # id_05

# In[ ]:


plot_categorical(train, test, 'id_05', 'isFraud', 10)


# # id_06

# In[ ]:


plot_categorical(train, test, 'id_06', 'isFraud', 10)


# # id_07

# In[ ]:


plot_categorical(train, test, 'id_07', 'isFraud', 10)


# # id_08

# In[ ]:


plot_numerical('id_08')


# In[ ]:


plot_categorical(train, test, 'id_08', 'isFraud', 10)


# # id_09

# In[ ]:


plot_categorical(train, test, 'id_09', 'isFraud', 10)


# # id_10

# In[ ]:


plot_numerical('id_10')


# In[ ]:


plot_categorical(train, test, 'id_10', 'isFraud', 10)


# # id_11

# In[ ]:


plot_numerical('id_11')


# # id_12

# In[ ]:


plot_categorical(train, test, 'id_12', 'isFraud', 3)


# # id_13

# In[ ]:


plot_categorical(train, test, 'id_13', 'isFraud', 10)


# # id_14

# In[ ]:


plot_categorical(train, test, 'id_14', 'isFraud', 10)


# # id_15

# In[ ]:


plot_categorical(train, test, 'id_15', 'isFraud', 4)


# # id_16

# In[ ]:


plot_categorical(train, test, 'id_16', 'isFraud', 3)


# # id_17

# In[ ]:


plot_numerical('id_17')


# In[ ]:


plot_categorical(train, test, 'id_17', 'isFraud', 10)


# # id_31

# In[ ]:


plot_categorical(train, test, 'id_31', 'isFraud', 6)


# In[ ]:


print('Covariate shift:', covariate_shift('id_31'))


# ## id_31 after data relaxation

# In[ ]:


train, test = relax_data(train, test, 'id_31')
plot_categorical(train, test, 'id_31', 'isFraud', 6)


# In[ ]:


plot_numerical('id_31')


# In[ ]:


print('Covariate shift after data relaxation:', covariate_shift('id_31'))

