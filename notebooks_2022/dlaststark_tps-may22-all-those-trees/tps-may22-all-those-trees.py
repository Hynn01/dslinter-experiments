#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import gc
import scipy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)


# ## Load source datasets

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
train.set_index('id', inplace=True)
print(f"train: {train.shape}")
train.head()


# In[ ]:


test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
test.set_index('id', inplace=True)
print(f"test: {test.shape}")
test.head()


# ## Feature Engineering

# In[ ]:


for df in [train, test]:
    for i in tqdm(range(10)):
        df[f'f_27_{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
        
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))


# In[ ]:


continuous_feat = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 
                   'f_06', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 
                   'f_24', 'f_25', 'f_26', 'f_28']

train['f_sum']  = train[continuous_feat].sum(axis=1)
train['f_min']  = train[continuous_feat].min(axis=1)
train['f_max']  = train[continuous_feat].max(axis=1)
train['f_std']  = train[continuous_feat].std(axis=1)    
train['f_mad']  = train[continuous_feat].mad(axis=1)
train['f_mean'] = train[continuous_feat].mean(axis=1)
train['f_kurt'] = train[continuous_feat].kurt(axis=1)
train.head()


# In[ ]:


test['f_sum']  = test[continuous_feat].sum(axis=1)
test['f_min']  = test[continuous_feat].min(axis=1)
test['f_max']  = test[continuous_feat].max(axis=1)
test['f_std']  = test[continuous_feat].std(axis=1)    
test['f_mad']  = test[continuous_feat].mad(axis=1)
test['f_mean'] = test[continuous_feat].mean(axis=1)
test['f_kurt'] = test[continuous_feat].kurt(axis=1)
test.head()


# In[ ]:


tfidf = TfidfVectorizer(analyzer='char').fit(train['f_27'].append(test['f_27']))

features = tfidf.transform(train['f_27']).toarray()
features_df = pd.DataFrame(features, 
                           columns=tfidf.get_feature_names(), 
                           index=train.index)

train = pd.merge(train, features_df, 
                 left_index=True, 
                 right_index=True)

train.drop('f_27', axis=1, inplace=True)
train.head()


# In[ ]:


features = tfidf.transform(test['f_27']).toarray()
features_df = pd.DataFrame(features, 
                           columns=tfidf.get_feature_names(), 
                           index=test.index)

test = pd.merge(test, features_df, 
                 left_index=True, 
                 right_index=True)

test.drop('f_27', axis=1, inplace=True)
train.head()


# In[ ]:


cat_cols = ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 
            'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18',
            'f_29', 'f_30', 'f_27_0', 'f_27_1', 'f_27_2', 
            'f_27_3', 'f_27_4', 'f_27_5', 'f_27_6', 'f_27_7', 
            'f_27_8', 'f_27_9', 'unique_characters']

train[cat_cols] = train[cat_cols].astype(int)
test[cat_cols] = test[cat_cols].astype(int)

cat_cols_indices = [test.columns.get_loc(col) for col in cat_cols]
print(cat_cols_indices)


# In[ ]:


features = test.columns.to_list()
print(features)


# ## Helper Function

# In[ ]:


def plot_confusion_matrix(cm, classes):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix', fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.tight_layout()


# ## Model Hyperparameters

# In[ ]:


FOLD = 5
SEEDS = [42]

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',
    'use_label_encoder': False,
    'n_jobs': -1,
    'n_estimators': 7500,
    'max_depth': 12,
    'subsample': 0.75,
    'colsample_bytree': 0.5,
    'learning_rate': 0.117,
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'random_state': 2021
}

cb_params = {
    'loss_function' : 'CrossEntropy',
    'eval_metric' : 'AUC',
    'iterations' : 7500,
    'grow_policy' : 'SymmetricTree',
    'use_best_model' : True,
    'depth' : 12,
    'l2_leaf_reg' : 3.0,
    'random_strength' : 1.0,
    'learning_rate' : 0.12,
    'task_type' : 'GPU',
    'devices' : '0',
    'verbose' : 0,
    'random_state': 2021
}

lgb_params = {
    'objective' : 'binary',
    'metric' : 'auc',
    'max_depth' : 12,
    'n_estimators' : 7500,
    'colsample_bytree' : 0.3,
    'subsample' : 0.75,
    'reg_alpha' : 18,
    'reg_lambda' : 0.17,
    'learning_rate' : 0.115,
    'device' : 'gpu',
    'random_state' : 2021
}


# ## XGBoost

# In[ ]:


counter = 0
oof_score = 0
y_pred_final_xgb = np.zeros((test.shape[0], 1))
y_pred_meta_xgb = np.zeros((train.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train_idx, val_idx) in enumerate(kfold.split(train[features], train['target'])):
        counter += 1

        train_x, train_y = train[features].iloc[train_idx], train['target'].iloc[train_idx]
        val_x, val_y = train[features].iloc[val_idx], train['target'].iloc[val_idx]

        model = XGBClassifier(**xgb_params)

        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], 
                  early_stopping_rounds=100, verbose=1000)
        
        y_pred = model.predict_proba(val_x, iteration_range=(0, model.best_iteration))[:,-1]
        y_pred_meta_xgb[val_idx] += np.array([y_pred]).T
        y_pred_final_xgb += np.array([
            scipy.stats.rankdata(
                model.predict_proba(test, iteration_range=(0, model.best_iteration))[:,-1]
            )]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nSeed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nSeed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_xgb = y_pred_meta_xgb / float(len(SEEDS))
y_pred_final_xgb = y_pred_final_xgb / float(counter)
oof_score /= float(counter)
print("Aggregate OOF Score: {}".format(oof_score))


# ## CatBoost

# In[ ]:


counter = 0
oof_score = 0
y_pred_final_cb = np.zeros((test.shape[0], 1))
y_pred_meta_cb = np.zeros((train.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train_idx, val_idx) in enumerate(kfold.split(train[features], train['target'])):
        counter += 1

        train_x, train_y = train[features].iloc[train_idx], train['target'].iloc[train_idx]
        val_x, val_y = train[features].iloc[val_idx], train['target'].iloc[val_idx]

        model = CatBoostClassifier(**cb_params)

        model.fit(train_x, train_y, eval_set=[(val_x, val_y)],
                  cat_features=cat_cols_indices,
                  early_stopping_rounds=100, verbose=1000)

        y_pred = model.predict_proba(val_x)[:,-1]
        y_pred_meta_cb[val_idx] += np.array([y_pred]).T
        y_pred_final_cb += np.array([
            scipy.stats.rankdata(
                model.predict_proba(test)[:,-1]
            )]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nSeed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nSeed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_cb = y_pred_meta_cb / float(len(SEEDS))
y_pred_final_cb = y_pred_final_cb / float(counter)
oof_score /= float(counter)
print("Aggregate OOF Score: {}".format(oof_score))


# ## LightGBM

# In[ ]:


counter = 0
oof_score = 0
y_pred_final_lgb = np.zeros((test.shape[0], 1))
y_pred_meta_lgb = np.zeros((train.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train_idx, val_idx) in enumerate(kfold.split(train[features], train['target'])):
        counter += 1

        train_x, train_y = train[features].iloc[train_idx], train['target'].iloc[train_idx]
        val_x, val_y = train[features].iloc[val_idx], train['target'].iloc[val_idx]

        model = LGBMClassifier(**lgb_params)
        
        model.fit(train_x, train_y, eval_metric='auc',
                  eval_set=[(train_x, train_y), (val_x, val_y)],
                  categorical_feature=cat_cols_indices, 
                  early_stopping_rounds=100, verbose=500)

        y_pred = model.predict_proba(val_x, num_iteration=model.best_iteration_)[:,-1]
        y_pred_meta_lgb[val_idx] += np.array([y_pred]).T
        y_pred_final_lgb += np.array([
            scipy.stats.rankdata(
                model.predict_proba(test, num_iteration=model.best_iteration_)[:,-1]
            )]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("\nSeed-{} | Fold-{} | OOF Score: {}\n".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nSeed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_lgb = y_pred_meta_lgb / float(len(SEEDS))
y_pred_final_lgb = y_pred_final_lgb / float(counter)
oof_score /= float(counter)
print("Aggregate OOF Score: {}".format(oof_score))


# ## Logistic Regression (Meta Model)

# In[ ]:


Xtrain_meta = np.concatenate((y_pred_meta_cb, y_pred_meta_lgb,
                              y_pred_meta_xgb), axis=1)
Xtest_meta = np.concatenate((y_pred_final_cb, y_pred_final_lgb, 
                             y_pred_final_xgb), axis=1)
Ytrain_meta = train['target'].values

print("Xtrain_meta: {}".format(Xtrain_meta.shape))
print("Ytrain_meta: {}".format(Ytrain_meta.shape))
print("Xtest_meta: {}".format(Xtest_meta.shape))


# In[ ]:


counter = 0
oof_score = 0
y_pred_final_lr = np.zeros((Xtest_meta.shape[0], 1))
y_pred_meta_lr = np.zeros((Xtrain_meta.shape[0], 1))


for sidx, seed in enumerate(SEEDS):
    seed_score = 0
    
    kfold = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)

    for idx, (train_idx, val_idx) in enumerate(kfold.split(Xtrain_meta, Ytrain_meta)):
        counter += 1

        train_x, train_y = Xtrain_meta[train_idx], Ytrain_meta[train_idx]
        val_x, val_y = Xtrain_meta[val_idx], Ytrain_meta[val_idx]

        model = LogisticRegression(
            max_iter=1500, 
            random_state=42
        )

        model.fit(train_x, train_y)

        y_pred = model.predict_proba(val_x)[:,-1]
        y_pred_meta_lr[val_idx] += np.array([y_pred]).T
        y_pred_final_lr += np.array([
            scipy.stats.rankdata(
                model.predict_proba(Xtest_meta)[:,-1]
            )]).T
        
        score = roc_auc_score(val_y, y_pred)
        oof_score += score
        seed_score += score
        print("Seed-{} | Fold-{} | OOF Score: {}".format(seed, idx, score))
        
        del model, y_pred
        del train_x, train_y
        del val_x, val_y
        gc.collect()
    
    print("\nSeed: {} | Aggregate OOF Score: {}\n\n".format(seed, (seed_score / FOLD)))


y_pred_meta_lr = y_pred_meta_lr / float(len(SEEDS))
y_pred_final_lr = y_pred_final_lr / float(counter)
oof_score /= float(counter)
print("Aggregate OOF Score: {}".format(oof_score))


# ## Validate predictions

# In[ ]:


y_pred = (y_pred_meta_xgb > 0.5).astype(int)
print(classification_report(train['target'], y_pred))

cnf_matrix = confusion_matrix(train['target'], y_pred, labels=[0, 1])
np.set_printoptions(precision=2)
plt.figure(figsize=(12, 5))
plot_confusion_matrix(cnf_matrix, classes=[0, 1])


# In[ ]:


y_pred = (y_pred_meta_lgb > 0.5).astype(int)
print(classification_report(train['target'], y_pred))

cnf_matrix = confusion_matrix(train['target'], y_pred, labels=[0, 1])
np.set_printoptions(precision=2)
plt.figure(figsize=(12, 5))
plot_confusion_matrix(cnf_matrix, classes=[0, 1])


# In[ ]:


y_pred = (y_pred_meta_cb > 0.5).astype(int)
print(classification_report(train['target'], y_pred))

cnf_matrix = confusion_matrix(train['target'], y_pred, labels=[0, 1])
np.set_printoptions(precision=2)
plt.figure(figsize=(12, 5))
plot_confusion_matrix(cnf_matrix, classes=[0, 1])


# In[ ]:


y_pred = (y_pred_meta_lr > 0.5).astype(int)
print(classification_report(train['target'], y_pred))

cnf_matrix = confusion_matrix(train['target'], y_pred, labels=[0, 1])
np.set_printoptions(precision=2)
plt.figure(figsize=(12, 5))
plot_confusion_matrix(cnf_matrix, classes=[0, 1])


# ## Create submission file

# In[ ]:


sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sub['target'] = y_pred_final_xgb.ravel()
sub.to_csv("./xgb_submission.csv", index=False)
sub.head()


# In[ ]:


sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sub['target'] = y_pred_final_cb.ravel()
sub.to_csv("./cb_submission.csv", index=False)
sub.head()


# In[ ]:


sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sub['target'] = y_pred_final_lgb.ravel()
sub.to_csv("./lgb_submission.csv", index=False)
sub.head()


# In[ ]:


sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sub['target'] = (y_pred_final_xgb * 0.35) + (y_pred_final_cb * 0.55) + (y_pred_final_lgb * 0.1)
sub.to_csv("./wa_submission.csv", index=False)
sub.head()


# In[ ]:


sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sub['target'] = y_pred_final_lr.ravel()
sub.to_csv("./meta_submission.csv", index=False)
sub.head()


# In[ ]:


## Good Day!!

