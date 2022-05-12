#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
        ...

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

import missingno as msno
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler 
from sklearn.metrics import roc_auc_score,roc_curve ,accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
import optuna
import lightgbm as lgbm


# In[ ]:


df = pd.read_csv("../input/coswara-dataset-heavy-cough/train2.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


X = df.drop(['date','status'],axis =1)
y = df['status']


# In[ ]:





# In[ ]:


X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:


print(f"X_train : {X.shape}\nX_test : {X_test.shape}\n\ny_train : {y.shape}\ny_test :  {y_test.shape}\n")


# In[ ]:


X


# In[ ]:


X_test


# In[ ]:


y_test


# In[ ]:


feat = [col for col in X.columns if col not in ("id", "path")]
scores = []


# In[ ]:


X = X[feat]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X


# In[ ]:


def run(trial, data=X,target=y):
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42)
    
    params = {
                'metric': 'auc', 
                'random_state': 22,
                'n_estimators': 4000,
                'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt"]),
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'bagging_fraction': trial.suggest_categorical('bagging_fraction', [0.6, 0.7, 0.80]),
                'feature_fraction': trial.suggest_categorical('feature_fraction', [0.6, 0.7, 0.80]),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0005 , 1),
                'max_depth': trial.suggest_int('max_depth', 2, 12, step=1),
                'num_leaves' : trial.suggest_int('num_leaves', 13, 148, step=5),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 96, step=5),
            }
    
    clf = lgbm.LGBMClassifier(**params)  
    clf.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid), (X_train, y_train)],
#             categorical_feature=cat_indices,
            callbacks=[lgbm.log_evaluation(period=100), 
                       lgbm.early_stopping(stopping_rounds=100)
                      ],
           )
    
    y_proba = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)
    return auc


# In[ ]:


study = optuna.create_study(direction='maximize')
study.optimize(run, n_trials=100)


# In[ ]:


optuna.visualization.plot_parallel_coordinate(study)


# In[ ]:


print(dir(optuna.visualization))


# In[ ]:


optuna.visualization.plot_optimization_history(study)


# In[ ]:


optuna.visualization.plot_param_importances(study)


# In[ ]:





# In[ ]:


study.best_params


# In[ ]:


print("Best parameters:")
print("*"*50)
for param, val in study.best_trial.params.items():
    print(f"{param} :\t {val}")
print("*"*50)
print(f"Best AUC score: {study.best_value}")


# In[ ]:


skf = StratifiedKFold(n_splits=18, shuffle=True, random_state=42)


# In[ ]:


x_test_fit = X_test[feat]
x_test_fit = scaler.fit_transform(x_test_fit)


# In[ ]:


lgbm_params = study.best_params
predictions = []
scores=[]
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    
    X_train, X_valid = X[train_idx], X[val_idx]
    y_train , y_valid = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgbm.LGBMClassifier(**lgbm_params)
    
    model.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid), (X_train, y_train)],
#             categorical_feature=cat_indices,
            callbacks=[lgbm.log_evaluation(period=100), 
                       lgbm.early_stopping(stopping_rounds=500)
                      ],
           )
    
    valid_preds = model.predict_proba(X_valid)[:,1]
    
    score = roc_auc_score(y_valid, valid_preds)
#     scores.append(score)
    print("*"*200)
    print(f"Fold: {fold}, AUC: {score}")
    print("*"*200)
    test_pred = model.predict_proba(x_test_fit)[:,1]
    predictions.append(test_pred)


# In[ ]:





# In[ ]:


preds = np.max(np.column_stack(predictions), axis=1)
sub = X_test[['id','path']]
sub['id'] = X_test['id']
sub['true_label']=y_test
sub['pred']=preds
sub


# In[ ]:


sub[sub['true_label']=='negative']['pred'].describe()


# In[ ]:


sub[sub['true_label']=='positive']['pred'].describe()


# In[ ]:


sub['prediction']= np.select( [sub.pred <0.5, sub.pred >0.5] , [0 ,1] )
sub


# In[ ]:


sub[sub['true_label']=='positive'].count()


# In[ ]:


sub[sub['prediction']==1].count()


# In[ ]:


encoder = LabelEncoder()
scaler = StandardScaler()
y_t = encoder.fit_transform(sub.true_label).astype(float)
# X = scaler.fit_transform(X)


# In[ ]:


sub['true']= y_t
sub


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[ ]:


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true=sub.true, y_pred=sub.prediction)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


y_true=y_t,
y_pred=sub.prediction
fpr1, tpr1, thresh1 = roc_curve(sub.true, sub.prediction, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(sub.prediction, sub.true, pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_t))]
p_fpr, p_tpr, _ = roc_curve(sub.true, random_probs, pos_label=1)


# In[ ]:


auc_score1 = roc_auc_score(sub.true, sub.prediction)
auc_score1


# In[ ]:


auc_score2 = roc_auc_score(sub.prediction, sub.true)
auc_score2


# In[ ]:


import matplotlib.pyplot as plt
# plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Lightgbm')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='reverse"')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
# plt.savefig('ROC',dpi=300)
plt.show();


# In[ ]:


conf_matrix[0][0]


# In[ ]:


TP =conf_matrix[0][0]/455*100  ; TN = conf_matrix[1][1]/455*100 ;FP =conf_matrix[1][0]/455*100  ;FN= conf_matrix[0][1]/455*100 
print("  FN    FP   TP     TN     pre   acc   rec   f1")
#print(FN, FP, TP, FN+FP+TP+TF)
precision = TP / (TP + FP)
accuracy = (TP + TN)/(TP + TN + FP + FN)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print(f"{FN:6.2f}{FP:6.2f} {TP:6.2f}{TN:6.2f}", end="")
print(f"{precision:6.2f}{accuracy:6.2f}{recall:6.2f}{f1_score:6.2f}")


# In[ ]:




