#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
import shap

from scipy import special

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt


# # Dataset

# In[ ]:


X, y =  sklearn.datasets.make_classification(n_samples=100000, n_features=10, n_informative=4, weights=[0.8], shift=0.0, scale=1.0,)

# 50%/50% split

X_train, y_train = X[:50000,:], y[:50000]
X_test, y_test = X[50000:,:], y[50000:]


# In[ ]:


pd.DataFrame(X).describe()


# # Logistic Regression

# In[ ]:


LRclf = LogisticRegression(random_state=0).fit(X_train, y_train)

fpr, tpr, thresholds = metrics.roc_curve(y_test,LRclf.predict_proba(X_test)[:,1])

print(f'AUC {metrics.auc(fpr, tpr):.2%}')


# abs(coeff) as feature importance

# In[ ]:


plt.bar([f'f_{i}' for i in range(10)],np.abs(LRclf.coef_[0]));
plt.title('Logistic Regression Feature Importance');


# In[ ]:


explainerLR = shap.LinearExplainer(LRclf, masker = shap.maskers.Independent(data = X_test))
shap_valuesLR = explainerLR.shap_values(X_test)
ind = 0


# In[ ]:


ind = 0

class ShapInput(object):
    def __init__(self, expectation, shap_values, features, feat_names):
        self.base_values = expectation
        self.values = shap_values
        self.data = features
        self.feature_names = feat_names

shap_input = ShapInput(explainerLR.expected_value, shap_valuesLR[ind], 
                       X_test[ind,:], feat_names=[f'f_{i}' for i in range(10)])

shap.waterfall_plot(shap_input)


# # Lightgbm - baseline

# In[ ]:


clf = lgb.LGBMClassifier().fit(X_train, y_train)

fpr, tpr, thresholds = metrics.roc_curve(y_test,clf.predict_proba(X_test)[:,1])

print(f'AUC {metrics.auc(fpr, tpr):.2%}')


# In[ ]:


plt.bar([f'f_{i}' for i in range(10)],clf.booster_.feature_importance(importance_type='gain'));
plt.title('Lgbm Feature Importance (Gain)');


# # Lgbm using logistic regression as initial score

# In[ ]:


p_train = LRclf.predict_proba(X_train)[:,1]
log_odds_train = np.log(p_train/(1-p_train))

p_test = LRclf.predict_proba(X_test)[:,1]
log_odds_test = np.log(p_test/(1-p_test))


# In[ ]:


# should init_score be log_odds ?
dtrain = lgb.Dataset(X_train, y_train, init_score = log_odds_train)
dtest = lgb.Dataset(X_test, y_test, init_score = log_odds_test)

clf2 = lgb.train(params={'objective':'binary'}, train_set = dtrain)


# How to predict correctly ?

# In[ ]:


preds = special.expit(log_odds_test + clf2.predict(X_test))

fpr, tpr, thresholds = metrics.roc_curve(y_test,preds)

print(f'AUC {metrics.auc(fpr, tpr):.2%}')


# In[ ]:


preds = clf2.predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test,preds)

print(f'AUC {metrics.auc(fpr, tpr):.2%}')


# # Feature importance + Gain of the baseline

# Seems ok.

# In[ ]:


LR_gain = (np.abs(LRclf.predict_proba(X_train)[:,1]-y_train)).sum() 

plt.bar(['BaseLine']+[f'f_{i}' for i in range(10)], [LR_gain]+list(clf2.feature_importance(importance_type='gain')));
plt.title('Lgbm Feature Importance (Gain)');


# # Global feature importance
# 
# 

# In[ ]:


# okay ? (LR gain as importance*n instances)

plt.bar([f'f_{i}' for i in range(10)], np.abs(LRclf.coef_[0])*50000 + np.array(clf2.feature_importance(importance_type='gain')));
plt.title('Lgbm Feature Importance (Gain)');


# # Individual importance

# lgbm alone

# In[ ]:


import shap

explainer = shap.TreeExplainer(clf2)
shap_values = explainer.shap_values(X_test)

ind = 0


# In[ ]:


class ShapInput(object):
    def __init__(self, expectation, shap_values, features, feat_names):
        self.base_values = expectation
        self.values = shap_values
        self.data = features
        self.feature_names = feat_names

shap_input = ShapInput(explainer.expected_value[0], shap_values[0][ind], 
                       X_test[ind,:], feat_names=[f'f_{i}' for i in range(10)])

shap.waterfall_plot(shap_input)


# # individual importance with logistic regeression

# In[ ]:


# final_pred = -np.log(preds[0]/(1-preds[0])) # seems to match (see above)
# base_pred_lgb = -np.log(preds.mean()/(1-preds.mean())) #not exactly matching
# base_pred_lR = -np.log(p_test.mean()/(1-p_test.mean())) #not exactly matching
# final_pred_LR = -np.log(p_test[0]/(1-p_test[0])) + base_pred_lR

# contrib = final_pred - shap_values[0][ind].sum() - base_pred_lR + final_pred_LR
#  contrib  = final_pred - shap_values[0][ind].sum() - base_pred_lR


# In[ ]:


model_diff = np.log(p_test.mean()) - np.log(p_test[ind])

shap_input = ShapInput(explainerLR.expected_value, 
                       +shap_values[0][ind]+shap_valuesLR[ind], 
                       X_test[ind,:], 
                       feat_names=[f'f_{i}' for i in range(10)])


shap.waterfall_plot(shap_input)


# Not getting the correct output...
