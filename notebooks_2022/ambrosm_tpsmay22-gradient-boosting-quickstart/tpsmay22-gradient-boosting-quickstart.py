#!/usr/bin/env python
# coding: utf-8

# # Gradient-Boosting Quickstart for TPSMAY22
# 
# This notebook shows how to train a gradient booster with minimal feature engineering. For the corresponding EDA, see the [separate EDA notebook](https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense).
# 
# Release notes:
# - V1: XGB
# - V2: LightGBM, one more feature

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from cycler import cycler
from IPython.display import display
import datetime
import scipy.stats

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibrationDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])


# # Feature engineering
# 
# We read the data and apply minimal feature engineering: We only split the `f_27` string into ten separate features as described in the [EDA](https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense), and we count the unique characters in the string.

# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
for df in [train, test]:
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    # unique_characters feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
features = [f for f in test.columns if f != 'id' and f != 'f_27']
test[features].head(2)


# # Cross-validation
# 
# For cross-validation, we use a simple KFold with five splits. It turned out that the scores of the five splits are very similar so that I usually run only the first split. This one split is good enough to evaluate the model.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Cross-validation of the classifier\n\ndef my_booster(random_state=1):\n#     return HistGradientBoostingClassifier(learning_rate=0.4, max_leaf_nodes=150,\n#                                           max_iter=1000, min_samples_leaf=4000,\n#                                           l2_regularization=1,\n#                                           validation_fraction=0.05,\n#                                           max_bins=255,\n#                                           random_state=random_state, verbose=1)\n#     return XGBClassifier(n_estimators=400, n_jobs=-1,\n#                          eval_metric=[\'logloss\'],\n#                          #max_depth=10,\n#                          colsample_bytree=0.8,\n#                          #gamma=1.4,\n#                          reg_alpha=6, reg_lambda=1.5,\n#                          tree_method=\'hist\',\n#                          #max_bin=511,\n#                          learning_rate=0.4,\n#                          verbosity=1,\n#                          use_label_encoder=False, random_state=random_state)\n    return LGBMClassifier(n_estimators=5000, min_child_samples=80,\n                          max_bins=511, random_state=random_state)\n      \nprint(f"{len(features)} features")\nscore_list = []\nkf = KFold(n_splits=5)\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train)):\n    X_tr = train.iloc[idx_tr][features]\n    X_va = train.iloc[idx_va][features]\n    y_tr = train.iloc[idx_tr].target\n    y_va = train.iloc[idx_va].target\n    \n    model = my_booster()\n\n    if True or type(model) != XGBClassifier:\n        model.fit(X_tr.values, y_tr)\n    else:\n        model.fit(X_tr.values, y_tr, eval_set = [(X_va.values, y_va)], \n                  early_stopping_rounds=30, verbose=10)\n    y_va_pred = model.predict_proba(X_va.values)[:,1]\n    score = roc_auc_score(y_va, y_va_pred)\n    try:\n        print(f"Fold {fold}: n_iter ={model.n_iter_:5d}    AUC = {score:.3f}")\n    except AttributeError:\n        print(f"Fold {fold}:                  AUC = {score:.3f}")\n    score_list.append(score)\n    break # we only need the first fold\n    \nprint(f"OOF AUC:                       {np.mean(score_list):.3f}")')


# # Three diagrams for model evaluation
# 
# We plot the ROC curve just because it looks nice. The area under the red curve is the score of our model.
# 

# In[ ]:


# Plot the roc curve for the last fold
def plot_roc_curve(y_va, y_va_pred):
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_va, y_va_pred)
    plt.plot(fpr, tpr, color='r', lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.gca().set_aspect('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.show()

plot_roc_curve(y_va, y_va_pred)


# Second, we plot a histogram of the out-of-fold predictions. Many predictions are near 0.0 or near 1.0; this means that in many cases the classifier's predictions have high confidence:

# In[ ]:


plt.figure(figsize=(12, 4))
plt.hist(y_va_pred, bins=25, density=True)
plt.title('Histogram of the oof predictions')
plt.show()


# Finally, we plot the calibration curve. The curve here is almost a straight line, which means that the predicted probabilities are almost exact: 

# In[ ]:


plt.figure(figsize=(12, 4))
CalibrationDisplay.from_predictions(y_va, y_va_pred, n_bins=20, strategy='quantile', ax=plt.gca())
plt.title('Probability calibration')
plt.show()


# # Submission
# 
# For the submission, we re-train the model on several different seeds and then submit the mean of the ranks.

# In[ ]:


# Create submission
print(f"{len(features)} features")

pred_list = []
for seed in range(10):
    X_tr = train[features]
    y_tr = train.target

    model = my_booster(random_state=seed)
    model.fit(X_tr.values, y_tr)
    pred_list.append(scipy.stats.rankdata(model.predict_proba(test[features].values)[:,1]))
    print(f"{seed:2}", pred_list[-1])
print()
submission = test[['id']].copy()
submission['target'] = np.array(pred_list).mean(axis=0)
submission.to_csv('submission.csv', index=False)
submission


# # What next?
# 
# Now it's your turn! Try to improve this model by
# - Engineering more features
# - Tuning hyperparameters
# - Replacing LightGBM by XGBoost, HistGradientBoostingClassifier or CatBoost 
