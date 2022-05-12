#!/usr/bin/env python
# coding: utf-8

# <h2 style="color:#2c3f51"> TPS MAY 2022 </h2>

# - Try LGBMClassifier
# - Experiment with new features
# - Search best params with Optuna 
# - Make Cross Validation

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import optuna
import gc

import warnings
warnings.simplefilter("ignore")


# # Loading Data

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")

train.shape, test.shape


# # Understanding Data

# - Describe

# In[ ]:


train.describe()


# In[ ]:


test.describe()


# - Infos

# In[ ]:


train.info()


# We have :
# - 16 float64 columns
# - 16 int64 columns
# - 1 object columns

# In[ ]:


test.info()


# - Check missing columns

# In[ ]:


train.isna().sum().any()


# In[ ]:


test.isna().sum().any()


# In[ ]:





# # Feature Engineering 
# 
# 
# ## FE from @ambrosm

# Nous remercions @ambrosm pour le travail fait et partagé sur cette section dans son [carnet](https://www.kaggle.com/code/ambrosm/tpsmay22-keras-quickstart#Feature-engineering).
# 
# "*We read the data and apply minimal feature engineering: We only split the f_27 string into ten separate features as described in the [EDA](https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense), and we count the unique characters in the string.*"

# In[ ]:


"""
for df in [train, test]:
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    # Next feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
features = [f for f in test.columns if f != 'id' and f != 'f_27']
test[features].head(2)
"""


# ## FE from @hasanbasriakcay

# Nous remercions @hasanbasriakcay pour le travail fait et partagé sur cette section dans son [carnet](https://www.kaggle.com/code/hasanbasriakcay/tpsmay22-my100-notebook-autoblendingfunc/notebook).

# In[ ]:


def create_features(data):
    object_data_cols = [f"f_27_{i+1}" for i in range(10)]
    object_data = pd.DataFrame(data['f_27'].apply(list).tolist(), columns=object_data_cols)
    for feature in object_data_cols:
        object_data[feature] = object_data[feature].apply(ord) - ord('A')
    
    '''
    object_data['f_27_sum'] = 0
    for feature in object_data_cols:
        object_data['f_27_sum'] += object_data[feature]
    '''
    
    data = pd.concat([data, object_data], 1)
    data["unique_characters"] = data.f_27.apply(lambda s: len(set(s)))
    
    ## sum
    # float
    #data['f_sum_1'] = (data['f_00']+data['f_01']+data['f_02']+data['f_05'])
    data['f_sum_2'] = (data['f_21']+data['f_22'])
    data['f_sum_3'] = (data['f_23']-data['f_20'])
    data['f_sum_4'] = (data['f_25']-data['f_28']/100)
    data['f_sum_5'] = (data['f_00']+data['f_01'])
    #data['f_sum_6'] = (data['f_02']+data['f_05'])
    #data['f_sum_7'] = (data['f_00']+data['f_02'])
    #data['f_sum_8'] = (data['f_01']+data['f_02'])
    # int
    #data['f_sum_9'] = (data['f_07']+data['f_08'])
    data['f_sum_10'] = (data['f_07']-data['f_10'])
    #data['f_sum_11'] = (data['f_07']-data['f_12'])
    #data['f_sum_12'] = (data['f_07']-data['f_15'])
    data['f_sum_13'] = (data['f_08']-data['f_10'])
    #data['f_sum_14'] = (data['f_08']-data['f_12'])
    #data['f_sum_15'] = (data['f_09']-data['f_11'])
    
    
    return data


# In[ ]:


train_fe = create_features(train.copy())
test_fe = create_features(test.copy())

train_fe.shape, test_fe.shape


# # Modeling

# In[ ]:


from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


# ### Split Data

# In[ ]:


y = train_fe["target"]
X = train_fe.drop(columns=["id","f_27", "target"])

test = test_fe.drop(columns=["id","f_27"])

X.shape, test.shape


# In[ ]:


del test_fe, train_fe, train
gc.collect()


# In[ ]:


from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ### Search best params with optuna

# In[ ]:


from lightgbm import LGBMClassifier

import time


# In[ ]:


#optuna.logging.set_verbosity(optuna.logging.WARNING)

def objectivesLGBM(trial):
    params = {
        #'max_depth' : trial.suggest_int("max_depth", 1, 16),
        'n_estimators': trial.suggest_int('n_estimators', 5, 5000),
        #'random_state': trial.suggest_int("random_state", 0, 722),
        'learning_rate': trial.suggest_float('learning_rate', 0, 1),
        
        'num_leaves': trial.suggest_int('num_leaves', 2, 200),
        'max_bin': trial.suggest_int('max_bin', 2, 100),
        
        'objective':'binary', 
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 500),
        #'max_bins': trial.suggest_int('max_bins', 1, 150),
        
        'device' : 'gpu',
        'n_jobs' : -1,
        'verbose': 0
    }

    model = LGBMClassifier(**params)
    model.fit(X,y)

    return model.score(X,y)

#opt = optuna.create_study(direction='maximize')
#opt.optimize(objectivesLGBM, n_trials=785)


# In[ ]:


"""
params = opt.best_params

params['device'] = 'gpu'
params['n_jobs'] = -1
params['verbose'] = 0
"""

params = {
     'n_estimators': 4740, 
     'learning_rate': 0.5224042427936195, 
     'num_leaves': 118, 
     'max_bin': 32, 
     'min_child_samples': 383,
     'objective':'binary',
     'device': 'gpu',
     'n_jobs': -1,
     'verbose': 0
    }

y_predict = []


# ### Check features important

# In[ ]:


model = LGBMClassifier(**params)
model.fit(X, y)

print("Training score :", model.score(X, y))

pred_y_test = model.predict(X_test)
print("Roc auc score  :", roc_auc_score(y_test, pred_y_test))

# Make test
y_predict.append(model.predict_proba(test)[:,1])


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)

feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# In[ ]:


feature_imp = feature_imp.sort_values(by = "Value", ascending=False)
feature_imp


# In[ ]:


# Selected 6 Best Features
selected_features = feature_imp[:6]['Feature'].tolist()
selected_features


# In[ ]:


#X = X[selected_features]
#test = test[selected_features]

#X.shape, test.shape


# ### Add Basics Features

# In[ ]:


def add_basics_features(data, features):
    
    for feature in features:
        """
        new_feature_name = str(feature) + '_min'
        data[new_feature_name] = data[feature].min()
        
        new_feature_name = str(feature) + '_max'
        data[new_feature_name] = data[feature].max()
        """
        
        new_feature_name = str(feature) + '_mean'
        data[new_feature_name] = data[feature].mean()
        
        new_feature_name = str(feature) + '_std'
        data[new_feature_name] = data[feature].std()
        
        new_feature_name = str(feature) + '_max_min'
        data[new_feature_name] = data[feature].max() - data[feature].min()
    
    return data


# In[ ]:


#X = add_basics_features(X, selected_features)
#test = add_basics_features(test, selected_features)

#X.shape, test.shape


# ### Cross Validation

# Nous remercions @cabaxiom pour son travail fait et partagé [ici](https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model#Feature-Importance).

# In[ ]:


kf = StratifiedKFold(n_splits=9, shuffle=True, random_state = 0)


for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print('*'*14, f" Fold {fold} ", '*'*14, '\n')
        
        # Split Data
        X_train = X.loc[train_index]
        X_val = X.loc[val_index]

        y_train = y.loc[train_index]
        y_val = y.loc[val_index]
        
        # Create Model here
        model = LGBMClassifier(**params)
        
        # Fit Model
        model.fit(X_train,y_train)

        # Make X_val prediction
        y_pred = model.predict_proba(X_val)[:,1]
        
        # Make Test prediction
        y_predict.append(model.predict_proba(test)[:,1])

        # Evaluate Model
        print("Training score :", model.score(X_train, y_train))
        print("Roc auc score  :", roc_auc_score(y_val, y_pred), '\n')
        
        # Free the memory
        del X_train, y_train, model, X_val, y_val, y_pred
        gc.collect()
        


# ### Training with complet data

# In[ ]:


# Create Model here
#model = LGBMClassifier(**params)

# Fit Model
#model.fit(X, y)

# Make Test prediction
#y_predict.append(model.predict_proba(test)[:,1])


# In[ ]:


del model, X, y, test, X_test, y_test
gc.collect()


# In[ ]:


test_preds = np.array(y_predict).mean(axis=0)
test_preds


# # Submission

# In[ ]:


submission = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
submission.shape


# In[ ]:


import scipy

submission['target'] = scipy.stats.rankdata(test_preds)
submission.to_csv('submission.csv', index=False)
submission


# In[ ]:





# In[ ]:





# <center>
#     <h2 style="color:#2c3f51"> Thanks for reading 👍 </h2>
