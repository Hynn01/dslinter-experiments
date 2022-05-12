#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


pwd


# In[ ]:


df_train = pd.read_csv('../input/spaceship-titanic/train.csv')
#df_train = df_train.set_index('PassengerId')
df_train.head()


# In[ ]:


df_test = pd.read_csv('../input/spaceship-titanic/test.csv')
#df_test = df_test.set_index('PassengerId')
df_test.head()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.isnull().sum()*100/len(df_train)


# In[ ]:


df_test.isnull().sum()*100/len(df_test)


# In[ ]:


df_train['isTrain'] = 'Yes'
df_test['isTrain'] = 'No'


# In[ ]:


tt = pd.concat([df_train, df_test])
tt


# In[ ]:


tt[["C1", 'C2', 'C3']] = tt["Cabin"].str.split('/', expand=True)
tt[["P_id_1", "P_id_2"]] = tt['PassengerId'].str.split("_", expand=True)
tt


# In[ ]:


tt.dtypes


# In[ ]:


tt.isnull().sum()*100/len(tt)


# In[ ]:


tt.head()


# In[ ]:


tt = tt.drop(['Name', 'Cabin'], axis = 1)
tt.head()


# In[ ]:


cat_cols = list(tt.select_dtypes(include=['category', object, 'bool']).columns)
cat_cols


# In[ ]:


cat_cols.remove('Transported')
cat_cols


# In[ ]:


num_cols = [i for i in tt.columns if i not in cat_cols + ['Transported']]
num_cols


# #### cat impute

# In[ ]:


cat_tt = tt[cat_cols]
cat_tt


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')
imp_cat_tt = imp.fit_transform(cat_tt)
imp_cat_tt = pd.DataFrame(imp_cat_tt, columns = cat_cols)
imp_cat_tt


# In[ ]:


imp_cat_tt.isnull().sum()


# In[ ]:


num_cols


# #### Num impute

# In[ ]:


num_tt = tt[num_cols]
num_tt


# In[ ]:


num_tt.isnull().sum()*100/len(num_tt)


# In[ ]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
imp_num_tt = pd.DataFrame(imp.fit_transform(num_tt), columns = num_tt.columns)
imp_num_tt


# In[ ]:


imp_num_tt.isnull().sum().sum()


# In[ ]:


imp_tt = pd.concat([imp_cat_tt, imp_num_tt], axis = 1)
imp_tt


# In[ ]:


imp_tt.dtypes


# In[ ]:


imp_tt[['CryoSleep', 'VIP']] = imp_tt[['CryoSleep', 'VIP']].astype('bool')
imp_tt.dtypes


# In[ ]:


df_train['Transported'] = df_train['Transported'].astype('int')
df_train['Transported']


# In[ ]:


train = imp_tt[imp_tt['isTrain'] == 'Yes']
train = train.drop('isTrain', axis = 1)
train = pd.concat([train, df_train['Transported']], axis = 1)
train = train.set_index('PassengerId')
train


# In[ ]:


train.isnull().sum()


# In[ ]:


test = imp_tt[imp_tt['isTrain'] == 'No']
test = test.set_index('PassengerId')
test = test.drop('isTrain', axis = 1)
test


# In[ ]:


test.isnull().sum().sum()


# In[ ]:


X = train.drop('Transported', axis = 1)
X


# In[ ]:


y = train['Transported']
y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101, stratify=y)


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

pipe = make_pipeline(OneHotEncoder(handle_unknown='ignore'), RandomForestClassifier())


# In[ ]:


pipe.fit(X_train, y_train)


# In[ ]:


RF_acc = pipe.score(X_test, y_test)
RF_acc


# #### catboost

# In[ ]:


cat_cols = list(X_train.select_dtypes(include=['category', object, 'bool']).columns)
cat_cols


# In[ ]:


from catboost import CatBoostClassifier, Pool
train_dataset = Pool(X_train,y_train, cat_features=cat_cols)
test_dataset = Pool(X_test,y_test,cat_features=cat_cols)                                                                                                                                    


# In[ ]:


model = CatBoostClassifier(random_state=101,
                          n_estimators=10000,
                          loss_function='CrossEntropy',  
                          eval_metric="Logloss",
                          task_type="GPU")


# In[ ]:


model.fit(train_dataset,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=150,
    verbose=False)


# In[ ]:


preds = model.predict(X_test)
preds


# In[ ]:


from sklearn.metrics import accuracy_score
cat_acc = accuracy_score(y_test, preds)
cat_acc


# #### Lightgbm

# In[ ]:


cat_cols = list(X_train.select_dtypes(include=['category', object]).columns)
cat_cols


# In[ ]:


X.dtypes


# In[ ]:


for c in cat_cols:
    train[c] = train[c].astype('category')


# In[ ]:


train.dtypes


# In[ ]:


cat_idx = [train.columns.get_loc(col) for col in cat_cols]
cat_idx


# In[ ]:


train.head()


# In[ ]:


X = train.drop('Transported', axis = 1)
y = train.Transported


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101, stratify=y)


# In[ ]:


import lightgbm as lgb

eval_set = [(X_test, y_test)]


lgbm_clf = lgb.LGBMClassifier(
    objective="binary",
    random_state=101,
    n_estimators=10000,
    boosting="gbdt")

lgbm_clf.fit(
    X_train,
    y_train,
    categorical_feature=cat_idx,  # Specify the categoricals
    eval_set=eval_set,
    early_stopping_rounds=150,
    eval_metric="logloss",
    verbose=False)


# In[ ]:


preds=lgbm_clf.predict(X_test)


# In[ ]:


lgb_acc = accuracy_score(y_test, preds)
lgb_acc


# #### Xgboost

# In[ ]:


X.dtypes


# In[ ]:


X['C2'] = X['C2'].astype('int')
X['P_id_1'] = X['P_id_1'].astype('int')
X['P_id_2'] = X['P_id_2'].astype('int')


# In[ ]:


# Encode categoricals
X_enc = pd.get_dummies(X)
X_enc


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.20, random_state=101, stratify=y)


# In[ ]:


import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    random_state=101,
    n_estimators=10000,
    tree_method="hist",  # enable histogram binning in XGB
)


# In[ ]:


xgb_clf.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="logloss",
    early_stopping_rounds=150,
    verbose=False,  # Disable logs
)


# In[ ]:


preds = xgb_clf.predict(X_test)
preds


# In[ ]:


xgb_acc = accuracy_score(y_test, preds)
xgb_acc


# In[ ]:


df = pd.DataFrame({'Model' : ['RF', 'CB', 'Lgbm', 'Xgb'], 'Accuracy' : np.round([RF_acc, cat_acc, lgb_acc, xgb_acc], 4)})
df


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
chart = sns.barplot(data=df, x='Model', y='Accuracy', estimator=sum, ci=None)

# new helper method to auto-label bars (matplotlib 3.4.0+)
chart.bar_label(chart.containers[0])
plt.show()


# #### CB CV (Best GBM)

# In[ ]:


X.head()


# In[ ]:


X.dtypes


# In[ ]:


test.dtypes


# In[ ]:


test['C2'] = test['C2'].astype('int')
test['P_id_1'] = test['P_id_1'].astype('int')
test['P_id_2'] = test['P_id_2'].astype('int')


# In[ ]:


for c in cat_cols:
    test[c] = test[c].astype('category')


# In[ ]:


test.dtypes


# In[ ]:


cat_cols


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 10
strat_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=101)

scores = np.empty(N_SPLITS)

for idx, (train_idx, test_idx) in enumerate(strat_kf.split(X, y)):
    print("=" * 12 + f"Training fold {idx}" + 12 * "=")
    

    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y[train_idx], y[test_idx]
    eval_set = [(X_val, y_val)]
    
    train_dataset = Pool(X_train,y_train,cat_features=cat_cols)
    test_dataset = Pool(X_val,y_val,cat_features=cat_cols)  

    model = CatBoostClassifier(random_state=101,
                          n_estimators=10000,
                          loss_function='CrossEntropy',  
                          eval_metric="Logloss",
                          task_type="GPU")
    
    
    model.fit(train_dataset,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=150,
    verbose=False)
    
    
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    scores[idx] = acc
    
    print(f"Fold {idx} finished with score: {acc:.5f}")


# In[ ]:


scores


# In[ ]:


scores.mean()


# In[ ]:


subb = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
subb


# In[ ]:


test.isnull().sum()


# In[ ]:


test


# In[ ]:


test.C2.unique()


# In[ ]:


preds = model.predict(test)


# In[ ]:


preds


# In[ ]:


subb.Transported = preds
subb


# In[ ]:


subb.Transported = subb.Transported.astype(bool)
subb


# In[ ]:


subb.to_csv('subb_cat.csv', index = None)


# In[ ]:


pd.read_csv('./subb_cat.csv')

