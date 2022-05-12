#!/usr/bin/env python
# coding: utf-8

# # Spaceship Titanic

# #### This was a binary classification problem that predicts whether Transported is True or False. 
# 
# #### There were 13 explanatory variables and all columns except PassengerId were partially missing.
# 
# 
# #### It is important how to extract the features of each column and how to link columns together to create new columns that have the potential to improve the accuracy of the model.
# 
# #### I used **XGBoost** and **lightGBM** to built models.

# # Basic EDA

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')

import warnings
warnings.filterwarnings('ignore')

import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import optuna

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')


test  = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


print(train.isnull().sum())


# In[ ]:


oe = OrdinalEncoder()

encoded = oe.fit_transform(train[['HomePlanet','CryoSleep','Destination','VIP','Transported']])
train[['HomePlanet','CryoSleep','Destination','VIP','Transported']] = encoded



encoded = oe.fit_transform(test[['HomePlanet','CryoSleep','Destination','VIP']])
test[['HomePlanet','CryoSleep','Destination','VIP']] = encoded


# In[ ]:


plt.subplots(2,5,figsize=(20,9))
plt.subplot(2,5,1)
sns.countplot(train['HomePlanet']); plt.ylabel('')
plt.subplot(2,5,2)
sns.countplot(train['CryoSleep']); plt.ylabel('')
plt.subplot(2,5,3)
sns.countplot(train['Destination']); plt.ylabel('')
plt.subplot(2,5,4)
plt.hist(train['Age'], bins=10, color='orange')
plt.xlabel('Age', fontsize=15); plt.ylabel('')
plt.ylabel('')
plt.subplot(2,5,5)
sns.countplot(train['VIP']); plt.ylabel('')
plt.subplot(2,5,6)
plt.hist(train['RoomService'], bins=3, color='orange'); plt.ylabel('')
plt.xlabel('RoomService', fontsize=15)
plt.subplot(2,5,7)
plt.hist(train['FoodCourt'], bins=3, color='orange'); plt.ylabel('')
plt.xlabel('FoodCourt', fontsize=15)
plt.subplot(2,5,8)
plt.hist(train['ShoppingMall'], bins=3, color='orange'); plt.ylabel('')
plt.xlabel('ShoppingMall', fontsize=15)
plt.subplot(2,5,9)
plt.hist(train['Spa'], bins=3, color='orange'); plt.ylabel('')
plt.xlabel('Spa', fontsize=15)
plt.subplot(2,5,10)
plt.hist(train['VRDeck'], bins=3, color='orange'); plt.ylabel('')
plt.xlabel('VRDeck', fontsize=15)
plt.show()


# In[ ]:


train['VIP'] = train['VIP'].fillna(0.0)



test['VIP']  = test['VIP'].fillna(0.0)


# # PassengerId

# #### Extract features from PassengerId; a sequence of the same number, such as 0003, 0003, indicates that they are a group.

# In[ ]:


train['PassengerId'].head()


# In[ ]:


train['PassengerId']  = train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)

train['Group']        = train['PassengerId'].duplicated(keep=False)

train['Group_size']   = train['PassengerId'].apply(lambda x: train['PassengerId'].value_counts()[x])


# In[ ]:


test['PassengerId']   = test['PassengerId'].apply(lambda x: x.split('_')[0])

test['Group']         = test['PassengerId'].duplicated(keep=False)

test['Group_size']    = test['PassengerId'].apply(lambda x: test['PassengerId'].value_counts()[x])


# In[ ]:


encoded = oe.fit_transform(train[['PassengerId','Group']])
train[['PassengerId','Group']] = encoded



encoded = oe.fit_transform(test[['PassengerId','Group']])
test[['PassengerId','Group']] = encoded


# # Name

# #### I thought I could identify the family from the LastName.

# In[ ]:


train['Name'].head()


# In[ ]:


train['Name'].replace(np.nan, 'No Name', inplace=True)

train['Last_Name']    = train['Name'].apply(lambda x: x.split()[1])

train['Familly']      = train['Last_Name'].duplicated(keep=False)

train['Familly_size'] = train['Last_Name'].apply(lambda x: train['Last_Name'].value_counts()[x])


# In[ ]:


test['Name'].replace(np.nan, 'No Name', inplace=True)

test['Last_Name']    = test['Name'].apply(lambda x: x.split()[1])

test['Familly']      = test['Last_Name'].duplicated(keep=False)

test['Familly_size'] = test['Last_Name'].apply(lambda x: test['Last_Name'].value_counts()[x])


# In[ ]:


encoded = oe.fit_transform(train[['Name','Last_Name','Familly']])
train[['Name','Last_Name','Familly']] = encoded



encoded = oe.fit_transform(test[['Name','Last_Name','Familly']])
test[['Name','Last_Name','Familly']] = encoded


# In[ ]:


train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)


# # Cabin

# #### Cabin consists of deck/num/side, and I divided the Cabin into each of those three categories and extracted features from them.

# In[ ]:


train['Cabin'] = train['Cabin'].fillna('Z/9999/Z')



test['Cabin']  = test['Cabin'].fillna('Z/9999/Z')


# In[ ]:


train['Cabin_0'] = train['Cabin'].apply(lambda x: x.split('/')[0])
train['Cabin_1'] = train['Cabin'].apply(lambda x: x.split('/')[1])
train['Cabin_2'] = train['Cabin'].apply(lambda x: x.split('/')[2])



test['Cabin_0']  = test['Cabin'].apply(lambda x: x.split('/')[0])
test['Cabin_1']  = test['Cabin'].apply(lambda x: x.split('/')[1])
test['Cabin_2']  = test['Cabin'].apply(lambda x: x.split('/')[2])


# In[ ]:


encoded = oe.fit_transform(train[['Cabin_0','Cabin_1','Cabin_2']])
train[['Cabin_0','Cabin_1','Cabin_2']] = encoded



encoded = oe.fit_transform(test[['Cabin_0','Cabin_1','Cabin_2']])
test[['Cabin_0','Cabin_1','Cabin_2']] = encoded


# In[ ]:


train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)


# # Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck

# #### For continuous variables, I complemented missing values with mean and mode values and created new variables to improve the accuracy of the model.

# In[ ]:


train.describe()[['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]


# In[ ]:


train['Age']          =  train['Age'].fillna(train['Age'].mean())
train['RoomService']  =  train['RoomService'].fillna(train['RoomService'].mode()[0])
train['FoodCourt']    =  train['FoodCourt'].fillna(train['FoodCourt'].mode()[0])
train['ShoppingMall'] =  train['ShoppingMall'].fillna(train['RoomService'].mode()[0])
train['Spa']          =  train['Spa'].fillna(train['Spa'].mode()[0])
train['VRDeck']       =  train['VRDeck'].fillna(train['VRDeck'].mode()[0])



test['Age']           =  test['Age'].fillna(test['Age'].mean())
test['RoomService']   =  test['RoomService'].fillna(test['RoomService'].mode()[0])
test['FoodCourt']     =  test['FoodCourt'].fillna(test['FoodCourt'].mode()[0])
test['ShoppingMall']  =  test['ShoppingMall'].fillna(test['RoomService'].mode()[0])
test['Spa']           =  test['Spa'].fillna(test['Spa'].mode()[0])
test['VRDeck']        =  test['VRDeck'].fillna(test['VRDeck'].mode()[0])


# #### I used mean and standard deviation to detect and exclude outliers.
# #### I excluded values that were more than two times the standard deviation away from the mean.

# In[ ]:


# outlier

def outlier(df, columns=None):
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        border    = np.abs(df[col] - mean) / std
        df = df[(border < 2.0)]
    return df


# In[ ]:


print(train.shape)
train = outlier(train, ['RoomService'])
print(train.shape)
train = outlier(train, ['FoodCourt'])
print(train.shape)
train = outlier(train, ['ShoppingMall'])
print(train.shape)
train = outlier(train, ['Spa'])
print(train.shape)
train = outlier(train, ['VRDeck'])
print(train.shape)


# In[ ]:


plt.subplots(2,3,figsize=(15,8))
plt.subplot(231)
plt.hist(train['RoomService'], bins=4); plt.ylabel('')
plt.xlabel('RoomService')
plt.subplot(232)
plt.hist(train['FoodCourt'], bins=4); plt.ylabel('')
plt.xlabel('FoodCourt')
plt.subplot(233)
plt.hist(train['ShoppingMall'], bins=4); plt.ylabel('')
plt.xlabel('ShoppingMall')
plt.subplot(234)
plt.hist(train['Spa'], bins=4); plt.ylabel('')
plt.xlabel('Spa')
plt.subplot(235)
plt.hist(train['VRDeck'], bins=4); plt.ylabel('')
plt.xlabel('VRDeck')
plt.show()


# # Total Expenditure

# In[ ]:


train['Total Expenditure'] = train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']



test['Total Expenditure']  = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']


# In[ ]:


#columns = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


#for i in range(10):
    #x = list(list(itertools.combinations(columns,2))[i])
    #train[i] = train[x[0]] + train[x[1]]
    
    
#for i in range(10):
    #x = list(list(itertools.combinations(columns,3))[i])
    #train[i+10] = train[x[0]] + train[x[1]] + train[x[2]]
    
    
#for i in range(5):
    #x = list(list(itertools.combinations(columns,4))[i])
    #train[i+20] = train[x[0]] + train[x[1]] + train[x[2]] + train[x[3]]


# In[ ]:


#for i in range(10):
    #x = list(list(itertools.combinations(columns,2))[i])
    #test[i] = test[x[0]] + test[x[1]]
    
    
#for i in range(10):
    #x = list(list(itertools.combinations(columns,3))[i])
    #test[i+10] = test[x[0]] + test[x[1]] + test[x[2]]
    
    
#for i in range(5):
    #x = list(list(itertools.combinations(columns,4))[i])
    #test[i+20] = test[x[0]] + test[x[1]] + test[x[2]] + test[x[3]]


# ## log-transformation

# In[ ]:


train['RoomService_log']  =  np.log1p(train['RoomService'])
train['FoodCourt_log']    =  np.log1p(train['FoodCourt'])
train['ShoppingMall_log'] =  np.log1p(train['ShoppingMall'])
train['Spa_log']          =  np.log1p(train['Spa'])
train['VRDeck_log']       =  np.log1p(train['VRDeck'])
train['Age_log']          =  np.log1p(train['Age'])



test['RoomService_log']   =  np.log1p(test['RoomService'])
test['FoodCourt_log']     =  np.log1p(test['FoodCourt'])
test['ShoppingMall_log']  =  np.log1p(test['ShoppingMall'])
test['Spa_log']           =  np.log1p(test['Spa'])
test['VRDeck_log']        =  np.log1p(test['VRDeck'])
test['Age_log']           =  np.log1p(test['Age'])


# ## StandardScaler

# In[ ]:


scaler = StandardScaler()


a = train['RoomService'].values.reshape(-1,1)
train['RoomService_scaler'] = scaler.fit_transform(a)

b = train['FoodCourt'].values.reshape(-1,1)
train['FoodCourt_scaler'] = scaler.fit_transform(b)

c = train['ShoppingMall'].values.reshape(-1,1)
train['ShoppingMall_scaler'] = scaler.fit_transform(c)

d = train['Spa'].values.reshape(-1,1)
train['Spa_scaler'] = scaler.fit_transform(d)

e = train['VRDeck'].values.reshape(-1,1)
train['VRDeck_scaler'] = scaler.fit_transform(e)

f = train['Age'].values.reshape(-1,1)
train['Age_scaler'] = scaler.fit_transform(f)


# In[ ]:


a = test['RoomService'].values.reshape(-1,1)
test['RoomService_scaler']  = scaler.fit_transform(a)

b = test['FoodCourt'].values.reshape(-1,1)
test['FoodCourt_scaler']    = scaler.fit_transform(b)

c = test['ShoppingMall'].values.reshape(-1,1)
test['ShoppingMall_scaler'] = scaler.fit_transform(c)

d = test['Spa'].values.reshape(-1,1)
test['Spa_scaler']          = scaler.fit_transform(d)

e = test['VRDeck'].values.reshape(-1,1)
test['VRDeck_scaler']       = scaler.fit_transform(e)

f = test['Age'].values.reshape(-1,1)
test['Age_scaler']          = scaler.fit_transform(f)


# ## PowerTransformation (yeo-johnson)

# In[ ]:


pt = PowerTransformer(method='yeo-johnson')


pt.fit(train['RoomService'].values.reshape(-1,1))
train['RoomService_yeo-johnson']  = pt.transform(train['RoomService'].values.reshape(-1,1))

pt.fit(train['FoodCourt'].values.reshape(-1,1))
train['FoodCourt_yeo-johnson']    = pt.transform(train['FoodCourt'].values.reshape(-1,1))

pt.fit(train['ShoppingMall'].values.reshape(-1,1))
train['ShoppingMall_yeo-johnson'] = pt.transform(train['ShoppingMall'].values.reshape(-1,1))

pt.fit(train['Spa'].values.reshape(-1,1))
train['Spa_yeo-johnson']          = pt.transform(train['Spa'].values.reshape(-1,1))

pt.fit(train['VRDeck'].values.reshape(-1,1))
train['VRDeck_yeo-johnson']       = pt.transform(train['VRDeck'].values.reshape(-1,1))

pt.fit(train['VRDeck'].values.reshape(-1,1))
train['VRDeck_yeo-johnson']       = pt.transform(train['VRDeck'].values.reshape(-1,1))

pt.fit(train['Age'].values.reshape(-1,1))
train['Age_yeo-johnson']          = pt.transform(train['Age'].values.reshape(-1,1))


# In[ ]:


pt.fit(test['RoomService'].values.reshape(-1,1))
test['RoomService_yeo-johnson']  = pt.transform(test['RoomService'].values.reshape(-1,1))

pt.fit(test['FoodCourt'].values.reshape(-1,1))
test['FoodCourt_yeo-johnson']    = pt.transform(test['FoodCourt'].values.reshape(-1,1))

pt.fit(test['ShoppingMall'].values.reshape(-1,1))
test['ShoppingMall_yeo-johnson'] = pt.transform(test['ShoppingMall'].values.reshape(-1,1))

pt.fit(test['Spa'].values.reshape(-1,1))
test['Spa_yeo-johnson']          = pt.transform(test['Spa'].values.reshape(-1,1))

pt.fit(test['VRDeck'].values.reshape(-1,1))
test['VRDeck_yeo-johnson']       = pt.transform(test['VRDeck'].values.reshape(-1,1))

pt.fit(test['Age'].values.reshape(-1,1))
test['Age_yeo-johnson']          = pt.transform(test['Age'].values.reshape(-1,1))


# #### I filled in the missing values using other columns.

# In[ ]:


# outlier

def check(df, column):
    col = np.abs(df.corr()[column])
    print(col.sort_values(ascending=False).head(10))

    
def missing_value(df,column,column2,column3,column4,column5,column6):
    target = df[[column,column2,column3,column4,column5,column6]]
    notnull = target[target[column].notnull()].values
    null = target[target[column].isnull()].values
    X = notnull[:, 1:]
    y = notnull[:, 0]
    rf = RandomForestClassifier(random_state=0,n_estimators=1000,n_jobs=-1)
    rf.fit(X,y)
    predict = rf.predict(null[:, 1::])
    print(predict)
    df.loc[(df[column].isnull(), column)] = predict


# In[ ]:


check(train, 'CryoSleep')


# In[ ]:


missing_value(train,'CryoSleep','Total Expenditure','RoomService_yeo-johnson','Spa_yeo-johnson','ShoppingMall_yeo-johnson','FoodCourt_yeo-johnson')


# In[ ]:


check(test, 'CryoSleep')


# In[ ]:


missing_value(test,'CryoSleep','Spa_yeo-johnson','FoodCourt_yeo-johnson','RoomService_yeo-johnson','ShoppingMall_yeo-johnson','VRDeck_yeo-johnson')


# In[ ]:


np.abs(train.corr()['Transported']).sort_values(ascending=False).head(30)


# In[ ]:


train.drop(['Destination','Age','HomePlanet'], axis=1, inplace=True)


test.drop(['Destination','Age','HomePlanet'], axis=1, inplace=True)


# In[ ]:


y = train['Transported']
X = train.copy()

X.drop('Transported', axis=1, inplace=True)


# In[ ]:


#kf = KFold(n_splits=5, shuffle=True, random_state=0)
#for tr_idx, va_idx in kf.split(X, y):
    #X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
    #y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, random_state=1205)


# In[ ]:


print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


# # Modeling

# ### Xgboost

# In[ ]:


import xgboost as xgb


# In[ ]:


model = XGBClassifier(random_state=1205)
model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[[X_valid, y_valid]])


# In[ ]:


params =  {'objective':['binary:logistic'],
           'n_estimators':[65],
           'max_depth':[2],
           'min_child_weight':[2],
           'subsample':[0.7],
           'colsample_bytree':[0.6],
           'eta':[0.03],
           'gamma':[0.2]
           }

grid_search = GridSearchCV(model,
                           params,
                           scoring='accuracy',
                           cv=2,
                           n_jobs=-1,
                           verbose=0)

grid_search.fit(X_train,y_train)


# In[ ]:


print(grid_search.best_estimator_)
print(grid_search.best_params_)


# In[ ]:


params = {'random_state':1205,
          'objective':'binary:logistic', 
          'n_estimators':65,
          'max_depth':2,
          'min_child_weight':2,
          'eta':0.03,
          'silent':1, 
          'gamma':0.2,
          'colsample_bytree':0.6,
          'subsample':0.7
          }

num_round = 100000000


# In[ ]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest  = xgb.DMatrix(test)


evals = [(dtrain, 'train'), (dvalid, 'eval')]


# In[ ]:


model = xgb.train(params, dtrain, num_round, evals=evals, early_stopping_rounds=150)


# In[ ]:


y_train_pred = model.predict(dtrain)
y_valid_pred = model.predict(dvalid)


# In[ ]:


y_train_pred = (y_train_pred > 0.5).astype(int)
y_valid_pred = (y_valid_pred > 0.5).astype(int)


print(np.mean(y_train_pred == y_train))
print(np.mean(y_valid_pred == y_valid))


# In[ ]:


y_pred_xgboost = model.predict(dtest)
y_pred_xgboost = (y_pred_xgboost > 0.5).astype(int)


# ### LightGBM

# In[ ]:


categorical_features = ['CryoSleep','Cabin_0','Group','Cabin_2','VIP','PassengerId']


lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
lgb_eval  = lgb.Dataset(X_valid, y_valid, reference=lgb_train,categorical_feature=categorical_features)


# In[ ]:


model = lgb.LGBMClassifier(seed=0, objective='binary')

param_grid = {
              "max_depth"     : [3,4,5,6,7,8],
              "learning_rate" : [0.055],
              "num_leaves"    : [5,7,10,15,20,25,30],
              "n_estimators"  : [60,65,70,300,400,500],
             }

grid_result = GridSearchCV(estimator = model,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 2,
                           return_train_score = False,
                           n_jobs = -1)

grid_result.fit(X_train,y_train)


print(grid_result.best_estimator_)
print((grid_result.best_params_))


# In[ ]:


lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
lgb_eval  = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

params={
    'seed':0,
    'num_leaves':15,
    'objective':'binary',
    'max_depth':7,
    'learning_rate':.055,
    'n_estimators':65,
    'early_stopping_rounds':10,
}
num_round=100000000000


# In[ ]:


model = lgb.train(params, 
                  lgb_train, 
                  valid_sets=[lgb_train, lgb_eval], 
                  num_boost_round=num_round, 
                  categorical_feature=categorical_features, 
                  verbose_eval=False)


# In[ ]:


y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)


# In[ ]:


y_train_pred = (y_train_pred > 0.5).astype(int)
y_valid_pred = (y_valid_pred > 0.5).astype(int)


print(np.mean(y_train_pred == y_train))
print(np.mean(y_valid_pred == y_valid))


# In[ ]:


y_pred_lightgbm = model.predict(test)
y_pred_lightgbm = (y_pred_lightgbm > 0.5).astype(int)


# In[ ]:


sub = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')
sub['Transported'] = list(map(int, y_pred_lightgbm))
sub['Transported'] = sub['Transported'].replace({0.0:'False',1.0:'True'})
sub.to_csv('submission.csv', index=False)


# In[ ]:


sub.head()

