#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install flaml')
get_ipython().system('git clone https://github.com/analokmaus/kuma_utils.git')


# # File and Data Field Descriptions
# * train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
#     * PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
#     * HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
#     * CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
#     * Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
#     * Destination - The planet the passenger will be debarking to.
#     * Age - The age of the passenger.
#     * VIP - Whether the passenger has paid for special VIP service during the voyage.
#     * RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
#     * Name - The first and last names of the passenger.
#     * Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
# 
# * test.csv - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.
# * sample_submission.csv - A submission file in the correct format.
#     * PassengerId - Id for each passenger in the test set.
#     * Transported - The target. For each passenger, predict either True or False.

# In[ ]:


import numpy as np
import pandas as pd
import sys
sys.path.append("kuma_utils/")
import seaborn as sns
import plotly.express as px
from kuma_utils.preprocessing.imputer import LGBMImputer
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_percentage_error


# In[ ]:


train=pd.read_csv('../input/spaceship-titanic/train.csv')
test=pd.read_csv('../input/spaceship-titanic/test.csv')
train.nunique().sort_values(ascending=False)
round(train.isnull().sum()*100/len(train),2).sort_values(ascending=False)
train=train.drop(['PassengerId'],axis=1)
test=test.drop(['PassengerId'],axis=1)
train=train.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)


# ##  HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.

# In[ ]:


df = px.data.tips()
fig = px.histogram(train, x="HomePlanet")
fig.show()


# In[ ]:


fig = px.histogram(test, x="HomePlanet")
fig.show()


# ## CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.

# In[ ]:


fig = px.histogram(train, x="CryoSleep")
fig.show()


# In[ ]:


fig = px.histogram(test, x="CryoSleep")
fig.show()


# ## Destination - The planet the passenger will be debarking to.

# In[ ]:


fig = px.histogram(train, x="Destination")
fig.show()


# In[ ]:


fig = px.histogram(test, x="Destination")
fig.show()


# ## VIP - Whether the passenger has paid for special VIP service during the voyage.

# In[ ]:


fig = px.histogram(train, x="VIP")
fig.show()


# In[ ]:


fig = px.histogram(test, x="VIP")
fig.show()


# ## Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# In[ ]:


fig = px.histogram(train, x="Transported")
fig.show()


# In[ ]:


print(train['Destination'].value_counts())
print('===========================')
train.info()


# In[ ]:


train[['deck', 'num','side']] = train['Cabin'].str.split('/', expand=True)
train=train.drop(['Cabin'],axis=1)
test[['deck', 'num','side']] = test['Cabin'].str.split('/', expand=True)
test=test.drop(['Cabin'],axis=1)
train.nunique().sort_values(ascending=False)


# In[ ]:


print(train['deck'].value_counts())
print('=======================')
print(train['deck'].unique().tolist())
print('=======================')
print(test['deck'].unique().tolist())


# In[ ]:


train['deck']=train['deck'].replace({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7})
test['deck']=test['deck'].replace({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7})
train[['Age','RoomService','FoodCourt',
       'ShoppingMall','Spa','VRDeck','deck','num']]=train[['Age','RoomService','FoodCourt',
       'ShoppingMall','Spa','VRDeck','deck','num']].astype('float')
test[['Age','RoomService','FoodCourt',
       'ShoppingMall','Spa','VRDeck','deck','num']]=test[['Age','RoomService','FoodCourt',
       'ShoppingMall','Spa','VRDeck','deck','num']].astype('float')
train=pd.get_dummies(train,prefix_sep='__')
test=pd.get_dummies(test,prefix_sep='__')


# In[ ]:


col=train.columns.tolist()
col.remove('Transported')
col


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lgbm_imtr = LGBMImputer(n_iter=500)\n\ntrain_iterimp = lgbm_imtr.fit_transform(train[col])\ntest_iterimp = lgbm_imtr.transform(test[col])\n\n# Create train test imputed dataframe\ntrain_ = pd.DataFrame(train_iterimp, columns=col)\ntest = pd.DataFrame(test_iterimp, columns=col)')


# In[ ]:


train_['Transported'] = train['Transported']


# In[ ]:


def undummify(df, prefix_sep="__"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


# In[ ]:


train=undummify(train_)
train.head()


# In[ ]:


test=undummify(test)
test.head()


# In[ ]:


automl = AutoML()


# In[ ]:


y = train.pop('Transported')
X = train


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42,shuffle=True, stratify=y)


# In[ ]:


automl.fit(X_train, y_train, task="classification",metric='ap',time_budget=300)


# In[ ]:


print(automl.best_estimator)
print(automl.best_config)
print(1-automl.best_loss)
print(automl.best_config_train_time)


# In[ ]:


classification_report(y_train, automl.predict(X_train))


# In[ ]:


classification_report(y_test, automl.predict(X_test))


# In[ ]:


y_pred = automl.predict(test)
y_pred[:5]


# In[ ]:


df = pd.DataFrame(y_pred,columns=['Transported'])
sol=pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
sol['Transported']=df['Transported']
sol.to_csv('./submission.csv',index=False)


# In[ ]:




