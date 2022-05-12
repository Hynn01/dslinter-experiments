#!/usr/bin/env python
# coding: utf-8

# #### Author: João Pedro Mantoan
# #### Linkedin: http://linkedin.com/in/jo%C3%A3o-pedro-mantoan

# # Introduction

# The objective of this competition is to forecast the sales for departments in Walmart stores based on historical sales data for 45 Walmart stores located in different regions. 

# # Importing dependencies

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import random


# # Loading the datasets

# In[ ]:


train=pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
test=pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
features=pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
sample_sub=pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')
stores=pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')


# # Data preparation, exploration and cleaning

# In[ ]:


feature_store = features.merge(stores, how='inner', on = "Store")


# In[ ]:


feature_store.head()


# In[ ]:


train = train.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
train.head()


# In[ ]:


test = test.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
test.head()


# In[ ]:


features.shape, train.shape, stores.shape, test.shape, sample_sub.shape


# ### Train and test data types

# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# In[ ]:


#Changing date's type from string to date
train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)


# In[ ]:


#Descriptive statistics of the numerical data
train.copy().drop(columns=['Date','IsHoliday','Type','Store']).describe().round(2)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.columns


# In[ ]:


train.Type.unique()


# #### Checking for negative sales

# In[ ]:


train[train['Weekly_Sales'] < 0]


# In[ ]:


train.drop(train[train.Weekly_Sales < 0].index, inplace=True)


# ### Encoding categorical data

# In[ ]:


sup_dict = {'A': 1,
           'B': 2,
           'C':3}
train['Type'] = train['Type'].map(lambda x: sup_dict[x])
train.Type.unique()


# In[ ]:


test['Type'] = test['Type'].map(lambda x: sup_dict[x])
test.Type.unique()


# In[ ]:


train['IsHoliday'] = train['IsHoliday'].map(lambda x: 0 if x == False else 1)
test['IsHoliday'] = test['IsHoliday'].map(lambda x: 0 if x == False else 1)


# ### Splitting Date into Year, Month, Week, Day
# 
# This allows a better understanding of the relationship between the target and the date info

# In[ ]:


train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Week'] = train['Date'].dt.week
train['Day'] = train['Date'].dt.day


# In[ ]:


test['Year'] = test['Date'].dt.year
test['Month'] = test['Date'].dt.month
test['Week'] = test['Date'].dt.week
test['Day'] = test['Date'].dt.day


# #### Plotting of Weekly sales' means for all the 3 years of data

# In[ ]:


weekly_sales2010 = train.loc[train['Year']==2010].groupby(['Week']).agg({'Weekly_Sales': ['mean']})
weekly_sales2011 = train.loc[train['Year']==2011].groupby(['Week']).agg({'Weekly_Sales': ['mean']})
weekly_sales2012 = train.loc[train['Year']==2012].groupby(['Week']).agg({'Weekly_Sales': ['mean']})
plt.figure(figsize=(20, 10))
sns.lineplot(weekly_sales2010['Weekly_Sales']['mean'].index, weekly_sales2010['Weekly_Sales']['mean'].values)
sns.lineplot(weekly_sales2011['Weekly_Sales']['mean'].index, weekly_sales2011['Weekly_Sales']['mean'].values)
sns.lineplot(weekly_sales2012['Weekly_Sales']['mean'].index, weekly_sales2012['Weekly_Sales']['mean'].values)

plt.grid()
plt.xticks(np.arange(1, 53, step=1))
plt.legend(['2010', '2011', '2012'])
plt.show()


# In[ ]:


# Changing the order of the variables in the dataset for better visualisation of the correlation betwen them
train=train[['Store', 'Dept', 'Date', 'Year', 'Month', 'Week', 'Day', 'IsHoliday', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Type', 'Size', 'Weekly_Sales']]
train.head()


# In[ ]:


test=test[['Store', 'Dept', 'Date', 'Year', 'Month', 'Week', 'Day', 'IsHoliday', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Type', 'Size']]
test.head()


# In[ ]:


#Droping the Date columns since it's now divided in multiple columns
train.drop(columns='Date')
test.drop(columns='Date')


# ### Plotting the Pearson correlation beetwen the variables

# In[ ]:


corr = train.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(3)


# In[ ]:


weekly_sales_corr = train.corr().iloc[-1].sort_values(ascending=False)
weekly_sales_corr


# #### Modulus of the correlations' values to decide wich variables will be used in the model

# In[ ]:


abs_corr = weekly_sales_corr.map(lambda x: abs(x)).sort_values(ascending=True).drop(index='IsHoliday')
abs_corr


# Only the variables with a value of correlation with 'WeeklySales' bigger than 0.025 will be used in the model, with the exception of 'IsHoliday' because it will be used in the model evaluation

# In[ ]:


drop_features = list(abs_corr.index.values[0:6])


# Being 'Size'and 'Type' strongly correlated and Size more correlated to WeeklySales, Type will be removed as well.

# In[ ]:


drop_features.append('Type')
drop_features


# In[ ]:


train_dataset = train.copy()
test_dataset = test.copy()


# In[ ]:


train_dataset=train_dataset.drop(columns=drop_features)
train_dataset


# In[ ]:


test_dataset=test_dataset.drop(columns=drop_features)
test_dataset


# #### Another correlation plot in order to see the correlation between the selected variables

# In[ ]:


corr = train_dataset.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(3)


# Being all the 'MarkDown' variables considerably correlated to other variables besides 'Weekly_Sales' and each other, they will be removed

# In[ ]:


train_dataset=train_dataset.drop(columns=['MarkDown1','MarkDown3','MarkDown4','MarkDown5'])
test_dataset=test_dataset.drop(columns=['MarkDown1','MarkDown3','MarkDown4','MarkDown5'])


# #### Treating any remaining null values in the datasets

# In[ ]:


train_dataset.isnull().sum()


# In[ ]:


test_dataset.isnull().sum()


# In[ ]:


test[test['Unemployment'].isnull()].shape


# In[ ]:


test_dataset['Unemployment'].max(), test_dataset['Unemployment'].min(), test_dataset['Unemployment'].mean()


# ### Unemployment rate plot

# In[ ]:


unemployment2010 = train.loc[train['Year']==2010].groupby(['Week']).agg({'Unemployment': ['mean']})
unemployment2011 = train.loc[train['Year']==2011].groupby(['Week']).agg({'Unemployment': ['mean']})
unemployment2012 = train.loc[train['Year']==2012].groupby(['Week']).agg({'Unemployment': ['mean']})
plt.figure(figsize=(20, 10))
sns.lineplot(unemployment2010['Unemployment']['mean'].index, unemployment2010['Unemployment']['mean'].values)
sns.lineplot(unemployment2011['Unemployment']['mean'].index, unemployment2011['Unemployment']['mean'].values)
sns.lineplot(unemployment2012['Unemployment']['mean'].index, unemployment2012['Unemployment']['mean'].values)

plt.grid()
plt.xticks(np.arange(1, 53, step=1))
plt.legend(['2010', '2011', '2012'])
plt.show()


# In[ ]:


test['Year'].unique()


# In[ ]:


test['Unemployment'].max()


# In[ ]:


unemployment2012 = test.loc[test['Year']==2012].groupby(['Week']).agg({'Unemployment': ['mean']})
unemployment2013 = test.loc[test['Year']==2013].groupby(['Week']).agg({'Unemployment': ['mean']})
plt.figure(figsize=(20, 10))
sns.lineplot(unemployment2012['Unemployment']['mean'].index, unemployment2012['Unemployment']['mean'].values)
sns.lineplot(unemployment2013['Unemployment']['mean'].index, unemployment2013['Unemployment']['mean'].values)

plt.grid()
plt.xticks(np.arange(1, 53, step=1))
plt.legend(['2012', '2013'])
plt.show()


# Since the training data shows a tendency of decreasing in the Unemployment rate by the end of the year and the missing data in the test dataset is from the middle to the end of 2013, it makes sense to use the minimum unemployment rate to fill the missing values.

# In[ ]:


test_dataset['Unemployment']=test_dataset['Unemployment'].fillna(test_dataset['Unemployment'].min())


# ## Definition of the Weighted Mean Absolute Error function 
# It's the one that will be used to evaluate the selected model's performance.

# In[ ]:


def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)


# In[ ]:


X = train_dataset[['Store', 'Dept', 'Month', 'Week', 'IsHoliday', 'Unemployment','Size']]
y = train_dataset['Weekly_Sales']


# ## Model Selection
# Random Forest Regressor and XGBoost Regressor are commonly selected for forecasting tasks because of theirs state-of-the-art performance, so they will be evaluated as baseline models.

# In[ ]:


models = {'XGBoost': xgb.XGBRegressor(random_state = 42, objective = 'reg:tweedie'),
          'Random Forest': RandomForestRegressor(random_state = 42) }


# In[ ]:


model_stats = []


# In[ ]:


def baseline_model_evaluation (name, model, X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Model fit
    model.fit(X_train, y_train)
    # Model predict
    predicted = model.predict(X_test)
    # RMSE
    rmse = round(np.sqrt(metrics.mean_squared_error(y_test, predicted)),3)
    
    model_stats.append({'Model': name, 'RMSE': rmse})
    print(f'Model: {name} | RMSE: {rmse}')

    return pd.DataFrame(model_stats)


# In[ ]:


for name, model in models.items():
    baseline_model_evaluation(name=name, model=model, X=X, y=y)


# Since the Random Forest baseline outperformed the XGboost, it will be optimized.

# ## Model Optimization
# 
# It was created a function for testing different parameters in the Random Forest Regressor. This function allowed the selection of the combination that resulted in the lowest WMAE, since GridSearch or RandomizedSearch couldn't be used for hyperparameter tuning.

# In[ ]:


result = []


# In[ ]:


def random_forest_evaluation(n_estimators: int, max_depth: int, max_features: int, min_samples_split: int, min_samples_leaf: int):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    RF = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, 
                                           min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_jobs=6, random_state=42)
    RF.fit(X_train, y_train)
    predicted = RF.predict(X_test)
    error = WMAE(X_test, y_test, predicted)
    mae = metrics.mean_absolute_error(y_test, predicted)
    mse = metrics.mean_squared_error(y_test, predicted)
    mape = np.mean(np.abs((y_test - predicted) / np.abs(y_test)))
    rmse = np.sqrt(mse)
    print(f'N_estimators: {n_estimators} | Max_Depth {max_depth} | Max_Features {max_features} | Min_Samples_Leaf {min_samples_leaf} | Min_Samples_Split {min_samples_split} | WMAE: {error}')
    result.append({'N_estimators': n_estimators,'Max_Depth': max_depth, 'Max_Features': max_features, 'Min_Samples_Leaf': min_samples_leaf, 'Min_Samples_Split': min_samples_split, 'WMAE': error, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})
    return pd.DataFrame(result)


# In[ ]:


for estim in range(0,100):
    random_forest_evaluation(n_estimators=random.choice([20, 30, 40, 50, 60]), max_depth=random.choice(range(20,60,2)), max_features=7, min_samples_split = 2, min_samples_leaf = 2)


# In[ ]:


results = pd.DataFrame(result)
results.sort_values(by='WMAE', ascending=True)


# In[ ]:


X_test = test_dataset[['Store', 'Dept', 'Month', 'Week', 'IsHoliday', 'Unemployment','Size']]


# In[ ]:


best_params = results.sort_values(by='WMAE', ascending=True).head(1)
RF = RandomForestRegressor(n_estimators=int(best_params['N_estimators']), max_depth=int(best_params['Max_Depth']), max_features=int(best_params['Max_Features']), min_samples_leaf=int(best_params['Min_Samples_Leaf']), min_samples_split=int(best_params['Min_Samples_Split']), n_jobs=6, random_state=42)
RF.fit(X, y)
y_predict = RF.predict(X_test)
sample_sub['Weekly_Sales'] = y_predict
sample_sub.to_csv('submission_2.csv', index=False)


# In[ ]:


final_submission = pd.read_csv('submission_2.csv')
final_submission

