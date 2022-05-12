#!/usr/bin/env python
# coding: utf-8

# ## Libraries Used

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_theme(style="darkgrid")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression

from sklearn import metrics


# ### Reading the Data

# In[ ]:


pd.set_option("display.max_colwidth", 200)

train = pd.read_csv('../input/bootcamp-challenge-1-used-cars-price-prediction/train.csv').drop('unique_id', axis=1)
train.head()


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


train.describe(include='all')


# ## Data Cleaning

# In[ ]:


train["Mileage"] = train["Mileage"].str.rstrip(" kmpl")
train["Mileage"] = train["Mileage"].str.rstrip(" km/g")
train["Engine"] = train["Engine"].str.rstrip(" CC")
train["Power"] = train["Power"].str.rstrip(" bhp")
train["Power"] = train["Power"].replace(regex="null",value=np.nan)

train.head()


# In[ ]:


train["Mileage"] = train["Mileage"].astype("float")
train["Power"] = train["Power"].astype("float")
train["Engine"] = train["Engine"].astype("float")


# In[ ]:


train.isnull().sum()


# In[ ]:


train.drop(columns="New_Price",inplace=True)

train['Mileage'].fillna(train['Mileage'].mode()[0],inplace=True)
train['Engine'].fillna(train['Engine'].median(),inplace=True)
train['Power'].fillna(train['Power'].mean(),inplace=True)
train['Seats'].fillna(train['Seats'].mode()[0],inplace=True)

print(train.head())
print(train.shape)


# ## Feature Engineering

# In[ ]:


train["Company"] = train["Name"].str.split(" ").str[0]
train["Model"] = train["Name"].str.split(" ").str[1]+train["Name"].str.split(" ").str[2]

del train['Name']

train.head()


# ## Data Visualization

# In[ ]:


train['Location'].value_counts()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='Location',data=train.sort_values('Location'),palette='Set2')
plt.title('Car Usage across Cities')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='Year',data=train.sort_values('Year'),palette='Set2')
plt.title('Yearly Car Usage')
plt.show()


# In[ ]:


sns.scatterplot(x="Kilometers_Driven",y="Price",data=train)
plt.plot()


# In[ ]:


#Removing outlier in Kilometers Driven

train.drop(train[train['Kilometers_Driven'] >= 6000000].index, axis=0, inplace=True)


# In[ ]:


#Fuel_Type consisted of Diesel, Petrol, CNG, LPG, Electric.
#The later 3 have a very few counts in the data, so we combined them and name them Clean_Fuel

train['Fuel_Type'] = train['Fuel_Type'].apply(lambda x: "Clean_Fuel" if x not in ['Diesel', 'Petrol'] else x)


# In[ ]:


train['Fuel_Type'].value_counts()


# In[ ]:


x = [train['Fuel_Type'].value_counts()[0],
     train['Fuel_Type'].value_counts()[1],
     train['Fuel_Type'].value_counts()[2]
    ]
labels=['Diesel','Petrol','Clean_Fuel']
pie_colors = ['tab:orange', 'tab:blue', 'tab:red']

plt.figure(figsize=(5,5))
plt.pie(x=x, autopct="%.1f%%", startangle=90, labels=labels, colors=pie_colors, wedgeprops={'edgecolor':'black', 'linewidth':1})
plt.title('Types of Fuel')


# In[ ]:


train['Transmission'].value_counts()


# In[ ]:


x = [train['Transmission'].value_counts()[0],
     train['Transmission'].value_counts()[1]
    ]
labels=['Manual','Automatic']
explode=[0.1,0.1]
pie_colors = ['tab:orange', 'tab:blue']

plt.figure(figsize=(5,5))
plt.pie(x=x, autopct="%.1f%%", startangle=90, explode=explode, labels=labels, colors=pie_colors, wedgeprops={'edgecolor':'black', 'linewidth':1})
plt.title('Types of Transmission')


# In[ ]:


train['Owner_Type'].value_counts()


# In[ ]:


x = [train['Owner_Type'].value_counts()[0],
     train['Owner_Type'].value_counts()[1],
     train['Owner_Type'].value_counts()[2],
     train['Owner_Type'].value_counts()[3]
    ]
labels=['First','Second','Third','Fourth']
explode=[0.1,0,0,0]
pie_colors = ['tab:orange', 'tab:blue', 'tab:red', 'tab:green', 'tab:brown']

plt.figure(figsize=(5,5))
plt.pie(x=x, autopct="%.1f%%", startangle=90, labels=labels, explode=explode, colors=pie_colors, wedgeprops={'edgecolor':'black', 'linewidth':1})
plt.title('Types of Fuel')


# In[ ]:


plt.figure()
train['Mileage'].plot(kind='box')
plt.show()


# In[ ]:


train.isin([0]).sum()


# In[ ]:


train['Mileage'].mode()


# In[ ]:


#Removing Outliers for Mileage

train["Mileage"].replace({0.0:17.0},inplace=True)


# In[ ]:


#Dropping a row with 0 seats

train.drop(train[train['Seats'] == 0].index, axis=0, inplace=True)


# In[ ]:


plt.figure()
train['Engine'].plot(kind='box')
plt.show()


# In[ ]:


plt.figure()
train['Power'].plot(kind='box')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='Seats',data=train,palette='Set2')
plt.title('Number of Seats')
plt.show()


# In[ ]:


train['Company'].value_counts()


# In[ ]:


plt.figure(figsize=(25,5))
sns.countplot(x='Company',data=train,palette='Set2')
plt.title('Car Companies')
plt.show()


# In[ ]:


sns.distplot(train['Price'])
print("Skewness: %f" % train['Price'].skew())
print("Kurtosis: %f" % train['Price'].kurt())


# #### Correlation Matrix

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train.corr(),cmap='coolwarm',annot=True)
plt.show()


# In[ ]:


train.head()


# ## Standard Scaling

# In[ ]:


scaler = StandardScaler()

train['Mileage'] = scaler.fit_transform(train['Mileage'].values.reshape(-1,1))
train['Engine'] = scaler.fit_transform(train['Engine'].values.reshape(-1,1))
train['Power'] = scaler.fit_transform(train['Power'].values.reshape(-1,1))

train.head()


# ## Label Encoding

# In[ ]:


le=LabelEncoder()

train['Location']=le.fit_transform(train['Location'])
train['Fuel_Type']=le.fit_transform(train['Fuel_Type'])
train['Transmission']=le.fit_transform(train['Transmission'])
train['Owner_Type']=le.fit_transform(train['Owner_Type'])
train['Company']=le.fit_transform(train['Company'])
train['Model']=le.fit_transform(train['Model'])

train.head()


# ## Train-Test Split

# In[ ]:


X = train[['Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats','Company','Model']]
y = train[['Price']]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25)


# In[ ]:


print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# ## Feature Importance

# In[ ]:


selection= ExtraTreesRegressor()

selection.fit(X_train,y_train)

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[ ]:


model=XGBRegressor()

model.fit(X_train,y_train)

plt.figure(figsize=(12,8))
importance=pd.Series(model.feature_importances_, index=X_train.columns)
importance.nlargest(20).plot(kind='barh')
plt.show()


# ## Linear Regression

# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


lr_pred = lr.predict(X_test)


# In[ ]:


print("Accuracy on Training set: %.2f " % lr.score(X_train,y_train))
print("Accuracy on Testing set: %.2f" % lr.score(X_test,y_test))

print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, lr_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, lr_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, lr_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, lr_pred))


# In[ ]:


plt.title('Actual Vs Predicted for Linear Regression Model')
plt.scatter(y_test,lr_pred,c='b',marker='.',s=36)
plt.plot([0, 50], [0, 50], 'r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# ## Random Forest Regressor

# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)


# In[ ]:


rf_pred = rf.predict(X_test)


# In[ ]:


print("Accuracy on Training set: %.2f " % rf.score(X_train,y_train))
print("Accuracy on Testing set: %.2f" % rf.score(X_test,y_test))

print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, rf_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, rf_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, rf_pred))


# In[ ]:


plt.title('Actual Vs Predicted for Random Forest Regressor')
plt.scatter(y_test,rf_pred,c='b',marker='.',s=36)
plt.plot([0, 50], [0, 50], 'r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# ## XGB Regressor

# In[ ]:


xgb=XGBRegressor()
xgb.fit(X_train, y_train)


# In[ ]:


xgb_pred = xgb.predict(X_test)


# In[ ]:


print("Accuracy on Training set: %.2f " % xgb.score(X_train,y_train))
print("Accuracy on Testing set: %.2f" % xgb.score(X_test,y_test))

print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, xgb_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, xgb_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, xgb_pred))


# In[ ]:


plt.title('Actual Vs Predicted for XGB Regressor')
plt.scatter(y_test,xgb_pred,c='b',marker='.',s=36)
plt.plot([0, 50], [0, 50], 'r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# ## Bagging Regressor

# In[ ]:


br = BaggingRegressor(base_estimator=RandomForestRegressor())
br.fit(X_train, y_train)


# In[ ]:


br_pred = br.predict(X_test)


# In[ ]:


print("Accuracy on Training set: %.2f " % br.score(X_train,y_train))
print("Accuracy on Testing set: %.2f" % br.score(X_test,y_test))

print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, br_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, br_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, br_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, br_pred))


# In[ ]:


plt.title('Actual Vs Predicted for Bagging Regressor')
plt.scatter(y_test,br_pred,c='b',marker='.',s=36)
plt.plot([0, 50], [0, 50], 'r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# ## HyperTuned Model

# In[ ]:


rf_reg = RandomForestRegressor(random_state=42,n_estimators=400,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=None,bootstrap=False)
rf_reg.fit(X_train, y_train)


# In[ ]:


pred = rf_reg.predict(X_test)


# In[ ]:


print("Accuracy on Training set: %.2f " % rf_reg.score(X_train,y_train))
print("Accuracy on Testing set: %.2f" % rf_reg.score(X_test,y_test))

print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, pred))


# In[ ]:


plt.title('Actual Vs Predicted for HyperTuned Model')
plt.scatter(y_test,pred,c='b',marker='.',s=36)
plt.plot([0, 50], [0, 50], 'r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# ## Preparing Test Data

# In[ ]:


test = pd.read_csv('../input/bootcamp-challenge-1-used-cars-price-prediction/test.csv').drop('unique_id', axis=1)
test.head()


# In[ ]:


test.shape


# In[ ]:


test.info()


# In[ ]:


test.describe(include='all')


# In[ ]:


test["Mileage"] = test["Mileage"].str.rstrip(" kmpl")
test["Mileage"] = test["Mileage"].str.rstrip(" km/g")
test["Engine"] = test["Engine"].str.rstrip(" CC")
test["Power"] = test["Power"].str.rstrip(" bhp")
test["Power"] = test["Power"].replace(regex="null",value=np.nan)

test.head()


# In[ ]:


test["Mileage"] = test["Mileage"].astype("float")
test["Power"] = test["Power"].astype("float")
test["Engine"] = test["Engine"].astype("float")


# In[ ]:


test.isnull().sum()


# In[ ]:


test.drop(columns="New_Price",inplace=True)

test['Engine'].fillna(test['Engine'].mean(),inplace=True)
test['Power'].fillna(test['Power'].mean(),inplace=True)
test['Seats'].fillna(test['Seats'].mode()[0],inplace=True)

print(test.head())
print(test.shape)


# In[ ]:


test["Company"] = test["Name"].str.split(" ").str[0]
test["Model"] = test["Name"].str.split(" ").str[1]+test["Name"].str.split(" ").str[2]

del test['Name']

test.head()


# In[ ]:


test['Fuel_Type'] = test['Fuel_Type'].apply(lambda x: "Clean_Fuel" if x not in ['Diesel', 'Petrol'] else x)
test.head()


# In[ ]:


test["Mileage"].replace({0.0:17.0},inplace=True)


# In[ ]:


test['Mileage'] = scaler.fit_transform(test['Mileage'].values.reshape(-1,1))
test['Engine'] = scaler.fit_transform(test['Engine'].values.reshape(-1,1))
test['Power'] = scaler.fit_transform(test['Power'].values.reshape(-1,1))

test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


le=LabelEncoder()

test['Location']=le.fit_transform(test['Location'])
test['Fuel_Type']=le.fit_transform(test['Fuel_Type'])
test['Transmission']=le.fit_transform(test['Transmission'])
test['Owner_Type']=le.fit_transform(test['Owner_Type'])
test['Company']=le.fit_transform(test['Company'])
test['Model']=le.fit_transform(test['Model'])

test.head()


# In[ ]:


test.shape


# In[ ]:


XX = test[['Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats','Company','Model']]


# In[ ]:


rf_reg = RandomForestRegressor(random_state=42,n_estimators=400,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=None,bootstrap=False)
rf_reg.fit(X_train, y_train)


# In[ ]:


pred = rf_reg.predict(XX)


# ## Final Submission

# In[ ]:


df_submit = pd.read_csv('../input/bootcamp-challenge-1-used-cars-price-prediction/sample_submission.csv', index_col=0)
df_submit['Price'] = pred
df_submit.to_csv('sample_submission.csv',index=True)

