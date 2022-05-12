#!/usr/bin/env python
# coding: utf-8

# # ****DATA PREPROCESSING****`

# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset=pd.read_csv('../input/calcofi/bottle.csv')
dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.isnull().sum()/len(dataset) 


# In[ ]:


null_cols= dataset.columns[dataset.isnull().sum()/len(dataset) > .70]
dataset.drop(null_cols,axis=1,inplace=True)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.columns


# In[ ]:


dataset.dtypes


# In[ ]:


sns.boxplot('Salnty',data=dataset)


# In[ ]:


dataset['Salnty']=dataset['Salnty'].fillna(dataset['Salnty'].median())


# In[ ]:


dataset['Salnty'].isnull().sum()


# In[ ]:


sns.boxplot('O2ml_L',data=dataset)


# In[ ]:


print(dataset['O2ml_L'].mean())
print(dataset['O2ml_L'].median())


# In[ ]:


dataset['O2ml_L']=dataset['O2ml_L'].fillna(dataset['O2ml_L'].median())


# In[ ]:


dataset['O2ml_L'].isnull().sum()


# In[ ]:


sns.boxplot('STheta',data=dataset)


# In[ ]:


dataset['STheta'].value_counts(ascending =True)


# In[ ]:


dataset['STheta']=dataset['STheta'].fillna(dataset['STheta'].median())


# In[ ]:


dataset['STheta'].isnull().sum()


# In[ ]:


sns.boxplot('O2Sat',data=dataset)


# In[ ]:


dataset['O2Sat']=dataset['O2Sat'].fillna(dataset['O2Sat'].median())


# In[ ]:


dataset['O2Sat'].isnull().sum()


# In[ ]:


sns.boxplot('Oxy_µmol/Kg',data=dataset)


# In[ ]:


dataset['Oxy_µmol/Kg']=dataset['Oxy_µmol/Kg'].fillna(dataset['Oxy_µmol/Kg'].median())


# In[ ]:


sns.boxplot('T_prec',data=dataset)


# In[ ]:


print(dataset['T_prec'].median())
print(dataset['T_prec'].mode())


# In[ ]:


dataset['T_prec']=dataset['T_prec'].fillna(dataset['T_prec'].median())


# In[ ]:


dataset['T_prec'].isnull().sum()


# In[ ]:


sns.boxplot('S_prec',data=dataset)


# In[ ]:


print(dataset['S_prec'].mean())
print(dataset['S_prec'].median())
print(dataset['S_prec'].mode())


# In[ ]:


dataset['S_prec']=dataset['S_prec'].fillna(dataset['S_prec'].median())


# In[ ]:


dataset['S_prec'].isnull().sum()


# In[ ]:


dataset['P_qual']=dataset['P_qual'].fillna(dataset['P_qual'].median())


# In[ ]:


dataset['Chlqua']=dataset['Chlqua'].fillna(dataset['Chlqua'].median())


# In[ ]:


dataset['Phaqua']=dataset['Phaqua'].fillna(dataset['Phaqua'].median())


# In[ ]:


dataset['PO4uM']=dataset['PO4uM'].fillna(dataset['PO4uM'].median())


# In[ ]:


dataset['PO4q']=dataset['PO4q'].fillna(dataset['PO4q'].median())


# In[ ]:


dataset['SiO3uM']=dataset['SiO3uM'].fillna(dataset['SiO3uM'].median())


# In[ ]:


dataset['SiO3qu']=dataset['SiO3qu'].fillna(dataset['SiO3qu'].median())


# In[ ]:


dataset['NO2uM']=dataset['NO2uM'].fillna(dataset['NO2uM'].median())


# In[ ]:


dataset['NO2q']=dataset['NO2q'].fillna(dataset['NO2q'].median())


# In[ ]:


dataset['NO3uM']=dataset['NO3uM'].fillna(dataset['NO3uM'].median())


# In[ ]:


dataset['NO3q']=dataset['NO3q'].fillna(dataset['NO3q'].median())


# In[ ]:


dataset['NH3q']=dataset['NH3q'].fillna(dataset['NH3q'].median())


# In[ ]:


dataset['C14A1q']=dataset['C14A1q'].fillna(dataset['C14A1q'].median())


# In[ ]:


dataset['C14A2q']=dataset['C14A2q'].fillna(dataset['C14A2q'].median())


# In[ ]:


dataset['DarkAq']=dataset['DarkAq'].fillna(dataset['DarkAq'].median())


# In[ ]:


dataset['MeanAq']=dataset['MeanAq'].fillna(dataset['MeanAq'].median())


# In[ ]:


dataset['R_TEMP']=dataset['R_TEMP'].fillna(dataset['R_TEMP'].median())


# In[ ]:


dataset['R_POTEMP']=dataset['R_POTEMP'].fillna(dataset['R_POTEMP'].median())


# In[ ]:


dataset['R_SALINITY']=dataset['R_SALINITY'].fillna(dataset['R_SALINITY'].median())


# In[ ]:


dataset['R_SIGMA']=dataset['R_SIGMA'].fillna(dataset['R_SIGMA'].median())


# In[ ]:


dataset['R_SVA']=dataset['R_SVA'].fillna(dataset['R_SVA'].median())


# In[ ]:


dataset['R_DYNHT']=dataset['R_DYNHT'].fillna(dataset['R_DYNHT'].median())


# In[ ]:


dataset['R_O2']=dataset['R_O2'].fillna(dataset['R_O2'].median())


# In[ ]:


dataset['R_O2Sat']=dataset['R_O2Sat'].fillna(dataset['R_O2Sat'].median())


# In[ ]:


dataset['R_SIO3']=dataset['R_SIO3'].fillna(dataset['R_SIO3'].median())


# In[ ]:


dataset['R_PO4']=dataset['R_PO4'].fillna(dataset['R_PO4'].median())


# In[ ]:


dataset['R_NO3']=dataset['R_NO3'].fillna(dataset['R_NO3'].median())


# In[ ]:


dataset['R_NO2']=dataset['R_NO2'].fillna(dataset['R_NO2'].median())


# In[ ]:


dataset['T_degC']=dataset['T_degC'].fillna(dataset['T_degC'].median())


# In[ ]:


dataset.to_csv('clean_data.csv')


# In[ ]:


new_data=pd.read_csv('./clean_data.csv')
new_data.head()


# In[ ]:


new_data.isnull().sum()


# In[ ]:


col_to_drop = []
describe = dataset.describe()
for x in new_data.columns:
    if describe[x]['std']==0:
        col_to_drop.append(x)
col_to_drop


# In[ ]:


new_data.dtypes


# In[ ]:


new_data.drop('Unnamed: 0',axis=1,inplace=True)
new_data.head()


# In[ ]:


new_data.drop('Sta_ID',axis=1,inplace=True)
new_data.drop('Depth_ID',axis=1,inplace=True)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))         
sns.heatmap(new_data.corr(),linewidths=.5, ax=ax)


# In[ ]:


sns.distplot(new_data['T_degC'])


# In[ ]:


new_data.to_csv('Preprocessed_Data_By_Median.csv')


# 

# # ****LINEAR REGRESSION MODEL****

# In[ ]:


data=pd.read_csv('./Preprocessed_Data_By_Median.csv')
data.head()


# In[ ]:


data.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,15))         
sns.heatmap(data.corr(),linewidths=.5, ax=ax)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x=data.drop('T_degC',axis=1)
y=data['T_degC']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# In[ ]:


reg.intercept_


# In[ ]:


reg.coef_


# In[ ]:


coeff=pd.DataFrame(reg.coef_,x.columns,columns=['Coefficient'])
coeff


# In[ ]:


predictions=reg.predict(x_test)
predictions


# In[ ]:


predictions1=reg.predict(x_train)
predictions1


# In[ ]:


from sklearn import metrics


# In[ ]:


print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('MSE :', metrics.mean_squared_error(y_test,predictions))
print('MAE :', metrics.mean_absolute_error(y_test,predictions))


# In[ ]:


from statsmodels.regression.linear_model import OLS
import statsmodels.regression.linear_model as sf


# In[ ]:


reg_model = sf.OLS(endog = y_train, exog = x_train).fit()
reg_model


# In[ ]:


reg_model.summary()


# In[ ]:


plt.scatter(y_test, predictions)


# In[ ]:


sns.distplot((y_test - predictions), bins=50)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test, predictions)


# In[ ]:


r2_score(y_train, predictions1)


# # ****DECISION TREE REGRESSOR MODEL****

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


data1=pd.read_csv('Preprocessed_Data_By_Median.csv')
data1.head()


# In[ ]:


data1.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


x1=data1.drop('T_degC',axis=1)
y1=data1['T_degC']


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(x1,y1,train_size=0.75,random_state=501)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[ ]:


dt_model = DecisionTreeRegressor()
dt_model.fit(xtrain, ytrain)


# In[ ]:


y_pred_train=dt_model.predict(xtrain)
y_pred_test=dt_model.predict(xtest)


# In[ ]:


print(r2_score(ytrain,y_pred_train))
print('\n')
print(r2_score(ytest,y_pred_test))


# # ****RANDOM FOREST REGRESSOR MODEL****

# In[ ]:


data_=pd.read_csv('Preprocessed_Data_By_Median.csv')
data_.head()


# In[ ]:


new_data1=data_.drop('Unnamed: 0',axis=1)
new_data1


# In[ ]:


new_data1.isnull().sum()


# In[ ]:


x_=new_data1.drop('T_degC',axis=1)
y_=new_data1['T_degC']


# In[ ]:


xtrain_,xtest_,ytrain_,ytest_=train_test_split(x_,y_,test_size=0.3,random_state=555)
print(xtrain_.shape)
print(xtest_.shape)
print(ytrain_.shape)
print(ytest_.shape)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf_reg=RandomForestRegressor()


# In[ ]:


rf_reg.fit(xtrain_,ytrain_)


# In[ ]:


ypred_train_=rf_reg.predict(xtrain_)
ypred_test_=rf_reg.predict(xtest_)


# In[ ]:


print(r2_score(ytrain_,ypred_train_))
print('\n')
print(r2_score(ytest_,ypred_test_))

