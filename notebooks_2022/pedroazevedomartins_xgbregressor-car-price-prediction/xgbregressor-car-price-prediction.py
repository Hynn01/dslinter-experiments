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
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


full_train = pd.read_csv("../input/task-02-car-price-prediction/train_car_details.csv")


# # Data Analysis

# In[ ]:


full_train.head(5)


# In[ ]:


full_train.info()


# Categorical values is found in object types columns
# Also, the target is unsualy at colmumn #3

# In[ ]:


full_train.describe(include = "all")


# It's decribed to have some missing values

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
#f, ax = plt.subplots(figsize=(16, 16))
sns.displot(full_train.get("selling_price"), kde=False)
plt.show()


# # Categorical and missing Values

# In[ ]:


miliage = []
for item in full_train['mileage']:
    if type(item) == str: # check if contains a value. Not null
        x = item.split('km') # split the string
        miliage.append(float(x[0])) #after the split the numerical values comes first
    else: miliage.append(0) # replace nan value to zero

avg = sum(miliage) / len(miliage)  # replace the zero values (former NaN) to the average. Imputation approach
for i in range(len(miliage)):
    if miliage[i] == 0: miliage[i] = avg

#print(miliage)

full_train = full_train.drop('mileage', 1) # replace the old columns with the proper treated one
df1 = pd.DataFrame (miliage, columns = ['miliage'])
full_train = full_train.join(df1)


# In[ ]:


max_power = []  #same code as miliage, however try excepet was neede as there was some values that contains no numerical data
for item in full_train['max_power']:
    if type(item) == str and item != 0:
        x = item.split('b')
        try:
            max_power.append(float(x[0]))
        except: max_power.append(0)
    else: max_power.append(0)


avg = sum(max_power) / len(max_power)
for i in range(len(max_power)):
    if max_power[i] == 0: max_power[i] = avg

#print(max_power)

full_train = full_train.drop('max_power', 1)
df1 = pd.DataFrame (max_power, columns = ['max_power'])
full_train = full_train.join(df1)
full_train.head(3)


# In[ ]:


torque = []
for item in full_train['torque']:
    if type(item) == str and item != 0:
        x = item.split('N')
        try:
            torque.append(float(x[0]))
        except: torque.append(0)
    else: torque.append(0)


avg = sum(torque) / len(torque)
for i in range(len(torque)):
    if torque[i] == 0: torque[i] = avg

#print(torque)

full_train = full_train.drop('torque', 1)
df1 = pd.DataFrame (torque, columns = ['torque'])
full_train = full_train.join(df1)
full_train.head(3)


# In[ ]:


engine = []
for item in full_train['engine']:
    if type(item) == str and item != 0:
        x = item.split('C')
        try:
            engine.append(float(x[0]))
        except: engine.append(0)
    else: engine.append(0)


avg = sum(engine) / len(engine)
for i in range(len(engine)):
    if engine[i] == 0: engine[i] = avg

#print(engine)

full_train = full_train.drop('engine', 1)
df1 = pd.DataFrame (engine, columns = ['engine'])
full_train = full_train.join(df1)
full_train.head(3)


# In[ ]:


seats = []
for item in full_train['seats']:
    if item > 0:
        seats.append(item)
    else: seats.append(4)
        
full_train = full_train.drop('seats', 1)
df1 = pd.DataFrame (seats, columns = ['seats'])
full_train = full_train.join(df1)
full_train.head(3)


# In[ ]:


full_train.isnull().sum()


# # Ordinal Enconder

# In[ ]:


full_train.info()


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder  #Encoding Categorical data into ordinal numbers

s = (full_train.dtypes == 'object')
object_cols = list(s[s].index)

df = full_train

ordinal_encoder = OrdinalEncoder()
df[object_cols] = ordinal_encoder.fit_transform(df[object_cols])

df.head(2)


# # Train Test Split

# In[ ]:


y = df['selling_price']

X_features = ['name','year','km_driven','fuel','transmission','owner','miliage','max_power','torque','engine','seats']
X = df[X_features]


# In[ ]:


from sklearn.model_selection import train_test_split


trainX, valX, trainy, valy = train_test_split(X, y,
                                              test_size=0.3, 
                                              random_state=12)
valX.shape


# # Model training

# In[ ]:


import xgboost
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(trainX, trainy)


# In[ ]:


y_preds = xgb_reg.predict(valX)
y_preds


# In[ ]:


from sklearn.linear_model import Lasso

lassomodel = Lasso(alpha=0.1, 
              precompute=True, 
#               warm_start=True, 
              positive=True, 
              selection='random',
              random_state=42)
lassomodel.fit(trainX, trainy)


# In[ ]:


LassoPred = lassomodel.predict(valX)


# # PCA

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principalcomponent1', 'principalcomponent2'])
principalDf


# In[ ]:


# PCA Split

PCAtrainX, PCAvalX, PCAtrainy, PCAvaly = train_test_split(principalDf, y,
                                              test_size=0.3, 
                                              random_state=12)


# In[ ]:


xgb_reg_PCA = xgboost.XGBRegressor()
xgb_reg_PCA.fit(PCAtrainX, PCAtrainy)
PCA_preds = xgb_reg_PCA.predict(PCAvalX)
PCA_preds


# # Metrics

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

print("XGBoost Mean Absolute Error: " + str(mean_absolute_error(y_preds, valy)))
print("XGBoost R²: " + str(r2_score(y_preds, valy)))

print("Lasso Mean Absolute Error: " + str(mean_absolute_error(LassoPred, valy)))
print("Lasso R²: " + str(r2_score(LassoPred, valy)))

print("XGBoost w/ PCA Mean Absolute Error: " + str(mean_absolute_error(PCA_preds, PCAvaly)))
print("XGBosst w/ PCA R²: " + str(r2_score(PCA_preds, PCAvaly)))


# # Complete Model

# In[ ]:


final_model = xgboost.XGBRegressor()


""""final_model = Lasso(alpha=0.1, 
              precompute=True, 
#               warm_start=True, 
              positive=True, 
              selection='random',
              random_state=42)  """


final_model.fit(X, y)


# # Final test

# In[ ]:


dftest = pd.read_csv('../input/task-02-car-price-prediction/test_car_details.csv')
dftest.info()


# In[ ]:


miliage = []
for item in dftest['mileage']:
    if type(item) == str: # check if contains a value. Not null
        x = item.split('km') # split the string
        miliage.append(float(x[0])) #after the split the numerical values comes first
    else: miliage.append(0) # replace nan value to zero

avg = sum(miliage) / len(miliage)  # replace the zero values (former NaN) to the average. Imputation approach
for i in range(len(miliage)):
    if miliage[i] == 0: miliage[i] = avg
        
max_power = []  #same code as miliage, however try excepet was neede as there was some values that contains no numerical data
for item in dftest['max_power']:
    if type(item) == str and item != 0:
        x = item.split('b')
        try:
            max_power.append(float(x[0]))
        except: max_power.append(0)
    else: max_power.append(0)


avg = sum(max_power) / len(max_power)
for i in range(len(max_power)):
    if max_power[i] == 0: max_power[i] = avg
        
torque = []
for item in dftest['torque']:
    if type(item) == str and item != 0:
        x = item.split('N')
        try:
            torque.append(float(x[0]))
        except: torque.append(0)
    else: torque.append(0)


avg = sum(torque) / len(torque)
for i in range(len(torque)):
    if torque[i] == 0: torque[i] = avg
        
engine = []
for item in dftest['engine']:
    if type(item) == str and item != 0:
        x = item.split('C')
        try:
            engine.append(float(x[0]))
        except: engine.append(0)
    else: engine.append(0)


avg = sum(engine) / len(engine)
for i in range(len(engine)):
    if engine[i] == 0: engine[i] = avg
        
dftest = dftest.drop(['mileage', 'max_power', 'torque', 'engine'], 1)
dftest.info()


# In[ ]:


df1 = pd.DataFrame (miliage, columns = ['miliage'])
dftest = dftest.join(df1)
df1 = pd.DataFrame (max_power, columns = ['max_power'])
dftest = dftest.join(df1)
df1 = pd.DataFrame (torque, columns = ['torque'])
dftest = dftest.join(df1)
df1 = pd.DataFrame (engine, columns = ['engine'])
dftest = dftest.join(df1)
dftest.info()


# In[ ]:


s = (dftest.dtypes == 'object')
object_cols = list(s[s].index)


ordinal_encoder = OrdinalEncoder()
dftest[object_cols] = ordinal_encoder.fit_transform(dftest[object_cols])
dftest.info()


# In[ ]:


X_test = dftest[X_features]
X_test


# In[ ]:


preds = final_model.predict(X_test)
preds


# In[ ]:


preds = pd.DataFrame(preds, columns = ['selling_price'])
dftest = dftest.join(preds)
dftest


# In[ ]:


submit = dftest[['Id', 'selling_price']]
submit.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:


submit

