#!/usr/bin/env python
# coding: utf-8

# # linear regression from scratch, Ridge, Lasso

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn import metrics
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


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


df_org = pd.read_csv("/kaggle/input/apartment-rental-offers-in-germany/immo_data.csv", 
                  usecols=['serviceCharge', 'heatingType', 'telekomUploadSpeed', 'totalRent']
                )


# In[ ]:


df = df_org.copy()


# In[ ]:


df.columns


# In[ ]:


df.head(2)


# In[ ]:


df.shape


# In[ ]:


df.info()


# # Cleansing

# In[ ]:


for col in df.columns:
    print(col," =>", df[col].isnull().sum()/len(df[col])*100)


# In[ ]:


df.serviceCharge.fillna(df.serviceCharge.median(), inplace=True)
df.heatingType.fillna(df.heatingType.mode()[0], inplace=True)
df.telekomUploadSpeed.fillna(df.telekomUploadSpeed.median(), inplace=True)
df.totalRent.fillna(df.totalRent.median(), inplace=True)


# In[ ]:


df.describe()


# # Outliers

# In[ ]:


# Return unbiased kurtosis over requested axis using Fisher's definition of kurtosis (kurtosis of normal == 0.0).
# The result is normalized by N-1
kurt = df.kurt(numeric_only=True)[:]
kurt


# In[ ]:


outliers = ['serviceCharge', 'totalRent']


# In[ ]:


for col in outliers:
    plt.figure(figsize=(4, 4))
    df.boxplot(column=[col])


# In[ ]:


def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))


# In[ ]:


for col in outliers:
    print(f'{col}: {len(outliers_iqr(df[col])[0])}')


# In[ ]:


def outliers_z_score(ys):
    threshold = 3
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)


# In[ ]:


for col in outliers:
    print(f'{col}: {len(outliers_z_score(df[col])[0])}')


# In[ ]:


for i in outliers:
    quartile_1, quartile_3 = np.percentile(df[i], [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    df = df[df[i]<upper_bound]
    df = df[df[i]>lower_bound]


# In[ ]:


#for i in outliers:
#     mean_y = np.mean(df[i])
#     stdev_y = np.std(df[i])
#     lower_bound = mean_y - (3 * stdev_y)
#     upper_bound = mean_y + (3 * stdev_y)
#     df = df[df[i]<upper_bound]
#     df = df[df[i]>lower_bound]


# # Feature engineering

# In[ ]:


df.heatingType.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.heatingType.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("heatingType")
plt.ylabel("heatingType")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# In[ ]:


def filter(x):
    if x in [
    'oil_heating', 'combined_heat_and_power_plant', 'heat_pump', 'night_storage_heater', 
    'wood_pellet_heating', 'electric_heating', 'stove_heating', 'solar_heating'
] :
        x = 'other'
        return x
    else:
        return x
df.heatingType = df.heatingType.apply(filter)


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.heatingType.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("heatingType")
plt.ylabel("heatingType")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# In[ ]:


# Visualizing the distribution for every "feature"
df.hist(edgecolor="black", linewidth=1.2, figsize=(20, 20))
plt.show()


# In[ ]:


df.corr()


# In[ ]:


df_copy = df.copy()


# In[ ]:


df_copy['serviceCharge-2'] = df.serviceCharge ** 2
df_copy['telekomUploadSpeed-2'] = df.telekomUploadSpeed ** 2
df_copy['serviceCharge-3'] = df.serviceCharge ** 3
df_copy['telekomUploadSpeed-3'] = df.telekomUploadSpeed ** 3


# In[ ]:


df_copy.corr()


# In[ ]:


for col in ["serviceCharge", "serviceCharge-2"]:
    plt.figure(figsize=(15,8))
    sns.scatterplot(x="totalRent", y=col, data=df_copy)


# In[ ]:


df['serviceCharge'] = df.serviceCharge ** 2


# # Preprocessing

# In[ ]:


df.columns


# In[ ]:


y = df.pop('totalRent')
X = df


# In[ ]:


X.columns


# In[ ]:


y = pd.DataFrame(data=y, columns=['totalRent'])


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

y_train.head(3)


# In[ ]:


ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ss = StandardScaler()


# In[ ]:


preprocessor_x = ColumnTransformer(
    transformers=[
            ('serviceCharge', ss, ['serviceCharge']),
            ('telekomUploadSpeed', ss, ['telekomUploadSpeed']),
            ('heatingType', ohe, ['heatingType']),       
])
preprocessor_y = ColumnTransformer(
    transformers=[
            ('totalRent', ss, ['totalRent']),       
])


# In[ ]:


fitter_y = preprocessor_y.fit(y_train)
fitter_x = preprocessor_x.fit(X_train)


# In[ ]:


y_train = fitter_y.transform(y_train)
y_test = fitter_y.transform(y_test)
X_train = fitter_x.transform(X_train)
X_test = fitter_x.transform(X_test)


# In[ ]:


m, n = y_train.shape
x, z = y_test.shape
y_train = y_train.reshape(m, )
y_test = y_test.reshape(x, )


# In[ ]:


print(f'train x shape: {X_train.shape}')
print(f'train y shape: {y_train.shape}')
print(f'test x shape: {X_test.shape}')
print(f'test y shape: {y_test.shape}')


# # Scratch

# In[ ]:


# Linear reggression
class Net() :   
    def __init__( self, learning_rate=0.01 , iterations=1000 , method='mse') :
        self.learning_rate = learning_rate
        self.iterations = iterations 
        self.method = method

    
    # Function for model training        
    def fit( self, X, Y ) : 
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape 
        # weight initialization 
        self.W = np.random.normal(loc=0.0, scale=0.001, size=self.n)
        self.b = 0 
        self.X = X
        self.Y = Y 
        # gradient descent learning         
        for i in range( self.iterations ) :
            self.update_weights(X, Y)  
        return self
    
    
    # Helper function to update weights in gradient descent 
    def update_weights( self, X, Y ) :
        Y_pred = self.predict( X )
        # calculate gradients
        
        # Linear reggression with minimum square error
        if self.method == 'mse':
            error = (Y - Y_pred)
            dW = - ( 2 * ( X.T ).dot( error )  ) / self.m
            db = - 2 * np.sum( error ) / self.m
    
            
        # Linear reggression with minimum absolute error
        elif self.method == 'ae':
            error = abs(Y - Y_pred)
            error = np.where(error == 0, 0.01, error)
            dW = ( np.sum( X.T, axis=1)/ 4)
            print(dW.shape)
            db = np.sum( error ) / 4
        
        # Linear reggression with Epsilon Sensitive Error
        elif self.method == 'ese':
            threshold = 0.00001
            error = abs(Y - Y_pred)
            error = np.where(error < threshold, 0.01, error)
            dW = ( ( X.T ).dot( error )  )
            db = np.sum( error ) 
        
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db 
        return self
    
    # Hypothetical function  h( x )  
    def predict( self, X ) :
        return (X.dot( self.W ) + self.b)


# # Linear reggression with minimum square error

# In[ ]:


model_mse = Net(method='mse')


# In[ ]:


model_mse.fit(X_train, y_train)


# In[ ]:


print(model_mse.W)
print(model_mse.b)
y_pre_test_mse = model_mse.predict(X_test)
print('linear reggression from scrach (MSE) R^2: test', metrics.r2_score(y_test, y_pre_test_mse))


# # Linear reggression with Absolute Error

# In[ ]:


model_ae = Net(method='ae')


# In[ ]:


#model_ae.fit(X_train, y_train)


# In[ ]:


# print(model_ae.W)
# print(model_ae.b)
# y_pre_test_ae = model_ae.predict(X_test)
# y_pre_test_ae = y_pre_test_ae.reshape(-1, 1)
# print('linear reggression from scrach (ae) R^2: test', metrics.r2_score(y_test, y_pre_test_ae))


# # Linear reggression with Epsilon Sensitive Error

# In[ ]:


# Linear reggression with Epsilon Sensitive Error
model_ese = Net(method='ese')


# In[ ]:


# model_ese.fit(X_train, y_train)


# In[ ]:


# print(model_ese.W)
# print(model_ese.b)
# y_pre_test_ese = model_ese.predict(X_test)
# y_pre_test_ese = y_pre_test_ese.reshape(-1, 1)
# print('linear reggression from scrach (ese) R^2: test', metrics.r2_score(y_test, y_pre_test_ese))


# # Sklearn pakage

# # LinearRegression

# In[ ]:


model_lr = LinearRegression()


# In[ ]:


model_lr.fit(X_train, y_train)


# In[ ]:


y_pred_test = model_lr.predict(X_test)


# In[ ]:


model_lr.coef_


# In[ ]:


print('linear reggression score train:', model_lr.score(X_train, y_train))
print('linear reggression R^2: test', metrics.r2_score(y_test, y_pred_test))


# # Ridge

# In[ ]:


model_ridge = Ridge(alpha=1)


# In[ ]:


model_ridge.fit(X_train, y_train)


# In[ ]:


y_pred_test_ridge = model_ridge.predict(X_test)


# In[ ]:


model_ridge.coef_


# In[ ]:


print('ridge reggression score train:', model_ridge.score(X_train, y_train))
print('ridge reggression R^2: test', metrics.r2_score(y_test, y_pred_test_ridge))


# # Lasso

# In[ ]:


model_lasso = Lasso(alpha=0.1)


# In[ ]:


model_lasso.fit(X_train, y_train)


# In[ ]:


y_pred_test_lasso = model_lasso.predict(X_test)


# In[ ]:


model_lasso.coef_


# In[ ]:


print('lasso reggression score train:', model_lasso.score(X_train, y_train))
print('lasso reggression R^2: test', metrics.r2_score(y_test, y_pred_test_lasso))


# # Only servicecharge

# In[ ]:


X2 = X['serviceCharge']


# In[ ]:


y = pd.DataFrame(data=y, columns=['totalRent'])
X2 = pd.DataFrame(data=X2, columns=['serviceCharge'])


# In[ ]:


X_train2,X_test2,y_train2,y_test2 = train_test_split(X2,y,test_size=0.2)
ss = StandardScaler()
preprocessor_x2 = ColumnTransformer(
    transformers=[
            ('serviceCharge', ss, ['serviceCharge']),
])
preprocessor_y2 = ColumnTransformer(
    transformers=[
            ('totalRent', ss, ['totalRent']),       
])
fitter_y2 = preprocessor_y2.fit(y_train2)
fitter_x2 = preprocessor_x2.fit(X_train2)
y_train2 = fitter_y2.transform(y_train2)
y_test2 = fitter_y2.transform(y_test2)
X_train2 = fitter_x2.transform(X_train2)
X_test2 = fitter_x2.transform(X_test2)


# In[ ]:


model_lr_ser = LinearRegression()


# In[ ]:


model_lr_ser.fit(X_train2, y_train2)


# In[ ]:


y_pred_test2 = model_lr_ser.predict(X_test2)


# In[ ]:


model_lr_ser.coef_


# In[ ]:


print('linear reggression serviceCharge score train:', model_lr_ser.score(X_train2, y_train2))
print('linear reggression serviceCharge R^2: test', metrics.r2_score(y_test2, y_pred_test2))


# # DecisionTreeRegressor

# In[ ]:


tree = DecisionTreeRegressor()


# In[ ]:


tree.fit(X_train, y_train)


# In[ ]:


y_pred_tree = tree.predict(X_test)


# In[ ]:


print('DecisionTreeRegressor: train', tree.score(X_train, y_train))
print('DecisionTreeRegressor R^2: test', metrics.r2_score(y_test, y_pred_tree))

