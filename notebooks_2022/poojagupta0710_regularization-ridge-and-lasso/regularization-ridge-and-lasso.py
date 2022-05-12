#!/usr/bin/env python
# coding: utf-8

# ## Reference Link 
# 
# * **Comprehensive beginners guide for Linear, Ridge and Lasso Regression -** https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/
# * **A Complete Tutorial on Ridge and Lasso Regression in Python -** https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial
# 
# * **Lasso and Ridge Regularization -** https://medium.com/all-about-ml/lasso-and-ridge-regularization-a0df473386d5
# * **7 Regression Techniques you should know -** https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/
# 

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


#Importing libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#Defining independent variable as angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(10,360,3)])


# In[ ]:


#Setting seed for reproducability
np.random.seed(10)  


# In[ ]:


#Defining the target/dependent variable as sine of the independent variable
y = np.sin(x) + np.random.normal(0,0.15,len(x))


# In[ ]:


#Creating the dataframe using independent and dependent variable
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])


# In[ ]:


data.shape


# In[ ]:


#Printing first 5 rows of the data
data.head()


# In[ ]:


#Plotting the dependent and independent variables
plt.figure(figsize=(12,8))
plt.plot(data['x'],data['y'],'.');


# In[ ]:


# polynomial regression with powers of x from 2 to 15
for i in range(2,16):  #power of 1 is already there, hence starting with 2
    colname = 'x_%d'%i      #new var will be x_power
    data[colname] = data['x']**i
data.head()


# In[ ]:


0.174533**3


# **Creating test and train**

# In[ ]:


data['randNumCol'] = np.random.randint(1, 6, data.shape[0])
train=data[data['randNumCol']<=3]
test=data[data['randNumCol']>3]
train = train.drop('randNumCol', axis=1)
test = test.drop('randNumCol', axis=1)


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


test.shape


# ## 1. Linear Regression

# In[ ]:


#Import Linear Regression model from scikit-learn.
from sklearn.linear_model import LinearRegression


# In[ ]:


#Separating the independent and dependent variables
X_train = train.drop('y', axis=1).values
y_train = train['y'].values
X_test = test.drop('y', axis=1).values
y_test = test['y'].values


# In[ ]:


train.drop('y', axis=1).head()


# In[ ]:


import seaborn as sns
sns.heatmap(train.drop('y', axis=1).corr().round(2),annot=True,vmin=-1, vmax=1);


# In[ ]:


#Linear Regression with one features
independent_variable_train = X_train[:,0:1]

linreg = LinearRegression(normalize=True)
linreg.fit(independent_variable_train,y_train)
y_train_pred = linreg.predict(independent_variable_train)

rss_train = sum((y_train_pred-y_train)**2) / X_train.shape[0]

independent_variable_test = X_test[:,0:1]
y_test_pred = linreg.predict(independent_variable_test)
rss_test = sum((y_test_pred-y_test)**2)/ X_test.shape[0]

print("Training Error", rss_train)
print("Testing Error",rss_test)


# In[ ]:


plt.plot(X_train[:,0:1],y_train,'.')
plt.plot(X_train[:,0:1],y_train_pred);


# **Linear regression with three features**

# In[ ]:


train.drop('y', axis=1).columns


# In[ ]:


independent_variable_train = X_train[:,0:3]

linreg = LinearRegression(normalize=True)
linreg.fit(independent_variable_train,y_train)
y_train_pred = linreg.predict(independent_variable_train)

rss_train = sum((y_train_pred-y_train)**2) / X_train.shape[0]

independent_variable_test = X_test[:,0:3]
y_test_pred = linreg.predict(independent_variable_test)
rss_test = sum((y_test_pred-y_test)**2)/ X_test.shape[0]

print("Training Error", rss_train)
print("Testing Error",rss_test)


# In[ ]:


plt.plot(X_train[:,0:1],y_train,'.')
plt.plot(X_train[:,0:1],y_train_pred);


# In[ ]:


linreg.coef_


# **Linear regression with Seven features**

# In[ ]:


train.drop('y', axis=1).columns[0:8]


# In[ ]:


independent_variable_train = X_train[:,0:8]

linreg = LinearRegression(normalize=True)
linreg.fit(independent_variable_train,y_train)
y_train_pred = linreg.predict(independent_variable_train)

rss_train = sum((y_train_pred-y_train)**2) / X_train.shape[0]

independent_variable_test = X_test[:,0:8]
y_test_pred = linreg.predict(independent_variable_test)
rss_test = sum((y_test_pred-y_test)**2)/ X_test.shape[0]

print("Training Error", rss_train)
print("Testing Error",rss_test)


# In[ ]:


plt.plot(X_train[:,0:1],y_train,'.')
plt.plot(X_train[:,0:1],y_train_pred);


# In[ ]:


linreg.coef_


# **Defining a function which will fit linear regression model, plot the results, and return the coefficients**

# In[ ]:


def linear_regression(train_x, train_y, test_x, test_y, features, models_to_plot):
        
    #Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(train_x,train_y)
    train_y_pred = linreg.predict(train_x)
    test_y_pred = linreg.predict(test_x)
    
    #Check if a plot is to be made for the entered features
    if features in models_to_plot:
        plt.subplot(models_to_plot[features])
        plt.tight_layout()
        plt.plot(train_x[:,0:1],train_y_pred)
        
        plt.plot(train_x[:,0:1],train_y,'.')
        
        plt.title('Number of Predictors: %d'%features)
    
    #Return the result in pre-defined format
    rss_train = sum((train_y_pred-train_y)**2)/train_x.shape[0]
    ret = [rss_train]
    
    rss_test = sum((test_y_pred-test_y)**2)/test_x.shape[0]
    ret.extend([rss_test])
    
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    
    return ret


# In[ ]:


#Initialize a dataframe to store the results:
col = ['mrss_train','mrss_test','intercept'] + ['coef_Var_%d'%i for i in range(1,16)]
ind = ['Number_of_variable_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)


# In[ ]:


#Define the number of features for which a plot is required:
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}


# In[ ]:


#Iterate through all powers and store the results in a matrix form
plt.figure(figsize=(12,8))
for i in range(1,16):
    train_x = X_train[:,0:i]
    train_y = y_train
    test_x = X_test[:,0:i]
    test_y = y_test
    
    coef_matrix_simple.iloc[i-1,0:i+3] = linear_regression(train_x,train_y, test_x, test_y, features=i, models_to_plot=models_to_plot)


# In[ ]:


#Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_simple


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(coef_matrix_simple['mrss_train'])
plt.plot(coef_matrix_simple['mrss_test'])
plt.xlabel('Features')
plt.ylabel('MSE')
plt.legend(['train', 'test'])
plt.xticks(rotation = 35)
plt.show();


# # 2. Ridge Regression

# The objective of Ridge is to minimize the MSE & Square of cofficient
# 
# $$J(\theta) = \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)} )^2 + \lambda(\theta^2) $$
# 
# where $h_{\theta}(x)$ is the hypothesis and given by the linear model
# 
# $$h_{\theta}(x) = \theta^Tx = \theta_0 + \theta_1x_1$$
# 
# where $\lambda$ is the penality term.

# * It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
# * It reduces the model complexity by coefficient shrinkage.
# * It uses L2 regularization technique.

# In[ ]:


# Importing ridge from sklearn's linear_model module
from sklearn.linear_model import Ridge


# In[ ]:


#Set the different values of alpha to be tested
alpha_ridge = [0, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20, 25]
alpha_ridge


# In[ ]:


# defining a function which will fit ridge regression model, plot the results, and return the coefficients
def ridge_regression(train_x, train_y, test_x, test_y, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(train_x,train_y)
    train_y_pred = ridgereg.predict(train_x)
    test_y_pred = ridgereg.predict(test_x)
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(train_x[:,0:1],train_y_pred)
        plt.plot(train_x[:,0:1],train_y,'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    mrss_train = sum((train_y_pred-train_y)**2)/train_x.shape[0]
    ret = [mrss_train]
    
    mrss_test = sum((test_y_pred-test_y)**2)/test_x.shape[0]
    ret.extend([mrss_test])
    
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    
    return ret


# In[ ]:


#Initialize the dataframe for storing coefficients.
col = ['mrss_train','mrss_test','intercept'] + ['coef_Var_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)


# In[ ]:


#Define the alpha value for which a plot is required:
models_to_plot = {0:231, 1e-4:232, 1e-3:233, 1e-2:234, 1:235, 5:236}


# In[ ]:


train.drop('y', axis=1).columns


# In[ ]:


#Iterate over the 10 alpha values:
plt.figure(figsize=(12,8))
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(train_x, train_y, test_x, test_y, alpha_ridge[i], models_to_plot)


# In[ ]:


#Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_ridge


# In[ ]:


coef_matrix_ridge['mrss_train']


# In[ ]:


coef_matrix_ridge['mrss_test']


# In[ ]:


coef_matrix_ridge[['mrss_train','mrss_test']].plot()
plt.xlabel('Alpha Values')
plt.ylabel('MRSS')
plt.legend(['train', 'test']);


# In[ ]:


alpha_ridge


# In[ ]:


#Printing number of zeros in each row of the coefficients dataset
coef_matrix_ridge.apply(lambda x: sum(x.values==0),axis=1)


# # 3. Lasso

# The objective of Lasso is to minimize the MSE & Square of cofficient
# 
# $$J(\theta) = \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)} )^2 + \lambda(\theta) $$
# 
# where $h_{\theta}(x)$ is the hypothesis and given by the linear model
# 
# $$h_{\theta}(x) = \theta^Tx = \theta_0 + \theta_1x_1$$
# 
# where $\lambda$ is the penality term.

# * It uses L1 regularization technique (will be discussed later in this article)
# * It is generally used when we have more number of features, because it automatically does feature selection.

# In[ ]:


#Importing Lasso model from sklearn's linear_model module
from sklearn.linear_model import Lasso


# In[ ]:


#Define the alpha values to test
alpha_lasso = [0, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]


# In[ ]:


# defining a function which will fit lasso regression model, plot the results, and return the coefficients
def lasso_regression(train_x, train_y, test_x, test_y, alpha, models_to_plot={}):
    #Fit the model
    if alpha == 0:
        lassoreg = LinearRegression(normalize=True)
        lassoreg.fit(train_x, train_y)
        train_y_pred = lassoreg.predict(train_x)
        test_y_pred = lassoreg.predict(test_x)
        
    else:
        lassoreg = Lasso(alpha=alpha,normalize=True)
        lassoreg.fit(train_x,train_y)
        train_y_pred = lassoreg.predict(train_x)
        test_y_pred = lassoreg.predict(test_x)
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(train_x[:,0:1],train_y_pred)
        plt.plot(train_x[:,0:1],train_y,'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    mrss_train = sum((train_y_pred-train_y)**2)/train_x.shape[0]
    ret = [mrss_train]
    
    mrss_test = sum((test_y_pred-test_y)**2)/test_x.shape[0]
    ret.extend([mrss_test])
    
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    
    return ret


# In[ ]:


#Initialize the dataframe to store coefficients
col = ['mrss_train','mrss_test','intercept'] + ['coef_Var_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)


# In[ ]:


#Define the models to plot
models_to_plot = {0:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}


# In[ ]:


#Iterate over the 10 alpha values:
plt.figure(figsize=(12,8))
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(train_x, train_y, test_x, test_y, alpha_lasso[i], models_to_plot)


# In[ ]:


#Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_lasso


# In[ ]:


coef_matrix_lasso[['mrss_train','mrss_test']].plot()
plt.xlabel('Alpha Values')
plt.ylabel('MRSS')
plt.legend(['train', 'test']);


# In[ ]:


coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)


# ## 4. Elastic Net Regression

# Elastic net is basically a combination of both L1 and L2 regularization. So if you know elastic net, you can implement both Ridge and Lasso by tuning the parameters. So it uses both L1 and L2 penality term, therefore its equation look like as follows:
# 
# $$J(\theta) = \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)} )^2 + \lambda_1(\theta) + \lambda_2(\theta^2) $$
# 
# where $h_{\theta}(x)$ is the hypothesis and given by the linear model
# 
# $$h_{\theta}(x) = \theta^Tx = \theta_0 + \theta_1x_1$$
# 
# where $\lambda$ is the penality term.

# In[ ]:


from sklearn.linear_model import ElasticNet


# In[ ]:


# define model
model = ElasticNet(alpha=0.5, l1_ratio=0.5)


# In[ ]:


model.fit(train_x, train_y)


# In[ ]:


model.coef_


# In[ ]:


model.predict(test_x)


# In[ ]:


sum((model.predict(train_x)-train_y)**2)/train_y.shape[0]


# In[ ]:


sum((model.predict(test_x)-test_y)**2)/test_y.shape[0]

