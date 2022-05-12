#!/usr/bin/env python
# coding: utf-8

# # Linear Regression and PCA - Boston Housing

# I  recently learned Regression and Principal Component Analysis and was very eager to try my hands on some intreasting dataset. I found the Boston housing data set as a perfect place to get my hands dirty on this.
# 
# Initially i did the analysis using SAS and when the results where really good, i thought of implementing the same in Python and publish my work.
# 
# Although i faced a few issues in Python as there were no easy implementation of analysis techniques like Variance Inflation Factor and Feature Selection, so therefore i tried to code them in myself.
# 
# Let me know your feedback on this, as this would help me improve and put more good quality work in it.

# ### Upvote if you find it helpful

# # About the Data

# This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and has been used extensively throughout the literature to benchmark algorithms. However, these comparisons were primarily done outside of Delve and are thus somewhat suspect. The dataset is small in size with only 506 cases.

# -Variables
# There are 14 attributes in each case of the dataset. They are:
# 
# CRIM - per capita crime rate by town
# 
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# INDUS - proportion of non-retail business acres per town.
# 
# 
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 
# NOX - nitric oxides concentration (parts per 10 million)
# 
# RM - average number of rooms per dwelling
# 
# AGE - proportion of owner-occupied units built prior to 1940
# 
# DIS - weighted distances to five Boston employment centres
# 
# RAD - index of accessibility to radial highways
# 
# TAX - full-value property-tax rate per 10,000
# 
# PTRATIO - pupil-teacher ratio by town
# 
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 
# LSTAT - per lower status of the population
# 
# MEDV - Median value of owner-occupied homes in $1000's

# # Data Loading

# In[ ]:


# Import Libraries needed to load the data
import pandas as pd
from sklearn.datasets import load_boston


# In[ ]:


# Load the data from sklearn module
df = pd.DataFrame(load_boston().data,columns=load_boston().feature_names)
df['MEDV'] = pd.DataFrame(load_boston().target)
print('Shape of Data is : {} rows and {} columns'.format(df.shape[0],df.shape[1]))


# In[ ]:


df.head()


# In[ ]:


# Lets look at the null values of the data
df.isna().sum()


# <p>Good! no null values in the  data.</p>

# In[ ]:


# Lets look at the datatype of the features
df.dtypes


# All numeric, great!!!

# # Exploratory Data Analysis

# In[ ]:


# import libraries needed to do EDA
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Lets look at the distribution plot of the features
pos = 1
fig = plt.figure(figsize=(16,24))
for i in df.columns:
    ax = fig.add_subplot(7,2,pos)
    pos = pos + 1
    sns.distplot(df[i],ax=ax)


# Except RM and MEDV, nothing else is normally distributed, this might be an issue, as most statistical assumptions hold true only when our data is normally distributed.

# In[ ]:


# lets look at some descriptive stats of our features
df.describe()


# #### Scale of our features are very different from each other, therefore we might have to rescale our data to improve our data quality, as we cannot apply PCA or Linear Regression on this data.

# In[ ]:


# Lets look at the correlation matrix of our data.
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111)
sns.heatmap(df.corr(),annot=True)


# #### Note :
# Our target variable, seems to be highly correlated, with LSTAT and RM, which makes sense, as these two are very important factors for house pricing, but there seems to be a lot of multicollinearity as well.
# 
# The issue here is, that there is a lot of collinearity between our predictor variables, for example DIS is highly correlated to INUDS, INOX and AGE.
# 
# This is not good, as multicollinearity can make our model unstable, we need to look at it a little more, before modeling our data,  I have explained, the probem of multicollinearity below.

# ## Variance Inflation Factor

# A variance inflation factor(VIF) detects multicollinearity in regression analysis. Multicollinearity is when there’s correlation between predictors (i.e. independent variables) in a model; it’s presence can adversely affect your regression results. The VIF estimates how much the variance of a regression coefficient is inflated due to multicollinearity in the model.

# \begin{align*}
# VIF = 1/(1 - R^2)
# \end{align*}

# Where <b>R Squared</b> is <b>coefficient of determination</b>, in simple terms, it is the proportion of variance in independent variable, which is explained by dependent variable. Formula of r squared is as follows

# \begin{align*}
# R^2 = 1 - (Residual sum of Squares)/(Total Sum of Squares)
# \end{align*}

# So what we do is, we perform Linear Regression using each variable as target and others as predictors and the calculate ther R-Squared, then calculate the VIF for them.
# 
# If VIF < 4, its okay to be used, other wise we need to find a way to remove collinearrity from these features.

# In[ ]:


# import libraries needed for this.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[ ]:


# lets get the VIF value to understand the multi collinearity
vifdf = []
for i in df.columns:
    X = np.array(df.drop(i,axis=1))
    y = np.array(df[i])
    lr = LinearRegression()
    lr.fit(X,y)
    y_pred = lr.predict(X)
    r2 = r2_score(y,y_pred)
    vif = 1/(1-r2)
    vifdf.append((i,vif))

vifdf = pd.DataFrame(vifdf,columns=['Features','Variance Inflation Factor'])
vifdf.sort_values(by='Variance Inflation Factor')


# ### Note :
# What we can see  is that, almost half of our features are having either VIF value greater than or near to 4, and TAX and RAD have VIF almost double of our threshold.
# 
# ### Problem of Multicollinearity
# 
# In the presence of multicollinearity, the estimate of one variable's impact on the dependent variable Y while controlling for the others tends to be less precise than if predictors were uncorrelated with one another.
# 
# The usual interpretation of a regression coefficient is that it provides an estimate of the effect of a one unit change in an independent variable, holding the other variables constant.
# 
# If X1 is highly correlated with another independent variable, X2 in the given data set, then we have a set of observations for which X1 and X2 have a particular linear stochastic relationship.
# 
# We don't have a set of observations for which all changes in X1 are independent of changes in X2, so we have an imprecise estimate of the effect of independent changes in X1.

# # Standardiztion of Data

# ### Rescaling the data
# 
# As our data comprises of many kinds of features, all of which have different scale. This is okay for Analysis, but not for data  modelling. As different scales can cause our model to be unstable and vary more than we would want.

# #### Z-score Normalization

# \begin{align*}
# x' = (x - mean)/std
# \end{align*}

# This will give us data with mean = 0 and std = 1

# In[ ]:


# Lets build our function which will perform the normaliztion
def rescale(X):
    mean = X.mean()
    std = X.std()
    scaled_X = [(i - mean)/std for i in X]
    return pd.Series(scaled_X)


# In[ ]:


# We will build a new dataframe
df_std = pd.DataFrame(columns=df.columns)
for i in df.columns:
    df_std[i] = rescale(df[i])


# In[ ]:


# Lets look at the descriptive stats now
df_std.describe().iloc[1:3:]


# ### Note : Shape of Data does not changes when rescaling, it just scales the data to give mean at 0, and standard deviation as 1 for all the features.

# In[ ]:


# lets look at the shape of data after scaling
pos = 1
fig = plt.figure(figsize=(16,24))
for i in df_std.columns:
    ax = fig.add_subplot(7,2,pos)
    pos = pos + 1
    sns.distplot(df_std[i],ax=ax)


# ### As you can see, shape did not change, only the mean value shifted to 0

# # Principal Component Analysis

# In simple words, PCA is a mathematical procedure, which takes a few linearly correlated features and returns few uncorrelated features.
# 
# It is often used in dimensionality reduction for reducing complexity of learning models or to visualize the multidimensional data into 2D or 3D data, making to easy to visualize.
# 
# But to say that, PCA is just a dimensionality reduction technique is like saying Java and Javascript are same.

# <b>Wiki</b>
# 
# Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.
# This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.
# The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.

# ### Why we need it here ?
# 
# Well we do not need it for dimesionality reduction of course, as our model is not that complex,
# We need to remove the multicollinearity problem in our data.
# 
# We are going to feed in our standardized predictor variables into the the PCA transformation and get a set of  uncorrelated features.

# In[ ]:


# import libraries for PCA
from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=13)
X = df_std.drop('MEDV',axis=1)
X_pca = pca.fit_transform(X)
df_std_pca = pd.DataFrame(X_pca,columns=['PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10','PCA11','PCA12','PCA13'])
df_std_pca['MEDV'] = df_std['MEDV']


# In[ ]:


# Lets look at the correlation matrix now.
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111)
sns.heatmap(df_std_pca.corr(),annot=True)


# ### Note : As you can see there is no correlation between predictor variables, thus removing multicollinearity.

# #### Note : The reason correlation between predictor variable and target varaiable is in sorted order is because, PCA takes all the explained variation and puts it into first components, and repeats the  process. The new feature are in no way related to the old ones, therfore it would be wrong to use the same name for them.
# 
# #### Note : PCA is often used to make the data Anonymous as it makes the data completely different from the original one, while still keeping the information in the data intact.

# In[ ]:


# Lets look at the distribution of our features after applying PCA
pos = 1
fig = plt.figure(figsize=(16,24))
for i in df_std_pca.columns:
    ax = fig.add_subplot(7,2,pos)
    pos = pos + 1
    sns.distplot(df_std_pca[i],ax=ax)


# # Data Modelling

# ### Now that our data is ready, we can apply our modelling techniques to it.

# ## Simple Linear Regression

# \begin{align*}
# y' = theta0 + theta1*x1 +theta2*x2 ...... + theta13*x13
# \end{align*}

# In[ ]:


# import libraires needed to perform our Regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[ ]:


# Split data into Training and testing
X = np.array(df_std_pca.drop('MEDV',axis=1))
y = np.array(df_std_pca['MEDV'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
for i in [X_train,X_test,y_train,y_test]:
    print("Shape of Data is {}".format(i.shape))


# In[ ]:


# Lets train our model on training data and predict also on training to see results
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_train)
r2 = r2_score(y_train,y_pred)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
print('R-Squared Score is : {} | Root Mean Square Error is : {}'.format(r2,rmse))


# In[ ]:


# Lets train our model on training data and predict on testing to see results
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print('R2 Score is : {} | Root Mean Square Error is : {}'.format(r2,rmse))


# ### This is better than I anticipated !!

# # Conclusion

# Regression Analysis has taken a back sit in todays world of Deep Learning and Complex Learning techniques, but I strongly believe that that these procedures can help explain your data quickly and give more explaination of the results.
# 
# Through this notebook I tried to explain the power of simple mathematical concepts and show there usage.
# 
# Building this notebook has helped me learn and clarify few key concepts like PCA and VIF, hope it does the same for you.

# I would try to improve the notebook if i get time, your suggestions would really help.

# ## Hope you liked the notebook, please leave a comment and upvote.

# ## Alright Folks, thats all for today!
