#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hello everyone,
# 
# This noteboot is an assignment of CBD Robotics Intern to utilize my acknowledge. It entails two main sections.
# 
# ***Cleaning data***, includes: dealing with missing data, outliers, scaling, and PCA.
# 
# ***Building and Tuning Linear Regression*** to get the best predictions. 

# In[ ]:


import numpy as np 
import pandas as pd 
import scipy
import random
random.seed(10)
np.random.seed(11)


from scipy import stats
from scipy.stats import norm
import missingno as msno
import datetime

#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

# Ploting libs

from plotly.offline import iplot, plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default = "notebook" 
# As after installing vscode, renderer changed to vscode, 
# which made graphs no more showed in jupyter.

from yellowbrick.regressor import ResidualsPlot


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_palette('RdBu')


# # 1. Take a look at the Dataset

# In[ ]:


df = pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')


# In[ ]:


print('Observations                 : ', df.shape[0])
print('Features -- exclude the Price: ', df.shape[1] - 1)


# In[ ]:


# Datatypes
df.info()


# In[ ]:


df.head(5)


# We have some zero in Landsize, let get a closer survey on zeroes.

# In[ ]:


# zero values
(df==0).sum().sort_values(ascending=False).head(6)


# ## Comments
# * ***Landsize and BuildingArea*** where equal zeros must be missing data. Convert them.
# * ***Date*** is time series, which is a big deal for Linear Regression, so better extract Month and Year from Date then delete it.
# * ***Suburb, Address, SellerG***: full of text with too many distinct values, should be removed, as Linear Regression can not deal with them.

# In[ ]:


# Zeroes to Missing in Landsize and BuildingArea
df['Landsize'].replace(0, np.nan, inplace=True)
df['BuildingArea'].replace(0, np.nan, inplace=True)


# In[ ]:


# Extract Month & Year from Date, then drop Date

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df.drop('Date', axis=1, inplace=True)


# In[ ]:


# Drop: Texts
df.drop(['Suburb', 'Address', 'SellerG'], axis=1, inplace=True)


# # 2. Missing Data: A Quick Glance

# In[ ]:


# A Brief of Missing data
total_miss   = df.isnull().sum().sort_values(ascending=False)

percent      = total_miss / df.shape[0]

table = pd.concat([total_miss, percent], axis=1, keys=['Numbers', 'Percent'])
print(table.head(15))


# ***More 40 percents*** must be unbearble for any imputation. I would drop those columns.
# 
# ***Bathroom and Bedroom2*** look like a twins. Notice them in EDA latter.
# 
# ***The target Price*** has 21% data missing. Should I try to impute them? No. I would not take risks of predicting missing of things already mystic. Lets listwise remove them.

# In[ ]:


# Drop: Missing > 40%
df.drop(['BuildingArea', 'YearBuilt', 'Landsize'], axis=1, inplace=True)


# In[ ]:


# Drop: Missing in Price
df.dropna(subset=['Price'], axis=0, inplace=True)


# In[ ]:


# Drop: Minorities
df.dropna(subset=['Propertycount', 'Regionname', 'CouncilArea', 'Postcode', 'Distance'],
          axis=0, inplace=True)


# # 3. Descriptive Statistic 

# In[ ]:


df.describe(percentiles=[0.01, 0.25, 0.75, 0.99])


# ## Comments on Numerics
# ***Datatype***
#  * ***Postcode and Propertycount:*** first, they should have been categorical by nature, but being numerical. Second, Postcode has 211 uniques and Propertycount has 342, so even with converts into categorical, one-hot-encode will be useless. I will remove them.
#  
# ***Abnormality***
#  * ***Some palaces on sale*** with: 30 Bedroom 2, 26 slots of Car parking.

# In[ ]:


# Texts with too many of uniques
df.drop(['Postcode', 'Propertycount'], axis=1, inplace=True)


# In[ ]:


df.describe(include='O').sort_values(axis=1, by=['unique'], ascending=False)


# ## Comments on Categories
# * ***CouncilArea*** has 33 of unique values, though still are able to apply one-hot-encode, but it will burden Linear Regression performance. It would be removed.
# * ***Regionname, Type, Method*** has pretty small number of distinct values. They are deserved to one-hot-encode.

# In[ ]:


df.drop('CouncilArea', axis=1, inplace=True)


# ## To-do latter
# * ***Regionname, Type, Method***: one-hot-encode.

# # 4. EDA

# In[ ]:


#Classify features based on Datatypes, helpful for EDA.

continuous_features = ['Price',      'Distance']

discrete_features  = ['Bathroom',    'Bedroom2',       'Car',        'Rooms']

category_features  = ['Type',        'Method',         'Regionname']
                     


# # 4.1 The Target: Price

# In[ ]:


sns.distplot(df['Price'], fit=norm);


# Price is ***skewed***, but be able to nomalized by removing extreme high points on the right.

# # 4.2 Univariate analyze: Features

# In[ ]:


df[continuous_features].hist(bins=40, figsize=(18,9))
plt.show()


# * ***Distance*** are skewed.

# In[ ]:


df[discrete_features].hist(bins=40, figsize=(20,20))
plt.show()


# ## Comments on Discrete Features
# ***Potential Outliers***
# Most of observations have:
# * Bathroom < 5,
# * Bedroom2 < 10,
# * Car      < 10,
# * Rooms    < 6.  
# 
# So, points standing out of these boundaries probaly are outliers.
# 

# # 4.3 Bivariate analyze

# In[ ]:


# First try for Total sales per Region

# plotly.offline.init_notebook_mode(connected=True)

regions = df.Regionname.unique()
total_values_per_region = [df['Price'][df.Regionname==region].sum() for region in regions]

fig = px.bar(y=regions, x=total_values_per_region,
             title='Total Sales per Regions', orientation='h',
             template='plotly_white')

fig.update_layout(xaxis={'title':'Price'},
                  yaxis={'title':'Regions'})

fig.show()


# ***Regions*** somehow play an important role in Sales.

# In[ ]:


fig = px.box(df, x='Regionname', y='Price', template='simple_white')
fig.update_layout(title='Price by Regions')


# ***Strange 'Outliers'***   
# A considerable number of Price are far from their quartiles, I afraid that Z-score, 3-sigma, or IQR - detecting outlier strategies will remove a lot of data, lets see.

# In[ ]:


# IQR score
def IQR_outlier_detect(data=df, features=[]):
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        outside_IQR = (data[feature]<=(Q1-1.5*IQR)) | ((Q3+1.5*IQR)<=data[feature])  
        outside_IQR = outside_IQR.sum()        
        
        print('Outside of IQR: %s -- Total: %d -- percent %2.2f'% (feature, outside_IQR, outside_IQR/df.shape[0]))
    return

IQR_outlier_detect(df, features=['Price'])


# ***No problem, Price is fine***.

# In[ ]:


fig = px.scatter(df, x='Longtitude', y='Lattitude', color='Price')
fig.update_layout(title='Price by Locations')


# Sale houses tend to locate in the map central.  

# # 4.4 Multivariate analyze

# In[ ]:


# Price vs Continuous Features

corr_matrix = df[continuous_features].corr()

figure = plt.figure(figsize=(16,12))

mask = np.triu(corr_matrix) # Hide the upper part.
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cmap="YlGnBu", mask=mask)

plt.show()


# Nothing seems to be meaningful.

# In[ ]:


# Price vs Discrete Features

corr_matrix = df[discrete_features + ['Price']].corr()

figure = plt.figure(figsize=(16,12))

mask = np.triu(corr_matrix) # Hide the upper part.
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cmap="YlGnBu", mask=mask)

plt.show()


# ***Rooms and Bedroom2*** is a twins, so keep Rooms and drop the latter.

# In[ ]:


df.drop('Bedroom2', axis=1, inplace=True)

discrete_features.remove('Bedroom2')


# # 5. Outliers

# ## Detection by IQR Rule
# 
# ***IQR Rule***  
# 
# This is a renowned technique to detecting outliers. To apply this rule, first we need to define several stuffs.
# 
# ***Q1***: the quantile at 25%.
# 
# ***Q3***: the quantile at 75%.
# 
# ***IQR*** = Q3 - Q1.
# 
# Then, any value stands out of range **[Q1 - 1.5 IQR, Q3 + 1.5 IQR]** would be considered an outlier.
# 
# The IQR rule would be praticed on numerical features only.

# In[ ]:


# First, detect Outliers
features = continuous_features + discrete_features
IQR_outlier_detect(df, features)


# In[ ]:


# Remove Outliers
def IQR_outlier_remove(data=df, features=[]):
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # the core: the ~ is a must to avoid removing NaN.
        outside_IQR = (data[feature]<=(Q1-1.5*IQR)) | ((Q3+1.5*IQR)<=data[feature])
        data = data[~outside_IQR]
        print('Cleaning: ', feature)
        print('Q1: %2.2f', Q1)
        print('Q2: %2.2f', Q3)
        print('After cleaning, data left: %d \n' % (data.shape[0]))
        
        # debug
        #inside_IQR = ((Q1-1.5*IQR)<= data[feature]) & (data[feature]<=(Q3+1.5*IQR))
        
    return data

# Driving code
features = continuous_features + discrete_features
df = IQR_outlier_remove(df, features)


# In[ ]:


# How much observations left?
df.shape


# # 6. Standardization

# In[ ]:


df.dtypes


# In[ ]:


features_to_scaler = ['Rooms', 'Distance', 'Bathroom', 'Car',
                        'Lattitude', 'Longtitude',
                        'Month', 'Year']


# In[ ]:


df_std = df


# In[ ]:


scaler = StandardScaler()

for feature in features_to_scaler:
    df_std[feature] = scaler.fit_transform(df_std[feature].values.reshape(-1, 1))


# In[ ]:


df_std.head()


# # 7. One Hot Encode
# With intending to knn imputing on missing data, but knn only works with numerical, not categorical, so the encoding is performed up front.

# In[ ]:


df_std.head()


# In[ ]:


df_encode = pd.get_dummies(df_std)


# In[ ]:


df_encode.dtypes


# # 8. Missing Data

# ## Strategies for  Missing Data
# <a href="https://ibb.co/fXC5QMG"><img src="https://i.ibb.co/TwHSrcq/Missing-Data.png" alt="Missing-Data" border="0"></a>

# ## Assumption: all MCAR.
# 
# Selection of methods must base on the nature of missing data, whether they are MCAR, MAR, or MNAR. I know a research on those are essential, but in this entry-level assignment, I will skip it to focus on the major section Modelling.
# 
# Therefore, let assumpt all columns are MCAR.

# In[ ]:


# A Brief of Missing data

total_miss   = df.isnull().sum().sort_values(ascending=False)

percent      = total_miss / df.shape[0]

table = pd.concat([total_miss, percent], axis=1, keys=['Numbers', 'Percent'])
print(table.head(8))


# ## Strategies on Choices
# 
#  * ***Hand-in-hand pattern***: If a row lacks Car value, moreoften lacks values in Bathroom, Longtitude, Lattitude, and vice versa. Please scroll a half page down then look at the graph, you'll see that most of cells are around 1, which means our missing data are very centralized in specific rows.
#       
#  
#  * ***K-nn Imputation is by far the best***. Reason is the way real estate market working: houses with similar specifications, close-by location, usually sold in the same price level. So, k-nn is a nice choice.
# 

# In[ ]:


msno.heatmap(df)


# ## Simply Put
# ***Listwise deleting***: Region Name and Distance.
# 
# ***K-nn approach***: all the rest.
# 
# 

# In[ ]:


df_encode.dtypes


# In[ ]:


# K-nn imputation
neighbors = 10
imputer = KNNImputer(n_neighbors=neighbors)

df_filled = imputer.fit_transform(df_encode)

# to Dataframe
df_filled = pd.DataFrame(df_filled)


# In[ ]:


df_filled.head()


# # 9. Assign to X, y

# In[ ]:


y = df_filled[1]

X = df_filled.drop(labels=1, axis=1)


# # 10. Linear Regression

# ### Assumptions of Linear Regression
# 
# Beforehand modeling or tuning, firstly we need to acknowledge of Assumptions of Linear Regression.
# 
# * ***Normality of X and y***. Or by a more specific term: multivariate normality. Hum, dangerous-look words..
# * ***Linearity of X and y***. Capital X means a plural of features, columns.
# * ***Homoscedasticity***. Namely, variance of residuals are constant. There are several others explanation of homoscedasticity, but this one is nice and simple at most, especially for Residual plots.
# 
# 
# It seems like a lot of works, but fortunely could be done just by Residual plots.

# ### Searching for Normality

# In[ ]:


# Normality of y
sns.distplot(y, fit=norm);
fig = plt.figure()
res = stats.probplot(y, plot=plt)


# The Price is not very normal. It shows peakedness, skewness and does not follow the diagonal line.
# 
# Let's transform it.
# 
# 

# In[ ]:


y = np.log(y)

# Check again
# Normality of y
sns.distplot(y, fit=norm);
fig = plt.figure()
res = stats.probplot(y, plot=plt)


# #### Normality of X
# 
# Take a look at X then we'll see lots of negative values, which are incompetence for log transformations. Of course, we can still perform np.log() for X, but it will return NaN for negative values, and damages our dataset.

# In[ ]:


X.head()


# ### Searching for Linearity and Homoscedasticity
# 
# I will leave them blank because I don't know how to do it for now. Sorry, it must be a gap in my knowledge.

# ### Modeling
# 
# The most important part of this sections is ***B - Linear Regression with Cross Validation***, that I am carrying on both modeling and tuning carefully. The reasons:
# * Linear regression with Holdout, aka 1-fold cross validation, are highly dependent on luck that I dislike, so it is removal.
# * Linear regression with PCA is just a CV linear regression with additional steps. It's better to detail the CV linear regression then assuming the PCA one are similiar.  
# 
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)


# X_train and y_train for models to learn with folding by the cross validation, where X_test, y_test would be untouch till the final scoring. 

# ## A - Linear Regression with Holdout

# In[ ]:


A = Ridge(alpha=0)

A.fit(X_train, y_train)
print("A's score: %2.4f" % A.score(X_test, y_test))


# Oh, a nice number.Then, 0.6959 would be our baseline for further tuning.

# ## B - Linear Regression with Cross Validation

# In[ ]:


# B is the same as A but with CV

B = RidgeCV(alphas=[0], cv=5, scoring='r2')

B.fit(X_train, y_train)
print("A's score: %2.4f" % B.score(X_test, y_test))


# In[ ]:


# Finding the best k-folds

B_score = []
cv = []

for i in range(2, 11):
    model = Ridge(alpha=0, normalize=True)
    score = cross_val_score(model, X_train, y_train, cv=i).mean()
    if score<0 : score = 0
    B_score.append(round(score, 5))
    cv.append(i)
    
    print("cv: %d --- score: %2.5f" % (i, score))
    
B_score = [0 if score<0 else score for score in B_score]
print(B_score)

px.line(x=cv, y=B_score, 
        template='simple_white', 
        title='<b>K-fold vs R2</b>',
        labels={'x':'K-fold', 'y':'R2'})


# *** 8 K-fold is the best***, though not the by far best. Let fix the k-fold down. 

# In[ ]:


cv = 8


# ## Tuning B - Linear Regression with cv=8

# We have 2 parameters to tune:
# * How strong the regularization.
# * Should we normalize the data?

# In[ ]:


params = {'alpha':[100, 30, 21, 20, 19.5, 19, 18.5, 18, 17, 17.5, 16, 15, 14, 13.5, 13, 12.5, 12, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.7, 7.6, 7.5, 7.4, 7.3, 7, 6, 5, 4.5, 4, 3.5, 3, 1, 0.3, 0.1, 0.03, 0.01, 0],
          'normalize': (True, False)}

model = Ridge()
gsc = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1)
gsc.fit(X_train, y_train)

best = gsc.best_params_
score = gsc.score(X_test, y_test)
print('With : ', best)
print('Score: %2.4f' % score)


# The best choice: alpha 7.6 and normalize False. 
# 
# Yes, normalize must be False - turned off, cause we perform a standard scaling already.

# In[ ]:


# With those best params, plot: Residuals vs Prediction

B = gsc.best_estimator_
B.fit(X_train, y_train)
print("B's score: %2.4f" % B.score(X_test, y_test))

visualizer = ResidualsPlot(B)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 


# #### Comments
# * Homoscedastic. Yes, we got it. The shape is not fan-out, not spreading. Points locate within a quite parallel limits.
# * Normality. We got it, too. On the scatter plot, there are a bit outliers on the upper, but no any dense on one side. Then, look at the histogram on the right, quite perfect nomarl, huh.
# * Linearity between X and y. I am not so sure. It is worth an extensive study.
# 
# * Outliers. There are some of them on the higher top. I afraid that somehow these outliers sneaked into data after all the scaling and normalizing to destroy our normality.

# ***Comments for the OLDER version of B***
# 
# > For my presentation at class.
# 
# In this graph of Residuals against Predicted values, the ***distribution*** is:
# 1. In fan-out shape: an identify of not constant variance of residuals, or namely ***Heteroscedasticity***.
# 2. A little curve or bend: probably is a proof of ***non-linear***.
# 
# So we got two violations here: ***non-homoscedasticity*** and ***non-linearity*** of X and y. Mention that both problems lay in natural of data, not in the linear model. Nothing in hell we can do with it.

# In[ ]:


from yellowbrick.regressor import PredictionError
from sklearn.linear_model import Lasso

model = PredictionError(B)
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.show()


# ## C - LR with PCA and Holdout

# In[ ]:


pca = PCA()
pca.fit(X_train)

cumsum = pca.explained_variance_ratio_.cumsum() // 0.01
n_comp = [i for i in range(1, len(cumsum)+1, 1)]

print(cumsum)
px.bar(y=cumsum, x=n_comp, text=cumsum)


# According to those numbers, with 10 principal components, we can loss no more than 10% information. Let's choose ***n_components=10***.

# In[ ]:


pipe = Pipeline([
                ('PCA', PCA(n_components=10)),
                ('Linear Regression', Ridge(alpha=0, normalize=True))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# Oh, dimensionality reduction means lossing in information as well as lossing in our R2 score.

# ## D - LR with PCA and Cross Validation

# In[ ]:


# D
step = [( 'PCA'     , PCA()   ),
        ( 'Lin_Reg' , RidgeCV(alphas=[0], cv=7) )]

D = Pipeline(step)
D.fit(X_train, y_train)
score = D.score(X_test, y_test)
print("D's score: %2.4f" % score)


# ### Tuning D

# We have 3 parameters to tune:
# * Number of Principle components in PCA,
# * How strong the regularization.
# * Should we normalize the data?

# In[ ]:


step = [( 'PCA'     , PCA()   ),
        ( 'Lin_Reg' , Ridge() )]
pipe = Pipeline(step)

params = {'PCA__n_components' : range(1,24),
          'Lin_Reg__alpha'    : [100, 30, 21, 20, 19.5, 19, 18.5, 18, 17, 17.5, 16, 15, 14, 13.5, 13, 12.5, 12, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.7, 7.6, 7.5, 7.4, 7.3, 7, 6, 5, 4.5, 4, 3.5, 3, 1, 0.3, 0.1, 0.03, 0.01, 0],
          'Lin_Reg__normalize': [True, False]}

gsc = GridSearchCV(pipe, param_grid=params, cv=7)
gsc.fit(X_train, y_train)

best = gsc.best_params_
score = gsc.score(X_test, y_test)
print('With : ', best)
print('Score: %2.4f' % score)


# In[ ]:


D = gsc.best_estimator_
D.fit(X_train, y_train)
print("B's score: %2.4f" % D.score(X_test, y_test))

visualizer = ResidualsPlot(D)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 


# With tons of tuning, the model advance only 0.0001 R2 score. It is not worth the effort.

# # 11. Conclusion.
# 
# 1. Cleaning data is very challenging. It took me dozens of hours and bunchs of effor to get data in shape. I doubt when people say most of time in data science, you will deal with data cleaning. Now I sadly know it's true.
# 2. In searching for Normality of X, I was unable to log transform X because of negative values. These negatives came from standard scaling performed beforehand. I am thinking that if I first perform log-transformation, then scale the data later, so I could get benefits from both process with nicer distributions of X features.
# 3. Tuning did not make sense as much as I hoped. The baseline of very simple linear regression was 0.69 in R2. Then I did not get any improvement after all the tuning. There are some reasonable explains:
#     * I perfectly cleaned the data. Oh, I hope so.
#     * Ridge - a linear regression with L2 regularization was too simple for predictions.
#     * Is there a better transform than a log, like a square root?
