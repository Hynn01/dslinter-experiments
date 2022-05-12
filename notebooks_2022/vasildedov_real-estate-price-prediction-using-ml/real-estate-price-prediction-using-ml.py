#!/usr/bin/env python
# coding: utf-8

# # Real estate price prediction using ML 

# This project aims to correctly predict real estate prices in Madrid based on a publicly available dataset, after doing some data cleaning and exploration. In particular, the project is meant more as a learning experience for the author, evaluating 4 different ML techniques to make predictions - linear regression using ordinary least squares (OLS), RandomForest Regressor, Catboost Regressor and LightGBM Regressor. Clarifying comments have been made throughout the report and there is also a conclusion at the end. 

# **Best result: 81.9% using Catboost** 

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# In[ ]:


#importing data from csv file
raw_data = pd.read_csv('/kaggle/input/madrid-real-estate-market/houses_Madrid.csv')

pd.set_option("display.max_columns", None)
# exploring the top 5 rows of the data
raw_data.head()


# In[ ]:


raw_data.describe(include='all')


# There is lots of data, some features including 21742 counts of data and
# 58 columns as outlined above. It was decided to include parameters that logically would increase 
# a given real estate's price. It makes sense that with the parameters below, the price would increase
# e.g. more space (sq_mt_built) would increase the price, a newer house would be more expensive, south orientation is also very valuable, etc.

# In[ ]:


#removing unnecessary data from the model
data = raw_data.filter(['sq_mt_built','n_rooms','n_bathrooms', 'buy_price','built_year','has_parking',
                        'is_orientation_south', 'has_lift','has_central_heating'])
#checking what's left:
data.head()


# In[ ]:


#checking for missing data points
data.describe(include='all')


# The built_year feature is important to the algorithm and its count of data points is only 10000, meaning more than half of the data available will have to be discarded. While it is generally bad practice to exclude as many data points, this and other parameters are important for building the algorithm and the missing data will be dropped. The good news is that the dataset is large enough for us to construct a somewhat reasonable regression in spite of the many lost datapoints.

# In[ ]:


#there is a huge disparity on the count of data points between different parameters
#summing the missing data points
data.isnull().sum()


# In[ ]:


#dropping missing values to be left only with valid data points
data_no_mv = data.dropna(axis=0)


# In[ ]:


data_no_mv.describe(include='all')


# We are hence left with less data points but they are more valuable to the model.

# # Dealing with outliers

# In[ ]:


#let's explore the data further to get to know what the price distribution is
#this is valuable since we are trying to train the model to predict buy price
sns.histplot(data_no_mv['buy_price'],kde=True, stat="density", linewidth=0)


# There are some outliers present. Here, the outliers are situated around the higher prices (right side of the graph) and if the right side is excluded, the prices seem normally distributed. Outliers are a great issue for the model we will first use (ordinary least squares OLS).

# In[ ]:


# Let's declare a variable that will be equal to the 95th percentile of the 'buy_price' variable
q = data_no_mv['buy_price'].quantile(0.95)

# Then we can create a new dataframe (df), with the condition that all prices must be below the 95th percentile 
data_1 = data_no_mv[data_no_mv['buy_price']<q]

data_1.describe(include='all')


# We can check the Probability Density Function (PDF) once again to ensure that the result is still distributed in the same way overall. It is, however, there are much fewer outliers:

# In[ ]:


sns.histplot(data_1['buy_price'],kde=True, stat="density", linewidth=0)


# By removing the outliers, the graph is much more concentrated on the real data points now,
# it looks more 'normal' as in normally distributed.

# In[ ]:


# The year built graph looks relatively strange, as in not normally distributed
sns.histplot(data_1['built_year'],kde=True, stat="density", linewidth=0)


# In[ ]:


#let's remove all buildings from the future, which have not yet been built (probably listed before building completion)
#those buildings' prices are different from completed buildings' prices and present uncertainty
#for example, how much (%) of the building is complete will affect price but no such info is available
# the future in this case refers to buildings after 2020, since the dataset was published in 2020

data_2 = data_1[data_1['built_year']<2020]
sns.histplot(data_2['built_year'],kde=True, stat="density", linewidth=0)


# In[ ]:


# resetting the index since some observations were dropped. 
data_cleaned = data_2.reset_index(drop=True)


# In[ ]:


data_cleaned.describe(include='all')


# In[ ]:


#plotting some parameters together

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['built_year'],data_cleaned['buy_price'])
ax1.set_title('Price and Year Built')
ax2.scatter(data_cleaned['sq_mt_built'],data_cleaned['buy_price'])
ax2.set_title('Price and Space')
ax3.scatter(data_cleaned['n_bathrooms'],data_cleaned['buy_price'])
ax3.set_title('Price and number of Bathrooms')


plt.show()


# In[ ]:


# log transforming price
log_price = np.log(data_cleaned['buy_price'])

# Then we add it to our data frame
data_cleaned['log_price'] = log_price
data_cleaned.head()


# In[ ]:


#log plots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['built_year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year Built')
ax2.scatter(data_cleaned['sq_mt_built'],data_cleaned['log_price'])
ax2.set_title('Log Price and Space')
ax3.scatter(data_cleaned['n_bathrooms'],data_cleaned['log_price'])
ax3.set_title('Log Price and number of Bathrooms')


plt.show()


# In[ ]:


# dropping the old buy price
data_cleaned = data_cleaned.drop(['buy_price'],axis=1)


# # Exploring parameter importance

# In[ ]:


#let's use the correlation feature to check for correlation between parameters
data_cleaned.corr()


# In[ ]:


#not much is clear from the table above, let's visualize the collinearity via a heatmap
import seaborn as sns

corrmat = data_cleaned.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

g=sns.heatmap(data_cleaned[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# Now it's much more clear what the correlation is between all the datasets. For the green, we have high correlation and for the red, we have little to no correlation between the parameters in the dataset. Most of the parameters have at least some correlation with log price, which is a good indicator that the logic applied at the beginning has some merit.

# In[ ]:


#let's use the last column as y value and the rest as x values

x = data_cleaned.iloc[:,0:8]
y = data_cleaned['log_price']


# In[ ]:


#checking what's included in the x parameter
x.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)


# In[ ]:


print(model.feature_importances_)


# Let's plot the feature importance values

# In[ ]:


feat_importances = pd.Series(model.feature_importances_, index = x.columns)
feat_importances.nlargest(11).plot(kind='barh')
plt.show()


# Although the number of bathrooms seems with high importance, it likely has high multicollinearity with sq_mt_built and n_rooms. These will have to be explored and if the multicollinearity they have is high, they will have to be removed from the analysis manually for techniques which can't do that automatically (OLS).

# This marks the end of the exploratory analysis of the data.

# Now, 4 different techniques for modelling the prediction of price will be used and compared via some graphs and other metrics.

# # Technique 1: Linear regression - Ordinary Least Squares

# In[ ]:


# statsmodels will be used for checking multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

# declaring a variable to put all features to check for multicollinearity, it cannot include categorical data
# so the features will be typed manually
variables = data_cleaned[['sq_mt_built', 'n_bathrooms', 'built_year', 'n_rooms']]

#create a new data frame which will include all the VIFs
vif = pd.DataFrame()

# make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# include names to make it easier to explore the result
vif["Features"] = variables.columns

vif


# Since number of bathrooms and number of rooms have high VIF, they will be removed from the model. This will drive the VIF of other variables down. So even if sq_mt_built seems with a high VIF, too, once the other 2 features are removed that will no longer be the case. sq_mt_built was chosen as the parameter to go forward with, despite the fact that it has higher VIF than n_rooms, since it better articulates actual space in the apartment and is logically a better predictor of price.

# In[ ]:


data_no_multicollinearity = data_cleaned.drop(['n_bathrooms','n_rooms'],axis=1)

#let's check vif again
variables = data_cleaned[['built_year','sq_mt_built']]

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif["Features"] = variables.columns

vif


# In[ ]:


data_no_multicollinearity.head()


#  let's include categorical data in the regression

# In[ ]:


# 'get_dummies' can be used from pandas
data_with_dummies = pd.get_dummies(data = data_no_multicollinearity,
                                   columns = ['has_parking', 'is_orientation_south',
                                    'has_lift', 'has_central_heating'], drop_first=True)

# the first is dropped to avoid multicollinearity
# the reason has_terrace and has_ac do not show up is they are all True

data_with_dummies.head()


# In[ ]:


#check the order of the variables and move the dependend variable in the beginning manually
data_with_dummies.columns.values

#declare a new variable that will contain the preferred order
#order: dependent variable, indepedendent numerical variables, dummies
cols = ['log_price', 'sq_mt_built', 'built_year','has_parking_True', 'is_orientation_south_True', 'has_lift_True',
       'has_central_heating_True']

# To implement the reordering, we will create a new df, which is equal to the old one but with the new order of features
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# In[ ]:


# The target (dependent variable) is 'log_price'
targets = data_preprocessed['log_price']

# The inputs are everything BUT the dependent variable, so we can simply drop it
inputs = data_preprocessed.drop(['log_price'],axis=1)


# The data will be scaled for the OLS to make sure the scale of each feature does not negatively impact the model.

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(inputs)

# storing the scaling inputs in a new variable
inputs_scaled = scaler.transform(inputs)


# In[ ]:


# Import the module for the split
from sklearn.model_selection import train_test_split

# Split the variables with an 80-20 split and some random state
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)

#let's check the shapes of inputs and targets
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)


# In[ ]:


#the test-train split is 80-20 as evident from above
# Create a linear regression object
reg = LinearRegression()
# Fit the regression with the scaled train inputs and targets
reg.fit(x_train,y_train)

# Let's check the outputs of the regression
# y_hat = predictions
y_hat = reg.predict(x_train)


# In[ ]:


# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_train, y_hat)

#naming the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)

plt.xlim(10,17)
plt.ylim(10,17)
plt.show()


# In[ ]:


# We can plot the PDF of the residuals and check for anomalies
sns.histplot(y_train - y_hat, kde=True, stat="density", linewidth=0)

# Include a title
plt.title("Residuals PDF", size=18)

# In the best case scenario this plot should be normally distributed


# It seems that the predictions are less accurate at the price outliers, in the lower and upper ranges

# In[ ]:


# Find the R-squared of the model
reg.score(x_train,y_train)

# Note that this is NOT the adjusted R-squared


# In[ ]:


# Obtain the bias (intercept) of the regression
reg.intercept_


# In[ ]:


# Obtain the weights (coefficients) of the regression
reg.coef_

# Note that they are barely interpretable if at all


# In[ ]:


# Create a regression summary where we can compare them with one-another
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# **Testing**

# In[ ]:


# Testing is done on a dataset that the algorithm has never seen

y_hat_test = reg.predict(x_test)


# In[ ]:


# Create a scatter plot with the test targets and the test predictions
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(10,17)
plt.ylim(10,17)
plt.show()


# There are many predictions outside the 45 degree line, suggesting that the algorithm struggled to correctly predict prices especially in lower and upper price ranges. The algorithm performed much better for predicting the prices in the middle of the range. This is consistent with OLS's inherent limitation in dealing with outliers.

# In[ ]:


# Finally, let's manually check these predictions
# To obtain the actual prices, we take the exponential of the log_price
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])

# drop the old indexing
y_test = y_test.reset_index(drop=True)

df_pf['Target'] = np.exp(y_test)

# Additionally, we can calculate the difference between the targets and the predictions
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# Since OLS is basically an algorithm which minimizes the total sum of squared errors (residuals),
# this comparison makes a lot of sense

# Finally, it makes sense to see how far off we are from the result percentage-wise
# Here, we take the absolute difference in %, so we can easily order the data frame
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[ ]:


# Exploring the descriptives here gives us additional insights
df_pf.describe()


# In[ ]:


pd.options.display.max_rows = 20
# to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'], ascending = False)


# In[ ]:


#as a final check, import metrics module and check the following metrics
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_hat_test, y_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_hat_test, y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_hat_test, y_test)))

from sklearn.metrics import r2_score
r2 = r2_score (y_hat_test, y_test)

print ('R-squared score', round (r2,2))

#adjusted R-squared:
adj_r2 = 1 - (1 - r2) * (254-1)/(254 - 6 - 1)
print ('Adjusted R-squared score', round (adj_r2,2))


# # Technique 2: Random forest regressor

# In[ ]:


# Import the scaling module to scale the data as done previously
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)

# Scale the features and store them in a new variable (the actual scaling procedure)
x_scaled = scaler.transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=365)


# In[ ]:


print (x_train.shape, y_train.shape)

print (x_test.shape, y_test.shape)


# Now the actual regression technique

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# RandomizedSearchCV will be used instead of GridSearchCV since it is faster

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Randomized Search CV
# these parameters would change depending on dataset size

#number of decision trees
n_estimators = [int(x_scaled) for x_scaled in np.linspace(start = 100, stop = 1000, num = 10)]
#number of features to consider at every split
max_features = ['auto', 'sqrt']
#maximum number of levels in tree
max_depth = [int(x_scaled) for x_scaled in np.linspace(5, 30, num = 6)]
#minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
#minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[ ]:


#creating a dictionary
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[ ]:


import time
start = time.time()
rf = RandomForestRegressor()


# In[ ]:


# Use the random grid to search for best hyperparameters
rf=RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', 
                      n_iter = 10, cv = 5, verbose=False, random_state=42, n_jobs = -1)


# In[ ]:


rf.fit(x_train,y_train)
end = time.time()
diff_rf = end - start
print ('Execution time RF:', round(diff_rf,2),'seconds')


# In[ ]:


# let's check what the best parameters were determined as
rf.best_params_


# In[ ]:


y_hat_rf = rf.predict(x_train)


# In[ ]:


plt.scatter(y_train, y_hat_rf)

plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat_rf)',size=18)


plt.xlim(10,17)
plt.ylim(10,17)
plt.show()


# In[ ]:


sns.histplot(y_train - y_hat_rf, kde=True, stat="density", linewidth=0)

plt.title("Residuals PDF", size=18)


# In[ ]:


# Find the R-squared of the model
from sklearn.metrics import r2_score

r2 = r2_score (y_train, y_hat_rf)

print ('R-squared score', round (r2,2))


# let's try the model on the test values now

# In[ ]:


y_hat_test_rf = rf.predict(x_test)


# In[ ]:


plt.scatter(y_test,y_hat_test_rf, alpha = 0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test_rf)',size=18)
plt.xlim(10,17)
plt.ylim(10,17)
plt.show()


# In[ ]:


sns.histplot(y_test-y_hat_test_rf, kde=True, stat="density", linewidth=0)

plt.title("Residuals PDF", size=18)


# In[ ]:


df_pf = pd.DataFrame(np.exp(y_hat_test_rf), columns=['Prediction'])
y_test = y_test.reset_index(drop=True)
df_pf['Target'] = np.exp(y_test)
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[ ]:


df_pf.describe()


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'], ascending = False)


# In[ ]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_hat_test_rf, y_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_hat_test_rf, y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_hat_test_rf, y_test)))

r2_rf = r2_score (y_hat_test_rf, y_test)

print ('R-squared score', round (r2_rf,2))

#adjusted R-squared:
adj_r2_rf = 1 - ((1 - r2_rf) * (254-1)/(254 - 8 - 1))
print ('Adjusted R-squared score', round (adj_r2_rf,2))


# # Technique 3: Catboost Regresssor

# In[ ]:


from catboost import CatBoostRegressor


# In[ ]:


import time
start = time.time()
cb=CatBoostRegressor()


# In[ ]:


#parameters to vary as in technique 2
grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}


# In[ ]:


cb = RandomizedSearchCV(estimator = cb, param_distributions = grid,scoring='neg_mean_squared_error', 
                        n_iter = 10, cv = 2, random_state=42, n_jobs = -1,)


# In[ ]:


cb.fit(x_train,y_train, verbose = False)
end = time.time()
diff_cb = end - start
print ('Execution time for CB:', round(diff_cb,2), 'seconds')


# In[ ]:


cb.best_params_


# In[ ]:


y_hat_cb = cb.predict(x_train)


# In[ ]:


plt.scatter(y_train, y_hat_cb)

plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat_cb)',size=18)


plt.xlim(10,17)
plt.ylim(10,17)
plt.show()


# In[ ]:


sns.histplot(y_train - y_hat_cb, kde=True, stat="density", linewidth=0)

plt.title("Residuals PDF", size=18)


# In[ ]:


from sklearn.metrics import r2_score

r2 = r2_score (y_train, y_hat_cb)

print ('R-squared score', round (r2,2))


# let's try the model on the test values now

# In[ ]:


y_hat_test_cb=cb.predict(x_test)


# In[ ]:


sns.histplot(y_test-y_hat_test_cb, kde=True, stat="density", linewidth=0)


# In[ ]:


plt.scatter(y_test,y_hat_test_cb, alpha = 0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test_cb)',size=18)
plt.xlim(10,17)
plt.ylim(10,17)
plt.show()


# In[ ]:


df_pf = pd.DataFrame(np.exp(y_hat_test_cb), columns=['Prediction'])

y_test = y_test.reset_index(drop=True)

df_pf['Target'] = np.exp(y_test)

df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[ ]:


df_pf.describe()


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'], ascending = False)


# In[ ]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_hat_test_cb, y_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_hat_test_cb, y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_hat_test_cb, y_test)))

r2_cb = r2_score (y_hat_test_cb, y_test)

print ('R-squared score', round (r2_cb,2))

#adjusted R-squared:
adj_r2_cb = 1 - ((1 - r2_cb) * (254-1)/(254 - 8 - 1))
print ('Adjusted R-squared score', round (adj_r2_cb,2))


# # Technique 4: LGBM Regressor

# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


import time
start = time.time()
lb=LGBMRegressor()  


# In[ ]:


params = {
    "learning_rate": (0.03, 0.3), # default 0.1 
    "max_depth": (2, 6), # default 3
    "n_estimators": (100, 150), # default 100
    "subsample": (0.6, 0.4)
}


# In[ ]:


lb = RandomizedSearchCV(estimator = lb, param_distributions = params,scoring='neg_mean_squared_error', 
                        n_iter = 10, cv = 5, verbose=False, random_state=42, n_jobs = -1)


# In[ ]:


lb.fit(x_train,y_train)
end = time.time()
diff_lb = end - start
print ('Execution time for LGBM:', round(diff_lb,2), 'seconds')


# In[ ]:


lb.best_params_


# In[ ]:


y_hat_lb = lb.predict(x_train)


# In[ ]:


plt.scatter(y_train, y_hat_lb)

plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat_lb)',size=18)


plt.xlim(10,17)
plt.ylim(10,17)
plt.show()


# In[ ]:


sns.histplot(y_train - y_hat_lb, kde=True, stat="density", linewidth=0)

plt.title("Residuals PDF", size=18)


# In[ ]:


from sklearn.metrics import r2_score

r2 = r2_score (y_train, y_hat_lb)

print ('R-squared score', round (r2,2))


# let's try the model on the test values now

# In[ ]:


y_hat_test_lb=lb.predict(x_test)


# In[ ]:


sns.histplot(y_test-y_hat_test_lb, kde=True, stat="density", linewidth=0)


# In[ ]:


plt.scatter(y_test,y_hat_test_lb, alpha = 0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test_lb)',size=18)
plt.xlim(10,17)
plt.ylim(10,17)
plt.show()


# In[ ]:


df_pf = pd.DataFrame(np.exp(y_hat_test_lb), columns=['Prediction'])

y_test = y_test.reset_index(drop=True)

df_pf['Target'] = np.exp(y_test)

df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[ ]:


df_pf.describe()


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'], ascending = False)


# In[ ]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_hat_test_lb, y_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_hat_test_lb, y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_hat_test_lb, y_test)))

r2_lb = r2_score (y_hat_test_lb, y_test)

print ('R-squared score', round (r2_lb,2))

#adjusted R-squared:
adj_r2_lb = 1 - ((1 - r2_lb) * (254-1)/(254 - 8 - 1))
print ('Adjusted R-squared score', round (adj_r2_lb,2))


# # Model comparison

# The linear OLS regression will not be considered further as it was obvious that both its graphs and metrics were much worse than the other techniques. The most likely reason is that the nature of the dataset is that there are many outliers and OLS has a very hard time dealing with outliers, leading to a high accumulated error and low R-squared value. 

# Furthermore, graphs will not be shown either as they are too similar and will not provide value to the comparison.

# 

# In[ ]:


from tabulate import tabulate

table = [["Mean Absolute Error", metrics.mean_absolute_error(y_hat_test_rf, y_test),
           metrics.mean_absolute_error(y_hat_test_cb, y_test),
          metrics.mean_absolute_error(y_hat_test_lb, y_test)], 
         
        ["Mean Squared Error", metrics.mean_squared_error(y_hat_test_rf, y_test),
        metrics.mean_squared_error(y_hat_test_cb, y_test),
        metrics.mean_squared_error(y_hat_test_lb, y_test)],
         
        ['Root Mean Squared Error', np.sqrt(metrics.mean_squared_error(y_hat_test_rf, y_test)),
        np.sqrt(metrics.mean_squared_error(y_hat_test_cb, y_test)),
        np.sqrt(metrics.mean_squared_error(y_hat_test_lb, y_test)) ],
         
        ['R-squared score', round (r2_rf,3), round (r2_cb,3), round (r2_lb,3)],
         
         ['Adjusted R-squared score', round (adj_r2_rf,3), round (adj_r2_cb,3), round (adj_r2_lb,3)],
         
         ['Execution time (seconds)', round(diff_rf,2), round(diff_cb,2), round(diff_lb,2)]]

print(tabulate(table, headers=["Parameter","RandomForest","Catboost","LGBM"], numalign = "left"))


# Catboost provides the best results in all metrics, with the minimal result in all 3 errors and the maximum result in R-squared/adjusted R-squared score. It is noteable that RandomForest, Catboost and LGBM performed similarly and outperformed OLS significantly. This is likely due to the dataset containing numerous outliers, which OLS struggles with inherently. Additionally, while OLS needs to be guided, i.e. shown what parameters to include, the other techniques determine automatically what the best paramaters for creating a model are. The get.dummies function had to be performed for the OLS, while the other techniques do not need that. In that sense, they are easier to implement. 
# 
# The caveat with those techniques are the hyperparameter tuning, which is different for each dataset and has to be done iteratively. The hyperparameter tuning in particular has allowed Catboost and RandomForest to slightly outperform LightGBM, while requiring significantly more amount for training. A separate script showed that with the default parameters the results between the 3 techniques come even closer, with their differences in performance becoming negligible. Without hyperparameter tuning, LGBM slightly outperforms RandomForest, and Catboost retains its first place, albeit with a smaller margin.
# 
# In conclusion, the hyperparameter tuning in this project did not manage to significantly improve results and if training speed is desired, the default parameters set by each technique can be employed with a good amount of result accuracy. The Catboost regression model was able to correctly predict 82% of the real estate test dataset's price correctly, which is a somewhat satisfiable result, considering that only 6 features were used to fit the regression on ~4000 datapoints.
