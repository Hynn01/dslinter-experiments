#!/usr/bin/env python
# coding: utf-8

# <a id='top'></a>

# # Introduction & Approach Taken
# This kernel represents my first personal exploration of an (almost) end-to-end data science process. While starting off as a kernel to complete the Kaggle Learn Machine Learning exercise, it has now been significantly modified to serve a broader learning experience for myself. If you find any errors, lapses in logic, or just have advice for me, please do post them. Any feedback would certainly be appreciated. <br>
# 
# The challenge is based on the Kaggle Housing Prices Competition that provides a dataset with different house attributes together with the price label. The aim is to develop a model capable of predicting the house price based on the data available.
# 
# Different data scientists espouse various approaches for the process of applied data science. The following is an adaptation of some that I have come across, which shall form the structure of this kernel and my approach taken for this particular competition:
# 
# [1. Exploratory Data Analysis](#explore) <br>
# > [1.1 Preliminary observations](#prelim_explore) <br>
#    [1.2 Exploring numerical attributes](#explore_num_columns) <br>
#    [1.3 Exploring categorical attributes](#explore_cat_columns)  <br>
# 
# [2. Data Cleaning & Preprocessing](#data_clean)
# > [2.1 Dealing with missing/null values](#fix_nans) <br>
#    [2.2 Addressing outliers](#address_outliers)<br>
#    [2.3 Transforming data to reduce skew](#transform_skew) <br>
# 
# [3. Feature Selection & Engineering](#feature_eng)
# 
# [4. Preliminary Assessment of Machine Learning Algorithms](#algorithms)
# 
# [5. Selection of Best Algorithm(s) and Fine-Tuning](#fine_tune)
# 

# ### Import modules

# In[ ]:


# Core
import pandas as pd
import numpy as np

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading in input file

# In[ ]:


# Path of the file to read.
file_path = '../input/train.csv'

# Load data into a pandas DataFrame. Note: 1st column is ID
home_data = pd.read_csv(file_path, index_col=0)


# <a id='explore'></a>

# # 1. Exploratory Data Analysis
# Amongst others, the data is explored to:
# 1. Gain a preliminary understanding of available data
# 2. Check for missing or null values
# 3. Find potential outliers
# 4. Assess correlations amongst attributes/features
# 5. Check for data skew
# 
# [Back to contents](#top)

# <a id='prelim_explore'></a>

# ## 1.1 Preliminary observations
# Examples from the dataset are shown below. <br><br>
# [Back to contents](#top)

# In[ ]:


home_data.tail()
# home_data.head()


# In[ ]:


home_data.shape


# The "shape" of the dataset shows that it has 1460 rows/instances, with data from 80 attributes. <br>
# Out of the 80 attributes, one is the target (SalePrice) that the model should predict. <br>
# Hence, there are 79 attributes that may be used for feature selection/engineering.

# ### Numerical columns within the dataset

# In[ ]:


# List of numerical attributes
home_data.select_dtypes(exclude=['object']).columns


# In[ ]:


len(home_data.select_dtypes(exclude='object').columns)


# The above shows that there appears to be 37 numerical columns, including the target, SalePrice. <br>
# Notes: <br>
# * Some potential anomalies in certain rows of the dataset may cause the column data type to become an 'object'. This may lead to an error in distinguishing between numerical and categorical columns. How can this be checked efficiently?
# * It is possible that there are numerical columns that have data in the form of discrete, and limited number of values. Such columns may also be interpreted as categorical data.
# <br>
# 
# The 37 numerical columns have the following general characteristics:

# In[ ]:


home_data.select_dtypes(exclude=['object']).describe().round(decimals=2)


# ### Categorical columns within the dataset

# In[ ]:


home_data.select_dtypes(include=['object']).columns


# In[ ]:


len(home_data.select_dtypes(include='object').columns)


# There are 43 categorical columns, with the following characteristics:

# In[ ]:


home_data.select_dtypes(include=['object']).describe()


# <a id='explore_num_columns'></a>

# ## 1.2 Exploring numerical columns
# [Back to contents](#top)

# ### Skew of target column
# It appears to be good practice to minimise the skew of the dataset. The reason often given is that skewed data adversely affects the prediction accuracy of regression models. <br>
# Note: While important for linear regression, correcting skew is not necessary for Decisions Trees and Random Forests.

# In[ ]:


target = home_data.SalePrice
plt.figure()
sns.distplot(target)
plt.title('Distribution of SalePrice')
plt.show()


# In[ ]:


sns.distplot(np.log(target))
plt.title('Distribution of Log-transformed SalePrice')
plt.xlabel('log(SalePrice)')
plt.show()


# In[ ]:


print('SalePrice has a skew of ' + str(target.skew().round(decimals=2)) + 
      ' while the log-transformed SalePrice improves the skew to ' + 
      str(np.log(target).skew().round(decimals=2)))


# __Notes for Data Cleaning & Preprocessing:__
# * Perform log transformation on SalePrice
# * Feature variables that are skewed should also be investigated to assess whether they require transformation

# ### Distributions of attributes

# In[ ]:


num_attributes = home_data.select_dtypes(exclude='object').drop('SalePrice', axis=1).copy()

fig = plt.figure(figsize=(12,18))
for i in range(len(num_attributes.columns)):
    fig.add_subplot(9,4,i+1)
    sns.distplot(num_attributes.iloc[:,i].dropna())
    plt.xlabel(num_attributes.columns[i])

plt.tight_layout()
plt.show()


# __Notes for Data Cleaning & Preprocessing:__ <br>
# Uni-modal, skewed distributions could potentially be log transformed: <br>
# > LotFrontage, LotArea, 1stFlrSF, GrLivArea, OpenPorchSF<br>
# 
# After-note: This will be a future addition.

# ### Finding Outliers
# Visualisation of data may support the discovery of possible outliers within the data. Examples of how this can be done include:
# 1. Within univariate analysis, for example through using box plots. Outliers are observations more than a multiple (1.5-3) of the IQR (inter-quartile range) beyond the upper or lower quartile. (If data is skewed, it may be helpful to transform them first to a more symmetric distribution shape)
# 2. Within bivariate analysis, for example scatterplots. Outliers have y-values that are unusual in relation to other observations with similar x-values. Alternatively, plots of the residuals from fitted least square line of bivariate regression can also indicate outliers.
# 
# The consensus is that all outliers should be carefully examined:
# * Go back to original data to check for recording or transcription errors
# * If no such errors, look carefully for unusual features of the individual unit to explain difference. This may lead to new theory/discoveries 
# * If data cannot be checked further, outlier is usually (often) dropped from the dataset.
# 
# The scatterplots of SalePrice against each numerical attribute is shown below, with the aim of employing method 2 above with bivariate analysis.
# 
# [Back to contents](#top)

# __Univariate analysis - box plots for numerical attributes__

# In[ ]:


fig = plt.figure(figsize=(12, 18))

for i in range(len(num_attributes.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.boxplot(y=num_attributes.iloc[:,i])

plt.tight_layout()
plt.show()


# __Bivariate analysis - scatter plots for target versus numerical attributes__

# In[ ]:


f = plt.figure(figsize=(12,20))

for i in range(len(num_attributes.columns)):
    f.add_subplot(9, 4, i+1)
    sns.scatterplot(num_attributes.iloc[:,i], target)
    
plt.tight_layout()
plt.show()


# <a id='notes_outliers'></a>

# __Notes for Data Cleaning & Preprocessing__ <br>
# Based on a first viewing of the scatter plots against SalePrice, there appears to be:
# * A few outliers on the LotFrontage (say, >200) and LotArea (>100000) data.
# * BsmtFinSF1 (>4000) and TotalBsmtSF (>6000)
# * 1stFlrSF (>4000)
# * GrLivArea (>4000 AND SalePrice <300000)
# * LowQualFinSF (>550) <br>
# 
# Reference: [quick link to the implementation](#address_outliers) <br>

# ### Assess correlations amongst attributes
# The linear correlation between two columns of data is shown below. There are various correlation calculation methods, but the Pearson correlation is often used and is the default method. It may be useful to note that:
# 1. A combination of the correlation figure and a scatter plot can support the understanding of whether there is a non-linear correlation (i.e. depending on the data, this may result in a low value of linear correlation, but the variables may still be strongly correlated in a non-linear fashion)
# 2. Correlation values may be heavily influenced by single outliers! 
# <br><br>
# 
# Several authors have suggested that _"to use linear regression for modelling, it is necessary to remove correlated variables to improve your model"_, and _"it's a good practice to remove correlated variables during feature selection"_<br><br>
# 
# Below is a heatmap of the correlation of the numerical columns:

# In[ ]:


correlation = home_data.corr()

f, ax = plt.subplots(figsize=(14,12))
plt.title('Correlation of numerical attributes', size=16)
sns.heatmap(correlation)
plt.show()

## Heatmap with annotation of correlation values
# sns.heatmap(home_data.corr(), annot=True)


# With reference to the target SalePrice, the top correlated attributes are:

# In[ ]:


correlation['SalePrice'].sort_values(ascending=False).head(15)


# Show scatter plots for each numerical attribute (again, but different, less-efficient code) and show correlation value:

# In[ ]:


num_columns = home_data.select_dtypes(exclude='object').columns
corr_to_price = correlation['SalePrice']
n_cols = 5
n_rows = 8
fig, ax_arr = plt.subplots(n_rows, n_cols, figsize=(16,20), sharey=True)
plt.subplots_adjust(bottom=-0.8)
for j in range(n_rows):
    for i in range(n_cols):
        plt.sca(ax_arr[j, i])
        index = i + j*n_cols
        if index < len(num_columns):
            plt.scatter(home_data[num_columns[index]], home_data.SalePrice)
            plt.xlabel(num_columns[index])
            plt.title('Corr to SalePrice = '+ str(np.around(corr_to_price[index], decimals=3)))
plt.show()


# __Notes for Feature Selection & Engineering:__
# Based on the scatter plots and correlation figures above, consider:
# * Excluding GarageArea - highly (0.88) correlated with GarageCars, which has a higher corr with Price
# * Excluding GarageYrBlt - highly (0.83) correlated with YearBuilt
# * Excluding all attributes with low corr with Price and unclear non-linear correlation - e.g. MSSubClass, MoSold, YrSold, MiscVal, BsmtFinSF2, BsmtUnfSF, LowQualFinSF?

# <a id='check_null'></a>

# ### Missing/null values in numerical columns
# 
# [Back to contents](#top)

# In[ ]:


# Show columns with most null values:
num_attributes.isna().sum().sort_values(ascending=False).head()


# __Notes for Data Cleaning & Processing:__
# * Not yet clear what to do with LotFrontage missing values. Simple imputation with median? LotFrontage correlation with Neigborhood?
# * GarageYrBlt is highly correlated with YearBuilt, and as an after-note, it is discarded before the machine learning step. Hence no action required.
# * MasVnrArea has 8 missing values, the same number as missing MasVnrType values. Likely not to have masonry veneer. Hence, fill with 0 

# <a id='explore_cat_columns'></a>

# ## 1.3 Exploring categorical columns
# [Back to contents](#top)
# ### Examples of box plots of SalePrice versus categorical values

# In[ ]:


cat_columns = home_data.select_dtypes(include='object').columns
print(cat_columns)


# In[ ]:


var = home_data['KitchenQual']
f, ax = plt.subplots(figsize=(10,6))
sns.boxplot(y=home_data.SalePrice, x=var)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(12,8))
sns.boxplot(y=home_data.SalePrice, x=home_data.Neighborhood)
plt.xticks(rotation=40)
plt.show()


# In[ ]:


## Count of categories within Neighborhood attribute
fig = plt.figure(figsize=(12.5,4))
sns.countplot(x='Neighborhood', data=home_data)
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.show()


# ### Missing/null values in categorical columns
# 
# [Back to contents](#top)

# In[ ]:


home_data[cat_columns].isna().sum().sort_values(ascending=False).head(17)


# __Notes for Data Cleaning & Preprocessing:__
# * For the moment, assume that PoolQC to Bsmt attributes are missing as the houses do not have them (pools, basements, etc.). Hence, the missing values could be filled in with "None".
# * MasVnrType has 8 missing values, the same number as missing MasVnrArea values. Likely not to have masonry veneer. Hence, fill with 'None' 

# <a id='data_clean'></a>

# # 2. Data Cleaning & Preprocessing
# 
# [Back to contents](#top)<br><br>

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# <a id='fix_nans'></a>

# ## 2.1 Dealing with missing/null values
# [Back to contents](#top)

# In[ ]:


# Create copy of dataset  ====================================
home_data_copy = home_data.copy()

# Dealing with missing/null values ===========================
# Numerical columns:
home_data_copy.MasVnrArea = home_data_copy.MasVnrArea.fillna(0)
# HOW TO TREAT LotFrontage - 259 missing values??

# Categorical columns:
cat_cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                     'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType',
                     'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',
                     'MasVnrType']
for cat in cat_cols_fill_none:
    home_data_copy[cat] = home_data_copy[cat].fillna("None")
    


# In[ ]:


# Check for outstanding missing/null values
# Scikit-learn's Imputer will be used to address these
home_data_copy.isna().sum().sort_values(ascending=False).head()


# <a id='address_outliers' ></a>

# ## 2.2 Addressing outliers
# [Back to contents](#top)<br>
# Reference: [Quick link to notes](#notes_outliers)

# In[ ]:


# Remove outliers based on observations on scatter plots against SalePrice:
home_data_copy = home_data_copy.drop(home_data_copy['LotFrontage']
                                     [home_data_copy['LotFrontage']>200].index)
home_data_copy = home_data_copy.drop(home_data_copy['LotArea']
                                     [home_data_copy['LotArea']>100000].index)
home_data_copy = home_data_copy.drop(home_data_copy['BsmtFinSF1']
                                     [home_data_copy['BsmtFinSF1']>4000].index)
home_data_copy = home_data_copy.drop(home_data_copy['TotalBsmtSF']
                                     [home_data_copy['TotalBsmtSF']>6000].index)
home_data_copy = home_data_copy.drop(home_data_copy['1stFlrSF']
                                     [home_data_copy['1stFlrSF']>4000].index)
home_data_copy = home_data_copy.drop(home_data_copy.GrLivArea
                                     [(home_data_copy['GrLivArea']>4000) & 
                                      (target<300000)].index)
home_data_copy = home_data_copy.drop(home_data_copy.LowQualFinSF
                                     [home_data_copy['LowQualFinSF']>550].index)


# <a id='transform_skew'></a>

# ## 2.3 Transforming data to reduce skew
# For the moment, this is restricted to the target variable.
# [Back to contents](#top)

# In[ ]:


home_data_copy['SalePrice'] = np.log(home_data_copy['SalePrice'])
home_data_copy = home_data_copy.rename(columns={'SalePrice': 'SalePrice_log'})


# <a id='feature_eng'></a>

# # 3. Feature Selection & Engineering
# 
# [Back to contents](#top)

# ### Considering highly-correlated features
# Feeding highly-correlated features to machine algorithms may cause a reduction in performance. Hence, these are addressed below:

# In[ ]:


transformed_corr = home_data_copy.corr()
plt.figure(figsize=(12,10))
sns.heatmap(transformed_corr)


# Highly-correlated attributes include (left attribute has higher correlation with SalePrice_log):
# * GarageCars and GarageArea (0.882)
# * YearBuilt and GarageYrBlt (0.826)
# * GrLivArea_log1p and TotRmsAbvGrd (0.826)
# * TotalBsmtSF and 1stFlrSF_log1p (0.780)
# 
# Perhaps choose to drop the column with the lower correlation against SalePrice_log from the above pairs with more than 0.8 correlation.

# ### Perform feature selection, and encoding of categorical columns

# In[ ]:


# Remove attributes that were identified for excluding when viewing scatter plots & corr values
attributes_drop = ['SalePrice_log', 'MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 
                   'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd'] # high corr with other attributes

X = home_data_copy.drop(attributes_drop, axis=1)

# Create target object and call it y
y = home_data_copy.SalePrice_log

# One-hot-encoding to transform all categorical data
X = pd.get_dummies(X)

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Normalisation - to be added later
# normaliser = StandardScaler()
# train_X = normaliser.fit_transform(train_X)
# val_X = normaliser.transform(val_X)

# Final imputation of missing data - to address those outstanding after previous section
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
val_X = my_imputer.transform(val_X)


# <a id='algorithms'></a>

# # 4. Preliminary Assessment of Machine Learning Algorithms
# [Back to contents](#top)

# ### Import machine learning modules

# In[ ]:


from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# __Reminder: target is now log(SalePrice). After prediction call, need to inverse-transform to obtain SalePrice!__

# In[ ]:


def inv_y(transformed_y):
    return np.exp(transformed_y)

# Series to collate mean absolute errors for each algorithm
mae_compare = pd.Series()
mae_compare.index.name = 'Algorithm'

# Specify Model ================================
# iowa_model = DecisionTreeRegressor(random_state=1)
# # Fit Model
# iowa_model.fit(train_X, train_y)

# # Make validation predictions and calculate mean absolute error
# val_predictions = iowa_model.predict(val_X)
# val_mae = mean_absolute_error(inv_y(val_predictions), inv_y(val_y))
# mae_compare['DecisionTree'] = val_mae
# # print("Validation MAE for Decision Tree when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Decision Tree. Using best value for max_leaf_nodes ==============
# iowa_model = DecisionTreeRegressor(max_leaf_nodes=90, random_state=1)
# iowa_model.fit(train_X, train_y)
# val_predictions = iowa_model.predict(val_X)
# val_mae = mean_absolute_error(inv_y(val_predictions), inv_y(val_y))
# mae_compare['DecisionTree_opt_max_leaf_nodes'] = val_mae
# # print("Validation MAE for Decision Tree with best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Random Forest. Define the model. =============================
rf_model = RandomForestRegressor(random_state=5)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(inv_y(rf_val_predictions), inv_y(val_y))

mae_compare['RandomForest'] = rf_val_mae
# print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# XGBoost. Define the model. ======================================
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_model.fit(train_X, train_y, early_stopping_rounds=5, 
              eval_set=[(val_X,val_y)], verbose=False)
xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(inv_y(xgb_val_predictions), inv_y(val_y))

mae_compare['XGBoost'] = xgb_val_mae
# print("Validation MAE for XGBoost Model: {:,.0f}".format(xgb_val_mae))

# Linear Regression =================================================
linear_model = LinearRegression()
linear_model.fit(train_X, train_y)
linear_val_predictions = linear_model.predict(val_X)
linear_val_mae = mean_absolute_error(inv_y(linear_val_predictions), inv_y(val_y))

mae_compare['LinearRegression'] = linear_val_mae
# print("Validation MAE for Linear Regression Model: {:,.0f}".format(linear_val_mae))

# Lasso ==============================================================
lasso_model = Lasso(alpha=0.0005, random_state=5)
lasso_model.fit(train_X, train_y)
lasso_val_predictions = lasso_model.predict(val_X)
lasso_val_mae = mean_absolute_error(inv_y(lasso_val_predictions), inv_y(val_y))

mae_compare['Lasso'] = lasso_val_mae
# print("Validation MAE for Lasso Model: {:,.0f}".format(lasso_val_mae))

# Ridge ===============================================================
ridge_model = Ridge(alpha=0.002, random_state=5)
ridge_model.fit(train_X, train_y)
ridge_val_predictions = ridge_model.predict(val_X)
ridge_val_mae = mean_absolute_error(inv_y(ridge_val_predictions), inv_y(val_y))

mae_compare['Ridge'] = ridge_val_mae
# print("Validation MAE for Ridge Regression Model: {:,.0f}".format(ridge_val_mae))

# ElasticNet ===========================================================
elastic_net_model = ElasticNet(alpha=0.02, random_state=5, l1_ratio=0.7)
elastic_net_model.fit(train_X, train_y)
elastic_net_val_predictions = elastic_net_model.predict(val_X)
elastic_net_val_mae = mean_absolute_error(inv_y(elastic_net_val_predictions), inv_y(val_y))

mae_compare['ElasticNet'] = elastic_net_val_mae
# print("Validation MAE for Elastic Net Model: {:,.0f}".format(elastic_net_val_mae))

# KNN Regression ========================================================
# knn_model = KNeighborsRegressor()
# knn_model.fit(train_X, train_y)
# knn_val_predictions = knn_model.predict(val_X)
# knn_val_mae = mean_absolute_error(inv_y(knn_val_predictions), inv_y(val_y))

# mae_compare['KNN'] = knn_val_mae
# # print("Validation MAE for KNN Model: {:,.0f}".format(knn_val_mae))

# Gradient Boosting Regression ==========================================
gbr_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, 
                                      max_depth=4, random_state=5)
gbr_model.fit(train_X, train_y)
gbr_val_predictions = gbr_model.predict(val_X)
gbr_val_mae = mean_absolute_error(inv_y(gbr_val_predictions), inv_y(val_y))

mae_compare['GradientBoosting'] = gbr_val_mae
# print("Validation MAE for Gradient Boosting Model: {:,.0f}".format(gbr_val_mae))

# # Ada Boost Regression ================================================
# ada_model = AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=5)
# ada_model.fit(train_X, train_y)
# ada_val_predictions = ada_model.predict(val_X)
# ada_val_mae = mean_absolute_error(inv_y(ada_val_predictions), inv_y(val_y))

# mae_compare['AdaBoost'] = ada_val_mae
# # print("Validation MAE for Ada Boost Model: {:,.0f}".format(ada_val_mae))

# # Support Vector Regression ===========================================
# svr_model = SVR(kernel='linear')
# svr_model.fit(train_X, train_y)
# svr_val_predictions = svr_model.predict(val_X)
# svr_val_mae = mean_absolute_error(inv_y(svr_val_predictions), inv_y(val_y))

# mae_compare['SVR'] = svr_val_mae
# print("Validation MAE for SVR Model: {:,.0f}".format(svr_val_mae))

print('MAE values for different algorithms:')
mae_compare.sort_values(ascending=True).round()


# ### Cross-validation
# Use scikit-learn's cross_val_score to try K-fold cross-validation. 

# In[ ]:


from sklearn.model_selection import cross_val_score

imputer = SimpleImputer()
imputed_X = imputer.fit_transform(X)
n_folds = 10

# =========================================================================
scores = cross_val_score(lasso_model, imputed_X, y, scoring='neg_mean_squared_error', 
                         cv=n_folds)
lasso_mae_scores = np.sqrt(-scores)

print('For LASSO model:')
# print(lasso_mae_scores.round(decimals=2))
print('Mean RMSE = ' + str(lasso_mae_scores.mean().round(decimals=3)))
print('Error std deviation = ' +str(lasso_mae_scores.std().round(decimals=3)))


# In[ ]:


scores = cross_val_score(gbr_model, imputed_X, y, scoring='neg_mean_squared_error', 
                         cv=n_folds)
gbr_mae_scores = np.sqrt(-scores)

print('For Gradient Boosting model:')
# print(lasso_mae_scores.round(decimals=2))
print('Mean RMSE = ' + str(gbr_mae_scores.mean().round(decimals=3)))
print('Error std deviation = ' +str(gbr_mae_scores.std().round(decimals=3)))


# In[ ]:


scores = cross_val_score(xgb_model, imputed_X, y, scoring='neg_mean_squared_error', 
                         cv=n_folds)
mae_scores = np.sqrt(-scores)

print('For XGBoost model:')
# print(mae_scores.round(decimals=2))
print('Mean RMSE = ' + str(mae_scores.mean().round(decimals=3)))
print('Error std deviation = ' +str(mae_scores.std().round(decimals=3)))


# In[ ]:


scores = cross_val_score(rf_model, imputed_X, y, scoring='neg_mean_squared_error', 
                         cv=n_folds)
mae_scores = np.sqrt(-scores)

print('For Random Forest model:')
# print(mae_scores.round(decimals=2))
print('Mean RMSE = ' + str(mae_scores.mean().round(decimals=3)))
print('Error std deviation = ' +str(mae_scores.std().round(decimals=3)))


# <a id='fine_tune'><a>

# # 5. Selection of Best Algorithm(s) & Fine-Tuning
# ### Create a Model and Make Predictions for the Competition
# Read the file of "test" data. And apply best model to make predictions
# 
# __Reminders:__
# * __Need to perform all transformations and normalisation, etc. to test data similar to when training the data__
# * __Make sure to inverse transform predictions to get predicted SalePrice__
# 
# [Back to contents](#top)

# ### Use scikit-learn's function to grid search for the best combination of hyperparameters

# In[ ]:


# Grid search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# # Tuning XGBoost
# param_grid = [{'n_estimators': [1000, 1500], 
#                'learning_rate': [0.01, 0.03] }]
# #               'max_depth': [3, 6, 9]}]

# top_reg = XGBRegressor()

# Tuning Lasso
param_grid = [{'alpha': [0.0007, 0.0005, 0.005]}]
top_reg = Lasso()

# -------------------------------------------------------
grid_search = GridSearchCV(top_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error')

grid_search.fit(imputed_X, y)

grid_search.best_params_


# In[ ]:


# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)


# ### Repeat pre-processing defined previously

# In[ ]:


# create test_X which to perform all previous pre-processing on
test_X = test_data.copy()

# Repeat treatments for missing/null values =====================================
# Numerical columns:
test_X.MasVnrArea = test_X.MasVnrArea.fillna(0)

# Categorical columns:
for cat in cat_cols_fill_none:
    test_X[cat] = test_X[cat].fillna("None")

# Repeat dropping of chosen attributes ==========================================
if 'SalePrice_log' in attributes_drop:
    attributes_drop.remove('SalePrice_log')

test_X = test_data.drop(attributes_drop, axis=1)

# One-hot encoding for categorical data =========================================
test_X = pd.get_dummies(test_X)


# ===============================================================================
# Ensure test data is encoded in the same manner as training data with align command
final_train, final_test = X.align(test_X, join='left', axis=1)

# Imputer for all other missing values in test data. Note: Do not 'fit_transform'
final_test_imputed = my_imputer.transform(final_test)


# ### Create final model

# In[ ]:


# Create model - on full set of data (training & validation)
# Best model = Lasso?
final_model = Lasso(alpha=0.0005, random_state=5)
# final_model = XGBRegressor(n_estimators=1500, learning_rate=0.03)
final_train_imputed = my_imputer.fit_transform(final_train)

# Fit the model using all the data - train it on all of X and y
final_model.fit(final_train_imputed, y)


# ### Make predictions for submission

# In[ ]:


# make predictions which we will submit. 
test_preds = final_model.predict(final_test_imputed)

# The lines below shows you how to save your data in the format needed to score it in the competition
# Reminder: predictions are in log(SalePrice). Need to inverse-transform.
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': inv_y(test_preds)})

output.to_csv('submission.csv', index=False)


# # Future Study
# ### Exploratory Data Analysis
# * How to assess outliers for categorical columns?
# * How to check for correlation amongst categorical columns?
# 
# ### Data Cleaning & Preprocessing
# * Add code to perform data normalisation/standardisation.
# * Create custom scikit-learn preprocessing class so that this can be easily used later with pipelines.
# * Consider scikit-learn's PowerTransformer preprocessing module to fit and transform columns to be more Gaussian-like. 
# 
# ### Feature Selection & Engineering
# * Think about possible combination of attributes to create useful new features, including polynomials.
# * Explore pros and cons of other methods to encode categorical attributes, besides one-hot encoding
# * Consider further advanced preprocessing techniques, such as Principal Components Analysis for dimensionality reduction, to create new features
# 
# ### Preliminary Assessment of ML Algorithms
# * Use scikit-learn's pipelines!
# * Learn how best to utilise ensemble methods.
# * Explore choices for performance criteria.
# * Plots to diagnose bias vs. variance, and learning curves
# 
# ### Selection of Best Algorithm(s) & Fine-tuning
# * Test newly engineered features as hyperparameters in grid-search cross-validation.
# * Partial-dependence plots for understanding the final model better.
# 
# [Back to contents](#top)
# 
# 
# 
