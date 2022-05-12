#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Welcome. This tutorial focuses on developing a model to predict California House Prices. It is intended for a beginner-level audience.
# 
# ## Purpose
# 
# This notebook serves two purposes:
# 
# * it acts as a beginner-level worked example of a machine learning regression problem
# * it consolidates my own understanding of the theory and practice of following the data science framework contained within this notebook.
# 
# 
# ## Competition Description
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# ## Goal
# 
# It is your job to predict the sales price for each house. For each __Id__ in the test set, you must predict the value of the __SalePrice__ variable. 
# 
# ## Metrics
# 
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)
# 
# ## Approach
# 
# We will use the following approach to predict which passengers survived:
# 
# * Problem Statement
# * Data Gathering
# * Data Exploration
# * Data Preparation
# * Model Building
# * Model Tuning
# * Feature Selection
# * Model Validation
# * Model Testing and Submission
# 
# __NOTE__ Early versions of this notebook will focus on obtaining baseline results for common regression algorithms and meta-learning algorithms. Early versions will purposely contain minimal Exploratory Data Analysis (EDA) and data preperation. Instead, the intial focus will be to use features that require minimal pre-processing and be restricted to numerical types. Once a working pipeline is in place, subsequent revisions will iteratively develop and refine each of the steps outlined in the approach above.  
# 
# # Data Gathering
# 
# The three data sources of interest are:
# 
# * test.csv
# * train.csv
# * data_description.txt
# 
# In this version of the notebook, the data is not enriched with external sources.

# In[ ]:


# All modules and packages are imported at the start of the notebook
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost
from xgboost import XGBRegressor
import lightgbm
from lightgbm import LGBMRegressor

# define constants we'll use in the modelling and validation stages.
RANDOM_SEED = 42 #ensure consistency of tests
CV_FOLDS = 12


# In[ ]:


# The accompanying data_description.txt file indicates the type of each of the test and training data features.
# 
# While setting the type of categorical features to 'category' can save memory and help with filtering activities such as selecting features based on dtype, it often causes problems.
# during data wrangling. For example, replacing missing categorical data with a category 'NA' that's not in the original category set will fail.
# We also run into problems with onehotencoding after removing minority categories. For this reason, I will limit setting of dtypes to numerical values only.

data_types = {'MSZoning': 'category',
              'Street': 'category',
              'Alley': 'category',
              'LotShape': 'category',
              'LandContour': 'category',
              'Utilities': 'category',
              'LotConfig': 'category',
              'LandSlope': 'category',
              'Neighborhood': 'category',
              'Condition1': 'category',
              'Condition2': 'category',
              'BldgType': 'category',
              'HouseStyle': 'category',
              'RoofStyle': 'category',
              'RoofMatl': 'category',
              'Exterior1st': 'category',
              'Exterior2nd': 'category',
              'MasVnrType': 'category',
              'ExterQual': 'category',
              'ExterCond': 'category',
              'Foundation': 'category',
              'BsmtQual': 'category',
              'BsmtCond': 'category',
              'BsmtExposure': 'category',
              'BsmtFinType1': 'category',
              'BsmtFinType2': 'category',
              'ExterQual': 'category',
              'Heating': 'category',
              'HeatingQC': 'category',
              'CentralAir': 'category',
              'Electrical': 'category',
              'KitchenQual': 'category',
              'Functional': 'category',
              'FireplaceQu': 'category',
              'GarageType': 'category',
              'GarageFinish': 'category',
              'GarageQual': 'category',
              'GarageCond': 'category',
              'PavedDrive': 'category',
              'PoolQC': 'category',
              'Fence': 'category',
              'MiscFeature': 'category',
              'SaleType': 'category',
              'SaleCondition': 'category',
              'MoSold': 'int64',
              'YrSold': 'int64',
              'YearBuilt': 'int64'}

data_types = {'MoSold': 'int64',
              'YrSold': 'int64',
              'YearBuilt': 'int64'}
              
# load the training an test data into seperate pandas DataFrames
#df_train = pd.read_csv('train.csv', index_col='Id', dtype=data_types)
#df_test = pd.read_csv('test.csv', index_col='Id', dtype=data_types)

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id', dtype=data_types)
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id', dtype=data_types)


# # Data Exploration

# Let's start with a brief summary of the DataFrame.

# In[ ]:


df_train.info()


# There are:
# 
# * 1460 rows of data
# * a mixture of category(43), float64(3), int64(34) data types.
# * 80 features and an index Id
# * the entire DataFrame only takes up 0.5 MB of memory. If we hadn't set the Dtype of objects upon reading from csv file, it would have been double this.
# 
# Let's now calculate missing (null) as percentage of the total number of records.

# In[ ]:


# Helper function to display missing data by feature
# https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe#26266451 
#
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'No. Missing Values', 1 : '%'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '%', ascending=False).round(1)
        print ("There are " + str(mis_val_table_ren_columns.shape[0]) +
               " features with missing values.")
        
        return mis_val_table_ren_columns

missing_values_table(df_train)


# Of the 19 features that have missing data, 6 (5 of type object, 1 of type float) have > 20% of missing data. 
# 
# * PoolQC (ojb)
# * MiscFeature (ojb)
# * Alley (ojb)
# * Fence (ojb)
# * FireplaceQu (num)
# 
# During the data preparation section we will:
# 
# **Features with 20% or more missing data**
# * Drop numerical features with more than 20% missing data.
# * Create a new category 'Missing_Data' for object/category features. This avoids losing potentially valuable categorical data while avoiding using mode.
# 
# **Features with less than 20% missing data**
# * Use either mean or median for missing numerical data. The degree of skewness will guide us (refer to EDA)
# * Use mode for missing object/category features.

# In[ ]:


df_train.describe()


# As expected, there's a large variation in scale between each of the features. Prescaling of data will be important for almost all of the modelling algorithms we use. 

# In[ ]:


df_train.head()


# In[ ]:


def num_summary(df, feature):
    """ Creates a distplot of the specified numerical column of a DataFrame.

    Args:
        df: DataFrame
        num_label: The column to be plotted

    Returns:
        None:
    """
    print("Feature: ", feature, '\n__________________________\n')
    print(df[feature].describe())
    print('skew: ', df[feature].skew())
    print('kurtosis: ', df[feature].kurtosis())
    
    minimum = df[feature].min()
    maximum = df[feature].max()
    sns.distplot(df[feature])
    plt.xlim((minimum-0.5, maximum+0.5))
    plt.title(feature)
    plt.show()
    
    sns.boxplot(data=df, x=feature)
    plt.show()
    
def cat_summary(df, feature, normalise=True):
    """ Creates a distplot of the specified nominal column of a DataFrame.

    Args:
        df: DataFrame
        num_label: The column to be plotted

    Returns:
        None:
    """
    print("Feature: ", feature, '\n__________________________\n')
    print('value_count: ', df[feature].value_counts(normalize=normalise))
    print('skew: ', df[feature].value_counts(normalize=normalise).skew())
    print('kurtosis: ', df[feature].value_counts(normalize=normalise).kurtosis())
    
    df[feature].value_counts().sort_values(ascending=False).plot(kind='bar')
    plt.title(feature)
    plt.show()


# In[ ]:


numerical_features = df_train.select_dtypes(include='number').columns.tolist()

# A number of features have zero KDE. This is causing distplot to throw an error. Removing features from visual exploration until later.
numerical_features.remove('BsmtFinSF2')
numerical_features.remove('LowQualFinSF')
numerical_features.remove('BsmtHalfBath')
numerical_features.remove('KitchenAbvGr')
numerical_features.remove('EnclosedPorch')
numerical_features.remove('3SsnPorch')
numerical_features.remove('ScreenPorch')
numerical_features.remove('PoolArea')
numerical_features.remove('MiscVal')

for feature in numerical_features:
    num_summary(df_train, feature)


# In this version of the notebook, let's keep observations to a high-level highlights
# 
# * A number of the distributions are skewed (mostly right skewed).
# * Most features have outliers to be considered and potentially removed.
# 
# 
# Let's look at how closely SalePrice follows a normal distrubtion. We saw from the earlier plot that it is rigt skewed (1.88), but let's take an even closer look.

# In[ ]:


#histogram and normal probability plot of SalePrice
sns.distplot(df_train['SalePrice'])
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# The actual ordered values deviate from the theoretical (normally distributed values). Let's log transform SalesPrice and see if that improves actual vs theoretical.

# In[ ]:


#histogram and normal probability plot of the natural log of SalePrice
sns.distplot(np.log(df_train['SalePrice']))
fig = plt.figure()
res = stats.probplot(np.log(df_train['SalePrice']), plot=plt)


# The log transform of SalePrice has made it significally more 'Normal'. A similar approach with 'GrLivArea' achieves a similar result.

# In[ ]:


#histogram and normal probability plot of the natural log of GrLivArea
sns.distplot(np.log(df_train['GrLivArea']))
fig = plt.figure()
res = stats.probplot(np.log(df_train['GrLivArea']), plot=plt)


# In[ ]:


categorical_features = df_train.select_dtypes(include='object').columns.tolist()

for feature in categorical_features:
    cat_summary(df_train, feature)


# In this version of the notebook, let's keep observations to high-level highlights
# 
# * Cardinality is low to medium across all the features
# * A number of features are dominated by 1~2 categories. We may be able to use this to quickly reduce dimensionality when categorical data is included into the modelling process. If one category dominates (>80%) we can probably drop the feature. If there are lot's of minority features, we can probably drop them during the pipeline transformation.
# 
# 

# Let's now look at how featues correlate and then pairplot of interesting feature pairs.

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap((df_train.select_dtypes(include='number')).astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=False)


# In[ ]:


def get_redundant_pairs(df):
    """ Determine the diagonal and lower triangular pairs of correlation matrix

    Args:
        df: DataFrame

    Returns:
        df: A list of tuples (row, column) of cells to drop in correlation
        matrix
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    """ Determine the correlation coefficients of features in a DataFrame
    and return the top n in order of importance.

    Args:
        df: DataFrame
        n:  The number of top coefficients to return

    Returns:
        df: The top n correlation coefficients
    """
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print("=========================\n")

print(get_top_abs_correlations(df_train[df_train.select_dtypes(include='number').columns.tolist()], 15))


# There's a high degree (> 0.7) of linear correlation between the following features:
# 
# * GarageCars <-> GarageArea
# * YearBuilt <-> GarageYrBlt
# * GrLivArea <-> TotRmsAbvGrd
# * TotalBsmtSF <-> 1stFlrS
# * OverallQual <-> SalePrice
# * GrLivArea <-> SalePrice
# 
# We want our model to consist of independent features, so let us drop the following features:
# 
# * GarageCars
# * GarageYrBlt
# * TotRmsAbvGrd
# * 1stFlrS

# # Data Preparation
# 
# In this version we'll be including categorical data (by performing one hot encoding) after dropping or imputing features as detailed earlier. This will be extensively developed in future versions.
# 
# I'll be using sklearn.pipeline.Pipeline to manage most (all in a later iteration) of the transformation of training and testing data and developing custom transform functions as required.

# In[ ]:


# let's start from a fresh copy of the data.
data_types = {'MoSold': 'int64',
              'YrSold': 'int64',
              'YearBuilt': 'int64'}

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id', dtype=data_types)
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id', dtype=data_types)


# In[ ]:


# let's perform additional feature engineering
for data in [df_train, df_test]:
    data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    # These features do not improve performance.
    #
    #data['YrBltAndRemod'] = data['YearBuilt']+data['YearRemodAdd']
    #data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    #data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +
    #                             data['1stFlrSF'] + data['2ndFlrSF'])
    #data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +
    #                           data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
    #data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +
    #                          data['EnclosedPorch'] + data['ScreenPorch'] +
    #                          data['WoodDeckSF'])


# In[ ]:


# create a list of numerical features without the target SalePrice
numerical_features = df_train.select_dtypes(include='number').columns.tolist()
numerical_features.remove('SalePrice')

# create a list of categorical features
categorical_features = df_train.select_dtypes(include='object').columns.tolist()


# In[ ]:


# numerical features to drop. More than 20% missing data OR correlation highly (>0.7) with other features
numerical_features_to_drop = ['FireplaceQu'] + ['GarageCars', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF']

for data in [df_train, df_test]:
    data.drop(numerical_features_to_drop,inplace=True, axis=1)     


# In[ ]:


# categorical features with more than 20% missing data. Replace missing data with new category 'missing_data'
categorical_features_na = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']

for data in [df_train, df_test]:
    for feature in categorical_features_na:
        data[feature].fillna("missing_value", inplace = True)


# Let's review how the transformations have altered df_train

# In[ ]:


missing_values_table(df_train)


# Good. All features now have less than 20% missing data. Imputation will address the remaing missing data. The rest of data preparation we'll do inside of the main transformation pipeline operation.

# In[ ]:


class TransLog(BaseEstimator, TransformerMixin):
    def __init__(self, feature_index):
        
        self.feature_index = feature_index

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_trans = np.log(X[:, self.feature_index])
        return np.c_[X[:, 0:self.feature_index - 1], X_trans, X[:, self.feature_index:]]


# In[ ]:


# Separate train data into X (input features) and y (output target) 

# Log transform y_train. Remember, we'll need to raise e to the power y_test before submission!
y_train = np.log(df_train['SalePrice'])

X_train = df_train
X_train.drop(columns=['SalePrice'], inplace=True)

# We only have input features for test, hence no need to drop 'SalePrice'
X_test = df_test

# Recompute the list of numerical and categorical features
numerical_features = df_test.select_dtypes(include='number').columns.tolist()
categorical_features = df_test.select_dtypes(include='object').columns.tolist()


# In[ ]:


# As the Pipeline works with Series not dataframes, we need the index of feature GrLivArea
log_feature_index = numerical_features.index('GrLivArea')


# In[ ]:


# Define the numerical transformation pipeline
num_pipeline = Pipeline([
                        ('simple_imp', SimpleImputer(strategy='median')),
                        ('log_trans', TransLog(feature_index=log_feature_index)),
                        ('std_scaler', StandardScaler()) #minmaxscaler is not an improvement.
                        ])


cat_pipeline = Pipeline([
                        ('simple_imp', SimpleImputer(strategy='most_frequent')),
                        ('one_hot_enc', OneHotEncoder(handle_unknown='ignore'))
                        ])

# Define a composite transformation pipeline.
comp_pipeline = ColumnTransformer([('num_pipeline', num_pipeline, numerical_features),
                                   ('cat_pipeline', cat_pipeline, categorical_features)
                                  ])

# fit AND transform X_train using the pipeline.
X_train_prepared = comp_pipeline.fit_transform(X_train)

# NOTE: we have already fit the pipeline using X_train. We therefore only call transform on X_test. We don't want to fit the test set. Not only is it wrong, but because there are minority categories in some features, X_train_prepared and X_test_prepared will be of different dimensions!
X_test_prepared = comp_pipeline.transform(X_test)


# Confirm the feature dimension of train and test are the same. If not, go back to the transformation pipeline and fix it.

# In[ ]:


print(X_train_prepared.shape)
print(X_test_prepared.shape)


# # Model Building
# 
# In this version of the notebook, let's limit modelling to only a handful of models:
# 
# * Linear Regression - maninly serving as a baseline.
# * Random Forest
# * AdaBoost
# * XGBoost
# * Lasso
# * ElasticNet
# * LightGBM
# * Stacking - mainly to plumb in meta-modelling for use in later versions of the notebook.

# In[ ]:


# Let's keep track of each model's performance
results = []


# In[ ]:


# Linear Regression classifier
lr_clf = LinearRegression()

get_ipython().run_line_magic('time', 'lr_clf.fit(X_train_prepared, y_train)')

# Apply the model to the train dataset.
y_train_predicted = lr_clf.predict(X_train_prepared)

results.append({'alg': 'lr_clf', 'model': lr_clf, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
result=results[-1]
print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# In[ ]:


# random forest classifier
rf_clf = RandomForestRegressor(random_state=RANDOM_SEED)

get_ipython().run_line_magic('time', 'rf_clf.fit(X_train_prepared, y_train)')

# Apply the model to the train dataset.
y_train_predicted = rf_clf.predict(X_train_prepared)

results.append({'alg': 'rf_clf', 'model': rf_clf, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
result=results[-1]
print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

ada_clf=AdaBoostRegressor(random_state=RANDOM_SEED, n_estimators=100)

get_ipython().run_line_magic('time', 'ada_clf.fit(X_train_prepared,y_train)')

# Apply the model to the train dataset.
y_train_predicted = ada_clf.predict(X_train_prepared)

print('Feature Importance Values\n\n',ada_clf.feature_importances_)

results.append({'alg': 'ada_clf', 'model': ada_clf, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
result=results[-1]
print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# Computation time has gone up dramatically, with no real improvemnt in MAE compared with Linear Regression. We can however see that many of the features coefficients are zero. We may be able to use this information to reduce features in later versions of the notebook.

# In[ ]:


# XGBoost Classifier
xgb_clf = XGBRegressor(objective="reg:squarederror", seed=RANDOM_SEED)

get_ipython().run_line_magic('time', 'xgb_clf.fit(X_train_prepared, y_train)')

# Apply the model to the train dataset.
y_train_predicted = xgb_clf.predict(X_train_prepared)

results.append({'alg': 'xgb_clf', 'model': xgb_clf, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
result=results[-1]
print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# XGBoost is currently leading the way with a significant reduction in MAE over other algorithms.
# 
# Eventually we'll have a full compliment of ensemble models and incoporate stacking. Right now, we don't have enought models to take advantage of it, but let's plumb it in ready for later versions of the notebook.

# In[ ]:


# Lasso Classifier
lso_clf = Lasso(random_state=RANDOM_SEED)

get_ipython().run_line_magic('time', 'lso_clf.fit(X_train_prepared, y_train)')

# Apply the model to the train dataset.
y_train_predicted = lso_clf.predict(X_train_prepared)

results.append({'alg': 'lso_clf', 'model': lso_clf, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
result=results[-1]
print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# In[ ]:


# ElasticNet Classifier
eln_clf = ElasticNet(random_state=RANDOM_SEED)

get_ipython().run_line_magic('time', 'eln_clf.fit(X_train_prepared, y_train)')

# Apply the model to the train dataset.
y_train_predicted = eln_clf.predict(X_train_prepared)

results.append({'alg': 'eln_clf', 'model': eln_clf, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
result=results[-1]
print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# In[ ]:


# LightGBM Classifier
lgbm_clf = LGBMRegressor(random_state=RANDOM_SEED)

get_ipython().run_line_magic('time', 'lgbm_clf.fit(X_train_prepared, y_train)')

# Apply the model to the train dataset.
y_train_predicted = lgbm_clf.predict(X_train_prepared)

results.append({'alg': 'lgbm_clf', 'model': lgbm_clf, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
result=results[-1]
print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# In[ ]:


# Summarise Results - Sorted by MAE
print('\n-------------------------Results-----------------------')
for result in sorted(results, key=lambda k: k['mae']) :
    print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# # Feature Selection
# Minor feature selection performed earlier based on correlation matrix. More feature selection will be performed in future versions of the notebook.
# 
# 
# # Model Tuning
# Let's start with grid search and cross-validation for each model in preparation for stacking.

# In[ ]:


# limit scope of variables
xgb_param_grid = {
        'min_child_weight': [1],
        'gamma': [0],
        'subsample': [1.0],
        'colsample_bytree': [0.2],
        'max_depth': [3],
        'n_estimators': [2500]
        }


#xgb_param_grid = {
#        'min_child_weight': [1, 2, 3, 4],
#        'gamma': [0, 0.1, 0.2],
#        'subsample': [0.8, 0.9, 1.0],
#        'colsample_bytree': [0.2, 0.25, 0.3, 0.35, 0.4],
#        'max_depth': [3, 4, 5, 6],
#        'n_estimators': [2000]
#        }

# Perform grid search with cross validation
xgb_grid_search = GridSearchCV(xgb_clf, xgb_param_grid, cv=CV_FOLDS, n_jobs=4, return_train_score=True, verbose=1)

get_ipython().run_line_magic('time', 'xgb_grid_search.fit(X_train_prepared, y_train)')

# Get the best model parameters
print(xgb_grid_search.best_params_)
print(xgb_grid_search.best_score_)

y_train_predicted = xgb_grid_search.best_estimator_.predict(X_train_prepared)

results.append({'alg': 'xgb_clf_tuned', 'model': xgb_grid_search.best_estimator_, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted), 'best_params': xgb_grid_search.best_params_})


# In[ ]:


# limit scope of variables to test
rf_param_grid = [{'ccp_alpha': [0.0],
                  'criterion': ['mse'],  # mae is causing a problem. Live lock? 
                  'max_depth': [None],
                  'max_features': [None],
                  'max_leaf_nodes': [None],
                  'min_impurity_decrease': [0.0],
                  'oob_score': [True],
                  'min_impurity_split': [None],
                  'min_samples_leaf': [2],
                  'min_samples_split': [4],
                  'min_weight_fraction_leaf': [0.0],
                  'random_state': [42],
                 }
                ]

#rf_param_grid = [{'ccp_alpha': [0.0],
#                  'criterion': ['mse'],  # mae is causing a problem. Live lock? 
#                  'max_depth': [None, 2, 5],
#                  'max_features': [None, 'auto'],
#                  'max_leaf_nodes': [None],
#                  'min_impurity_decrease': [0.0],
#                  'oob_score': [True],
#                  'min_impurity_split': [None],
#                  'min_samples_leaf': [1, 2],
#                  'min_samples_split': [4, 5, 6],
#                  'min_weight_fraction_leaf': [0.0],
#                  'random_state': [42],
#                 }
#                ]

# Perform grid search.
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, n_jobs=4, cv=CV_FOLDS, return_train_score=True, verbose=1)

get_ipython().run_line_magic('time', 'rf_grid_search.fit(X_train_prepared, y_train)')

# Get the best model parameters
print(rf_grid_search.best_params_)
print(rf_grid_search.best_score_)

y_train_predicted = rf_grid_search.best_estimator_.predict(X_train_prepared)

results.append({'alg': 'rf_clf_tuned', 'model': rf_grid_search.best_estimator_, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted), 'best_params': rf_grid_search.best_params_})


# In[ ]:


# limit scope of variables to test
ada_param_grid = [{'n_estimators': [2000],
                   'loss': ['square'],
                   'random_state': [RANDOM_SEED]
                 }
                ]

#ada_param_grid = [{'n_estimators': [50, 100, 300, 2000],
#                   'loss': ['linear', 'square', 'exponential'],
#                   'random_state': [RANDOM_SEED]
#                 }
#                ]

# Perform grid search.
ada_grid_search = GridSearchCV(ada_clf, ada_param_grid, n_jobs=4, cv=CV_FOLDS, return_train_score=True, verbose=1)

get_ipython().run_line_magic('time', 'ada_grid_search.fit(X_train_prepared, y_train)')

# Get the best model parameters
print(ada_grid_search.best_params_)
print(ada_grid_search.best_score_)

y_train_predicted = ada_grid_search.best_estimator_.predict(X_train_prepared)


results.append({'alg': 'ada_clf_tuned', 'model': ada_grid_search.best_estimator_, 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
result=results[-1]
print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# In[ ]:


#from sklearn.linear_model import Lasso

# Stacking
#estimators = [
#              ('ada_clf', ada_grid_search.best_estimator_),
#              ('xgb_clf', xgb_grid_search.best_estimator_),
#              ('rf_clf', rf_grid_search.best_estimator_)             ]
             
#stack_clf = StackingRegressor(estimators=estimators, final_estimator=Lasso())

#%time  stack_clf.fit(X_train_prepared, y_train)

#y_train_predicted = stack_clf.predict(X_train_prepared)

#results.append({'alg': 'stack_clf', 'mse': mean_squared_error(y_train, y_train_predicted), 'mae': mean_absolute_error(y_train, y_train_predicted)})
#result=results[-1]
#print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])


# # Model Testing & Submission

# In[ ]:


# Summarise Results - Sorted by MAE
print('\n-------------------------Results-----------------------')
for result in sorted(results, key=lambda k: k['mae']) :
    print('Algorithm:', result['alg'], 'MSE:', result['mse'], 'MAE:', result['mae'])
    
best_model = sorted(results, key=lambda k: k['mae'])[0]['model']
print(best_model)

y_test_predicted = np.exp(best_model.predict(X_test_prepared))

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': y_test_predicted})
#output.to_csv('submission.csv', index=False)
output.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:




