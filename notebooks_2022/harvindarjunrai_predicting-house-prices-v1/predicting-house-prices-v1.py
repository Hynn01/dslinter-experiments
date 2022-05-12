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


# # Predicting House Prices (Ames Housing Dataset)

# For this project, the house prices from the Ames Housing test dataset will be predicted. There are two datasets that will be imported from this competition:
# 
# - Training dataset "train.csv" for whom the Supervised Learning Models will be trained upon;
# - Test dataset "test.csv" where the house prices will be predicted.

# ## Acknowledgements
# 
# The work presented was inspired from the Machine Learning Tutorials I have gone through here on Kaggle and from other sources. I would also like to thank [A. Qua](https://www.kaggle.com/adibouayjan) and his [notebook](https://www.kaggle.com/code/adibouayjan/house-price-step-by-step-modeling) which provided the inspiration for this notebook and I highly recommended looking through if you want to pick up some important techniques with regards to exploratory data analysis, feature engineering and training and analysis of supervised learning models.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Model functions
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Statistics functions
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# Functions to calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Function to split data into different groups
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold

# Function to deal with missing values via imputation
from sklearn.impute import SimpleImputer

# Function that converts categorical values into numerical values via ordinal encoding or one-hot encoding
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# First, we import the training and test datasets from the House Prices Advanced Regression competition into data_train and data_test respectively. As seen by the shape of the two datasets, data_test has one less column than data_train as it does not contain the target variable "SalePrice".

# In[ ]:


data_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
data_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(f"Training set shape: {data_train.shape}\n")
print(f"Test set shape: {data_test.shape}\n")


# In[ ]:


data_train.info()


# Looking at the information from the data_train DataFrame, we see that some of these columns have missing values. These missing values will be dealt with later. But before we do some exploratory data analysis and feature engineering, we check to see that both datasets have the same feature variables (i.e. they both have the same columns except for "SalePrice" in data_train).

# In[ ]:


# Checking if column headings are the same in both data set
dif_1 = [x for x in data_train.columns if x not in data_test.columns]
print(f"Columns present in df_train and absent in df_test: {dif_1}\n")

dif_2 = [x for x in data_test.columns if x not in data_train.columns]
print(f"Columns present in df_test set and absent in df_train: {dif_2}")


# Before we proceed investigating the numerical and categorical features from both datasets, the "Id" column will be dropped from both datasets as well. The list of IDs from the test set will be saved so that it can be used later on when we predict the house prices from that dataset.

# In[ ]:


# Drop the 'Id' column from the training set
data_train.drop(["Id"], axis=1, inplace=True)

# Save the 'Id' list before dropping it from the test set
Id_test_list = data_test["Id"].tolist()
data_test.drop(["Id"], axis=1, inplace=True)


# At certain stages of this project, the shape of both datasets will be printed out to clarify that both datasets have had the same features either added on or removed. The shape of both datasets now before we investigate the numerical and categorical features from both datasets stands as follows:

# In[ ]:


print(f"Training set shape: {data_train.shape}\n")
print(f"Test set shape: {data_test.shape}\n")


# ## Numerical Data

# Firstly we will focus on the numerical features from both datasets where we will do some exploratory data analysis and some feature engineering to
# 
# * Investigate the distribution of these numerical features;
# * Investigate the correlation of these numerical features with "SalePrice" and drop any features that have low correlation;
# * Deal with missing values through imputation.
# 
# To start off with, we select only the numerical columns from data_train (including "SalePrice") by obtaining the list of numerical columns from data_train in numerical_cols, and using this to take out the numerical columns from data_train and store them in a new DataFrame data_train_num (we also do the same thing with the test dataset).

# In[ ]:


# Select numerical columns from the training dataset
numerical_cols = [cname for cname in data_train.columns if 
                  data_train[cname].dtype in ['int64', 'float64']]

numerical_cols_test = [cname for cname in data_test.columns if 
                      data_test[cname].dtype in ['int64', 'float64']]

data_train_num = data_train[numerical_cols].copy()
data_test_num = data_test[numerical_cols_test].copy()

data_train_num.head()


# ### Distribution

# In[ ]:


# Plot the distribution of all the numerical features
fig_ = data_train_num.hist(figsize=(16, 20), bins=50, color="deepskyblue",
                           edgecolor="black", xlabelsize=8, ylabelsize=8)


# Looking at the distriutions of each numerical feature, we see that there are a variety of distributions on show and that we have a mixture of discrete and continuous variables in our dataset. One thing that we can spot from these plots are variables that has small variation (i.e. they have very similar values) - these variables are likely to only have a small impact on the final price of the house. We will drop any variables where 95% of the values are similar or constant.

# In[ ]:


# Drop any quasi-constant features where 95% of the values are similar or constant
sel = VarianceThreshold(threshold=0.05) # 0.05: drop column where 95% of the values are constant

# The fit finds the features with constant variance
sel.fit(data_train_num.iloc[:, :-1])


# Get the number of features that are not constant
print(f"Number of retained features: {sum(sel.get_support())}")
print(f"\nNumber of quasi_constant features: {len(data_train_num.iloc[:, :-1].columns) - sum(sel.get_support())}")

quasi_constant_features_list = [x for x in data_train_num.iloc[:, :-1].columns if x not in data_train_num.iloc[:, :-1].columns[sel.get_support()]]

print(f"\nQuasi-constant features to be dropped: {quasi_constant_features_list}")


# From this, we see that only one variable "KitchenAbvGr" (Kitchens above the grade) has at least 95% of variables that are similar and constant, so this will be removed from both the training and test datasets.

# In[ ]:


# Drop quasi-constant features from both datasets
data_train_num.drop(quasi_constant_features_list, axis=1, inplace=True)
data_test_num.drop(quasi_constant_features_list, axis=1, inplace=True)


# In[ ]:


print(f"Training set shape (Numerical features): {data_train_num.shape}\n")
print(f"Test set shape (Numerical features): {data_test_num.shape}\n")


# ### Correlation

# We will now produce a correlation heatmap showing the correlation between all of the numerical variables including the correlation of each numerical feature with "SalePrice". Any variables that have a low correlation with "SalePrice" is likely not to have a huge impact on the final sale price of the house, any variables with correlation less than |0.3| will be replaced by 0 for simplicity.

# In[ ]:


# Heatmap for all the remaining numerical data including the taget 'SalePrice'

# Define the heatmap parameters
pd.options.display.float_format = "{:,.2f}".format

# Define correlation matrix
corr_matrix = data_train_num.corr()

# Replace any correlation < |0.3| by 0 for a better visibility
corr_matrix[(corr_matrix < 0.3) & (corr_matrix > -0.3)] = 0

# Mask the upper part of the heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Choose the color map
cmap = "viridis"

# plot the heatmap
sns.set(rc = {'figure.figsize':(20,15)})
sns.heatmap(corr_matrix, mask=mask, vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot_kws={"size": 9, "color": "black"}, square=True, cmap=cmap, annot=True)


# Looking at the heatmap above, we see that 18 of these numerical features has some noticeable correlation with "SalePrice". There are also a few variables that have a very strong positive correlation (of at least 0.8) including
# 
# * "GarageArea" and "GarageCars" (0.88)
# * "GarageYrBlt" and "YearBuilt" (0.83)
# * "TotRmsAbvGrd" and "GrLivArea" (0.83)
# * "1stFlrSF" and "TotalBsmtSF" (0.82)
# 
# The other correlations will be dealt with later, but first we will focus on features that have a correlaton of more than |0.3| with "SalePrice".

# In[ ]:


# Select features where the correlation with 'SalePrice' is higher than |0.3|
# -1 because the latest row is SalePrice
data_num_corr = data_train_num.corr()["SalePrice"][:-1]

# Correlated features (r2 > 0.5)
high_features_list = data_num_corr[abs(data_num_corr) >= 0.5].sort_values(ascending=False)
print(f"{len(high_features_list)} strongly correlated values with SalePrice:\n{high_features_list}\n")

# Correlated features (0.3 < r2 < 0.5)
low_features_list = data_num_corr[(abs(data_num_corr) < 0.5) & (abs(data_num_corr) >= 0.3)].sort_values(ascending=False)
print(f"{len(low_features_list)} slightly correlated values with SalePrice:\n{low_features_list}")


# In[ ]:


# Features with high correlation (higher than 0.5)
strong_features = data_num_corr[abs(data_num_corr) >= 0.5].index.tolist()
strong_features.append("SalePrice")

data_strong_features = data_train_num.loc[:, strong_features]

plt.style.use("seaborn-whitegrid")  # define figures style
fig, ax = plt.subplots(round(len(strong_features) / 3), 3)

for i, ax in enumerate(fig.axes):
    # plot the correlation of each feature with SalePrice
    if i < len(strong_features)-1:
        sns.regplot(x=strong_features[i], y="SalePrice", data=data_strong_features, ax=ax, scatter_kws={
                    "color": "deepskyblue"}, line_kws={"color": "black"})


# In[ ]:


# Features with low correlation (between 0.3 and 0.5)
low_features = data_num_corr[(abs(data_num_corr) >= 0.3) & (abs(data_num_corr) < 0.5)].index.tolist()
low_features.append("SalePrice")

data_low_features = data_train_num.loc[:, low_features]

plt.style.use("seaborn-whitegrid")  # define figures style
fig, ax = plt.subplots(round(len(low_features) / 3), 3)

for i, ax in enumerate(fig.axes):
    # plot the correlation of each feature with SalePrice
    if i < len(low_features) - 1:
        sns.regplot(x=low_features[i], y="SalePrice", data=data_low_features, ax=ax, scatter_kws={
                    "color": "deepskyblue"}, line_kws={"color": "black"},)


# Looking at some of these scatter plots related to the surface area of certain rooms in the house ("1stFlrSF", "TotalBsmtSF", "GrLivingArea"), there is at least one house whose price is relatively inexpensive given their surface area (located on the bottom right-hand-side of those graphs) which suggests the presence of outliers in the dataset. These will be dealt with in a future version of this project, but for now we will keep these numerical features shown in the scatter plot above.

# In[ ]:


# Define the list of numerical fetaures to keep
list_of_numerical_features = strong_features[:-1] + low_features

print("List of features to be kept in the dataset:")
print(list_of_numerical_features)

# Select these features form our training set
data_train_num = data_train_num.loc[:, list_of_numerical_features]

# Select the same features from the test set (-1 -> except 'SalePrice')
data_test_num = data_test_num.loc[:, list_of_numerical_features[:-1]]


# Looking at the other correlation values beyond "SalePrice", since the correlation between "GarageCars" and "GarageArea" is so high, "GarageCars" will be dropped from both datasets as this is not likely to add in any relevant new information with regards to the final price of the house.

# In[ ]:


data_train_num.drop(["GarageCars"], axis = 1, inplace = True)
data_test_num.drop(["GarageCars"], axis = 1, inplace = True)


# In[ ]:


print(f"Training set shape (Numerical features): {data_train_num.shape}\n")
print(f"Test set shape (Numerical features): {data_test_num.shape}\n")


# ### Missing Values

# In[ ]:


# Get names of columns with missing values (training set)
cols_with_missing = [col for col in data_train_num.columns
                     if data_train_num[col].isnull().any()]

print("Columns with missing (NA) values:")
print(cols_with_missing)

# Count how many NA values are in each of those columns
cols_nan_count = list(map(lambda col: round(data_train_num[col].isna().sum()*100/len(data_train_num)), cols_with_missing))


tab = pd.DataFrame(cols_with_missing, columns=["Column"])
tab["Percent_NaN"] = cols_nan_count
tab.sort_values(by=["Percent_NaN"], ascending=False, inplace=True)


# Define figure parameters
sns.set(rc={"figure.figsize": (10, 7)})
sns.set_style("whitegrid")

# Plot results
p = sns.barplot(x="Percent_NaN", y="Column", data=tab,
                edgecolor="black", color="deepskyblue")

p.set_title("Percent of NaN per column of the train set\n", fontsize=20)
p.set_xlabel("\nPercent of NaN (%)", fontsize=20)
p.set_ylabel("Column Name\n", fontsize=20)


# From the updated numerical features training set, we see that there are three columns with missing values, with "LotFrontage" having around 18% missing values. The missing values from these columns will be imputed where they will be replaced with the median value from each column (which is sensible given that one of the columns is "GarageYrBlt" where the values are strictly discrete, plus they're less affected by outliers).

# In[ ]:


# Imputation of missing values (NaNs) with SimpleImputer
my_imputer = SimpleImputer(strategy="median")
data_train_imputed = pd.DataFrame(my_imputer.fit_transform(data_train_num))
data_train_imputed.columns = data_train_num.columns


# We will also check the distributions to ensure that the distribution of these variables after imputation is not heavily impacted by these changes.

# In[ ]:


# Check the distribution of each imputed feature before and after imputation

# Define figure parameters
sns.set(rc={"figure.figsize": (14, 12)})
sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 2)

# Plot the results
for feature, fig_pos in zip(["LotFrontage", "GarageYrBlt", "MasVnrArea"], [0, 1, 2]):

    """Features distribution before and after imputation"""

    # before imputation
    p = sns.histplot(ax=axes[fig_pos, 0], x=data_train_num[feature],
                     kde=True, bins=30, color="deepskyblue", edgecolor="black")
    p.set_ylabel(f"Before imputation", fontsize=14)

    # after imputation
    q = sns.histplot(ax=axes[fig_pos, 1], x=data_train_imputed[feature],
                     kde=True, bins=30, color="darkorange", edgecolor="black")
    q.set_ylabel(f"After imputation", fontsize=14)


# Looking at the distribution plots, we see that the distributions for "LotFrontage" and "GarageYrBlt" have been changed by imputation (with heavy bias towards the median for "LotFrontage"). Since there is noticeable bias towards the median class for both "LotFrontage" and "GarageYrBlt", these features will be removed from both datasets. The distribution of "MasVnrArea" has not changed that much at all after imputation, and will be kept on.

# In[ ]:


# Drop 'LotFrontage' and 'GarageYrBlt'
data_train_imputed.drop(["LotFrontage", "GarageYrBlt"], axis=1, inplace=True)
data_train_imputed.head()


# In[ ]:


# Drop these same features from test set
data_test_num.drop(["LotFrontage", "GarageYrBlt"], axis=1, inplace=True)


# Now we will deal with missing values from the test dataset.

# In[ ]:


# Get names of columns with missing values (test set)
cols_with_missing_b = [col for col in data_test_num.columns
                       if data_test_num[col].isnull().any()]

print("Columns with missing (NA) values:")
print(cols_with_missing_b)

# Count how many NA values are in each of those columns
cols_nan_count_b = list(map(lambda col: round(data_test_num[col].isna().sum()*100/len(data_test_num)), cols_with_missing_b))


tab = pd.DataFrame(cols_with_missing_b, columns=["Column"])
tab["Percent_NaN"] = cols_nan_count_b
tab.sort_values(by=["Percent_NaN"], ascending=False, inplace=True)


# Define figure parameters
sns.set(rc={"figure.figsize": (10, 7)})
sns.set_style("whitegrid")

# Plot results
p = sns.barplot(x="Percent_NaN", y="Column", data=tab,
                edgecolor="black", color="deepskyblue")

p.set_title("Percent of NaN per column of the train set\n", fontsize=20)
p.set_xlabel("\nPercent of NaN (%)", fontsize=20)
p.set_ylabel("Column Name\n", fontsize=20)


# Here there are 4 feature variables that have missing values, but they all have a very small percentage of missing values. Each of these values will be filled in with the median from each perspective column.

# In[ ]:


# Imputation of missing values (NaNs) with SimpleImputer
my_imputer = SimpleImputer(strategy="median")
data_test_imputed = pd.DataFrame(my_imputer.fit_transform(data_test_num))
data_test_imputed.columns = data_test_num.columns


# In[ ]:


# Check the distribution of each imputed feature before and after imputation

# Define figure parameters
sns.set(rc={"figure.figsize": (20, 18)})
sns.set_style("whitegrid")
fig, axes = plt.subplots(4, 2)

# Plot the results
for feature, fig_pos in zip(tab["Column"].tolist(), range(0, 6)):

    """Features distribution before and after imputation"""

    # before imputation
    p = sns.histplot(ax=axes[fig_pos, 0], x=data_test_num[feature],
                     kde=True, bins=30, color="deepskyblue", edgecolor="black")
    p.set_ylabel(f"Before imputation", fontsize=14)

    # after imputation
    q = sns.histplot(ax=axes[fig_pos, 1], x=data_test_imputed[feature],
                     kde=True, bins=30, color="darkorange", edgecolor="black",)
    q.set_ylabel(f"After imputation", fontsize=14)


# The distribution of each of these variables before and after imputation does not really change at all, so all of these variables will be kept on. The shape of both datasets consisting of numerical features is shown below.

# In[ ]:


print(f"Training set shape (Numerical features): {data_train_imputed.shape}\n")
print(f"Test set shape (Numerical features): {data_test_imputed.shape}\n")


# ## Categorical Data

# Now we focus on the categorical features from each dataset. Firstly we separate the categorical features from both datasets into their own DataFrames data_train_categ and data_test_categ.

# In[ ]:


# Categorical to Quantitative relationship
categorical_features = [
    i for i in data_train.columns if data_train.dtypes[i] == "object"]
categorical_features.append("SalePrice")

# Train set
data_train_categ = data_train[categorical_features]

# Test set (-1 because test set don't have 'Sale Price')
data_test_categ = data_test[categorical_features[:-1]]


# In[ ]:


print(f"Training set shape (Categorical features): {data_train_categ.shape}\n")
print(f"Test set shape (Categorical features): {data_test_categ.shape}\n")


# ### Distribution

# In[ ]:


# Countplot for each of the categorical features in the training set
# Determine which categorical features are dominated by one outcome
fig, axes = plt.subplots(round(len(data_train_categ.columns) / 3), 3, figsize=(12, 35))

for i, ax in enumerate(fig.axes):
    # plot barplot of each feature
    if i < len(data_train_categ.columns) - 1:
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        sns.countplot(x=data_train_categ.columns[i], alpha=0.7, data=data_train_categ, ax=ax)

fig.tight_layout()


# Looking at the Count Plots above, we see that some of these variables are highly dominated by one feature. Since those variables will have minimal impact on the final house prices, those variables dominated by one outcome will be removed from both datasets.

# In[ ]:


cols_to_drop = [
    'Street',
    'LandContour',
    'Utilities',
    'LandSlope',
    'Condition2',
    'RoofMatl',
    'BsmtCond',
    'BsmtFinType2',
    'Heating',
    'CentralAir',
    'Electrical',
    'Functional',
    'GarageQual',
    'GarageCond',
    'PavedDrive'
]

# Training set
data_train_categ.drop(cols_to_drop, axis=1, inplace=True)

# Test set
data_test_categ.drop(cols_to_drop, axis=1, inplace=True)


# In[ ]:


print(f"Training set shape (Categorical features): {data_train_categ.shape}\n")
print(f"Test set shape (Categorical features): {data_test_categ.shape}\n")


# ### Variation of target variable with each categorical feature

# In[ ]:


# With the boxplot we can see the variation of the target 'SalePrice' in each of the categorical features
fig, axes = plt.subplots(
    round(len(data_train_categ.columns)/3), 3, figsize=(15, 30))

for i, ax in enumerate(fig.axes):
    # plot the variation of SalePrice in each feature
    if i < len(data_train_categ.columns) - 1:
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=75)
        sns.boxplot(
            x=data_train_categ.columns[i], y="SalePrice", data=data_train_categ, ax=ax)

fig.tight_layout()


# Looking at these box plots, we see that the distribution of sale price for certain categorical variables are similar to each other (like "Exterior1st" and "Exterior2nd") which suggests that certain categorical variables are co-dependent on each other. There are three pairs of categorical variables for which the distribution of sale price is very similar:
# 
# * "Exterior1st" and "Exterior2nd"
# * "ExterQual" and "MasVnrType"
# * "BsmtQual" and "BsmtExposure"
# 
# We will perform a Chi-squared test for each pair of variables at a 5% significance level to determine whether or not there is a strong dependency between these variables.

# In[ ]:


# Plot contingency table

sns.set(rc={"figure.figsize": (10, 7)})

X = ["Exterior1st", "ExterQual", "BsmtQual"]
Y = ["Exterior2nd", "MasVnrType", "BsmtExposure"]

# Parameters for Chi-squared test (5% significance level)
prob = 0.95
alpha = 1.0 - prob

for i, j in zip(X, Y):

    # Contingency table
    cont = data_train_categ[[i, j]].pivot_table(
        index=i, columns=j, aggfunc=len, margins=True, margins_name="Total")
    tx = cont.loc[:, ["Total"]]
    ty = cont.loc[["Total"], :]
    n = len(data_train_categ)
    indep = tx.dot(ty) / n
    c = cont.fillna(0)  # Replace NaN with 0 in the contingency table
    measure = (c - indep) ** 2 / indep
    xi_n = measure.sum().sum()
    table = measure / xi_n

    # Plot contingency table
    p = sns.heatmap(table.iloc[:-1, :-1],
                    annot=c.iloc[:-1, :-1], fmt=".0f", cmap="Oranges")
    p.set_xlabel(j, fontsize=18)
    p.set_ylabel(i, fontsize=18)
    p.set_title(f"\nχ² test between groups {i} and groups {j}\n", size=18)
    plt.show()

    # Performing Chi-sq test
    CrosstabResult = pd.crosstab(
        index=data_train_categ[i], columns=data_train_categ[j])
    ChiSqResult = chi2_contingency(CrosstabResult)
    # P-Value is the Probability of H0 being True
    print(f"P-Value of the ChiSq Test bewteen {i} and {j} is: {ChiSqResult[1]}\n")
    print('significance=%.3f, p=%.3f' % (alpha, ChiSqResult[1]))
    if ChiSqResult[1] <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')


# Here, the contigency tables for each pair of categorical variables considered is printed as a heatmap, with the colours representing the relative error between the actual value of the table with its expected value. After performing the Chi-squared test for each pairs of variables considered, we see that there is strong co-dependency for each of these variables. Since highly dependent/correlated variables do not add much relevant new information with regards to the value of the target variable, we will drop one of each co-dependent variable from the dataset.

# In[ ]:


# Drop the one of each co-dependent variables
# Training set
data_train_categ.drop(Y, axis=1, inplace=True)

# Test set
data_test_categ.drop(Y, axis=1, inplace=True)


# In[ ]:


print(f"Training set shape (Categorical features): {data_train_categ.shape}\n")
print(f"Test set shape (Categorical features): {data_test_categ.shape}\n")


# ### Missing Values

# In[ ]:


# Get names of categorical columns with missing values (training set)
cat_cols_with_missing = [col for col in data_train_categ.columns
                         if data_train_categ[col].isnull().any()]

print("Categorical Columns with missing (NA) values:")
print(cat_cols_with_missing)

# Count how many NA values are in each of those columns
cat_cols_nan_count = list(map(lambda col: round(data_train_categ[col].isna().sum()*100/len(data_train_categ)), 
                              cat_cols_with_missing))


tab_cat = pd.DataFrame(cat_cols_with_missing, columns=["Column"])
tab_cat["Percent_NaN"] = cat_cols_nan_count
tab_cat.sort_values(by=["Percent_NaN"], ascending=False, inplace=True)


# Define figure parameters
sns.set(rc={"figure.figsize": (10, 7)})
sns.set_style("whitegrid")

# Plot results
p = sns.barplot(x="Percent_NaN", y="Column", data=tab_cat,
                edgecolor="black", color="deepskyblue")

p.set_title("Percent of NaN per column of the train set\n", fontsize=20)
p.set_xlabel("\nPercent of NaN (%)", fontsize=20)
p.set_ylabel("Column Name\n", fontsize=20)


# In the training set, there are five categorical variables that have a significant amount of missing values. To help reduce the error, we will remove any columns with more than 30% NaN entries from both datasets. Imputation will be used to fill in the missing entries from the remaining columns in the training set using the modal class.

# In[ ]:


# Drop categorical columns that have at least 30% missing values
large_na = [col for col in cat_cols_with_missing if (data_train_categ[col].isna().sum()/data_train_categ.shape[0]) > 0.3]

print("Columns to be dropped:")
print(large_na)

data_train_categ.drop(large_na, axis=1, inplace=True)


# In[ ]:


# Fill the NaN of each feature by the corresponding modal class
categ_fill_null = {"GarageType": data_train_categ["GarageType"].mode().iloc[0],
                   "GarageFinish": data_train_categ["GarageFinish"].mode().iloc[0],
                   "BsmtQual": data_train_categ["BsmtQual"].mode().iloc[0],
                   "BsmtFinType1": data_train_categ["BsmtFinType1"].mode().iloc[0]}

data_train_categ = data_train_categ.fillna(value=categ_fill_null)


# After dropping those same columns from the test dataset, we investigate the categorical variables from that dataset that have missing values.

# In[ ]:


# Drop the same categorical columns from the test set
data_test_categ.drop(large_na, axis=1, inplace=True)

# Get names of categorical columns with missing values (test set)
cat_cols_with_missing_t = [col for col in data_test_categ.columns
                           if data_test_categ[col].isnull().any()]

print("Categorical Columns with missing (NA) values:")
print(cat_cols_with_missing_t)

# Count how many NA values are in each of those columns
cat_cols_nan_count_t = list(map(lambda col: round(data_test_categ[col].isna().sum()*100/len(data_test_categ)), 
                              cat_cols_with_missing_t))


tab_cat_t = pd.DataFrame(cat_cols_with_missing_t, columns=["Column"])
tab_cat_t["Percent_NaN"] = cat_cols_nan_count_t
tab_cat_t.sort_values(by=["Percent_NaN"], ascending=False, inplace=True)


# Define figure parameters
sns.set(rc={"figure.figsize": (10, 7)})
sns.set_style("whitegrid")

# Plot results
p = sns.barplot(x="Percent_NaN", y="Column", data=tab_cat_t,
                edgecolor="black", color="deepskyblue")

p.set_title("Percent of NaN per column of the test set\n", fontsize=20)
p.set_xlabel("\nPercent of NaN (%)", fontsize=20)
p.set_ylabel("Column Name\n", fontsize=20)


# There are a few more columns in the test dataset that have missing values, but none of them have more than 5% missing values. Therefore, we will fill in each NaN entry for each feature using it's corresponding modal class like before.

# In[ ]:


# Fill the NaN of each feature by the corresponding modal class
categ_fill_null = {"GarageType": data_test_categ["GarageType"].mode().iloc[0],
                   "GarageFinish": data_test_categ["GarageFinish"].mode().iloc[0],
                   "BsmtQual": data_test_categ["BsmtQual"].mode().iloc[0],
                   "BsmtFinType1": data_test_categ["BsmtFinType1"].mode().iloc[0],
                   "MSZoning": data_test_categ["MSZoning"].mode().iloc[0],
                   "Exterior1st": data_test_categ["Exterior1st"].mode().iloc[0],
                   "KitchenQual": data_test_categ["KitchenQual"].mode().iloc[0],
                   "SaleType": data_test_categ["SaleType"].mode().iloc[0]}

data_test_categ = data_test_categ.fillna(value=categ_fill_null)


# The shape of each dataset (containing categorical features only) after imputation is shown below.

# In[ ]:


print(f"Training set shape (Categorical features): {data_train_categ.shape}\n")
print(f"Test set shape (Categorical features): {data_test_categ.shape}\n")


# ### Transformation into numerical values (get_dummies())

# Before we combine the categorical data back with the numerical data, we need to transform the categorical entries into numerical entries. This will be done using the get_dummies() function where each categorical feature will be transformed into a binary feature.

# In[ ]:


# Drop the SalePrice column from the training dataset
data_train_categ.drop(["SalePrice"], axis = 1, inplace = True)

# Use get_dummies to transform the Categorical features into Binary features (Training dataset)
data_train_dummies = pd.get_dummies(data_train_categ)
data_train_dummies.head()


# In[ ]:


# Apply get_dummies to the test dataset as well
data_test_dummies = pd.get_dummies(data_test_categ)
data_test_dummies.head()


# From the first few rows of each modified dataset we see that they do not have the same number of columns. We need to check which columns are missing from the test dataset.

# In[ ]:


# Check if the column headings are the same in both data sets: data_train_dummies and data_test_dummies
dif_1 = [x for x in data_train_dummies.columns if x not in data_test_dummies.columns]
print(f"Features present in df_train_categ and absent in df_test_categ: {dif_1}\n")

dif_2 = [x for x in data_test_dummies.columns if x not in data_test_dummies.columns]
print(f"Features present in df_test_categ set and absent in df_train_categ: {dif_2}")


# Three of these columns from the training dataset are not present in the test dataset. Thus they will be dropped from the training dataset to ensure that both datasets have exactly the same features.

# In[ ]:


# Drop the columns listed in dif_1 from data_train_dummies
data_train_dummies.drop(dif_1, axis=1, inplace=True)

# Check again if the column headings are the same in both data sets: data_train_dummies and data_test_dummies
dif_1 = [x for x in data_train_dummies.columns if x not in data_test_dummies.columns]
print(f"Features present in df_train_categ and absent in df_test_categ: {dif_1}\n")

dif_2 = [x for x in data_test_dummies.columns if x not in data_test_dummies.columns]
print(f"Features present in df_test_categ set and absent in df_train_categ: {dif_2}")


# The shape of both datasets (categorical features only) after all of these changes are given below.

# In[ ]:


print(f"Training set shape (Categorical features): {data_train_dummies.shape}\n")
print(f"Test set shape (Categorical features): {data_test_dummies.shape}\n")


# ## Preparing Data for modelling

# In[ ]:


# Join numerical and categorical datasets together
# Training set
data_train_new = pd.concat([data_train_imputed, data_train_dummies], axis = 1)
print(f"Train set: {data_train_new.shape}")

# Test set
data_test_new = pd.concat([data_test_imputed, data_test_dummies], axis = 1)
print(f"Test set: {data_test_new.shape}")


# ### Further Feature Engineering

# The Year of construction and the Year of Remodelling variables will be transformed into new variables representing the Age of the House and the Age since the house was remodelled - this will enable us to apply a log transform to normalize those variables. After the transformation, the variables "YearBuilt" and "YearRemodAdd" will be removed.

# In[ ]:


# Convert Year of construction to the Age of the house since the construction
data_train_new["AgeSinceConst"] = (data_train_new["YearBuilt"].max() - data_train_new["YearBuilt"])

data_test_new["AgeSinceConst"] = (data_test_new["YearBuilt"].max() - data_test_new["YearBuilt"])

# Drop "YearBuilt"
data_train_new.drop(["YearBuilt"], axis=1, inplace=True)
data_test_new.drop(["YearBuilt"], axis=1, inplace=True)


# In[ ]:


# Convert Year of remodeling to the Age of the house since the remodeling
data_train_new["AgeSinceRemod"] = (data_train_new["YearRemodAdd"].max() - data_train_new["YearRemodAdd"])

data_test_new["AgeSinceRemod"] = (data_test_new["YearRemodAdd"].max() - data_test_new["YearRemodAdd"])

# Drop "YearRemodAdd"
data_train_new.drop(["YearRemodAdd"], axis=1, inplace=True)
data_test_new.drop(["YearRemodAdd"], axis=1, inplace=True)


# Now, we consider the continuous numerical variables that are skewed. A Log transformation will be applied to the skewed numerical variables to help mitigate the strong variation of some variables, and to reduce redundancy. The continuous features are defined below.

# In[ ]:


continuous_features = ['AgeSinceConst', 'AgeSinceRemod', 'MasVnrArea', 'BsmtFinSF1',
                      'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',
                      'WoodDeckSF', 'OpenPorchSF']


# To obtain the skewed features, we take out variables that are more than 50% skewed.

# In[ ]:


data_skew_verify = data_train_new.loc[:, continuous_features]

# Select features with absolute Skew higher than 0.5
skew_ft = []

for i in continuous_features:
    # list of skew for each corresponding feature
    skew_ft.append(abs(data_skew_verify[i].skew()))

data_skewed = pd.DataFrame({"Columns": continuous_features, "Abs_Skew": skew_ft})

sk_features = data_skewed[data_skewed["Abs_Skew"] > 0.5]["Columns"].tolist()
print(f"List of skewed features: {sk_features}")


# A log transformation is then applied to the skewed features listed above.

# In[ ]:


# Log transformation of the skewed features
for i in sk_features:
    # loop over i (features) to calculate Log of surfaces
    # Training set
    data_train_new[i] = np.log((data_train_new[i])+1)
    
    # Test set
    data_test_new[i] = np.log((data_test_new[i])+1)


# Looking at the distribution of the numerical features near the start, we noticed that "SalePrice" is skewed as well. To help normalize this variable, a log transformation will be applied to "SalePrice" as well.

# In[ ]:


# Log transformation of the target variable "SalePrice"
data_train_new["SalePriceLog"] = np.log(data_train_new.SalePrice)

# Drop the original SalePrice
data_train_new.drop(["SalePrice"], axis=1, inplace=True)


# In[ ]:


numerical_cols_new = [cname for cname in data_train_new.columns if 
                      data_train_new[cname].dtype in ['int64', 'float64']]

data_train_new_num = data_train_new[numerical_cols_new].copy()

# Plot the distribution of all the numerical features
fig_ = data_train_new_num.hist(figsize=(16, 20), bins=50, color="deepskyblue",
                               edgecolor="black", xlabelsize=8, ylabelsize=8)


# Looking at the distribution of the numerical features, we notice that most of the previously skewed variables have a more normal distribution (excluding zero values) with exception of the Age variables, which should result in better predictons.

# ### Splitting the data into training and test sets

# In[ ]:


# TRAINING DATASET
# Feature variables
X = data_train_new.copy().drop(["SalePriceLog"], axis = 1)

# Target Variable
y = data_train_new.loc[:, "SalePriceLog"]

print(X.shape)
print(y.shape)


# In[ ]:


# Split the data into Training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")
print(f"\nX_test:{X_test.shape}\ny_test:{y_test.shape}")


# ## Modelling

# For this project, six supervised learning models will be considered:
# 
# * Linear Regression
# * Ridge Regression
# * Lasso Regression
# * Decision Tree Regressor
# * Random Forest Regressor
# * XGBoost Regressor
# 
# To measure model performance and their predicitons the RMSE and R^{2} scores will be used, and 5-fold cross-validation will also be used.

# In[ ]:


# Define models
model_lin = LinearRegression()
model_ridge = Ridge(alpha = 0.001)
model_lasso = Lasso(alpha = 0.001)
model_tree = DecisionTreeRegressor()
model_ran = RandomForestRegressor()
model_xg = XGBRegressor()


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score

# Define a function for each metric
# R²
def rsqr_score(test, pred):
    """Calculate R squared score 

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        R squared score 
    """
    r2_ = r2_score(test, pred)
    return r2_


# RMSE
def rmse_score(test, pred):
    """Calculate Root Mean Square Error score 

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        Root Mean Square Error score
    """
    rmse_ = np.sqrt(mean_squared_error(test, pred))
    return rmse_


# Print the scores
def print_score(test, pred, model):
    """Print calculated score 

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        print the regressor name
        print the R squared score
        print Root Mean Square Error score
    """

    print(f"- Regressor: {model}")
    print(f"R²: {rsqr_score(test, pred)}")
    print(f"RMSE: {rmse_score(test, pred)}\n")


# ## Linear Regression

# In[ ]:


scores_lin = -1 * cross_val_score(model_lin, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_squared_error')

print("MSE scores (Linear Model):\n", scores_lin)
print("Mean MSE scores:", scores_lin.mean())


# In[ ]:


model_lin.fit(X_train, y_train)
y_pred_lin = model_lin.predict(X_test)
print_score(y_test, y_pred_lin, "Linear")


# In[ ]:


plt.figure()
plt.title("Actual vs. Predicted house prices\n (Linear)", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_lin),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlim(0, 800000)
plt.ylim(0, 800000)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# ## Ridge Regression

# In[ ]:


scores_ridge = -1 * cross_val_score(model_ridge, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_squared_error')

print("MSE scores (Ridge Model):\n", scores_ridge)
print("Mean MSE scores:", scores_ridge.mean())


# In[ ]:


model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)
print_score(y_test, y_pred_ridge, "Ridge")


# In[ ]:


plt.figure()
plt.title("Actual vs. Predicted house prices\n (Ridge)", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_ridge),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlim(0, 800000)
plt.ylim(0, 800000)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# ### Hyperparamter Tuning (Ridge)

# In[ ]:


# Define hyperparameters
alphas = np.linspace(0, 10, 100).tolist()

tuned_parameters = {"alpha": alphas}

# GridSearch
ridge_cv = GridSearchCV(Ridge(), tuned_parameters, cv=10, n_jobs=-1, verbose=1)

# fit the GridSearch on train set
ridge_cv.fit(X_train, y_train)

# print best params and the corresponding R²
print(f"Best hyperparameters: {ridge_cv.best_params_}")
print(f"Best R² (train): {ridge_cv.best_score_}")


# In[ ]:


model_ridge_opt = Ridge(alpha = ridge_cv.best_params_["alpha"])
model_ridge_opt.fit(X_train, y_train)
y_pred_ridge_opt = model_ridge_opt.predict(X_test)
print_score(y_test, y_pred_ridge_opt, "Ridge")


# In[ ]:


plt.figure()
plt.title("Actual vs. Predicted house prices\n (Ridge - Optimal alpha value)", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_ridge_opt),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlim(0, 800000)
plt.ylim(0, 800000)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# ## Lasso Regression

# In[ ]:


scores_lasso = -1 * cross_val_score(model_lasso, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_squared_error')

print("MSE scores (Lasso Model):\n", scores_lasso)
print("Mean MSE scores:", scores_lasso.mean())


# In[ ]:


model_lasso.fit(X_train, y_train)
y_pred_lasso = model_lasso.predict(X_test)
print_score(y_test, y_pred_lasso, "Lasso")


# In[ ]:


plt.figure()
plt.title("Actual vs. Predicted house prices\n (Lasso)", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_lasso),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlim(0, 800000)
plt.ylim(0, 800000)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# ### Hyperparameter Tuning (Lasso)

# In[ ]:


# Define hyperparameters
alphas = np.logspace(-5, 5, 100).tolist()

tuned_parameters = {"alpha": alphas}

# GridSearch
lasso_cv = GridSearchCV(Lasso(), tuned_parameters, cv=10, n_jobs=-1, verbose=1)

# fit the GridSearch on train set
lasso_cv.fit(X_train, y_train)

# print best params and the corresponding R²
print(f"Best hyperparameters: {lasso_cv.best_params_}")
print(f"Best R² (train): {lasso_cv.best_score_}")


# In[ ]:


model_lasso_opt = Lasso(alpha = lasso_cv.best_params_["alpha"])

model_lasso_opt.fit(X_train, y_train)
y_pred_lasso_opt = model_lasso_opt.predict(X_test)
print_score(y_test, y_pred_lasso_opt, "Lasso")


# In[ ]:


plt.figure()
plt.title("Actual vs. Predicted house prices\n (Lasso - Optimal alpha value)", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_lasso_opt),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlim(0, 800000)
plt.ylim(0, 800000)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# ## Decision Tree Regressor

# In[ ]:


scores_tree = -1 * cross_val_score(model_tree, X_train, y_train,
                                   cv=5,
                                   scoring='neg_mean_squared_error')

print("MSE scores (Decision Tree Model):\n", scores_tree)
print("Mean MSE scores:", scores_tree.mean())


# In[ ]:


model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)
print_score(y_test, y_pred_tree, "Decision Tree")


# In[ ]:


plt.figure()
plt.title("Actual vs. Predicted house prices\n (Decision Tree)", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_tree),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlim(0, 800000)
plt.ylim(0, 800000)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# ## Random Forest Regressor

# In[ ]:


scores_ran = -1 * cross_val_score(model_ran, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_squared_error')

print("MSE scores (Random Forest Model):\n", scores_ran)
print("Mean MSE scores:", scores_ran.mean())


# In[ ]:


model_ran.fit(X_train, y_train)
y_pred_ran = model_ran.predict(X_test)
print_score(y_test, y_pred_ran, "Random Forest")


# In[ ]:


plt.figure()
plt.title("Actual vs. Predicted house prices\n (Random Forest)", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_ran),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlim(0, 800000)
plt.ylim(0, 800000)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# ## XGBoost Regression

# In[ ]:


# Define hyperparameters
tuned_parameters_xgb = {"max_depth": [3],
                        "colsample_bytree": [0.3, 0.7],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "n_estimators": [100, 500, 1000]}

# GridSearch
xgbr_cv = GridSearchCV(estimator=XGBRegressor(),
                       param_grid=tuned_parameters_xgb,
                       cv=5,
                       n_jobs=-1,
                       verbose=1)

# fit the GridSearch on train set
xgbr_cv.fit(X_train, y_train)

# print best params and the corresponding R²
print(f"Best hyperparameters: {xgbr_cv.best_params_}\n")
print(f"Best R²: {xgbr_cv.best_score_}")


# In[ ]:


model_xgb_opt = XGBRegressor(colsample_bytree = xgbr_cv.best_params_["colsample_bytree"],
                             learning_rate = xgbr_cv.best_params_["learning_rate"],
                             max_depth = xgbr_cv.best_params_["max_depth"],
                             n_estimators = xgbr_cv.best_params_["n_estimators"])

model_xgb_opt.fit(X_train, y_train)
y_pred_xgb_opt = model_xgb_opt.predict(X_test)
print_score(y_test, y_pred_xgb_opt, "XGBoost")


# In[ ]:


plt.figure()
plt.title("Actual vs. Predicted house prices\n (XGBoost Regressor)", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_xgb_opt),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlim(0, 800000)
plt.ylim(0, 800000)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# After application of hyperparameter tuning, the Lasso model returned the best R^{2} score and the lowest RMSE value, achieving an accuracy of around 90.7%. Based on this, the optimized Lasso Regression model will be used to predict the Sale Price of houses from the test dataset. The predicted sale prices will be saved into a csv file named "submission.csv".

# In[ ]:


# Prediction of House Prices using the Optimal Lasso Regression Model

y_pred = np.exp(model_lasso_opt.predict(data_test_new))

output = pd.DataFrame({"Id": Id_test_list,
                       "SalePrice": y_pred})

output.head(10)


# In[ ]:


# Save the output
output.to_csv("submission.csv", index=False)

