#!/usr/bin/env python
# coding: utf-8

# ### Data describtion:
# 
# - MSSubClass: Identifies the type of dwelling involved in the sale.
# - MSZoning: Identifies the general zoning classification of the sale.
# - LotFrontage: Linear feet of street connected to property.
# - LotArea: Lot size in square feet.
# - Street: Type of road access to property.
# - Alley: Type of alley access to property.
# - LotShape: General shape of property.
# - LandContour: Flatness of the property.
# - Utilities: Type of utilities available.
# - LotConfig: Lot configuration.
# - LandSlope: Slope of property.
# - Neighborhood: Physical locations within Ames city limits.
# - Condition1: Proximity to various conditions.
# - Condition2: Proximity to various conditions (if more than one is present).
# - BldgType: Type of dwelling.
# - HouseStyle: Style of dwelling.
# - OverallQual: Rates the overall material and finish of the house.
# - OverallCond: Rates the overall condition of the house.
# - YearBuilt: Original construction date.
# - YearRemodAdd: Remodel date (same as construction date if no remodeling or additions).
# - RoofStyle: Type of roof.
# - RoofMatl: Roof material.
# - Exterior1st: Exterior covering on house.
# - Exterior2nd: Exterior covering on house (if more than one material).
# - MasVnrType: Masonry veneer type **(None - None)**.
# - MasVnrArea: Masonry veneer area in square feet.
# - ExterQual: Evaluates the quality of the material on the exterior/
# - ExterCond: Evaluates the present condition of the material on the exterior.
# - Foundation: Type of foundation.
# - BsmtQual: Evaluates the height of the basement **(NA - No Basement)**.
# - BsmtCond: Evaluates the general condition of the basement **(NA - No Basement)**.
# - BsmtExposure: Refers to walkout or garden level walls **(NA - No Basement)**.
# - BsmtFinType1: Rating of basement finished area **(NA - No Basement)**.
# - BsmtFinSF1: Type 1 finished square feet.
# - BsmtFinType2: Rating of basement finished area (if multiple types) **(NA - No Basement)**.
# - BsmtFinSF2: Type 2 finished square feet.
# - BsmtUnfSF: Unfinished square feet of basement area.
# - TotalBsmtSF: Total square feet of basement area.
# - Heating: Type of heating.
# - HeatingQC: Heating quality and condition.
# - CentralAir: Central air conditioning.
# - Electrical: Electrical system.
# - 1stFlrSF: First Floor square feet.
# - 2ndFlrSF: Second floor square feet.
# - LowQualFinSF: Low quality finished square feet (all floors).
# - GrLivArea: Above grade (ground) living area square feet.
# - BsmtFullBath: Basement full bathrooms.
# - BsmtHalfBath: Basement half bathrooms.
# - FullBath: Full bathrooms above grade.
# - HalfBath: Half baths above grade.
# - Bedroom: Bedrooms above grade (does NOT include basement bedrooms).
# - Kitchen: Kitchens above grade.
# - KitchenQual: Kitchen quality.
# - TotRmsAbvGrd: Total rooms above grade (does not include bathrooms).
# - Functional: Home functionality (Assume typical unless deductions are warranted).
# - Fireplaces: Number of fireplaces.
# - FireplaceQu: Fireplace quality **(NA - No Fireplace)**.
# - GarageType: Garage location **(NA - No Garage)**.
# - GarageYrBlt: Year garage was built.
# - GarageFinish: Interior finish of the garage **(NA - No Garage)**.
# - GarageCars: Size of garage in car capacity.
# - GarageArea: Size of garage in square feet.
# - GarageQual: Garage quality **(NA - No Garage)**.
# - GarageCond: Garage condition **(NA - No Garage)**.
# - PavedDrive: Paved driveway.
# - WoodDeckSF: Wood deck area in square feet.
# - OpenPorchSF: Open porch area in square feet.
# - EnclosedPorch: Enclosed porch area in square feet.
# - 3SsnPorch: Three season porch area in square feet.
# - ScreenPorch: Screen porch area in square feet.
# - PoolArea: Pool area in square feet.
# - PoolQC: Pool quality **(NA - No Pool)**.
# - Fence: Fence quality **(NA - No Fence)**.
# - MiscFeature: Miscellaneous feature not covered in other categories **(NA - None)**.
# - MiscVal: Value of miscellaneous feature.
# - MoSold: Month Sold (MM).
# - YrSold: Year Sold (YYYY).
# - SaleType: Type of sale.
# - SaleCondition: Condition of sale.

# ## 1. Download the data and explore

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgbm
import xgboost as xgb
import catboost as cat

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, SGDRegressor, ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from IPython.display import display
pd.options.display.max_columns = None


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


test_ID = test['Id']


# **Let's look at the data - samples, missing values, info, correlation**

# In[ ]:


train.head(5)


# In[ ]:


train.shape


# In[ ]:


train_missing = train.isnull().sum().sort_values(ascending=False)
train_missing.head(20)


# Fill in missing values

# In[ ]:


train['MiscFeature'].fillna('None', inplace=True)
train['Fence'].fillna('No Fence', inplace=True)
train['PoolQC'].fillna('No Pool', inplace=True)
train['Alley'].fillna('No alley access', inplace=True)
train['FireplaceQu'].fillna('No Fireplace', inplace=True)
train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
train['GarageCond'].fillna('No Garage', inplace=True)
train['GarageType'].fillna('No Garage', inplace=True)
train['GarageYrBlt'].fillna(round(train['GarageYrBlt'].median(), 1), inplace=True)
train['GarageFinish'].fillna('No Garage', inplace=True)
train['GarageQual'].fillna('No Garage', inplace=True)
train['BsmtExposure'].fillna('No Basement', inplace=True)
train['BsmtFinType2'].fillna('No Basement', inplace=True)
train['BsmtFinType1'].fillna('No Basement', inplace=True)
train['BsmtCond'].fillna('No Basement', inplace=True)
train['BsmtQual'].fillna('No Basement', inplace=True)
train['MasVnrArea'].fillna(0.0, inplace=True)
train['MasVnrType'].fillna('None', inplace=True)
train['Electrical'].fillna('Mixed', inplace=True)


# Correlation

# In[ ]:


plt.subplots(figsize = (30,20))

mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(train.corr(), 
            cmap=sns.diverging_palette(20, 220, n=200), 
            mask = mask, 
            annot=True, 
            center = 0, 
);

plt.title("Heatmap of all the Features", fontsize=30);


# In[ ]:


plt.figure(figsize=(5,20))
sns.heatmap(train.corr()[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(50), vmin=-1, annot=True);


# Correlation more than 0.3

# In[ ]:


best_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',               'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',                'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces',                'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF']


# In[ ]:


train.info()


# Some feature have wrong format

# In[ ]:


test.head(5)


# In[ ]:


test.shape


# In[ ]:


test.info()


# In[ ]:


test_missing = test.isnull().sum().sort_values(ascending=False)
test_missing.head(35)


# In[ ]:


test['MiscFeature'].fillna('None', inplace=True)
test['Fence'].fillna('No Fence', inplace=True)
test['PoolQC'].fillna('No Pool', inplace=True)
test['Alley'].fillna('No alley access', inplace=True)
test['FireplaceQu'].fillna('No Fireplace', inplace=True)
test['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
test['GarageCond'].fillna('No Garage', inplace=True)
test['GarageType'].fillna('No Garage', inplace=True)
test['GarageYrBlt'].fillna(round(test['GarageYrBlt'].median(), 1), inplace=True)
test['GarageFinish'].fillna('No Garage', inplace=True)
test['GarageQual'].fillna('No Garage', inplace=True)
test['BsmtExposure'].fillna('No Basement', inplace=True)
test['BsmtFinType2'].fillna('No Basement', inplace=True)
test['BsmtFinType1'].fillna('No Basement', inplace=True)
test['BsmtCond'].fillna('No Basement', inplace=True)
test['BsmtQual'].fillna('No Basement', inplace=True)
test['MasVnrArea'].fillna(0.0, inplace=True)
test['MasVnrType'].fillna('None', inplace=True)
test['MSZoning'].fillna('RL', inplace=True)
test['Utilities'].fillna('AllPub', inplace=True)
test['Functional'].fillna('Typ', inplace=True)
test['BsmtFullBath'].fillna(0.0, inplace=True)
test['BsmtHalfBath'].fillna(0.0, inplace=True)
test['BsmtFinSF2'].fillna(0.0, inplace=True)
test['BsmtUnfSF'].fillna(0.0, inplace=True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(), inplace=True)
test['SaleType'].fillna('WD', inplace=True)
test['BsmtFinSF1'].fillna(0.0, inplace=True)
test['GarageCars'].fillna(2.0, inplace=True)
test['GarageArea'].fillna(0.0, inplace=True)
test['KitchenQual'].fillna('TA', inplace=True)
test['Exterior1st'].fillna('VinylSd', inplace=True)
test['Exterior2nd'].fillna('VinylSd', inplace=True)


# **Concatenate train and test**

# In[ ]:


train_test = pd.concat([train,test], axis=0, sort=False)


# In[ ]:


train_test.sample(10)


# Drop irrelevant features

# In[ ]:


train.drop(['Id', 'Utilities', 'Street', 'LowQualFinSF', 'PoolArea'], axis=1, inplace=True)
test.drop(['Id', 'Utilities', 'Street', 'LowQualFinSF', 'PoolArea'], axis=1, inplace=True)
train_test.drop(['Id', 'Utilities', 'Street', 'LowQualFinSF', 'PoolArea'], axis=1, inplace=True)


# ## 2. EDA and preprocessing

# In[ ]:


train['MSSubClass'] = train['MSSubClass'].apply(str)
train['OverallCond'] = train['OverallCond'].astype(str)
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
#train['GarageYrBlt'] = train['GarageYrBlt'].astype(str)

test['MSSubClass'] = test['MSSubClass'].apply(str)
test['OverallCond'] = test['OverallCond'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)
#test['GarageYrBlt'] = test['GarageYrBlt'].astype(str)

train_test['MSSubClass'] = train_test['MSSubClass'].apply(str)
train_test['OverallCond'] = train_test['OverallCond'].astype(str)
train_test['YrSold'] = train_test['YrSold'].astype(str)
train_test['MoSold'] = train_test['MoSold'].astype(str)
#train_test['GarageYrBlt'] = train_test['GarageYrBlt'].astype(str)


# In[ ]:


train_test.head()


# In[ ]:


columns = train.columns


# In[ ]:


columns


# In[ ]:


numeric = ['LotFrontage', 
           'LotArea',  
           'YearBuilt', 
           'YearRemodAdd',
           'MasVnrArea', 
           'BsmtFinSF1', 
           'BsmtFinSF2', 
           'BsmtUnfSF', 
           'TotalBsmtSF',
           '1stFlrSF', 
           '2ndFlrSF', 
           'GrLivArea',  
           'GarageArea', 
           'WoodDeckSF', 
           'OpenPorchSF',
           'EnclosedPorch', 
           '3SsnPorch', 
           'ScreenPorch', 
           'MiscVal', 
           'GarageYrBlt']


# In[ ]:


categorical = set(columns) - set(numeric)


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


def num_plot(data):
    for col in numeric:
        if col != 'SalePrice':
            print(col)
            print(data[col].dtype)
    
            plt.figure(figsize=(10, 10))
            ax = sns.scatterplot(x=col, y='SalePrice', data=data)
            plt.show();


# In[ ]:


num_plot(train)


# In[ ]:


def cat_plot(data):
    for col in categorical:
        if col != 'SalePrice':
            print(col)
            print(data[col].dtype)
    
            ax = sns.catplot(x=col, y='SalePrice', data=data, size=10)
            plt.show();


# In[ ]:


cat_plot(train)


# Drop noisy features

# In[ ]:


train.drop(['PoolQC', 'Condition2'], axis=1, inplace=True)
test.drop(['PoolQC', 'Condition2'], axis=1, inplace=True)
train_test.drop(['PoolQC', 'Condition2'], axis=1, inplace=True)


# In[ ]:


categorical.remove('PoolQC')
categorical.remove('Condition2')
categorical.remove('SalePrice')


# ## 3. Check feature importance on RandomForestRegressor and final preparation

# In[ ]:


LE = LabelEncoder()


# In[ ]:


for col in categorical:
    train[col] = LE.fit_transform(train[col])


# In[ ]:


train.head()


# In[ ]:


rf_clf = RandomForestRegressor(random_state=42, n_estimators=500, max_depth=8, criterion='mse')


# In[ ]:


get_ipython().run_line_magic('time', "rf_clf.fit(train.drop(['SalePrice'], axis=1), train['SalePrice'])")


# In[ ]:


get_ipython().run_line_magic('time', "cross_val_score(rf_clf, train.drop(['SalePrice'], axis=1), train['SalePrice'], cv=5)")


# In[ ]:


importances = rf_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(train.shape[1] - 1):
    print("%d. %s -- feature %d (%f)" % (f + 1, train.columns[indices[f]], indices[f], importances[indices[f]]))

plt.figure(figsize=(18, 12))
plt.title("Feature importances")
plt.bar(range(train.drop(['SalePrice'], axis=1).shape[1]), importances[indices],
       yerr=std[indices], align="center")
plt.xticks(range(train.shape[1]), indices)
plt.xlim([-1, train.shape[1]])
plt.show()


# Leave first 40 features

# In[ ]:


indices[:40]


# In[ ]:


columns_to_models = train.columns[indices[:40]]


# In[ ]:


columns_to_models = list(columns_to_models)


# In[ ]:


columns_to_models.append('SalePrice')


# In[ ]:


columns_to_models


# In[ ]:


train_test = train_test[columns_to_models]


# In[ ]:


train_test.shape


# In[ ]:


train_test.head()


# **Scaling and dummy encoding**

# In[ ]:


scaler = StandardScaler()


# In[ ]:


new_numeric = []

for col in numeric:
    if col in columns_to_models:
        new_numeric.append(col)


# In[ ]:


new_numeric


# In[ ]:


new_cat = []

for col in categorical:
    if col in columns_to_models:
        new_cat.append(col)


# In[ ]:


new_cat


# In[ ]:


train_test[new_numeric] = scaler.fit_transform(train_test[new_numeric])


# In[ ]:


train_test.head()


# In[ ]:


for col in new_cat:
    if col != 'SalePrice':
        train_test[col] = LE.fit_transform(train_test[col])


# In[ ]:


train_test.head()


# In[ ]:


train_test_lin = train_test.copy()


# In[ ]:


train_test_lin = pd.concat([train_test_lin, 
                     pd.get_dummies(train_test_lin['BedroomAbvGr'], prefix='BedroomAbvGr', drop_first=True),
                     pd.get_dummies(train_test_lin['GarageFinish'], prefix='GarageFinish', drop_first=True), 
                     pd.get_dummies(train_test_lin['BsmtQual'], prefix='BsmtQual', drop_first=True),
                     pd.get_dummies(train_test_lin['HalfBath'], prefix='HalfBath', drop_first=True),
                     pd.get_dummies(train_test_lin['MoSold'], prefix='MoSold', drop_first=True), 
                     pd.get_dummies(train_test_lin['SaleCondition'], prefix='SaleCondition', drop_first=True),
                     pd.get_dummies(train_test_lin['BsmtFinType1'], prefix='BsmtFinType1', drop_first=True),
                     pd.get_dummies(train_test_lin['KitchenQual'], prefix='KitchenQual', drop_first=True),
                     pd.get_dummies(train_test_lin['OverallQual'], prefix='OverallQual', drop_first=True),
                     pd.get_dummies(train_test_lin['LandContour'], prefix='LandContour', drop_first=True),
                     pd.get_dummies(train_test_lin['TotRmsAbvGrd'], prefix='TotRmsAbvGrd', drop_first=True),
                     pd.get_dummies(train_test_lin['ExterQual'], prefix='ExterQual', drop_first=True),
                     pd.get_dummies(train_test_lin['Exterior2nd'], prefix='Exterior2nd', drop_first=True),
                     pd.get_dummies(train_test_lin['OverallCond'], prefix='OverallCond', drop_first=True),   
                     pd.get_dummies(train_test_lin['GarageType'], prefix='GarageType', drop_first=True),
                     pd.get_dummies(train_test_lin['Fireplaces'], prefix='Fireplaces', drop_first=True),
                     pd.get_dummies(train_test_lin['CentralAir'], prefix='CentralAir', drop_first=True),
                     pd.get_dummies(train_test_lin['LotShape'], prefix='LotShape', drop_first=True),   
                     pd.get_dummies(train_test_lin['Exterior1st'], prefix='Exterior1st', drop_first=True),
                     pd.get_dummies(train_test_lin['BsmtExposure'], prefix='BsmtExposure', drop_first=True),
                     pd.get_dummies(train_test_lin['Neighborhood'], prefix='Neighborhood', drop_first=True),   
                     pd.get_dummies(train_test_lin['FullBath'], prefix='FullBath', drop_first=True),         
                     pd.get_dummies(train_test_lin['GarageCars'], prefix='GarageCars', drop_first=True),         
                     pd.get_dummies(train_test_lin['MSZoning'], prefix='MSZoning', drop_first=True)
                       ],
                    axis=1)


# In[ ]:


train_test_lin.drop(new_cat, axis=1, inplace=True)


# In[ ]:


train_test.sample(10)


# In[ ]:


train_test_lin.sample(10)


# In[ ]:


train_test.shape, train_test_lin.shape


# **SPLIT THE DATA FOR LINEAR MODEL AND BOOSTS + NN**

# In[ ]:


train_to_model = train_test[train_test['SalePrice'].isna() == False]
test_to_model = train_test[train_test['SalePrice'].isna() == True]
train_to_model_lin = train_test_lin[train_test_lin['SalePrice'].isna() == False]
test_to_model_lin = train_test_lin[train_test_lin['SalePrice'].isna() == True]


# In[ ]:


train_to_model.head()


# In[ ]:


test_to_model.head()


# In[ ]:


train_to_model_lin.head()


# In[ ]:


test_to_model_lin.head()


# In[ ]:


test_to_model.drop(['SalePrice'], axis=1, inplace=True)
test_to_model_lin.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


train_to_model.shape, test_to_model.shape, train_to_model_lin.shape, test_to_model_lin.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_to_model.drop(['SalePrice'], axis=1),                                                     train_to_model['SalePrice'], test_size=0.1, random_state=42)


# In[ ]:


X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(train_to_model_lin.drop(['SalePrice'], axis=1),                                                     train_to_model_lin['SalePrice'], test_size=0.1, random_state=42)


# In[ ]:


print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# In[ ]:


print('X_train_lin shape :', X_train_lin.shape)
print('X_test_lin shape :', X_test_lin.shape)
print('y_train_lin shape :', y_train_lin.shape)
print('y_test_lin shape :', y_test_lin.shape)


# ## 4. Modeling

# In[ ]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lgb_reg = lgbm.LGBMRegressor(objective=\'regression\', learning_rate=0.01, n_estimators=100, n_jobs=-1, subsample=0.5)\nlgb_param_grid = {"learning_rate":[0.1, 0.01, 0.001],\n                   "n_estimators":[250, 500, 750, 1000],\n                  "subsample":[0.3, 0.4, 0.5, 0.6],\n                   "max_depth":[4, 5, 6, 7]\n                  }\n                  \ngrid_search = GridSearchCV(lgb_reg, param_grid=lgb_param_grid, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)\ngrid_search.fit(X_train, y_train)\n\nlgb_reg = grid_search.best_estimator_\nprint(grid_search.best_params_)\n\ny_pred = lgb_reg.predict(X_test)\n\nprint(\'-\' * 10 + \'LGBM\' + \'-\' * 10)\nprint(\'R square Accuracy: \', r2_score(y_test, y_pred))\nprint(\'Mean Absolute Error: \', mean_absolute_error(y_test, y_pred))\nprint(\'Mean Squared Error: \', mean_squared_error(y_test, y_pred))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xgb_reg = xgb.XGBRegressor(learning_rate=0.01, n_estimators=100, n_jobs=-1, booster=\'gbtree\', random_state=42, subsample=0.5)\nxgb_param_grid = {"learning_rate":[0.1, 0.01, 0.001],\n                   "n_estimators":[250, 500, 750, 1000],\n                  "subsample":[0.3, 0.4, 0.5, 0.6],\n                   "max_depth":[4, 5, 6, 7]\n                  }\n                  \ngrid_search = GridSearchCV(xgb_reg, param_grid=xgb_param_grid, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)\ngrid_search.fit(X_train, y_train)\n\nxgb_reg = grid_search.best_estimator_\nprint(grid_search.best_params_)\n\ny_pred = xgb_reg.predict(X_test)\n\nprint(\'-\' * 10 + \'XGB\' + \'-\' * 10)\nprint(\'R square Accuracy: \', r2_score(y_test, y_pred))\nprint(\'Mean Absolute Error: \', mean_absolute_error(y_test, y_pred))\nprint(\'Mean Squared Error: \', mean_squared_error(y_test, y_pred))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cat_reg = cat.CatBoostRegressor(learning_rate=0.01, n_estimators=100, objective=\'RMSE\', loss_function=\'R2\', random_state=42, subsample=0.5)\ncat_param_grid = {"learning_rate":[0.1, 0.01, 0.001],\n                   "n_estimators":[250, 500, 750, 1000],\n                  "subsample":[0.3, 0.4, 0.5, 0.6],\n                   "max_depth":[4, 5, 6, 7]\n                  }\n                  \ngrid_search = GridSearchCV(cat_reg, param_grid=cat_param_grid, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)\ngrid_search.fit(X_train, y_train)\n\ncat_reg = grid_search.best_estimator_\nprint(grid_search.best_params_)\n\ny_pred = cat_reg.predict(X_test)\n\nprint(\'-\' * 10 + \'CatBoost\' + \'-\' * 10)\nprint(\'R square Accuracy: \', r2_score(y_test, y_pred))\nprint(\'Mean Absolute Error: \', mean_absolute_error(y_test, y_pred))\nprint(\'Mean Squared Error: \', mean_squared_error(y_test, y_pred))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, random_state=42, subsample=0.5)\ngbdt_param_grid = {"learning_rate":[0.1, 0.01, 0.001],\n                   "n_estimators":[250, 500, 750, 1000],\n                  "subsample":[0.3, 0.4, 0.5, 0.6],\n                   "max_depth":[4, 5, 6, 7]\n                  }\n\n                  \ngrid_search = GridSearchCV(gb_reg, param_grid=gbdt_param_grid, cv=kfolds, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)\ngrid_search.fit(X_train, y_train)\n\ngb_reg = grid_search.best_estimator_\nprint(grid_search.best_params_)\n\ny_pred = gb_reg.predict(X_test)\n\nprint(\'-\' * 10 + \'GBM\' + \'-\' * 10)\nprint(\'R square Accuracy: \', r2_score(y_test, y_pred))\nprint(\'Mean Absolute Error: \', mean_absolute_error(y_test, y_pred))\nprint(\'Mean Squared Error: \', mean_squared_error(y_test, y_pred))')


# In[ ]:


alphas = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1, 1, 10]

ridgecv_reg = make_pipeline(RidgeCV(alphas=alphas, cv=kfolds))
ridgecv_reg.fit(X_train_lin, y_train_lin)
y_pred = ridgecv_reg.predict(X_test_lin)

print('-' * 10 + 'RidgeCV' + '-' * 10)
print('R square: ', r2_score(y_test_lin, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test_lin, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test_lin, y_pred))


# In[ ]:


lassocv_reg = make_pipeline(LassoCV(alphas=alphas, cv=kfolds))
lassocv_reg.fit(X_train_lin, y_train_lin)
y_pred = lassocv_reg.predict(X_test_lin)

print('-' * 10 + 'LassoCV' + '-' * 10)
print('R square: ', r2_score(y_test_lin, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test_lin, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test_lin, y_pred))


# In[ ]:


alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
l1ratio = [0.8, 0.83, 0.85, 0.87, 0.9, 0.92, 0.95, 0.97, 0.99, 1]

elasticv_reg = make_pipeline(ElasticNetCV(alphas=alphas, cv=kfolds, l1_ratio=l1ratio))
elasticv_reg.fit(X_train_lin, y_train_lin)
y_pred = elasticv_reg.predict(X_test_lin)

print('-' * 10 + 'ElasticNetCV' + '-' * 10)
print('R square: ', r2_score(y_test_lin, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test_lin, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test_lin, y_pred))


# Final prediction + blending

# In[ ]:


ensemble = lgb_reg.predict(test_to_model) * 0.2 + xgb_reg.predict(test_to_model) * 0.2 + cat_reg.predict(test_to_model) * 0.2             + gb_reg.predict(test_to_model) * 0.2 + lassocv_reg.predict(test_to_model_lin) * 0.05 +             ridgecv_reg.predict(test_to_model_lin) * 0.05 + elasticv_reg.predict(test_to_model_lin) * 0.1


# In[ ]:


ensemble[:10]


# In[ ]:


#sub = pd.DataFrame()
#sub['Id'] = test_ID
#sub['SalePrice'] = ensemble
#sub.to_csv('submission_blend_free.csv',index=False)


# **Simple keras NN**

# In[ ]:


import keras
from keras import models, layers, regularizers


# In[ ]:


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1))
    
optimizer = keras.optimizers.RMSprop(0.001)
    
model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['mae', 'mse'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(X_train, y_train,
                   epochs=250,
                   batch_size=5,
                   validation_data=(X_test, y_test))


# In[ ]:


import matplotlib.pyplot as plt

mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(1, len(mae) + 1)

plt.figure(figsize=(12, 12))
plt.plot(epochs[20:], mae[20:], 'bo', label='Training mae')
plt.plot(epochs[20:], val_mae[20:], 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


test_predictions = model.predict(X_test).flatten()

plt.figure(figsize=(10, 10))
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-1000000, 1000000], [-1000000, 1000000])


# In[ ]:


y_pred = model.predict(X_test)

print('-' * 10 + 'NN' + '-' * 10)
print('R square Accuracy: ', r2_score(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))


# In[ ]:


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1))
    
optimizer = keras.optimizers.RMSprop(0.001)
    
model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['mae', 'mse'])


# In[ ]:


history = model.fit(train_to_model.drop(['SalePrice'], axis=1), train_to_model['SalePrice'],
                   epochs=300,
                   batch_size=5,
                   validation_split=0.05)


# In[ ]:


mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(1, len(mae) + 1)

plt.figure(figsize=(12, 12))
plt.plot(epochs[20:], mae[20:], 'bo', label='Training mae')
plt.plot(epochs[20:], val_mae[20:], 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


y_pred_model = model.predict(test_to_model)


# In[ ]:


model_pred = []

for pred in range(len(y_pred_model.tolist())):
    model_pred.append(y_pred_model.tolist()[pred][0])


# In[ ]:


model_pred[:10]


# In[ ]:


ensemble[:10]


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = model_pred
sub.to_csv('submission_NN.csv',index=False)


# In[ ]:


model_pred = np.array(model_pred)
model_pred


# In[ ]:


final_blend = ensemble * 0.3 + model_pred * 0.7


# In[ ]:


final_blend[:10]


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = final_blend
sub.to_csv('submission_blend_NN7.csv',index=False)


# In[ ]:


final_blend = ensemble * 0.4 + model_pred * 0.6
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = final_blend
sub.to_csv('submission_blend_NN6.csv',index=False)


# In[ ]:


final_blend = ensemble * 0.5 + model_pred * 0.5
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = final_blend
sub.to_csv('submission_blend_NN5.csv',index=False)


# In[ ]:


final_blend = ensemble * 0.6 + model_pred * 0.4
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = final_blend
sub.to_csv('submission_blend_NN4.csv',index=False)


# In[ ]:


final_blend = ensemble * 0.2 + model_pred * 0.8
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = final_blend
sub.to_csv('submission_blend_NN8.csv',index=False)

