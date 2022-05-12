#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement:
# ### Assignment Part-I
# A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia. The data is provided in the CSV file below.
# 
#  
# 
# The company is looking at prospective properties to buy to enter the market. You are required to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not.
# 
#  
# 
# The company wants to know the following things about the prospective properties (Our target):
# 
#             a) Which variables are significant in predicting the price of a house, and
# 
#             b) How well those variables describe the price of a house.
#             
#             c) Determine the optimal value of lambda for ridge and lasso regression.
#             
# **P.S. : Please let me know how you liked my work and do suggest!! :)**

# ## Step 1 : Read and find the details of the Data

# In[ ]:


# Importing necessary packages
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# reading the Dataset
df_house = pd.read_csv("../input/regression-technique-eda/House Price Regression Tecnique/train.csv")
df_house.head()


# In[ ]:


#Reading number of rows and columns
df_house.shape


# In[ ]:


df_house.info()


# In[ ]:


#Details of the dataset
df_house.describe()


# ## Step 2 : Data Understanding, Preparation and EDA

# In[ ]:


#finding columns containing null values.
df_house.select_dtypes(include='object').isnull().sum()[df_house.select_dtypes(include='object').isnull().sum()>0]


# In[ ]:


#replacing NaN values with None
na_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure' ,'BsmtFinType1','BsmtFinType2', 'Electrical',
          'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC','Fence', 'MiscFeature']

for i in na_cols:
    df_house[i] =df_house[i].fillna("None")
    


# In[ ]:


#double check on the NaN values
df_house.select_dtypes(include='object').isnull().sum()[df_house.select_dtypes(include='object').isnull().sum()>0]


# No null values are present in the categorical variables anymore

# ### Removing skewed attributes
#       - We cannot keep categorical and numerical variables that contain more than 85% of the same data(skewed data),
#       else they might corrupt the results.
#       - Hence we will find out the columns that contain skewed and drop those columns.
# 

# In[ ]:


col_list = list()
for i in df_house.select_dtypes("object"):
    print(i, ' = ' , df_house[i].value_counts(normalize=True).max()) 
    if df_house[i].value_counts(normalize=True).max()>0.85:
        col_list.append(i)
        
print(col_list)


# In[ ]:


#Dropping the columns filtered above

df_house.drop(col_list, axis=1, inplace = True)


# In[ ]:


col_num_list = list()

for i in df_house.select_dtypes(include=['int64','float64']):
    if df_house[i].value_counts(normalize=True).max() > 0.85:
        col_num_list.append(i)
        
df_house.drop(col_num_list, axis=1, inplace=True)


# In[ ]:


df_house.shape


# ### Find the null values in numerical attributes and replace them.
# 

# In[ ]:


num_cols = df_house.select_dtypes(include=['int64','float64'])

num_cols.isnull().sum()[num_cols.isnull().sum()>0]


# In[ ]:


#Replacing these 3 column data to their respective mean values
df_house['LotFrontage'] = df_house['LotFrontage'].replace(np.nan ,df_house['LotFrontage'].median())

df_house['MasVnrArea'] = df_house['MasVnrArea'].replace(np.nan, df_house['MasVnrArea'].median())


# In[ ]:


df_house['GarageYrBlt'] = df_house['GarageYrBlt'].replace(np.nan, df_house['GarageYrBlt'].median())


# In[ ]:


df_house.isnull().sum()


# No more null values in the dataset

# In[ ]:


#Further cleaning is required for column "MoSold" as it is not important for analysis
df_house.drop('MoSold', axis=1, inplace=True)
df_house.shape


# ### Creating new column by combining existing columns

# In[ ]:


#Adding Column 'Remodelled' if a house is remodelled or not

def Remodel(r):
    if(r['YearBuilt'] == r['YearRemodAdd']):
        return 0
    elif(r['YearBuilt'] < r['YearRemodAdd']):
        return 1
    else:
        return 2
    
df_house['Remodelled'] = df_house.apply(Remodel, axis=1)


# In[ ]:


#Adding Column 'BuildRemodelAge' to quantify age of house after being built or remodelled.

def buildRemodelAge(r):
    if(r['YearBuilt'] == r['YearRemodAdd']):
        return r['YrSold'] - r['YearBuilt']
    else:
        return r['YrSold'] - r['YearRemodAdd']
       
df_house['BuildRemodelAge'] = df_house.apply(buildRemodelAge, axis=1)


# In[ ]:


# setting new Column  'oldNewGarage' to check if garage is old(0) or new(1).
def constructionPeriod(r):
    if r == 0:
        return 0
    elif r >= 1900 and r < 2000:        
        return 0
    else:   
        return 1
    
df_house['oldNewGarage'] = df_house['GarageYrBlt'].apply(constructionPeriod)


# In[ ]:


#Dropping old features , as they are not required and we have created new features based on them. 

df_house.drop(['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt'], axis = 1, inplace = True)


# In[ ]:


df_house.shape


# #### Removing Duplicates

# In[ ]:


#Check for duplicate values in the dataset

df_house.drop_duplicates()


# Hence we see that there are no duplicates in the dataset.

# In[ ]:


# Checking outliers at 25%,50%,75%,90%,95% and above

df_house.describe(percentiles=[0.25,0.5,0.75,0.90,0.95])


# In[ ]:


num_list = ['LotArea','MasVnrArea','WoodDeckSF','OpenPorchSF','TotalBsmtSF']

plt.figure(figsize=[15,20])
count=1
for i in num_list:
    plt.subplot(5,3,count)
    sns.boxplot(y=i, palette="Set2", data=df_house)
    count= count+1
plt.show()


# In[ ]:


# Treating Outliers

# Removing values beyond 98% for MasVnrArea

qtl_MasVnrArea = df_house['MasVnrArea'].quantile(0.98)
df_house = df_house[df_house["MasVnrArea"] < qtl_MasVnrArea]

# Removing data beyond 98% for LotArea

qtl_LotArea = df_house['LotArea'].quantile(0.98)
df_house = df_house[df_house["LotArea"] < qtl_LotArea]

# Removing values beyond 99% for TotalBsmtSF

qtl_TotalBsmtSF = df_house['TotalBsmtSF'].quantile(0.99)
df_house = df_house[df_house["TotalBsmtSF"] < qtl_TotalBsmtSF]

# Removing values beyond 99% for WoodDeckSF

qtl_WoodDeckSF = df_house['WoodDeckSF'].quantile(0.99)
df_house = df_house[df_house["WoodDeckSF"] < qtl_WoodDeckSF]

# Removing values beyond 99% for OpenPorchSF

qtl_OpenPorchSF = df_house['OpenPorchSF'].quantile(0.99)
df_house = df_house[df_house["OpenPorchSF"] < qtl_OpenPorchSF]


# In[ ]:


# Finding the retained data after removal

print((round(df_house.shape[0])/1460)*100)


# ### Step 3: Data Visualisation

# In[ ]:


#Correlation matrix for the dataset
plt.figure(figsize=(20,10))
sns.heatmap(df_house.corr(), annot=True)
plt.show()


# ### Comments:
#     - We need to remove highly collinear features except those which affect the dependent variable.
#     - We can see 'TotRmsAbvGrd' and 'GrLivArea' have a collinearity of 83% .
#     - Also 'GarageCars' and 'GarageArea' have a collinearity of 89%.

# In[ ]:


sns.distplot(df_house['SalePrice'])


# In[ ]:


print("Skewness: %f" % df_house['SalePrice'].skew())
print("Kurtosis: %f" % df_house['SalePrice'].kurt())


# In[ ]:


var = 'GrLivArea'
data = pd.concat([df_house['SalePrice'], df_house[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


var = 'TotalBsmtSF'
data = pd.concat([df_house['SalePrice'], df_house[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_house['SalePrice'], df_house[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


plt.figure(figsize=(10,5))
sns.pairplot(df_house, x_vars=['MSSubClass','LotFrontage','LotArea'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(df_house, x_vars=['OverallQual', 'OverallCond','MasVnrArea'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(df_house, x_vars=['BsmtFinSF1', 'BsmtUnfSF','TotalBsmtSF'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(df_house, x_vars=['1stFlrSF','2ndFlrSF', 'GrLivArea'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(df_house, x_vars=['BsmtFullBath','FullBath', 'HalfBath'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(df_house, x_vars=['BedroomAbvGr','TotRmsAbvGrd', 'Fireplaces'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(df_house, x_vars=['GarageCars','GarageArea', 'WoodDeckSF'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(df_house, x_vars=['OpenPorchSF','SalePrice', 'Remodelled'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(df_house, x_vars=['BuildRemodelAge'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')

plt.show()


# ### Comments:
#     - Most of the feature data are scattered except 'LotArea', '1stFlrSF'and 'GrLivArea' which show some correlation with 'SalePrice'.
#     - We will analyse the other features further below.

# In[ ]:


#Removing highly collinear features as discussed above.
df_house.drop(['TotRmsAbvGrd', 'GarageArea'], axis=1, inplace=True)


# In[ ]:


df_house.shape


# In[ ]:


#Lets have a closer look at the correlation matrix with only high correlations with SalePrice
corMat = df_house.corr()
k = 10 #number of variables for heatmap
cols = corMat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_house[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# ## Step 4: Data Preparation

# In[ ]:


#We are converting categorical variables to numerical features


# In[ ]:


df_house.select_dtypes("object").columns


# In[ ]:


df_house['LotShape'] = df_house['LotShape'].map({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0})
df_house['ExterQual'] = df_house['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0 })
df_house['BsmtQual'] = df_house['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
df_house['BsmtExposure'] = df_house['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0})
df_house['BsmtFinType1'] = df_house['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 
                                                                 'None': 0})
df_house['HeatingQC'] = df_house['HeatingQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
df_house['KitchenQual'] = df_house['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
df_house['FireplaceQu'] = df_house['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
df_house['GarageFinish'] = df_house['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0 })
df_house['BldgType'] = df_house['BldgType'].map({'Twnhs': 5, 'TwnhsE': 4, 'Duplex': 3, '2fmCon': 2, '1Fam': 1, 
                                                                 'None': 0 })
df_house['HouseStyle'] = df_house['HouseStyle'].map({'SLvl': 8, 'SFoyer': 7, '2.5Fin': 6, '2.5Unf': 5, '2Story': 4, 
                                                                 '1.5Fin': 3, '1.5Unf': 2, '1Story': 1, 'None': 0 })
df_house['Fence'] = df_house['Fence'].map({'GdPrv': 4, 'GdWo': 3, 'MnPrv': 2, 'MnWw': 1, 'None': 0 })
df_house['LotConfig'] = df_house['LotConfig'].map({'Inside': 5, 'Corner': 4, 'CulDSac': 3, 'FR2': 2, 'FR3': 1, 
                                                           'None': 0  })
df_house['MasVnrType'] = df_house['MasVnrType'].map({'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0 })
df_house['SaleCondition'] = df_house['SaleCondition'].map({'Normal': 1, 'Partial': 1, 'Abnorml': 0, 'Family': 0, 
                                                                   'Alloca': 0, 'AdjLand': 0, 'None': 0})


# In[ ]:


df_house.shape


# In[ ]:


#Dummy creation for remaining categorical variables
dum_MSZ = pd.get_dummies(df_house['MSZoning'],prefix = 'MSZoning', drop_first=True)
df_house = pd.concat([df_house,dum_MSZ] , axis=1)

dum_Neighborhood = pd.get_dummies(df_house['Neighborhood'],prefix = 'Neighborhood', drop_first=True)
df_house = pd.concat([df_house,dum_Neighborhood] , axis=1)

dum_RoofStyle = pd.get_dummies(df_house['RoofStyle'],prefix = 'RoofStyle', drop_first=True)
df_house = pd.concat([df_house,dum_RoofStyle] , axis=1)

dum_Exterior1st = pd.get_dummies(df_house['Exterior1st'],prefix = 'Exterior1st', drop_first=True)
df_house = pd.concat([df_house,dum_Exterior1st] , axis=1)

dum_Exterior2nd = pd.get_dummies(df_house['Exterior2nd'],prefix = 'Exterior2nd', drop_first=True)
df_house = pd.concat([df_house,dum_Exterior2nd] , axis=1)

dum_Foundation = pd.get_dummies(df_house['Foundation'],prefix = 'Foundation', drop_first=True)
df_house = pd.concat([df_house,dum_Foundation] , axis=1)

dum_GarageType = pd.get_dummies(df_house['GarageType'],prefix = 'GarageType', drop_first=True)
df_house = pd.concat([df_house,dum_GarageType] , axis=1)


# In[ ]:


df_house.head()


# In[ ]:


# drop the below columns as we now have new columns derived from these columns

df_house = df_house.drop(['MSZoning', 'Neighborhood', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'Foundation', 
                                'GarageType'], axis=1)

df_house.head()


# In[ ]:


df_house.info()


# ## Step 5: Model Building

# In[ ]:


y = df_house['SalePrice']
y.head(5)


# In[ ]:


X = df_house.drop(['SalePrice'],axis=1)
X.head(5)


# In[ ]:


# scaling the features

from sklearn.preprocessing import scale

# storing column names in cols
# scaling (the dataframe is converted to a numpy array)

cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# In[ ]:


from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


# split into train and test
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=42)


# In[ ]:





# In[ ]:


# Running RFE with the output number of the variable equal to 50

lm = LinearRegression()
lm.fit(X_train, y_train)

# running RFE
rfe = RFE(lm)            
rfe = rfe.fit(X_train, y_train)


# In[ ]:


col = X_train.columns[rfe.support_]
X_train_rfe = X_train[col]


# In[ ]:



# Associate the new 50 columns to X_train and X_test for further analysis

X_train = X_train_rfe[X_train_rfe.columns]
X_test =  X_test[X_train.columns]


# ### Ridge Regression:

# In[ ]:


# list pf alphas

params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
                    9.0, 10.0, 20, 50, 100, 500, 1000 ]}

ridge = Ridge()

# cross validation

folds = 5
ridge_model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
ridge_model_cv.fit(X_train, y_train)


# In[ ]:


# display the mean scores

ridge_cv_results = pd.DataFrame(ridge_model_cv.cv_results_)
ridge_cv_results = ridge_cv_results[ridge_cv_results['param_alpha']<=500]
ridge_cv_results[['param_alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']].sort_values(by = ['rank_test_score'])


# In[ ]:


# plotting mean test and train scoes with alpha 

ridge_cv_results['param_alpha'] = ridge_cv_results['param_alpha'].astype('int32')

# plotting

plt.plot(ridge_cv_results['param_alpha'], ridge_cv_results['mean_train_score'])
plt.plot(ridge_cv_results['param_alpha'], ridge_cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()


# In[ ]:


# get the best estimator for lambda

ridge_model_cv.best_estimator_


# In[ ]:


# check the coefficient values with alpha = 100

alpha = 100
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_


# In[ ]:


mean_squared_error(y_test, ridge.predict(X_test))


# In[ ]:


# Put the Features and coefficienst in a dataframe

ridge_df = pd.DataFrame({'Features':X_train.columns, 'Coefficient':ridge.coef_.round(4)})
ridge_df.reset_index(drop=True, inplace=True)
ridge_df


# In[ ]:


# Assign the Features and their coefficient values to a dictionary which would be used while plotting the bar plot

ridge_coeff_dict = dict(pd.Series(ridge.coef_.round(4), index = X_train.columns))
ridge_coeff_dict


# In[ ]:


#RFE

# Do an RFE to minimise the features to 15
X_train_ridge = X_train[ridge_df.Features]

lm = LinearRegression()
lm.fit(X_train_ridge, y_train)

# running RFE
rfe = RFE(lm)            
rfe = rfe.fit(X_train_ridge, y_train)


# In[ ]:


# Method to get the coefficient values

def find(x):
    return ridge_coeff_dict[x]

# Assign top 10 features to a temp dataframe for further display in the bar plot

temp1_df = pd.DataFrame(list(zip( X_train_ridge.columns, rfe.support_, rfe.ranking_)), columns=['Features', 'rfe_support', 'rfe_ranking'])
temp1_df = temp1_df.loc[temp1_df['rfe_support'] == True]
temp1_df.reset_index(drop=True, inplace=True)

temp1_df['Coefficient'] = temp1_df['Features'].apply(find)
temp1_df = temp1_df.sort_values(by=['Coefficient'], ascending=False)
temp1_df = temp1_df.head(10)
temp1_df


# In[ ]:


print("Ridge Score for Train Set: ",round(ridge.score(X_train,y_train),2))


# In[ ]:


print("Ridge Score for Test Set: ",round(ridge.score(X_test, y_test),2))


# In[ ]:


from sklearn.metrics import r2_score
y_pred_train = ridge.predict(X_train)
y_pred_test = ridge.predict(X_test)
print("Ridge Train R2-Score : ",round(r2_score(y_train, y_pred_train),2))
print("Ridge Test R2-Score : ",round(r2_score(y_test, y_pred_test),2))


# In[ ]:


# bar plot to determine the variables that would affect pricing most using ridge regression

plt.figure(figsize=(20,20))
plt.subplot(4,3,1)
sns.barplot(y = 'Features', x='Coefficient', palette='Set2', data = temp1_df)
plt.show()


# 
# The above graph displays the top 10 variables based on the Ridge Regression model that are significant in predicting the price of a house.

# ### Lasso Regression:

# In[ ]:


lasso = Lasso()

# list of alphas

params = {'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]}

# cross validation

folds = 5
lasso_model_cv = GridSearchCV(estimator = lasso,                         
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

lasso_model_cv.fit(X_train, y_train)


# In[ ]:


# display the mean scores

lasso_cv_results = pd.DataFrame(lasso_model_cv.cv_results_)
lasso_cv_results[['param_alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']].sort_values(by = ['rank_test_score'])


# In[ ]:


# plotting mean test and train scores with alpha 

lasso_cv_results['param_alpha'] = lasso_cv_results['param_alpha'].astype('float64')

#plotting Alpha against Negative Mean Absolute Error
plt.plot(lasso_cv_results['param_alpha'], lasso_cv_results['mean_train_score'])
plt.plot(lasso_cv_results['param_alpha'], lasso_cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()


# In[ ]:


#getting the best estimator for lambda

lasso_model_cv.best_estimator_


# In[ ]:


#checking the coefficient values with lambda = 0.01

alpha = 0.01

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 
lasso.coef_


# In[ ]:


# Checking the mean squared error

mean_squared_error(y_test, lasso.predict(X_test))


# In[ ]:



# Putting the shortlisted Features and coefficienst in a dataframe

lasso_df = pd.DataFrame({'Features':X_train.columns, 'Coefficient':lasso.coef_.round(4)})
lasso_df = lasso_df[lasso_df['Coefficient'] != 0.00]
lasso_df.reset_index(drop=True, inplace=True)
lasso_df


# In[ ]:


# Putting the Features and Coefficients in dictionary

lasso_coeff_dict = dict(pd.Series(lasso.coef_, index = X_train.columns))
lasso_coeff_dict


# In[ ]:


# RFE to minimise the features to 15

X_train_lasso = X_train[lasso_df.Features]

lm = LinearRegression()
lm.fit(X_train_lasso, y_train)

# running RFE

rfe = RFE(lm)            
rfe = rfe.fit(X_train_lasso, y_train)


# In[ ]:


# Method to get the coefficient values

def find(x):
    return lasso_coeff_dict[x]

# Assigning top 10 features to a temp dataframe for further display in the bar plot

temp2_df = pd.DataFrame(list(zip( X_train_lasso.columns, rfe.support_, rfe.ranking_)), columns=['Features', 'rfe_support', 'rfe_ranking'])
temp2_df = temp2_df.loc[temp2_df['rfe_support'] == True]
temp2_df.reset_index(drop=True, inplace=True)

temp2_df['Coefficient'] = temp2_df['Features'].apply(find)
temp2_df = temp2_df.sort_values(by=['Coefficient'], ascending=False)
temp2_df = temp2_df.head(10)
temp2_df


# In[ ]:


print("Lasso Score for Train Set: ", round(lasso.score(X_train,y_train),2))
print("Lasso Score for Test Set: ", round(lasso.score(X_test,y_test),2))


# In[ ]:


from sklearn.metrics import r2_score
y_pred_train = lasso.predict(X_train)
y_pred_test = lasso.predict(X_test)
print("Lasso Train R2-Score : ",round(r2_score(y_train, y_pred_train),2))
print("Lasso Test R2-Score : ",round(r2_score(y_test, y_pred_test),2))


# In[ ]:


# bar plot to determine the variables that would affect pricing most using ridge regression

plt.figure(figsize=(20,20))
plt.subplot(4,3,1)
sns.barplot(y = 'Features', x='Coefficient', palette='Set2', data = temp2_df)
plt.show()


# ## Conclusion:
# 
#     - Observations from analysis above: 
#            - Score(approx.) for Ridge Regression:-
#            > Train Set : 0.89
#            > Test Set : 0.88
#            
#            - Score for Lasso Regression:-
#            > Train Set : 0.90
#            > Test Set : 0.88
#            
#            - Ridge Train R2-Score :  0.9
#              Ridge Test R2-Score :  0.88
#            
#            - Lasso Train R2-Score :  0.9
#              Lasso Test R2-Score :  0.88
#              
#      - The Mean Squared error in case of Ridge and Lasso are:
#            > Ridge : 581840731.24
#            > Lasso : 570841157.43
#        (MSE for Lasso is lesser than that of Ridge)
#        
#      - Lasso Regression helps in Feature Reduction , we select the features with high coefficients in Lasso Regression.
#      - Second floor square feet ,Overall Quality of the house, Exterior covering on house, First Floor square feet,   Above            grade (ground) living area square feet , Physical locations within Ames city limits, Type of foundation, Type 1 finished        square feet, Size of garage in car capacity and walkout or garden level walls are the top 10 factors that affect the            Sale Price of the houses most.
#     

# In[ ]:




