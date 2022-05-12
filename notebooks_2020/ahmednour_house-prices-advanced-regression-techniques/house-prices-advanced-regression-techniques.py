#!/usr/bin/env python
# coding: utf-8

# ## Kaggle Competition for House Prices: Advanced Regression Techniques 

# In[ ]:


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df.head()


# # =================Handle Test Data==================================
# 

# In[ ]:


test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df.shape
test_df.head()


# In[ ]:


#check null values
test_df.isnull().sum()


# In[ ]:


## Fill Missing Values

test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())


# In[ ]:


test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])


# In[ ]:


test_df.shape


# In[ ]:


test_df.drop(['Alley'],axis=1,inplace=True)


# In[ ]:


test_df.shape


# In[ ]:


test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])


# In[ ]:


test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])


# In[ ]:


test_df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[ ]:


test_df.shape


# In[ ]:


test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])

test_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


test_df.shape


# In[ ]:


test_df.drop(['Id'],axis=1,inplace=True)


# In[ ]:


test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])


# In[ ]:


sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])


# In[ ]:


sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])


# In[ ]:


test_df.loc[:, test_df.isnull().any()].head()


# In[ ]:


test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])


# In[ ]:


test_df.shape


# In[ ]:


test_df.to_csv('formulatedtest.csv',index=False)


# # ====================================

# In[ ]:





# In[ ]:


df['MSZoning'].value_counts()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


## Fill Missing Values

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[ ]:


df.drop(['Alley'],axis=1,inplace=True)


# In[ ]:


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])


# In[ ]:


df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[ ]:


df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[ ]:


df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])


# In[ ]:


df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.drop(['Id'],axis=1,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[ ]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# In[ ]:


df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


##HAndle Categorical Features


# In[ ]:


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[ ]:


len(columns)


# In[ ]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[ ]:


main_df=df.copy()


# In[ ]:


## Combine Test Data 

test_df=pd.read_csv('formulatedtest.csv')


# In[ ]:


test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


final_df=pd.concat([df,test_df],axis=0)


# In[ ]:


final_df['SalePrice']


# In[ ]:


final_df.shape


# In[ ]:


final_df=category_onehot_multcols(columns)


# In[ ]:


final_df.shape


# In[ ]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[ ]:


final_df.shape


# In[ ]:


final_df


# In[ ]:


df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[ ]:


df_Train.head()


# In[ ]:


df_Test.head()


# In[ ]:


df_Train.shape


# In[ ]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[ ]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# ## Prediciton and selecting the Algorithm

# In[ ]:


import xgboost
classifier=xgboost.XGBRegressor()


# In[ ]:


import xgboost
regressor=xgboost.XGBRegressor()


# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]


# In[ ]:


## Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[ ]:


# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=regressor, 
                               param_distributions=hyperparameter_grid,
                               cv=5, n_iter=50,
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = 4,
                               verbose = 5,
                               return_train_score = True,
                               random_state=42)


# In[ ]:


random_cv.fit(X_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


random_cv.best_estimator_


# In[ ]:


regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:


df_Test


# In[ ]:


#df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[ ]:


df_Test.shape


# In[ ]:


df_Test.head()


# In[ ]:


y_pred=regressor.predict(df_Test)


# In[ ]:


rf_pred =regressor.predict(df_Test)


# In[ ]:


#df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[ ]:


##Create Sample Submission file and Submit using ANN
#pred=pd.DataFrame(ann_pred)
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)


# ## Step2

# In[ ]:


pred.columns=['SalePrice']


# In[ ]:


temp_df=df_Train['SalePrice'].copy()


# In[ ]:


temp_df.column=['SalePrice']


# In[ ]:


df_Train.drop(['SalePrice'],axis=1,inplace=True)


# In[ ]:


df_Train=pd.concat([df_Train,temp_df],axis=1)


# In[ ]:


df_Test.head()


# In[ ]:


df_Test=pd.concat([df_Test,pred],axis=1)


# In[ ]:


df_Train=pd.concat([df_Train,df_Test],axis=0)


# In[ ]:


df_Train.shape


# In[ ]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# ## Artificial Neural Network Implementation

# In[ ]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
from keras import backend as K

# def root_mean_squared_error(y_true, y_pred):
#         return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


    
    
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu',input_dim = 174))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'he_uniform'))

# Compiling the ANN
classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 10, nb_epoch = 1000)


# In[ ]:


ann_pred=classifier.predict(df_Test.drop(['SalePrice'],axis=1).values)

