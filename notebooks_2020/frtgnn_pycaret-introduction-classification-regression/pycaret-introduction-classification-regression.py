#!/usr/bin/env python
# coding: utf-8

# # Introduction to PyCaret - An open source low-code ML library
# 
# ## This notebook consists 2 parts
#  - Classification part using Titanic DataSet
#  - Regression part using House Price Regression DataSet

# ![](https://pycaret.org/wp-content/uploads/2020/03/Divi93_43.png)
# 
# You can reach pycaret website and documentation from https://pycaret.org
# 
# PyCaret is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within seconds in your choice of notebook environment.
# 
# PyCaret being a low-code library makes you more productive. With less time spent coding, you and your team can now focus on business problems.
# 
# PyCaret is simple and easy to use machine learning library that will help you to perform end-to-end ML experiments with less lines of code. 
# 
# PyCaret is a business ready solution. It allows you to do prototyping quickly and efficiently from your choice of notebook environment.
# 

# # let's install pycaret ! 

# In[ ]:


get_ipython().system('pip install pycaret')


# # Part 1 Classification
# 
# ![](https://www.sciencealert.com/images/articles/processed/titanic-1_1024.jpg)

# # We start by loading the libraries

# In[ ]:


import numpy as np 
import pandas as pd 


# # Read our files

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test  = pd.read_csv('../input/titanic/test.csv')
sub   = pd.read_csv('../input/titanic/gender_submission.csv')


# # Import whole classification

# In[ ]:


from pycaret.classification import *


# # let's see what we're dealing with

# In[ ]:


train.head()


# In[ ]:


train.info()


# # Set up our dataset (preprocessing)

# In[ ]:


clf1 = setup(data = train, 
             target = 'Survived',
             numeric_imputation = 'mean',
             categorical_features = ['Sex','Embarked'], 
             ignore_features = ['Name','Ticket','Cabin'],
             silent = True)


# # Compare the models

# In[ ]:


compare_models()


# # let's create a Light GBM Model

# In[ ]:


lgbm  = create_model('lightgbm')      


# # Let's tune it!

# In[ ]:


tuned_lightgbm = tune_model(lgbm)


# # Learning Curve

# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'learning')


# # AUC Curve

# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'auc')


# # Confusion Matrix

# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')


# # Feature Importance

# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'feature')


# # whole thing!

# In[ ]:


evaluate_model(tuned_lightgbm)


# # Interpretation

# In[ ]:


interpret_model(tuned_lightgbm)


# # Predictions

# In[ ]:


predict_model(tuned_lightgbm, data=test)


# In[ ]:


predictions = predict_model(tuned_lightgbm, data=test)
predictions.head()


# In[ ]:


sub['Survived'] = round(predictions['Score']).astype(int)
sub.to_csv('submission.csv',index=False)
sub.head()


# # Extra: Blending made easy!

# In[ ]:


logr  = create_model('lr');          

blend = blend_models(estimator_list=[tuned_lightgbm,logr])


# # Part2 - Regression

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSYeyNpaoAW-3rFX9-ORmiJ-uLAAswYBRhszs2QzllV7MCfFPvk&usqp=CAU)

# # Import Whole Regression

# In[ ]:


from pycaret.regression import *


# # let's see the data

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample= pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# # Set up our dataset (preprocessing)

# In[ ]:


reg = setup(data = train, 
             target = 'SalePrice',
             numeric_imputation = 'mean',
             categorical_features = ['MSZoning','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType',
                                     'Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',   
                                     'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',    
                                     'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',   
                                     'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',   
                                     'Electrical','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                                     'SaleCondition']  , 
             ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],
             normalize = True,
             silent = True)


# # let's compare different regression models!

# In[ ]:


compare_models()


# # let's do LGBM

# In[ ]:


lgb = create_model('lightgbm')


# # gotta tune it

# In[ ]:


tuned_lgb = tune_model(lgb)


# # SHAP Values (impact on model output)

# In[ ]:


interpret_model(tuned_lgb)


# In[ ]:


predictions = predict_model(tuned_lgb, data = test)
sample['SalePrice'] = predictions['Label']
sample.to_csv('submission_house_price.csv',index=False)
sample.head()


# # thank you very much for checking my notebook!
