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


# In[ ]:


# Load required libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

pd.pandas.set_option('display.max_columns',None)


# In[ ]:


df_data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_data.head()


# In[ ]:


df_data.info()


# In[ ]:


df_data.shape


# In[ ]:


#Check for null values
df_data.isnull().sum()


# ## Feature description:
# * 'customerID': Customer ID
# * 'gender': Whether the customer is a male or a female
# * 'SeniorCitizen': Whether the customer is a senior citizen or not (1, 0)
# * 'Partner': Whether the customer has a partner or not (Yes, No)
# * 'Dependents': Whether the customer has dependents or not (Yes, No)
# * 'tenure': Number of months the customer has stayed with the company
# * 'PhoneService': Whether the customer has a phone service or not (Yes, No)
# * 'MultipleLines': Whether the customer has multiple lines or not (Yes, No, No phone service)
# * 'InternetService': Customer’s internet service provider (DSL, Fiber optic, No)
# * 'OnlineSecurity': Whether the customer has online security or not (Yes, No, No internet service)
# * 'OnlineBackup': Whether the customer has online backup or not (Yes, No, No internet service)
# * 'DeviceProtection': Whether the customer has device protection or not (Yes, No, No internet service)
# * 'TechSupport': Whether the customer has tech support or not (Yes, No, No internet service)
# * 'StreamingTV': Whether the customer has streaming TV or not (Yes, No, No internet service)
# * 'StreamingMovies': Whether the customer has streaming movies or not (Yes, No, No internet service)
# * 'Contract': The contract term of the customer (Month-to-month, One year, Two year)
# * 'PaperlessBilling': Whether the customer has paperless billing or not (Yes, No)
# * 'PaymentMethod': The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# * 'MonthlyCharges': The amount charged to the customer monthly
# * 'TotalCharges': The total amount charged to the customer
# * 'Churn': Whether the customer churned or not (Yes or No)
# 
# 
# * **'Churn' is the target feature**
# 

# ## Data cleaning

# ### The feature 'TotalCharges' has float values but it's data type is object. So we will check on this.
# First we will find the index positions that have the space(i.e missing value). Then we will replace the spaces with null value and convert the data-type of 'TotalCharges' feature to 'float64'. Next we will impute the missing values with the median value of this feature.

# In[ ]:


#Index of rows that have a blank space i.e. it is a null value
na_index = df_data[df_data['TotalCharges'].apply(lambda x: x.isspace())==True].index
print(na_index)


# In[ ]:


# Fill the 11 blank values with the np.nan
df_data['TotalCharges'] = df_data['TotalCharges'].replace(' ', np.nan)

#Convert to float type
df_data['TotalCharges'] = df_data['TotalCharges'].astype('float64')


# In[ ]:


#Replace the 11 missing values with median of the feature
df_data['TotalCharges']=df_data['TotalCharges'].fillna(df_data['TotalCharges'].median())


# In[ ]:


# Drop customerID feature as it is not required
df_data.drop('customerID', axis=1, inplace=True)


# In[ ]:


#Apart from 'SeniorCitizen' feature, all the other features have values like Yes/No. So we will map 0 to No and 1 to Yes for the 'SeniorCitizen' feature.
df_data['SeniorCitizen']=df_data['SeniorCitizen'].map({0:'No', 1:'Yes'})


# In[ ]:


df_data.head()


# In[ ]:





# ## EDA

# ### Check the data distribution of Target feature

# In[ ]:


sns.countplot(x="Churn", data=df_data)


# In[ ]:


df_data['Churn'].value_counts()


# #### This is an imbalanced data as the number of 'No' is far greater than the number of 'Yes' in our dataset
# #### 73% data is for 'No' and remaining 27% data is for 'Yes'

# In[ ]:


# Getting categorical and numerical features
cat_cols = [cname for cname in df_data.columns if df_data[cname].dtype=='object' and cname!='Churn']
num_cols = [cname for cname in df_data.columns if df_data[cname].dtype!='object']

print('categorical features: ', cat_cols)
print('numerical features: ', num_cols)


# ### Univariate analysis

# In[ ]:


#Plotting the impact of categorical features on 'Churn'
plt.figure(figsize=(25,25))
for i,cat in enumerate(cat_cols):
    plt.subplot(6,3,i+1)
    sns.countplot(data = df_data, x= cat, hue = "Churn")
plt.show()


# In[ ]:


# Plotting the impact of continuous features on 'Churn'
plt.figure(figsize=(15,5))
for j,con in enumerate(num_cols):
    plt.subplot(1,3,j+1)
    sns.histplot(data = df_data, x= con, hue = "Churn", multiple="stack")
plt.show()


# In[ ]:


#We will try to create groups based on the 'tenure' feature
df_data['tenure'].describe()


# In[ ]:


df_data['tenure_grp'] = pd.cut(df_data['tenure'], bins=[0,12,24,36,48,60,np.inf], labels=['0-12', '13-24', '25-36', '37-48', '49-60', '60+'])


# In[ ]:


df_data['tenure_grp'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(data=df_data, x='tenure_grp',hue = "Churn")


# In[ ]:


df_data.drop('tenure', axis=1, inplace=True)


# In[ ]:


df_data.head()


# In[ ]:


#Mapping target feature
df_data['Churn']=df_data['Churn'].map({'No':0, 'Yes':1})


# In[ ]:


#convert categorical data into dummy variables
df_data_dummy = pd.get_dummies(df_data,drop_first=True)
df_data_dummy.head()


# In[ ]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', data=df_data_dummy, hue='Churn')


# Insight:
# Total charges increase as the Monthly charges increase

# In[ ]:


plt.figure(figsize=(15,5))
df_data_dummy.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# In[ ]:





# ## Model building

# In[ ]:


df_data_model = df_data_dummy.copy(deep=True)
df_data_model.head()


# In[ ]:


#Seperate independent and dependent features
X = df_data_model.loc[:, df_data_model.columns!='Churn']
y = df_data_model['Churn']

X.shape, y.shape


# In[ ]:


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_valid.shape, y_valid.shape)


# In[ ]:


#from pycaret.classification import *


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




