#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Let declare the libraries that will be used for the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#importing the dataset
df=pd.read_csv('../input/fraudulent-transactions-prediction/Fraud.csv')
df.head()


# In[ ]:


### Let get more info about data
df.info()


# In[ ]:


df.shape


# In[ ]:


### Let check if there is any missing value

df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


## Correlation
import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


#check the fraud transaction 
df["isFraud"].value_counts()


# In[ ]:


df["isFraud"].value_counts(normalize=True)


# In[ ]:


# checking the payment type of the transactions
df["type"].value_counts()


# In[ ]:


# counting the number of transactions per type
plt.figure(figsize=(15,8))
sns.countplot(x="type", data=df,hue="isFraud" , palette="Set2")


# In[ ]:


# counting the number of transactions per type
plt.figure(figsize=(15,8))
sns.countplot(x="type", data=df,hue="isFlaggedFraud" , palette="Set2")


# ### Feature Selection Techniques
# 

# In[ ]:


df.head()


# In[ ]:


# dropping the variables that are not needed
# making a copy of the data
df.columns
#df_copy = df.copy()
#df_copy.columns


# In[ ]:


# columns with object type
df.select_dtypes(include=["object"]).columns


# In[ ]:


# droping NameOrig and NameDest

df.drop(['nameOrig','nameDest'], axis=1)


# In[ ]:


## Handle categorical feature Age
df['type'].unique()


# In[ ]:


##second technqiue
from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
df['type']= label_encoder.fit_transform(df['type'])
 
df['type'].unique()


# In[ ]:


# droping NameOrig and NameDest

df1=df.drop(['nameOrig','nameDest'], axis=1)
df1.head()


# In[ ]:


### Let divide the dataset into dependent and independent features
X=df1.iloc[:,:-1]##independent features
y=df1.iloc[:,-1]## dependent features


# In[ ]:



X.head()


# In[ ]:


y.head()


# In[ ]:


corrmat.index


# In[ ]:


df1.shape


# In[ ]:


###Linear KnearestNeightborRegressor

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[ ]:


print(model.feature_importances_)


# In[ ]:


## Plot graph of feature importance for better visualisation
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(9).plot(kind='barh')
plt.show()


# ### K Nearest Neighbor Regression 
# 
# 

# In[ ]:


sns.distplot(y)


# ### Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


regressor=KNeighborsRegressor(n_neighbors=1)
regressor.fit(X_train,y_train)


# In[ ]:


print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))


# In[ ]:


print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)


# In[ ]:


score.mean()


# ### Model Evaluation

# In[ ]:


prediction=regressor.predict(X_test)


# In[ ]:


sns.distplot(y_test-prediction)


# In[ ]:


plt.scatter(y_test,prediction)


# ### Conclusion
# 
# #### After analysing dataset
# 
# #### We can conclude that the probability of Credit Card Fraud Prediction might be null.
# 
# #### We can ensure that the bank system is secure but they still have to pay attention about the security measures.

# In[ ]:





# In[ ]:




