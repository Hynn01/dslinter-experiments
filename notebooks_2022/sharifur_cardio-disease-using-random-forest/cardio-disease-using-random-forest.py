#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/cardiovascular-disease-dataset/cardio_train.csv',sep=';') 


# In[ ]:


df


# In[ ]:


df['cardio'].value_counts()


# In[ ]:


sns.countplot(df['cardio'])


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.countplot(x='gender',hue='cardio',data=df,palette='colorblind',edgecolor=sns.color_palette('dark',n_colors=1))


# In[ ]:





# In[ ]:


df['yr'] = (df['age']/365).round(0)


# In[ ]:


df['yr']


# In[ ]:


sns.countplot(x='yr',hue='cardio',data=df,palette='colorblind',edgecolor=sns.color_palette('dark',n_colors=1))


# # Random Forest

# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


df=df.drop(['yr','id'],axis=1)


# In[ ]:


x=df.iloc[:,:-1]


# In[ ]:


x


# In[ ]:


y=df.iloc[:,11]


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.30,random_state=1)


# In[ ]:


xtrain


# # Now we will use Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


Rclf = RandomForestClassifier()


# In[ ]:


Rclf.fit(xtrain,ytrain)


# In[ ]:


Rclf.score(xtest,ytest)


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf = DecisionTreeClassifier()


# In[ ]:


clf.fit(xtrain,ytrain)


# In[ ]:


clf.score(xtest,ytest)


# # Save ML Models

# In[ ]:


import pickle as pk
with open('Cardio_Check_Model_1', 'wb') as file:
    pk.dump(Rclf,file)


# # Feature Selection

# In[ ]:


from sklearn.feature_selection import SelectKBest


# In[ ]:


from sklearn.feature_selection import f_classif


# In[ ]:


FIT_FEATURES = SelectKBest(score_func=f_classif)


# In[ ]:


FIT_FEATURES.fit(x,y)


# In[ ]:


score_col = pd.DataFrame(FIT_FEATURES.scores_,columns = ['score value'])


# In[ ]:


name_col = pd.DataFrame(x.columns)


# In[ ]:


top_features = pd.concat([name_col,score_col],axis=1)


# In[ ]:


top_features


# In[ ]:


top_features.nlargest(7,'score value')


# In[ ]:


x_new = df[["age", "cholesterol","weight","gluc","ap_lo","ap_hi","active","smoke"]]


# In[ ]:


x_new


# In[ ]:


y_new = df["cardio"]


# In[ ]:


y_new


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_new,y_new,test_size=.30,random_state=1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


Rfc = RandomForestClassifier()


# In[ ]:


Rfc.fit(xtrain,ytrain)


# In[ ]:


Rfc.score(xtest,ytest)


# In[ ]:


xtest


# # Feature Importance

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


model = ExtraTreesClassifier()


# In[ ]:


model.fit(x,y)


# In[ ]:


model.feature_importances_


# In[ ]:


top = pd.Series(model.feature_importances_, index = x.columns)


# In[ ]:


top


# In[ ]:


top.nlargest(8).plot(kind='pie')


# In[ ]:




