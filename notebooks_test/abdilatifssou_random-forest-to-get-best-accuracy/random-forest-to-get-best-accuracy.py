#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Analyse the Data** 

# In[ ]:


data=pd.read_csv('../input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
data.head(5)


# In[ ]:


data.columns


# In[ ]:


#♠move the target data to the end of the dataframe 
def ToTheEnd(df,column):
    Target_data=df[column]
    df=df.drop([column],axis=1)
    df[column]=Target_data
    return df

data=ToTheEnd(data,'class')


# check for null values

# In[ ]:


data.info()


# great!! no null values 

# In[ ]:


data['class'].head(10)
ax = sns.countplot(x='class',data=data)


# deal with categorical features 

# In[ ]:


data['class']=data['class'].map({'STAR':0,'GALAXY':1,'QSO':2}).astype(int)


# In[ ]:


ax = plt.subplots(figsize=(10,8))
sns.heatmap(data.corr(), vmax=.8 ,annot=True,square=True,fmt=".2f")


# Something not right 
# yes ! we should remove the unnecessary features 

# In[ ]:


data=data.drop(['objid','rerun','specobjid','fiberid'],axis=1)
#recall the heatmap
ax = plt.subplots(figsize=(10,8))
sns.heatmap(data.corr(), vmax=.8 , annot=True,square=True,fmt=".2f")


# Look Cool!!

# In[ ]:


data.describe()


# move to features distribution

# In[ ]:


def feature_dist(df0,df1,df2,label0,label1,label2,features):
    plt.figure()
    fig,ax=plt.subplots(13,1,figsize=(8,45))
    i=0
    for ft in features:
        i+=1
        plt.subplot(13,1,i)
        # plt.figure()
        sns.distplot(df0[ft], hist=False,label=label0)
        sns.distplot(df1[ft], hist=False,label=label1)
        sns.distplot(df2[ft], hist=False,label=label2)
        plt.xlabel(ft, fontsize=11)
        #locs, labels = plt.xticks()
        plt.tick_params(axis='x', labelsize=9)
        plt.tick_params(axis='y', labelsize=9)
    plt.show()
t0 = data.loc[data['class'] == 0]
t1 = data.loc[data['class'] == 1]
t2 = data.loc[data['class'] == 2]
features = data.columns.values[:13]
feature_dist(t0,t1,t2, 'STAR', 'GALAXY','QSO', features)


# check the redshift feature "notice the graph" 

# In[ ]:


#data['redshift'].describe()
data[['redshift','class']].groupby(['class'],as_index=False).mean().sort_values(by='class',ascending=False)


# now everything look clear right ?

# **Prepare data**

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


X=data.drop(['class'],axis=1)
y=data['class']


# In[ ]:


X=StandardScaler().fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)
X_train


# **Let's Predict**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, y_train)


# In[ ]:


y_pred = random_forest.predict(X_test)

Test_acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 3)
print("Score ",Test_acc_random_forest)


# In[ ]:


sns.heatmap(confusion_matrix(y_test , y_pred), center=True,annot=True,fmt='.1f')


# **Fin**
