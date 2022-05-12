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


# # TITANIC DATASET PREDICTIONS WITH 1. LOGISTIC REGRESSION                                                                              2. DECISION TREE                                                                                    3. K NEAREST NEIGHBOR

# <p style="color:blue;font-size:21px;">The titanic dataset contains information regarding the passengers of the historic Titanics ship. We are all aware of the tragic fate of Titanic ship as it sank in the North Atlantic Ocean on 15th of April 1912. The dataset comprises of the various features about the survivors one of which is their survival status.
# The detailed list of the features are as follows</p>
#     <p style="color:green;font-size:20px"><b>1.'PassengerId',</b></p>
#     <p style="color:green;font-size:20px"><b>2.'Survived',</b></p>
#     <p style="color:green;font-size:20px"><b>3.'Pclass',</b></p> 
#     <p style="color:green;font-size:20px"><b>4.'Name',</b></p> 
#     <p style="color:green;font-size:20px"><b>5.'Sex',</b></p> 
#     <p style="color:green;font-size:20px"><b>6.'Age',</b></p> 
#     <p style="color:green;font-size:20px"><b>7.'SibSp',</b></p>
#     <p style="color:green;font-size:20px"><b>8.'Parch',</b></p> 
#     <p style="color:green;font-size:20px"><b>9.'Ticket',</b></p> 
#     <p style="color:green;font-size:20px"><b>10.'Fare',</b></p> 
#     <p style="color:green;font-size:20px"><b>11.'Cabin',</b></p> 
#     <p style="color:green;font-size:20px"><b>12.'Embarked'</b></p>

# ***IMPORTING THE LIBRARIES***

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


df.head()
    


# # EDA

# ***1.redundant features***

# In[ ]:


df.columns


# In[ ]:


for f in df.columns:
    print(f,df[f].nunique())


# In[ ]:


df=df.drop(columns=['PassengerId','Cabin','Ticket','Fare','Name'])


# In[ ]:


test_pid=test_df['PassengerId']
test_df=test_df.drop(columns=['PassengerId','Cabin','Ticket','Fare','Name'])


# ***1.Null Values***

# 

# 

# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.dropna()
df = df.reset_index(drop=True)


# In[ ]:


test_df.isnull().sum()


# In[ ]:


non_nf=[f for f in df.columns if df[f].dtype=='O']
non_nf


# In[ ]:


nf=[f for f in df.columns if df[f].dtype!='O']


# In[ ]:


c_nf=[f for f in nf if df[f].nunique()>=30]
d_nf=[f for f in nf if df[f].nunique()<30]


# In[ ]:


non_nf


# In[ ]:


for f in non_nf:
    sns.countplot(x=f,data=df,hue='Survived')
    plt.show()


# In[ ]:


d_nf


# In[ ]:


for f in d_nf:
    sns.countplot(x=f,data=df,hue='Survived')
    plt.show()


# In[ ]:


df.groupby('Survived').count()


# In[ ]:


for f in c_nf:
    sns.displot(x=f,data=df,hue='Survived',palette=['r','g'])
    plt.axvline(df[f].mean(), linestyle = '--', color = "red")
    plt.show()


# In[ ]:


df.groupby('Survived')[['Age']].mean()


# In[ ]:


df.groupby('Survived')[['Age']].median()


# In[ ]:


sns.heatmap(df.corr(),vmin=-1,vmax=1,annot=True)


# # Feature Engineering

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV


# In[ ]:


l=LabelEncoder()
for f in non_nf:
    df[f]=l.fit_transform(df[f])
    


# In[ ]:


m=MinMaxScaler()
for f in c_nf:
    f_scaled=m.fit_transform(df[f].values.reshape(-1,1))
    df[f]=pd.DataFrame(f_scaled.reshape(-1,1))


# In[ ]:


df


# In[ ]:


sdf=df['Survived']
df=df.drop(columns=['Survived'])
df


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df, sdf, test_size = 0.1, random_state = 42)


# # 1. LOGISTIC REGRESSION

# In[ ]:


lr=LogisticRegression()
parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
clf1 = GridSearchCV(lr,                    
                   param_grid = parameters,   
                   scoring='accuracy',        
                   cv=5)                     
clf1.fit(X_train,y_train)


# In[ ]:


y_pred=clf1.predict(X_test)


# In[ ]:


clf1.best_params_


# In[ ]:


clf1.score(X_test,y_test)


# In[ ]:


cm=confusion_matrix(y_test,y_pred)


# In[ ]:


sns.heatmap(cm,annot=True,fmt='d')
plt.show()


# # 2. DECISION TREE

# In[ ]:


dt=DecisionTreeClassifier()
tree_para = [{'criterion':['gini','entropy'],'max_depth':list(range(1,20))}]
clf = GridSearchCV(dt, tree_para, cv=10)
clf.fit(X_train,y_train)


# In[ ]:


print(clf.best_params_)
yt_pred=clf.predict(X_test)


# In[ ]:


clf.score(X_test,y_test)


# # 3. K NEAREST NEIGHBOR

# In[ ]:


kn=KNeighborsClassifier()
n_neighbors=list(range(1,31))
weights=['uniform','distance']
metric=['euclidean','manhattan','minikowski']
param_grid=dict(n_neighbors=n_neighbors,weights=weights,metric=metric)

clf2 = GridSearchCV(kn,param_grid, cv=5)
clf2.fit(X_train,y_train)


# In[ ]:


yk_pred=clf2.predict(X_test)


# In[ ]:


clf2.score(X_test,y_test)


# In[ ]:


cm=confusion_matrix(y_test,yk_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.show()


# In[ ]:


test_df


# In[ ]:


test_df.isnull().sum()


# In[ ]:


age_test=pd.DataFrame(df['Age'])
age_test=age_test.dropna()
m=age_test.median()


# In[ ]:


test_df=test_df.fillna(m)


# In[ ]:


l=LabelEncoder()
for f in non_nf:
    test_df[f]=l.fit_transform(test_df[f])
    


# In[ ]:


mi=MinMaxScaler()
for f in c_nf:
    f_scaled=mi.fit_transform(test_df[f].values.reshape(-1,1))
    test_df[f]=pd.DataFrame(f_scaled.reshape(-1,1))


# In[ ]:


test_pred=clf.predict(test_df)


# In[ ]:


test_pid=pd.Series(test_pid)
test_pred=pd.Series(test_pred)


# In[ ]:


Submission=pd.concat({'PassengerId':test_pid,'Survived':test_pred},axis=1)


# In[ ]:


Submission


# In[ ]:


Submission.to_csv('Submission.csv', index=False)


# ***Thank you for going through the notebook. Do share your thoughts in the comment section. And above all, do not forget to upvote if you have found it useful.***
