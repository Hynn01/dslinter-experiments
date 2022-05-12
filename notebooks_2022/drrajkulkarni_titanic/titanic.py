#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sn
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# In[ ]:


pf_test = pd.read_csv("../input/titanic/test.csv")
pf_train = pd.read_csv("../input/titanic/train.csv")

pf_train.info()


# In[ ]:


pf_test.info()


# # What is Age distribution among passengers
# 

# In[ ]:



plt.hist(pf_train["Age"],bins=[10,20,30,40,50,60,70,80,90],ec='red',color='blue',histtype='bar',rwidth=0.80)
plt.xlabel("Age")
plt.ylabel("No of Passengers")
plt.title("Age distibution among passengers")
plt.show()


# In[ ]:



plt.hist(pf_train["Sex"],bins=2,ec='red',color='blue',histtype='bar',rwidth=0.80)
plt.xlabel("SEX")
plt.ylabel("No of Passengers")
plt.title("Male Vs Female")
plt.show()


# In[ ]:


plt.hist(pf_train["Fare"],bins=10,ec='red',color='blue',histtype='bar',rwidth=0.80)
plt.xlabel("Fare")
plt.ylabel("No of Passengers")
plt.title("Fare distribution")
plt.show()


# In[ ]:


sns.countplot(x="Survived",hue="Sex", data = pf_train)


# In[ ]:


sns.countplot(x="Embarked",hue="Sex", data = pf_train)


# # Lets look for Null values in Data and remove the appropriate feature

# In[ ]:


pf_train.isnull().sum()


# In[ ]:


pf_test.isnull().sum()


# In[ ]:


pf_train['Age'].median()


# In[ ]:


pf_test["Age"].median()


# In[ ]:


pf_test['Age'].fillna(pf_test['Age'].median(), inplace = True)
pf_train['Age'].fillna(pf_train['Age'].median(), inplace = True)


# In[ ]:



pf_test['Fare'].fillna(pf_test['Fare'].median(), inplace = True)


# In[ ]:


pf_train.drop("Cabin",axis=1,inplace=True)
pf_test.drop("Cabin",axis=1,inplace=True)


# In[ ]:


sns.heatmap(pf_train.isnull())


# In[ ]:


sns.heatmap(pf_test.isnull())


# In[ ]:


sex = pd.get_dummies(pf_train['Sex'], drop_first=True)
sex.head(5)


# In[ ]:


sex1 = pd.get_dummies(pf_test['Sex'], drop_first=True)
sex1.head(5)


# In[ ]:


embark = pd.get_dummies(pf_train["Embarked"],drop_first=True)
embark.head(5)


# In[ ]:


embark1 = pd.get_dummies(pf_test["Embarked"],drop_first=True)
embark1.head(5)


# In[ ]:


pcl = pd.get_dummies(pf_train["Pclass"],drop_first=True)
pcl.head(5)


# In[ ]:


pcl1 = pd.get_dummies(pf_test["Pclass"],drop_first=True)
pcl1.head(5)


# In[ ]:


pf_train= pd.concat([pf_train,sex,embark,pcl],axis=1)
pf_train.head(5)


# In[ ]:


pf_test= pd.concat([pf_test,sex1,embark1,pcl1],axis=1)
pf_test.head(5)


# In[ ]:


pf_train.drop(["Sex","Name", "Pclass","Ticket", "Embarked","Fare"], axis=1, inplace=True)
pf_train.head(5)


# In[ ]:


pf_test.drop(["Sex","Name", "Pclass","Ticket", "Embarked","Fare"], axis=1, inplace=True)
pf_test.head(5)


# In[ ]:


pf_train.info()


# In[ ]:


pf_test.info()


# # Modelling the data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix 
all_features = pf_train.drop("Survived",axis=1)
Targeted_feature = pf_train["Survived"]
X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.3,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[ ]:


# Logistic Regression 
from sklearn.linear_model import LogisticRegression # Logistic Regression

model = LogisticRegression()
model.fit(X_train,y_train)
prediction_lr=model.predict(X_test)
print('--------------Final report ----------------------------')
print('The accuracy of the Logistic Regression is',round(accuracy_score(prediction_lr,y_test)*100,2))
kfold = KFold(n_splits=10, random_state=None,shuffle=True) # k=10, split the data into 10 equal parts
result_lr=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')
print('The cross validated score for Logistic REgression is:',round(result_lr.mean()*100,2))
y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)
sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)


# In[ ]:


prediction_lr=model.predict(pf_test)


# In[ ]:


submission= pf_test[["PassengerId"]]


# In[ ]:


submission.shape


# In[ ]:


submission["Survived"] = prediction_lr


# In[ ]:


submission.head(10)


# In[ ]:


submission.shape


# In[ ]:


submission.to_csv("submission.csv",index = None)


# In[ ]:


pd.read_csv("submission.csv")


# In[ ]:




