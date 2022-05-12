#!/usr/bin/env python
# coding: utf-8

# # Introduction 😃😃😃
# 
# 
# ## Heart Attack Analysis & Prediction Dataset
# 
# - A dataset for heart attack classification
# 
# 
# ### About this dataset
# 
# - Age : Age of the patient
# - Sex : Sex of the patient
# - exang: exercise induced angina (1 = yes; 0 = no)
# - ca: number of major vessels (0-3)
# - cp : Chest Pain type chest pain type
# 
# - Value 1: typical angina
# - Value 2: atypical angina
# - Value 3: non-anginal pain
# - Value 4: asymptomatic
# 
# 
# - trtbps : resting blood pressure (in mm Hg)
# - chol : cholestoral in mg/dl fetched via BMI sensor
# - fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - rest_ecg : resting electrocardiographic results
# 
# - Value 0: normal
# - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# - thalach : maximum heart rate achieved
# 
# - target : 0= less chance of heart attack 1= more chance of heart attack
# 
# 
# 
# # Work plan 🤝🤝🤝🤝🤝
# 
# - 1- Data Exploration & Analysis 🤝🤝🤝
# - 2- Building a Machine Learning Model / classification score Volume

# # Data Exploration & Analysis 🤝🤝🤝

# In[ ]:


#Importing the basic librarires fot analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
import plotly.express as px


# In[ ]:


#Importing the dataset
df =pd.read_csv("../input/heart-attack-analysis-prediction-dataset/heart.csv")


# look the data set
df.head()


# In[ ]:


# looking the shape DataSet
df.shape


# In[ ]:


#Checking the dtypes of all the columns

df.info()


# In[ ]:


#checking null value 
df.isna().sum()


# - No any missing value 

# In[ ]:


# look  describe data set
df.describe().round(2)


# In[ ]:


# check unique value
df.nunique().sort_values()


# ## Some visual for Analysis

# In[ ]:


# interactive graph between Age and cholesterol in mg/dl

px.scatter(df,x="age",y="chol",color="output",title="Influence Age and cholesterol")


# - 0 = less chance of heart attack
# - 1 = more chance of heart attack
# - maybe when increasing old years have more chance of heart attack

# In[ ]:


# make groupby and pie graph to see how the percentage [ male and female]

df.groupby('sex')["output"].count().plot(kind="pie",autopct='%1.1f%%',shadow=True,figsize=(10,10),title="Male & Female have heart attack")


# ### Gender of the patients
# - 1 = 68.3% are **male**,
# - 0 = 31.7% are **female**

# In[ ]:


# make groupby and pie graph to see how the percentage [ Chest Pain type chest pain type]

df.groupby('cp')["output"].count().plot(kind="pie",autopct='%1.1f%%',shadow=True,figsize=(10,10),title="Chest Pain type chest pain type")


# ## cp : Chest Pain type chest pain type
# - 0 : asymptomatic 47.2%
# - 1 : typical angina 16.5%
# - 2 : atypical angina 28.7 %
# - 3 : non-anginal pain 7.6 %

# In[ ]:


# make groupby and pie graph to see how the percentage [ fasting blood sugar]


df.groupby('fbs')["output"].count().plot(kind="pie",autopct='%1.1f%%',shadow=True,figsize=(10,10),title="fasting blood sugar")


# ### fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - 0 = 85.1 %
# - 1 = 14.9 %

# In[ ]:


# make groupby and pie graph to see how the percentage [ Electrocardiographic results]


df.groupby('restecg')["output"].count().plot(kind="pie",autopct='%1.1f%%',shadow=True, figsize=(10,10),title="Electrocardiographic results")


# ## rest_ecg : resting electrocardiographic results
# 
# - Value 0: normal = 48.5%
# - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) = 50.2 %
# - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria = 1.3 %

# In[ ]:


# make groupby and pie graph to see how the percentage [ Target Variable - output]


df.groupby('output')["output"].count().plot(kind="pie",autopct='%1.1f%%',shadow=True,figsize=(10,10),title="Target Variable - output")


# ## Target Variable - output
# - More than half of the patients, **54.5 percent**, have a heart attack risk. The remaining **45.5 percent** .
# 

# In[ ]:


# make groupby and bar graph to see relationship between Age and have patients heart 


df.groupby('age')["output"].count().plot(kind="bar",figsize=(17,6), title="Relationship between Age and have patients heart ")


# ## Age Variable
# - The vast majority of patients are between 50 and 60.

# # Analysis Results 🙉🙈🙊
# 
# - After make some analysis , visual graph  and explore the data set , I see some results .
# 
# 
# #### The vast majority of patients are between 50 and 60.
# - Maybe when increasing old years have more chance of heart attack
# 
# #### Gender of the patients
# -  68.3% are Male,
# -  31.7% are Female
# 
# 
# #### Chest Pain type chest pain type
# - Asymptomatic 47.2%
# - Typical angina 16.5%
# - Atypical angina 28.7 %
# - Non-anginal pain 7.6 %
# 
# 
# #### (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - 0 = 85.1 %
# - 1 = 14.9 %
# 
# 
# #### Resting electrocardiographic results
# - Normal = 48.5%
# - Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) = 50.2 %
# - Showing probable or definite left ventricular hypertrophy by Estes' criteria = 1.3 %
# 
# 
# #### Target Variable - output
# - More than half of the patients, 54.5 percent, have a heart attack risk. The remaining 45.5 percent 
# 
# 

# # Building a Machine Learning Model / classification score Volume

# In[ ]:


#Importing the basic librarires for building model - classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,r2_score


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import  MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[ ]:


#Defined X value and y value , and split the data train
X = df.drop(columns="output")           
y = df["output"]    # y = quality

# split the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)


# In[ ]:


#Defined object from library classification 

LR = LogisticRegression()
DTR = DecisionTreeClassifier()
RFR = RandomForestClassifier()
KNR = KNeighborsClassifier()
MLP = MLPClassifier()
XGB = XGBClassifier()
SVR=SVC()


# In[ ]:


# make for loop for classification 

li = [LR,DTR,RFR,KNR,MLP,KNR,XGB,SVR]
d = {}
for i in li:
    i.fit(X_train,y_train)
    ypred = i.predict(X_test)
    print(i,":",accuracy_score(y_test,ypred)*100)
    d.update({str(i):i.score(X_test,y_test)*100})


# - we see the best model  **LogisticRegression = 88%**
# 

# In[ ]:


# make graph about Accuracy

plt.figure(figsize=(30, 6))
plt.title("Algorithm vs Accuracy")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(d.keys(),d.values(),marker='o',color='red')
plt.show()


# # Model Selection Results 😃😃😃
# 
# - Logistic Regression =  88.1 %
# - Decision Tree Classifier = 76.3 %
# - Random Forest Classifier =  81.5 %
# - K Neighbors Classifier = 69.7 %
# - MLP Classifier = 82.89 %
# - K Neighbors Classifier =  69.7 %
# - XGB Classifier =  81.5 %
# - SVC =  69.7 %
# 
# ### So , the best model Logistic Regression
# 
# ### You can change parameter in the library , maybe get better accuracy 

# # Notes 😃😃😃😃
# 
# - Thank for reading my analysis and my classification. 😃😃😃😃
# 
# - If you any questions or advice me please write in the comment . ❤️❤️❤️❤️
# 
# - If anyone has a model with a higher percentage, please tell me 🤝🤝🤝, it`s will support me .
# 
# # Vote ❤️😃
# 
# - If you liked my work upvote me ,
# 
# 
# # The End 🤝🎉🤝🎉
