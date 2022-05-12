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


df=pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head().T


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# In[ ]:


# Dataset first look

print(df.shape)
print()
print(df.columns)
print()
print(df.isnull().sum())
print()
print(df.dtypes)


# In[ ]:


df.info()


# TotalCharges, as MontlyCharges must be float64 dtype

# In[ ]:


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# In[ ]:


for col in df.columns:
    print(col, " : ",df[col].isna().sum())


# The eleven rows with nan values will be dropped

# In[ ]:


df.dropna(axis=0, inplace=True)


# In[ ]:


df.describe().T


# In[ ]:


df.dtypes


# In[ ]:


df.head().T


# In[ ]:


plt.figure(figsize=(15,15))
corr=df.corr()
sns.heatmap(df.corr(), annot=True, cmap="inferno")


# In[ ]:


#  look into uniques elements for every column
df.apply(lambda x: x.unique())


# In[ ]:


df.apply(lambda x: x.unique())


# In[ ]:


#  "No" and "No internet service" has no meaning to stay toghether so I will repalce them with "No"
df["OnlineSecurity"] = df["OnlineSecurity"].apply(lambda x: x.replace("No internet service", "No"))
df["OnlineBackup"] = df["OnlineBackup"].apply(lambda x: x.replace("No internet service", "No"))
df["DeviceProtection"] = df["DeviceProtection"].apply(lambda x: x.replace("No internet service", "No"))
df["TechSupport"] = df["TechSupport"].apply(lambda x: x.replace("No internet service", "No"))
df["StreamingTV"] = df["StreamingTV"].apply(lambda x: x.replace("No internet service", "No"))
df["StreamingMovies"] = df["StreamingMovies"].apply(lambda x: x.replace("No internet service", "No"))

# Same for "No phone service"
df["MultipleLines"] = df["MultipleLines"].apply(lambda x: x.replace("No phone service", "No"))

df.apply(lambda x: x.unique())


# In[ ]:


df.drop(columns=["customerID"], axis=1, inplace=True)


# In[ ]:


# tenure = Number of months the customer has stayed with the company
# churn = This sample data module tracks a fictional telco company's customer churn based on a variety of possible factors. 
#         The churn column indicates whether or not the customer left within the last month. 


#  I'm looking to plot the distribution of customers which stayed with the company or not
#  in base of the number of months

no=df[df["Churn"]=="No"]["tenure"]
yes=df[df["Churn"]=="Yes"]["tenure"]

sns.set()
plt.figure(figsize=(10,7))

plt.hist([no,yes], label=["NO", "YES"])
plt.legend(loc="upper left")
plt.title("Churn Distribution on monthly based")
plt.xlabel("Number of months")
plt.ylabel("Number of Customers")


# As from the plot I can see that the customer staying in a inverse proportinal order
# In the first months most customers stay, but more time pass more people leave
# 
# This customer behaviour creates an incosistent and imballanced df

# In[ ]:


print("NO:", df[df["Churn"]=="No"]["tenure"].sort_values().sum())
print("YES:",df[df["Churn"]=="Yes"]["tenure"].sort_values().sum())


# In[ ]:


df.apply(lambda x: x.unique())


# Dataframe preprocessing for ML

# In[ ]:


#  In a new df I will replace no with 0, yest with 1
#  InternetService will not be touched at this moment

# 
dfx=df.copy()


# In[ ]:


dfx.drop("InternetService", axis=1, inplace=True)
dfx.columns


# In[ ]:


dfx.replace("No", 0, inplace=True)
dfx.replace("Yes", 1, inplace=True)

dfx.apply(lambda x: x.unique())


# In[ ]:


dfx["InternetService"] = df["InternetService"] 

dfx.head()


# In[ ]:


dfx["gender"].replace("Female", 0, inplace=True)
dfx["gender"].replace("Male", 1, inplace=True)


# In[ ]:


dfx.dtypes


# In[ ]:


# Label Encoding


# I will Use LabelEncoder instead of the most common pd.get_dummies() because i will have less features at the end
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


for col in dfx.columns:
    if dfx[col].dtypes == "object":
        dfx[col]=le.fit_transform(dfx[col])
        
dfx.head()


# In[ ]:


dfx.apply(lambda x: x.unique())


# In[ ]:


# Float columns must be scaled

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()

dfx["TotalCharges"] = mms.fit_transform(np.array(dfx["TotalCharges"]).reshape(-1, 1))
dfx["MonthlyCharges"] = mms.fit_transform(np.array(dfx["MonthlyCharges"]).reshape(-1, 1))
dfx["tenure"] = mms.fit_transform(np.array(dfx["tenure"]).reshape(-1, 1))


# In[ ]:


dfx.head()


# In[ ]:


#  Split the df for ML
#  I'm looking to predict the Churn colum
from sklearn.model_selection import train_test_split

X = dfx.drop("Churn", axis=1)
y = dfx["Churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 101)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print()
print(y_train.shape)
print(y_test.shape)


# In[ ]:


import tensorflow as tf
from tensorflow import keras as ks


# In[ ]:


model = ks.Sequential(
    [
        ks.layers.Dense(25,input_shape=(19,), activation="relu"),
        ks.layers.Dense(20, activation="relu"),
        ks.layers.Dense(15, activation="relu"),
        ks.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy", # I'm looking for 0 and 1 response
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs = 10)


# The epoch 6 is giving me about 80% accuracy

# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:


# This is an 2D array, but i need it into 1D as y_test
pred=model.predict(X_test)
pred


# In[ ]:


y1D_pred=[]

for x in pred:
    if x>0.5:
        y1D_pred.append(1)
    else:
        y1D_pred.append(0)
        
y1D_pred[:8]


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score

print("Confusion Matrix:")
print(confusion_matrix(y_test, y1D_pred))
print()

print("Classification Report")
print(classification_report(y_test, y1D_pred))

print("F1 Report")
print(f1_score(y_test, y1D_pred))


# In[ ]:


# plot Confusion matrix With Seaborn

cm = confusion_matrix(y_test, y1D_pred)

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("Confusion Matrix Predictions vs Actual values")


# CORRECT PREDICTION 1221 + 215 = 1436
# 
# 
# WRONG PREDICTION   108 + 214  = 322
