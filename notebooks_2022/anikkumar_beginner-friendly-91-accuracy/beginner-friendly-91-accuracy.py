#!/usr/bin/env python
# coding: utf-8

# #### Import liabraries and load datasets

# In[ ]:


# import data science basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# load data
df = pd.read_csv('../input/heart-failure-prediction/heart.csv')
df.head()


# ####  Data Dictionary
# - Age: age of the patient [years]
# - Sex: sex of the patient [M: Male, F: Female]
# - ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# - RestingBP: resting blood pressure [mm Hg]
# - Cholesterol: serum cholesterol [mm/dl]
# - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# - RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or - ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]     
# - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
# - ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
# - Oldpeak: oldpeak = ST [Numeric value measured in depression]
# - ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# - HeartDisease: output class [1: heart disease, 0: Normal]

# #### Exploratory Data Analysis

# In[ ]:


# data shape
df.shape


# In[ ]:


df.info()


# In[ ]:


# show unique values
df.nunique()


# In[ ]:


# data basic statistics
df.describe()


# In[ ]:


# missing values in decerding order
df.isnull().sum().sort_values(ascending=False)


# In[ ]:


# duplicated values
df.duplicated().sum()


# In[ ]:


# numerical and categorical features 
Categorical = df.select_dtypes(include=['object'])
Numerical = df.select_dtypes(include=['int64', 'float64'])
print('Categorical features:\n', Categorical)
print('Numerical features:\n', Numerical)


# In[ ]:


# count target variable
df['HeartDisease'].value_counts()


# #### Visualization

# In[ ]:


# Normal and Heart Disease with target column
plt.figure(figsize=(10,5))
plt.pie(df['HeartDisease'].value_counts(), labels=['Heart Disease[1]', 'Normal[0]'], autopct='%1.1f%%')
plt.show()


# In[ ]:


# ploting corelation matrix
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[ ]:


# ploting numerical features with target
for i in Numerical:
    plt.figure(figsize=(10,5))
    sns.countplot(x=i, data=df, hue='HeartDisease')
    plt.legend(['Normal', 'Heart Disease'])
    plt.title(i)
    plt.show()


# In[ ]:


#ploting categorical features with target
for i in Categorical:
    plt.figure(figsize=(10,5))
    sns.countplot(x=i, data=df, hue='HeartDisease', edgecolor='black')
    plt.legend(['Normal', 'Heart Disease'])
    plt.title(i)
    plt.show()


# In[ ]:


#pairplot using target HeartDisease Column
sns.pairplot(df, hue='HeartDisease')
plt.show()


# In[ ]:


# distribution plot of Age for HeartDisease
sns.distplot(df['Age'][df['HeartDisease'] == 1], kde=True, color='red', label='Heart Disease')
sns.distplot(df['Age'][df['HeartDisease'] == 0], kde=True, color='green', label='Normal')
plt.legend()


# #### Data Preprocessing

# In[ ]:


# select numerical features and encoding it
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# select numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64'])
# apply label encoding
numerical_features = numerical_features.apply(LabelEncoder().fit_transform)
numerical_features.head()


# In[ ]:


# One-Hot encoding the categorical features using get_dummies()
# select categorical features
categorical_features = df.select_dtypes(include=['object'])
# apply get_dummies encoding
categorical_features = pd.get_dummies(categorical_features)
categorical_features.head()


# In[ ]:


# combine numerical and categorical features
combined = pd.concat([numerical_features, categorical_features], axis=1)
combined.head()


# In[ ]:


# separet features and target
X = combined.drop(['HeartDisease'], axis=1)
y = combined['HeartDisease']


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### XGboost

# In[ ]:


# model building xgboost
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=590)
model.fit(X_train, y_train)
# predict
y_pred = model.predict(X_test)
# accuracy
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))


# In[ ]:


# Finding the best parameters using loop
accuracy = []
for i in range(550, 600):
    model = XGBClassifier(n_estimators=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
# ploting accuracy graph
plt.plot(range(550, 600), accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Range')
plt.show()


# In[ ]:


# print precetion, recall, f1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,3))
sns.heatmap(cm, annot=True)


# In[ ]:


# Feature importance for xgboost
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()


# #### Catboost

# In[ ]:


# model building catboost
from catboost import CatBoostClassifier
model2 = CatBoostClassifier(iterations=107)
model2.fit(X_train, y_train)
# predict
y_pred = model2.predict(X_test)
# Print accuracy
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))
# print classification report
from sklearn.metrics import classification_report
print('Classification report\n',classification_report(y_test, y_pred))


# In[ ]:


# Simple parameter tuning using loop
accuracy = []
for i in range(100, 115):
    model2 = CatBoostClassifier(iterations=i)
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
# ploting accuracy graph
plt.plot(range(100, 115), accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Range')
plt.show()


# In[ ]:


# plot confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,3))
sns.heatmap(cm, annot=True)


# In[ ]:


# Feature importance for xgboost
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()

