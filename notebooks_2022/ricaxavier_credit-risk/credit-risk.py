#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install -U scikit-learn


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualization packaes
import matplotlib.pyplot as plt
import seaborn as sns

#Machine learning
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score,recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Parametro de configuração para os gráficos
from matplotlib import rcParams
rcParams['figure.figsize'] = 12,4
rcParams['lines.linewidth'] = 3
rcParams['xtick.labelsize'] = 'x-large'
rcParams['ytick.labelsize'] = 'x-large'


# The main purpose of this analysis is to predict whether a new customer can be a reliable customer. It's a way to avoid default and increase the company's revenue.
# 
# 
# __Features of Dataset__
# 
# - person_age - Customer Age
# - person_income - Annual Income
# - personhomeownership - Home ownership
# - personemplength - Employment length (in years)
# - loan_intent - Loan intent
# - loan_grade - Loan grade
# - loan_amnt- Loan amount
# - loanintrat - Interest rate
# - loan_status - Loan status (0 is non default 1 is default)
# - loanpercentincome - Percent income
# - cbpersondefaultonfile - Historical default
# - cbpresoncredhistlength - Credit history length

# ### Loading Credit Risk Dataset

# In[ ]:


credit_risk_df = pd.read_csv('../input/credit-risk-dataset/credit_risk_dataset.csv')


# In[ ]:


#Copy of the dataset
credit_risk_df_copy = credit_risk_df


# In[ ]:


#First 5 dataset rows
credit_risk_df.head()


# ### Exploratory Analysis

# In[ ]:


#Checking information about the type of features
credit_risk_df.info()


# There are null fields in features 'loan_int_rate' and 'person_emp_length' and 4 features with the type string

# In[ ]:


#Dataset size
data_rows = credit_risk_df.shape[0]
data_colunms = credit_risk_df.shape[1]

print(f'This dataset have {data_rows} rows and {data_colunms} columns.')


# In[ ]:


#Null values
credit_risk_df.isnull().sum()


# In[ ]:


#N/A values
credit_risk_df.isna().sum()


# In[ ]:


#Summary statistics
credit_risk_df.describe()


# - With this summary statistics we can analyse the atribute person_age and person_emp_length and see that the maximum 144 and 123 could be an outliers because those represent age and years of employment.
# 

# In[ ]:


#We can see that the maximum age is 144 years, so probabilly it is an outlier
features = ['person_age','person_emp_length','loan_percent_income','cb_person_cred_hist_length']
plt.figure(figsize=(10,5))
for i in range(0,len(features)):
    plt.subplot(1, len(features), i + 1)
    sns.boxplot(y=credit_risk_df[features[i]], color='CornflowerBlue', orient='v')
    plt.tight_layout()


# In[ ]:


#Checking the distribution - we have assimetric distribuition
plt.figure(figsize=(10,5))
credit_risk_df.hist()
plt.show()


# In[ ]:


#Checking balance columns target
credit_risk_df.loan_status.value_counts()


# In[ ]:


plt.figure(figsize=(8,4))
sns.countplot(x='loan_status', data=credit_risk_df)
plt.show()


# - This an imbalanced feature; could be a problem with the trainning because the positive class is less than negative class. I will test the algorithm with balanced and imbalanced class.

# In[ ]:


plt.figure(figsize=(10,5))
sns.histplot(data=credit_risk_df, x= 'person_age', bins=15)
plt.show()


# - The most customer are young people

# In[ ]:


plt.figure(figsize=(10,5))
sns.histplot(data=credit_risk_df, x= 'person_emp_length', bins=15)
plt.show()


# - the most custumer have less than ten years of employment

# __Analysing the correlation between variables__

# In[ ]:


variables = ['person_age','person_income','person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_status']
credit_risk_corr = credit_risk_df[variables].corr()
credit_risk_corr


# In[ ]:


plt.figure(figsize=(8,8))
sns.heatmap(credit_risk_corr, cmap='Blues', annot=True, fmt='.2f')
plt.show()


# _Variables with most correlation_
# - cb_person_cred_hist_length x person_age
# - loan_percent_income x loan_amount
# - person_income x loan_amount
# 

# __Categorical Variables__

# In[ ]:


var_categorical = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for i in var_categorical:
    print(f'Total row of variable {i}')
    print(credit_risk_df[i].value_counts())
    print()
   


# In[ ]:


#Analysing categorical variables with target
plt.figure(figsize=(15,9))
for i in range(0,len(var_categorical)):
    plt.subplot(2,2,i + 1)
    sns.countplot(data= credit_risk_df, x = var_categorical[i], hue='loan_status')
    plt.tight_layout()


# - The categorical variable that has the most correlation with the target is the person_home_ownership, the customer who lives in a rent house can have more probability to be default customer

# __Summary of exploratory analysis__
# 
# - We have to transform null fields in features 'loan_int_rate' and 'person_emp_length' 
# - Transform categorical variable type string in numerical into their corresponding number
# - the outliers can be deleted
# - The data are not simetric
# - the dataset is imbalanced
# - There are some important variables correlation with the target
# 
# 

# ### Feature Engeneering

# __Deal with Outliers__

# In[ ]:


#Looking for the outliers
credit_risk_df['person_age'].sort_values(ascending=False).head(10)


# Drop index 81, 32297, 183, 747, 575      

# In[ ]:


credit_risk_df['person_emp_length'].sort_values(ascending=False).head(10)


# __Drop outliers__

# In[ ]:


credit_risk_df.drop(credit_risk_df.loc[credit_risk_df['person_emp_length'] == 123].index, inplace=True)


# In[ ]:


credit_risk_df.loc[credit_risk_df['person_emp_length'] == 123]


# In[ ]:


credit_risk_df.drop(credit_risk_df.loc[credit_risk_df['person_age'] >= 123].index, inplace=True)


# In[ ]:


credit_risk_df.loc[credit_risk_df['person_age'] >= 123].index


# In[ ]:


#Confirm there is no outliers 
features = ['person_age','person_emp_length','loan_percent_income','cb_person_cred_hist_length']
plt.figure(figsize=(10,5))
for i in range(0,len(features)):
    plt.subplot(1, len(features), i + 1)
    sns.boxplot(y=credit_risk_df[features[i]], color='CornflowerBlue', orient='v')
    plt.tight_layout()


# __Deal with N/A and Null values__

# In[ ]:


credit_risk_df.loc[credit_risk_df['person_emp_length'].isna()]


# In[ ]:


credit_risk_df.loc[credit_risk_df['loan_int_rate'].isna()]


# In[ ]:


#After drop outliers
mean_person_emp_length = credit_risk_df['person_emp_length'].mean()
mean_loan_int_rate = credit_risk_df['loan_int_rate'].mean()


# In[ ]:


print(f'Mean variable person_emp_length, {mean_person_emp_length} and loan_int_rate, {mean_loan_int_rate}.')


# In[ ]:


#Fill na with mean
credit_risk_df['person_emp_length'] = credit_risk_df['person_emp_length'].fillna(mean_person_emp_length) 


# In[ ]:


credit_risk_df['person_emp_length'].isna().sum()


# In[ ]:


credit_risk_df['loan_int_rate'] = credit_risk_df['loan_int_rate'].fillna(mean_loan_int_rate) 


# In[ ]:


credit_risk_df['loan_int_rate'].isna().sum()


# In[ ]:


credit_risk_df.info()


# __Transform categorical variables__

# In[ ]:


def categorical_variables(df):
    object_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    label_encoder = LabelEncoder()
    for col in object_cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df


# In[ ]:


credit_df = categorical_variables(credit_risk_df)


# In[ ]:


credit_df.head()


# In[ ]:


credit_df.info()


# __Applying Train_test_split__

# In[ ]:


y = credit_df['loan_status'].values
print(y)


# In[ ]:


x_df = credit_df.drop('loan_status', axis=1)
x = x_df.values
print(x)


# __Applying train_test_split to split train and test data__

# In[ ]:


seed = 27
size = 0.3
X_train, X_test, Y_train, y_test = train_test_split(x, y, test_size=size, random_state=seed)


# In[ ]:


print(f'X_train size {X_train.shape},X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test {y_test.shape}')


# __Applying Normalization__
# - Transform features on a similar scale

# In[ ]:


scaler = MinMaxScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)
    
    


# In[ ]:


#X_train scaled
X_train_norm


# In[ ]:


#X_test scaled
X_test_norm


# ### Machine Learning Model Trainning

# In[ ]:


#Logistic Regression
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train_norm, Y_train)
score = modelo.score(X_train_norm, Y_train)
print('Accuracy train data: %.2f%%' % (score * 100))


# In[ ]:


predict = modelo.predict(X_test_norm)


# In[ ]:


predict


# __Score__

# In[ ]:


accuracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict)
recall = recall_score(y_test, predict)
fscore = f1_score(y_test, predict)

print('Accuracy predict: %.2f%%' % (accuracy * 100.0))
print('Precision: %.2f%%' % (precision * 100.0))
print('Recall: %.2f%%' % (recall * 100.0))
print('F1_score: %.2f%%' % (fscore * 100.0))


# __Confusion Matrix__

# In[ ]:


plt.figure(figsize=(5,5))

sns.heatmap(confusion_matrix(y_test,predict), annot=True, cmap='Blues', fmt='g', cbar=False)

plt.title("Confusion Matrix", fontsize=14, fontweight='bold')

plt.show()


# __Imbalanced classes__
# 
# - This dataset has the target class imbalanced and i'm gonna try to get a better performance using the SMOTE algoritm to balance the class.
# 

# In[ ]:


print('Distribution target class before oversample: ', Counter(Y_train))


# In[ ]:


oversample= SMOTE(sampling_strategy=1, random_state=42, k_neighbors=3)
X_train_over, Y_train_over = oversample.fit_resample(X_train_norm, Y_train)
print('Distribuition target class after oversample: ', Counter(Y_train_over))


# In[ ]:


#Trainning model with oversmaple examples
modelo.fit(X_train_over, Y_train_over)
score = modelo.score(X_train_over, Y_train_over)
print('Accuracy train oversample data: %.2f%%' % (score * 100))


# In[ ]:


predict_over = modelo.predict(X_test_norm)


# In[ ]:


#Score with oversample
accuracy_oversample = accuracy_score(y_test, predict_over)
precision_oversample = precision_score(y_test, predict_over)
recall_oversample = recall_score(y_test, predict_over)
f1_score_oversample = f1_score(y_test, predict_over)

print('Accuracy predict: %.2f%%' % (accuracy_oversample * 100.0))
print('Precision: %.2f%%' % (precision_oversample * 100.0))
print('Recall: %.2f%%' % (recall_oversample * 100.0))
print('F1_score: %.2f%%' % (f1_score_oversample * 100.0))



# __Summary of the metrics__
# - Accuracy means that the ratio of the correctly labeled subjects to the whole pool of subjects, so this metric decrease after the oversample, but after that, this metrics is not a better choice.
# - Precision is the ratio of the correctly positive labeled to all positive labeled.
# - Recall is the ratio of positive predicted to all who are real positive
# - f1_score considers both precision and recall
# 
# <p>I can see that the recall and f1_score are better after the oversample</p>

# __Confusion Matrix after oversample__

# In[ ]:


plt.figure(figsize=(5,5))

sns.heatmap(confusion_matrix(y_test,predict_over), annot=True, cmap='Blues', fmt='g', cbar=False)

plt.title("Confusion Matrix", fontsize=14, fontweight='bold')

plt.show()


# ### Trainning with RandomForestClassifier
# 
# - Random Forest could be a better choice because it has a good handle with overfitting problem.
# - the number of th decision trees participating in the process could have a high accurate

# In[ ]:


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)


# In[ ]:


Y_predict_clf = clf.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, Y_predict_clf)
precision = precision_score(y_test, Y_predict_clf)
recall = recall_score(y_test, Y_predict_clf)
fscore = f1_score(y_test, Y_predict_clf)

print('Accuracy predict: %.2f%%' % (accuracy * 100.0))
print('Precision: %.2f%%' % (precision * 100.0))
print('Recall: %.2f%%' % (recall * 100.0))
print('F1_score: %.2f%%' % (fscore * 100.0))


# In[ ]:


plt.figure(figsize=(5,5))

sns.heatmap(confusion_matrix(y_test,Y_predict_clf), annot=True, cmap='Blues', fmt='g', cbar=False)

plt.title("Confusion Matrix", fontsize=14, fontweight='bold')

plt.show()


# - We had a better metrics using the RandomForestClassifier algorithm decreasing the false positive and negative results and promoting the increase the right results

# In[ ]:




