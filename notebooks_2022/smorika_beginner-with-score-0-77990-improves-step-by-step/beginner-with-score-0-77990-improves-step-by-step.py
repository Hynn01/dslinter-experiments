#!/usr/bin/env python
# coding: utf-8

# Kaggle Titanic Submission 5 - GridSearchCV
# 
# Adding improvement one by one to see if the score gets improved from the past best score of 0.78229
# 
# for Submission 5, I will try GridSearchCV

# In[ ]:


# Loading libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# delete submission.csv if it is already in Output folder

# os.remove('submission.csv')


# In[ ]:


# Loading the Train dataset (train.csv)

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data.head()


# In[ ]:


# Loading the Test dataset (test.csv)

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()


# In[ ]:


# Concat train_data and test_data

# before .concat() drop Survived feature from train dataset 
# as test dataset do not have this feature

# join with train dataset later on
Survived_data = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)

df = pd.concat([train_data, test_data], sort=False, ignore_index=True)

df2 = df.copy()

# check ignore_index is working and index is continuous in whole dataset
df.tail()


# ## 1. Exploratory Data Analysis (EDA) to understand the dataset

# In[ ]:


df.shape


# In[ ]:


df.info()


# so train and test datasets combined, now 1309 rows

# In[ ]:


df.describe().T


# ## 2. Data Pre-processing - preparing the data for modelling

# Omit features that are not important 
# 
# removing columns which I think are not important as input features to predict the Survived feature based on EDA earlier, 
# 
# I may adjust this at later stage when I evaluate the model accuracy

# In[ ]:


columns_to_drop = ['PassengerId', 'Name','Ticket', 'Cabin', 'Embarked']

df = df.drop(columns_to_drop, axis=1)

df.head()


# Non numeric feature to numeric feature

# In[ ]:


df.info()


# Sex feature needs to be turned into a numerical feature

# In[ ]:


# There is no missing values in Sex column but checking if there are values other than female and male

df['Sex'].value_counts(dropna=False)


# In[ ]:


df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)
df['Sex'].value_counts()


# Missing values

# In[ ]:


df.info()


# Fare feature has missing value but only for 1 row,
# so this will be replaced by the mean value of Fare feature

# In[ ]:


df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# In[ ]:


df.info()


# In[ ]:


df['Age'].isnull().sum()


# Age feature has 263 missing values
# 
# As tested in my previous notebook version, I will impute this with Iterative imputation with RandomForestRegressor

# In[ ]:


# before imputation

df['Age'].describe()


# In[ ]:


df_rfr = df.copy()


# Iterative imputation with RandomForestRegressor
# 
# I will only use columns that did not have any missing values to start with (exclusing columns I already dropped)
# 
# to predict missing Age values
# 
# 

# In[ ]:


df_rfr.info()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# define features to predict the age
age_df = df_rfr[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']]

# separate age_df into train (with Age) and test (Age is NaN) and to ndarray
have_age = age_df[age_df['Age'].notnull()].values
no_age = age_df[age_df['Age'].isnull()].values

# separate train data to X and y

X = have_age[:, 1:]
y = have_age[:, 0]


# In[ ]:


# build age prediction model with RandomForest

rfr = RandomForestRegressor(random_state = 0, n_estimators = 100, n_jobs = -1)
rfr.fit(X, y)


# In[ ]:


# use the model to predict the Age for test data

# fix df_copy later
age_predicted = rfr.predict(no_age[:, 1:])

# age_predicted = np.round_(age_predicted, decimals=1)
print(age_predicted)


# In[ ]:


# replace missing Age values with age_predicted

df_rfr.loc[(df_rfr['Age'].isnull()), 'Age'] = age_predicted


# In[ ]:


df_rfr['Age'] = df_rfr['Age'].map(lambda x: round(x, 2))


# In[ ]:


df_rfr['Age'].isnull().sum()


# In[ ]:


# Compare Age feature before imputation and after

print('Before imputation')
print(df['Age'].describe())
print(df['Age'].value_counts().sort_index())
print(" ")

print('Age feature for df_rfr')
print(df_rfr['Age'].describe())
print(df_rfr['Age'].value_counts().sort_index())


# Checking if there are any outliers or unknown values

# In[ ]:


df_rfr.describe().T


# In[ ]:


for i in df_rfr.columns:
  print("Unique values for column: " + i)
  print(df_rfr[i].value_counts(dropna=False))
  print(" ")


# In[ ]:


df_rfr['Fare'].value_counts().sort_index()


# The .min() value for Fare feature is 0 and it appears 17 times
# 
# it is questionable whether this is because the passenger did not pay at all 
# 
# or 0 value because the fare paid by the 17 passengers are unknown

# In[ ]:


fare_unknown = df_rfr[df_rfr['Fare'] == 0]
fare_unknown


# Some of the passengers with 'unknown' fare are Pclass 1 (i.e. first class) passengers
# 
# and it is very unlikely that first passengers did not pay or pay little fare (unless they were invited passegners or so) 
# 
# so I assume that 'fare_unknown' passengers, how much fare they paid is unknown, rather than they did not pay any fare at all

# In my earlier notebooks/versions I left the 0 fare values as they were, and replaced 1 NaN value with the mean Fare value 
# 
# but in my previous notebook/version I tested and compared different ways to impute 0 Fare values
# 
# As the result, I impute 0 Fare values by median of Pclass and Embarked features
# 
# I have to get the row which had NaN in the original dataset, so I can also treat it as a row with 0 Fare value and impute

# In[ ]:


# df2 is the copied dataframe of the original dataframe

df2[df2['Fare'].isnull()].index


# Index of the row with missing Fare value is 1043 and Fare replaced by 0 value
# 
# so I can transform all 0 Fare value rows together

# In[ ]:


df_rfr3 = df_rfr.copy()


# In[ ]:


df_rfr3.loc[1043, :]

# Fare value here is the mean value imputed


# In[ ]:


df_rfr3.loc[1043, 'Fare'] = 0


# In[ ]:


len(df_rfr3[df_rfr3['Fare'] == 0])


# Fare is determined by Pclass as well as by Embarked
# 
# so I impute 0 Fare value with the median Fare of Pclass and Embarked
# 
# I have dropped Embarked column earlier in the process so I put it back temporarily to get the median value with Pclass

# In[ ]:


df_rfr3.head()


# In[ ]:


df_rfr3['Embarked'] = df2['Embarked']

df_rfr3.head()


# Check the mean and median of Fare values by Pclass and Embarked

# In[ ]:


data2 = df_rfr3.loc[df_rfr3['Fare'] != 0,:].groupby(['Pclass', 'Embarked']).agg(['mean', 'median', 'count'])['Fare']
print(data2)


# In[ ]:


for i in df_rfr3['Pclass'].unique():
  for location in df_rfr3['Embarked'].unique():

    df_rfr3.loc[(df_rfr3['Fare'] == 0) & (df_rfr3['Pclass'] == i) & (df_rfr3['Embarked'] == location), 'Fare'] = df_rfr3.loc[(df_rfr3['Fare'] == 0) & (df_rfr3['Pclass'] == i) & (df_rfr3['Embarked'] == location), 'Fare'].map(
        lambda x: df_rfr3[(df_rfr3['Fare'] != 0) & (df_rfr3['Pclass'] == i) & (df_rfr3['Embarked'] == location)]['Fare'].median()
    )
    
    # print(i,location) 
    # print(df_rfr3[(df_rfr3['Fare'] != 0) & (df_rfr3['Pclass'] == i) & (df_rfr3['Embarked'] == location)]['Fare'].median())


# In[ ]:


# All 0 Fare values have been imputed

df_rfr3[df_rfr3['Fare'] == 0]


# In[ ]:


print("Before imputation")
print(data2)
print("")
print("After imputation")
print(df_rfr3.groupby(['Pclass', 'Embarked']).agg(['mean', 'median', 'count'])['Fare'])


# In[ ]:


df_rfr3 = df_rfr3.drop(columns='Embarked', axis=1)


# ## 3. Building the model

# Now that full dataset (train and test combined) has been preprocessed 
# 
# I will now separate the full dataset back to train and test datasets then build the model

# Build the model
# 
# I will build a **random forest model** first (as recommended by Kaggle example notebook) 
# 
# and compare the accuracy with other algorithms at later stage

# In[ ]:


# separate the full dataset back to train and test

train = df_rfr3.iloc[:891, :]
test = df_rfr3.iloc[891:,:]


# In[ ]:


# adding back the Survived feature 
# that I separated from train_data earlier before datasets concat
train['Survived'] = Survived_data

train.info()


# In[ ]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# I am budilng the model just like how I did previously but this time with GridSearchCV
# 
# so that it uses cross validation to find the optimal hyperparameters out of different combinations of parameters passed
# 
# 

# In[ ]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
clf = RandomForestClassifier() 

# cross validation of all combinations of the parameters
param_grid = [
  {"n_estimators": [i for i in range(10, 100, 10)], 
   "criterion": ["gini", "entropy"],
   "max_depth": [i for i in range(10, 15, 1)],
   "min_samples_split": [2, 4, 10, 12, 16]},
]

# cv=3 means cross validation with 3 folds
# performance scoring metrics is accuracy
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X, y)


# In[ ]:


# get the combination with the best performing score

final_clf = grid_search.best_estimator_
final_clf


# In[ ]:


# get the mean score of the model with best combination of parameters on the X_test

scores = []

for rs in np.arange(0, 1000, 50):
  score = final_clf.score(X_test, y_test)
  scores.append(score)

print(scores)
print("MEAN SCORE")
print(np.array(scores).mean())      


# The model with optimal combination of parameters is producing good score on average
# 
# and now I will use the model on test data to make submission prediction

# In[ ]:


# check feature importance

final_clf.feature_importances_


# In[ ]:


index_list = X_train.columns
print(index_list)


# In[ ]:


feature_importance = pd.DataFrame(
    { "score": final_clf.feature_importances_},
    index = index_list
)

feature_importance.sort_values('score', ascending=False)


# ## 4. Prediction on test dataset for submission

# In[ ]:


test = df_rfr3.iloc[891:, :]

test.head()


# In[ ]:


test.info()


# In[ ]:


# I will need PassengerId column for submission later on 

id_column = test_data['PassengerId']
id_column


# Now the test data is ready to make prediction

# In[ ]:


submission_prediction = final_clf.predict(test)


# In[ ]:


submission_prediction


# ## 5. Prediction submission

# In[ ]:


final_df = pd.DataFrame(id_column)

final_df['Survived'] = submission_prediction

final_df.head()


# In[ ]:


final_df.info()


# In[ ]:


final_df.to_csv('submission.csv', index=False)

