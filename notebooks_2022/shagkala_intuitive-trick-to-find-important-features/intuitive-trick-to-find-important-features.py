#!/usr/bin/env python
# coding: utf-8

# Hi Kagglers, 
# 
# In this short notebook I will showcase a simple yet intuitive trick to check the relevance of features in the model. I have learned this trick from the top grandmasters on this platform. Using this technique, we can quickly check the significance of each feature and try to increase their significance by applying feature transformations or even drop them if nothing works.
# 
# # Intuition:
# 
# #### Ideally, any relevant feature should add some value to help the model in making the right predictions. It should be at least more significant than a variable containing random noise.
# 
# Let's try to implement this intuition and see if any of the features are performing worse than random noise in our data!

# # Loading Libraries

# In[ ]:


#Importing Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score
from statistics import mean

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

sns.set_palette("muted")


# # Reading the data files

# In[ ]:


#Reading the data files

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
sample = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


#Data Overview

print(f'Shape of train data: {train.shape}')
print(f'Shape of train data: {test.shape}')

train.head()


# # Preprocessing

# In[ ]:


# Filling missing values
test['Embarked'].fillna((train['Embarked'].mode()), inplace=True)
train['Embarked'].fillna((train['Embarked'].mode()), inplace=True)

test['Fare'].fillna((train['Fare'].median()), inplace=True)
train['Fare'].fillna((train['Fare'].median()), inplace=True)

test['Age'].fillna((train['Age'].median()), inplace=True)
train['Age'].fillna((train['Age'].median()), inplace=True)


# In[ ]:


# Dropping unimportant features
train.drop(['Name','Ticket','Cabin', 'PassengerId'], axis=1, inplace=True)
test.drop(['Name','Ticket','Cabin', 'PassengerId'], axis=1, inplace=True)


# In[ ]:


# Encodeing categorical features
obj_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in obj_cols:
    le = LabelEncoder()
    le.fit(train[col])
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


# # Adding a random noise feature

# In[ ]:


np.random.seed(44)

train['Noise'] = np.random.normal(0,1,train.shape[0])
test['Noise'] = np.random.normal(0,1,test.shape[0])


# # Modeling

# In[ ]:


# Storing the target variable separately

X_train = StandardScaler().fit_transform(train.drop('Survived', axis = 1))
X_test = StandardScaler().fit_transform(test)
y_train = train['Survived']

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))


# #### Finding model importance using Logistic Regression and K Fold Cross Validation

# In[ ]:


#Stratified K fold Cross Validation

def train_and_validate(model, N):
    
    regex = '^[^\(]+'
    match = re.findall(regex, str(model))
    print(f'Running {N} Fold CV with {match[0]} Model.')
    
    probs = pd.DataFrame(np.zeros((len(X_test), N * 2)), columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])
    importances = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)], index=train.drop('Survived', axis = 1).columns)
    fprs, tprs, scores = [], [], []

    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print('Fold {}\n'.format(fold))
        
        # Fitting the model
        model.fit(X_train[trn_idx], y_train[trn_idx])

        # Computing Train AUC score
        trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx], model.predict_proba(X_train[trn_idx])[:, 1])
        trn_auc_score = auc(trn_fpr, trn_tpr)
        # Computing Validation AUC score
        val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx], model.predict_proba(X_train[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)  

        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)

        # X_test probabilities
        probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = model.predict_proba(X_test)[:, 0]
        probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = model.predict_proba(X_test)[:, 1]
        importances.iloc[:, fold - 1] = model.coef_[0] 
        
        print(scores[-1])    
    
    trauc = mean([i[0] for i in scores])
    cvauc = mean([i[1] for i in scores])
    print(f'\nAverage Training AUC: {trauc}, Average CV AUC: {cvauc}')
    
    return trauc, cvauc, importances, probs


# In[ ]:


#Testing multiple ML models using stratified K fold CV

df_row = []
N = 3
model = LogisticRegression()
    
trauc, cvauc, importances, probs = train_and_validate(model, N)

regex = '^[^\(]+'
match = re.findall(regex, str(model))

df_row.append([match[0], trauc, cvauc])

df = pd.DataFrame(df_row, columns = ['Model', f'{N} Fold Training AUC', f'{N} Fold CV AUC'])
df


# # Feature Importance

# In[ ]:


#Plotting the feature importance

importances['Mean_Importance'] = abs(importances.mean(axis=1))
importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)

plt.figure(figsize=(8,8))
bar = sns.barplot(x='Mean_Importance', y=importances.index, data=importances, palette=['orange' if x!='Noise' else 'black' for x in importances.index])

plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
plt.title('Logistic Regression Mean Feature Importance Between Folds', size=15)

plt.show()


# #### We can observe that Fare, Embarked, and Parch variables are less important than a random noise feature. Hence, these features can be considered insignificant for the model predictions because their importance is lower than the importance of a random noise feature containing randomly generated numbers.

# # Points to note:

# 1. Please note that it is not a good idea to blindly drop the lesser significant variables.
# 2. Instead, we should preprocess, and transform the insignificant features to make them add value to predictions.
# 3. Internally every machine learning model works differently. Hence each model can give different order of feature importance. So before taking any decision we should first check the importance of a few models.
# 4. The noise feature can lower the accuracy. Hence, do not forget to remove the Noise feature before training the final model for submission.

# ## The End!
# 
# Thank you for reading this notebook. I hope you have learned something new today.
# Kindly share feedback if you find any flaws or have a better approach.
# 
# Please upvote the notebook if you liked this kernel!
# 
# Have a good day!
