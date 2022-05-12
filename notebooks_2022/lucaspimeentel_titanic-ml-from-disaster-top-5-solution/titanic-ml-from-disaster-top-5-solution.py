#!/usr/bin/env python
# coding: utf-8

# In this notebook, I am going to show you the model that landed me on the top 5% of the leaderboard of this competition. 
# 
# I hope that by making this notebook public those who just got into this famous competition can build upon my findings and create even better models.
# 
# Now, the key takeaway I got from my participation is that you should be extremely familiar with the dataset. I simply can't stress this enough, exploratory data analysis is crucial, and most of the insights I needed to build the model and create unique features I got in the EDA phase.
# 
# However, I won't dive deep in EDA in this notebook because I already did it [here](https://www.kaggle.com/code/lucaspimeentel/titanic-dataset-exploratory-data-analysis), so please check it out if you haven't already and feel free to use it as a starting point.
# 
# **In short, here are the steps I took to tackle this competition:**
# <li>Exploratory Data Analysis (already covered in the other notebook)</li>
# <li>Feature Engineering</li>
# <li>Imput Missing Values</li>
# <li>Data Preprocessing</li>
# <li>Hyperparameter Tuning</li>
# <li>Predict & Create Submission File</li>
# 
# And without further ado, let's start:

# <h2> Importing Libraries & Loading the Data </h2>

# In[ ]:


# Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np

# Loading the data the datasets

x_train = pd.read_csv('../input/titanic/train.csv')
y_train = x_train.Survived
x_pred = pd.read_csv('../input/titanic/test.csv')


x_train.set_index('PassengerId',inplace=True)
x_pred.set_index('PassengerId',inplace=True)


# <h2> Feature Engineering</h2>

# Now, this dataset only has 11 columns, and not all of them are useful as they have been stored on the dataset, and so based on my fidings of the previous exploratory data analysis I did, we will engineer some features to help us improve our model.
# 

# <h3> Make a 'Title' Feature From the 'Name' column </h3>

# First, let's extract the title pronouns from the names of each one of the passengers.
# 
# After all, our previuous EDA already showed that this feature uncovers lots of information about survival chance, and it would be foolish of us to not to use it.

# In[ ]:


x_train["Title"] = x_train['Name'].apply(lambda x: x[x.find(',') : x.find('.')][1:].strip())
x_pred["Title"] = x_pred['Name'].apply(lambda x: x[x.find(',') : x.find('.')][1:].strip())


# <h3> Make a Role Feature From Title </h3>

# However, we cannot forget that title pronouns are closely related to social class and sex.
# 
# So let's group the title feature based on class and sex and create a feature called Role out of it. (got this idea from this notebook from @??'s notebook)
# 
# Here's my thought process behind the classification below:
# 
# Crew: These are the ??? that ??? that a given person belongs to the crew of the Titanic.
# 
# VIP: These are the ??? that ??? that a given person is of high importance.
# 
# Average Joe: Generic male ??? pronouns
# 
# Average Jane: Generic female ?? pronouns

# In[ ]:


def assign_role(row):
    if row['Title'] in ['Capt', 'Col', 'Major']:
        return 'Crew'
    elif row['Title'] in ['Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Master', 'Rev', 'Sir', 'the Countess']:
        return 'VIP'
    elif row['Title'] in ['Mr']:
        return 'Average Joe'
    elif row['Title'] in ['Miss','Mme','Mlle','Mrs', 'Ms']:
        return 'Average Jane'
    
x_train['Role'] = x_train.apply(lambda row: assign_role(row), axis=1)
x_pred['Role'] = x_pred.apply(lambda row: assign_role(row), axis=1)


# <h3> Create a 'Cabin Category' Feature From the Cabin Column </h3>

# The Cabin column that our dataset provided us with is also another feature that is not useful on its own, so let's get the only the category from which each cabin belongs to.

# In[ ]:


x_train['Cabin_Category'] = x_train['Cabin'].str[0]
x_pred['Cabin_Category'] = x_pred['Cabin'].str[0]


# <h3> Create a Family Size Feature from SibSp and Parch </h3>

# Our EDA also uncovered that people who went to the titanic alnone or in larger families had lower chances of survival, so let's create a family size feature to use that information in our model.
# 
# Now, our family size feature will be composed of 3 categories...
# 
# **Alone:** For those who went to the cruise alone (that is, SibSp + Parch = 0)
# 
# **Small family:** Those who went to the Titanic with a small family (that is, SibSp + Parch between 1 and 3)
# 
# **Big Family:** People with SibSp + Parch = 4 or above

# In[ ]:


x_train['FamilySize'] = x_train['Parch']+x_train['SibSp']
x_train['FamilySize'].replace(to_replace={0:'Alone',1:'Small Family',2:'Small Family',3:'Small Family',4:'Big Family',5:'Big Family',6:'Big Family',7:'Big Family',8:'Big Family',9:'Big Family',10:'Big Family'},inplace=True)


x_pred['FamilySize'] = x_pred['Parch']+x_pred['SibSp']
x_pred['FamilySize'].replace(to_replace={0:'Alone',1:'Small Family',2:'Small Family',3:'Small Family',4:'Big Family',5:'Big Family',6:'Big Family',7:'Big Family',8:'Big Family',9:'Big Family',10:'Big Family'},inplace=True)


# <h3> Create a GroupSize Feature </h3>

# Our EDA also uncovered that people who went in groups (as measured by the number of people with the same ticket) also had higher survival rates, so let's create a feature called GroupSize in order to use that information to build our model.
# 
# Have in mind that the GroupSize feature is a little broader than the familySize feature.
# 
# After all, we are assign people to the same groups based on their ticket IDs (which probably indicates that they bought their tickets together but don't necessarily come from the same family)

# In[ ]:


x_train = x_train.merge(pd.DataFrame(x_train['Ticket'].value_counts()).reset_index().rename({'Ticket':'Ppl_in_group','index':'Ticket'},axis=1),how='left',left_on='Ticket',right_on='Ticket')
x_pred = x_pred.merge(pd.DataFrame(x_pred['Ticket'].value_counts()).reset_index().rename({'Ticket':'Ppl_in_group','index':'Ticket'},axis=1),how='left',left_on='Ticket',right_on='Ticket')


# <h2> Imputting Missing Values </h2>

# <h3> Imput Cabin Category NaN Values With '?'</h3>

# The Cabin Category column has lot of missing values, but since they hold important about survival rates, we won't drop it. Instead, what we will do is assign the value **'?'** to all the rows with NaN values.

# In[ ]:


x_train['Cabin_Category'].fillna('?',inplace=True)
x_pred['Cabin_Category'].fillna('?',inplace=True)


# <h3>Imput Embark NaN Values With the Mode

# Since we only have 2 NaN values in the embarked column, we will assign them with the most frequent value seen in this distribution.

# In[ ]:


mode_imputer = SimpleImputer(strategy='most_frequent')
x_train['Embarked'] = mode_imputer.fit_transform(np.array(x_train['Embarked']).reshape(-1,1))


# <h2>Data Preprocessing</h2>

# Now, let's assign some categories to our columns....
# 
# **drop_columns:** these columns will be dropped from the database since they cannot be used to generate prediction in their current form, or - in the case of the title column, they have extremely high correlation with other features.
# 
# **categorical_coluns:** These columns have categorical values and will be one hot encoded shortly.
# 
# **ordinal columns:** these columns have a categorical nature that is hierarchical in nature.
# 
# **numerical_columns:** these columns have a numerical and continuous nature. All of them will be standardized and some of them - such as age and fare - will be categorized in order to help our prediction model.

# In[ ]:


# Categorizing all columns of the dataset based on the criteria above

drop_columns = ['Survived','Name','Ticket','Cabin','Title']
categorical_columns = ['Sex','Embarked','Cabin_Category','FamilySize','Role']
ordinal_columns = ['Pclass']
numerical_columns = ['Age','Fare','Ppl_in_group','SibSp','Parch']

# Standardizing numerical columns

from sklearn.preprocessing import StandardScaler

s_scaler = StandardScaler()
scaled_values = s_scaler.fit_transform(x_train[numerical_columns])
x_train[numerical_columns] = scaled_values

# Dropping Columns

x_train = x_train.drop(axis=1,labels=drop_columns,errors='ignore')
x_pred = x_pred.drop(axis=1,labels=drop_columns,errors='ignore')


# One Hot Encoding columns

x_train = pd.get_dummies(data=x_train,columns=categorical_columns,drop_first=True)
x_pred = pd.get_dummies(data=x_pred,columns=categorical_columns,drop_first=True)

# Adding columns that are not present in both ones in order for them to have the same shape

columns_to_add_x_pred = list(set(x_train.columns)-set(x_pred.columns))
columns_to_add_x_train = list(set(x_pred.columns)-set(x_train.columns))
x_train[columns_to_add_x_train] = 0
x_pred[columns_to_add_x_pred] = 0

# Encoding ordinal columns

ordinal_encoder = OrdinalEncoder()

encoded_ordinals = ordinal_encoder.fit_transform(x_train[ordinal_columns])
x_train[ordinal_columns] = encoded_ordinals

encoded_ordinals = ordinal_encoder.fit_transform(x_pred[ordinal_columns])
x_pred[ordinal_columns] = encoded_ordinals


# <h3> Imput 'Age' and 'Fare' NaN Values With K-Nearest Neighbors and Create Group Categories For Both Of Them </h3>

# We still have some missing values in a few of the columns, so let's imput values to them.
# 
# I waited this long to input values in these columns because I am going to use the K-Nearest Neighbors imputer. Which will give us way better imputs in this case instead of gross generalizations like mean or mode.

# In[ ]:


from sklearn.impute import KNNImputer

knn = KNNImputer()
x_train[['Age','Fare','FamilySize_Big Family','FamilySize_Small Family','Sex_male']] = knn.fit_transform(x_train[['Age','Fare','FamilySize_Big Family','FamilySize_Small Family','Sex_male']])

x_pred[['Age','Fare','FamilySize_Big Family','FamilySize_Small Family','Sex_male']] = knn.fit_transform(x_pred[['Age','Fare','FamilySize_Big Family','FamilySize_Small Family','Sex_male']])


# And now that we don't have missing values on the Age and Fare columns anymore, let's also categorize them according to the information that our EDA has uncovered.
# 
# For grouping both of these columns, we will use a technique called binning (which you can learn more about here)

# In[ ]:


kbins = KBinsDiscretizer(n_bins=5,strategy='quantile',encode='ordinal')
x_train['AgeGroup'] = kbins.fit_transform(np.array(x_train['Age']).reshape(-1,1))
x_pred['AgeGroup'] = kbins.fit_transform(np.array(x_pred['Age']).reshape(-1,1))

kbins = KBinsDiscretizer(n_bins=4,strategy='kmeans',encode='ordinal')
x_train['FareGroup'] = kbins.fit_transform(np.array(x_train['Fare']).reshape(-1,1))
x_pred['FareGroup'] = kbins.fit_transform(np.array(x_pred['Fare']).reshape(-1,1))


# <h2> Hyperparameter Tuning With Gridsearch </h2>

# Now that our model is complete and ready to predict, let's find out what are the best hyperparameters for this model.
# 
# I commented out the gridsearch because it took too long, but you can see the best parameters on the next block of code

# In[ ]:


#from catboost import CatBoostClassifier  

#model = CatBoostClassifier(verbose=False)

#parameters = {'border_count':[1,2,3],'depth':[3,5,7,8,9,10],'learning_rate':[1.0,1.4,1.5,1.6, 1.7],'iterations':[2500,5000,7500,1000]}

#from sklearn.model_selection import GridSearchCV

#grid_search = GridSearchCV(model,parameters)
#grid_search.fit(x_train,y_train)


# <h2>Predict & Create CSV Submission File</h2>

# And now that we have found the best hyper parameters based on our grid search, all we have to do is run our prediction model and submit our findings.
# 
# Thanks for reading this far into my notebook. Feel free to fork it and expand upon it to build your own models.
# 
# Here are a few notebooks that I took inspiration from, please check them out:
# 
# [Titanic Dataset Exploratory Data Analysis](https://www.kaggle.com/code/lucaspimeentel/titanic-dataset-exploratory-data-analysis).
# 
# [Titanic Survival Prediction with Random Forest](https://www.kaggle.com/code/naveenkonam1985/titanic-survival-prediction-with-random-forest#Problem-Definition)
# 
# [EDA & Machine Learning (Top 3%)](https://www.kaggle.com/code/enisteper1/titanic-eda-machine-learning-top-3/notebook?scriptVersionId=89272225)

# In[ ]:


from catboost import CatBoostClassifier  

model = CatBoostClassifier(verbose=False,iterations=5000,learning_rate=0.158,depth=8,border_count=2,l2_leaf_reg=0.0038)
model.fit(x_train,y_train)
y_pred = model.predict(x_pred)

