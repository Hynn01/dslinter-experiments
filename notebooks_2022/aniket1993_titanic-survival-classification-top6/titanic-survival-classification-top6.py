#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')


# # Exploratory Data Analysis

# In[ ]:


train['Survived'].value_counts()


# Amongst 891 people, 342 people survived and 549 people did not survive the titanic disaster. 

# In[ ]:


train.head()


# In[ ]:


train.info()


# This dataset contains features of the float, integer and object types. Few features have missing values as well. 

# In[ ]:


train.describe().T


# **Next few graphs and tables help us understand deeper connection of the characteristics such as age distribution, economic status etc to the target variable.**

# In[ ]:


sns.set_theme()
fig=plt.figure(figsize=(20,6))
fig.add_subplot(1,2,1)
sns.histplot(x=train['Age'], shrink=0.8, bins=12, hue=train['Survived'], multiple="dodge",palette='brg' )
fig.add_subplot(1,2,2)
sns.histplot(x=train['Age'], shrink=0.8, bins=12, hue=train['Sex'], multiple="dodge", palette='Blues_r')
plt.show()

fig=plt.figure(figsize=(18,6))
sns.histplot(x=train['Age'], shrink=0.8, bins=12, hue=train['Pclass'], multiple="dodge", palette='dark')
plt.show()


# In[ ]:


#Pie-Chart1
fig = plt.figure(figsize=(18,10))

#define data
data = [train['Pclass'].value_counts()[1], train['Pclass'].value_counts()[2], train['Pclass'].value_counts()[3]]
labels = ['Class-1', 'Class-2', 'Class-3']
#define Seaborn color palette to use
colors = sns.color_palette('CMRmap_r')[0:3]
ax1 = plt.subplot2grid((1,3),(0,0))
#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Number of people in different passenger classes')

# Pie Chart 2
data = [train['Embarked'].value_counts()['S'], train['Embarked'].value_counts()['C'], train['Embarked'].value_counts()['Q']]
labels = ['S', 'C', 'Q']
#define Seaborn color palette to use
colors = sns.color_palette('Blues_r')[0:3]
ax2 = plt.subplot2grid((1,3),(0,1))
#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Number of people embarked titanic ship from different locations')

# Pie Chart 3
data = [train['Sex'].value_counts()['female'], train['Sex'].value_counts()['male']]
labels = ['female', 'male']
#define Seaborn color palette to use
colors = sns.color_palette('Greens_r')[0:3]
ax3 = plt.subplot2grid((1,3),(0,2))
#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('percentage of makes and females in the titanic ship')
plt.show()


# In[ ]:


fig=plt.figure(figsize=(24,10))

for i in range(1,4,1):
    fig.add_subplot(1,3,i)
    sns.histplot(data=train[train['Pclass']==i], x='Age', hue='Survived',multiple="dodge", bins=10, palette='gist_rainbow_r')
    plt.title('Age distribution of the Passenger Class {}'.format(i))


# In[ ]:


fig=plt.figure(figsize=(24,10))
i=1
em=['S', 'C', 'Q']
for j in em:
    fig.add_subplot(2,2,i)
    sns.histplot(data=train[train['Embarked']==j], x='Age', hue='Survived',multiple="dodge", bins=10, palette='gist_rainbow_r')
    plt.title('Age distribution of the people Embarked at {}'.format(j))
    i+=1


# In[ ]:


fig=plt.figure(figsize=(24,10))
fig.add_subplot(1,2,1)
sns.histplot(data=train, x='SibSp', hue='Survived',multiple="dodge", bins=10, palette='brg_r', shrink=0.85)
fig.add_subplot(1,2,2)
sns.histplot(data=train, x='Parch', hue='Survived',multiple="dodge", bins=10, palette='brg_r', shrink=0.85)


# If only X=0 and X=1 values are considered in these graphs, an interesting insight is observed. People who had parents / children aboard the Titanic had more chance of being alive than the people who were alone aboard the Titanic. 
# 
# Feature creation during the data-preprocessing phase. The feature 'Alone' is created. 

# In[ ]:


train['Fam_Size']= train['Parch']+train['SibSp']
train['Fam_Size']=train['Fam_Size'].apply(lambda x: 1 if x>0 else 0)


# In[ ]:


plt.figure(figsize=(10,5))
sns.histplot(data=train, x='Fam_Size', hue='Survived',multiple="dodge", bins=10, palette='brg_r', shrink=0.85)


# This graph provides clear idea about survival probability between people who had boarded the Titanic alone and people who had boarded the Titanic with a family member. 

# # Missing data Analysis

# In[ ]:


pd.DataFrame((train.isna().sum()/train.shape[0])*100).T


# In[ ]:


pd.DataFrame(train.isna().sum()).T


# **Following statements provide insights from the two tables obtained in the previous two cells.**
# 
# **Age**: Above table conveys information about the missing values in every column. There are three tables with the missing values. In order to address the missing values issue, we will delve deep into different types graphs displaying the relation of different characteristics with the Age column and target variable. This will give us a better idea about different imputation techniques one can use to solve the missing values issue.
# 
# **Cabin**: The percentage of missing values is enormously high. Hence, the column is not taken into consideration in this analysis. 
# 
# **Embarked**: Only two values are missing in this column. These values are replaced by analysing the relationship of other characteristics with this column(**Embarked**). The reasoning is communicated below. 

# **Now let's look into the missing observations for the 'Age' column.**

# In[ ]:


columns=['Survived', 'Pclass', 'Sex', 'Embarked']
for i in columns:
    print(pd.DataFrame(train[train['Age'].isnull()][i].value_counts()))


# ***Characteristics of the missing values' observations are:*** 
#     
#    1. 52 people ***survived*** vs 125 people ***did not survive***.
#     
#    2. 30 people belonged to ***Passenger Class 1***, 11 people belonged to ***Passenger Class 2*** and 136 people belonged to ***Passenger Class 3***.
#     
#    3. 124 ***males*** and 53 ***females***.
#    
#    4. 90 people embarked at location ***S***, 49 people embarked at location ***Q*** and 38 people embarked at location ***C***.

# In[ ]:


fig=plt.figure(figsize=(24,10))

for i in range(1,4,1):
    fig.add_subplot(1,3,i)
    sns.histplot(data=train[train['Pclass']==i], x='Age', hue='Survived',multiple="dodge", bins=10, palette='gist_rainbow_r')
    plt.title('Age distribution of the Passenger Class {}'.format(i))


# Previous table certain highlights different percentages of survival probabilities for different passenger classes. The survival
# probabilities were highest in class 1 and was lowest in class 3. 

# These missing values in **Age column** can addressed by grouping the different observation based on the columns **Embarked**,
# **Sex** and **Passenger Class**. Let's find the mean age of observations for groups formed using unique values of these three columns. Following code is used for the required imputation.  

# In[ ]:


train['Age'] = train['Age'].fillna(train.groupby(['Sex', 'Embarked', 'Pclass'])['Age'].transform('mean'))


# Let's confirm there are no remaining missing values for the age column anymore.

# In[ ]:


train['Age'].isna().sum()


# In[ ]:


train['Age']=train['Age'].apply(lambda x: round(x,2))


# **Let's move our attention to the missing values remaining the Embarked column.**

# In[ ]:


columns=['Survived', 'Pclass', 'Sex', 'Age', 'Fare']
for i in columns:
    print(pd.DataFrame(train[train['Embarked'].isnull()][i].value_counts()))


# ***Both persons with missing values in 'Embarked' column were females, belonged to passenger class 1 and survived in titanic 
# disaster. Both paid a fare of 80.***

# Let's understand more about Embarked locations from the already present observations.

# In[ ]:


fig=plt.figure(figsize=(24,12))
cols=['S', 'C', 'Q']
j=1
for i in cols:
    fig.add_subplot(1,3,j)
    sns.histplot(data=train[(train['Pclass']==1) & (train['Sex']=='female') & (train['Embarked']==i)], x='Age', hue='Survived',multiple="dodge", bins=10, palette='brg_r',shrink=.8)
    plt.title('Age distribution of the people Embarked at location {}'.format(i))
    j+=1


# **First,** this graph indicates ***age distribution for females in passenger class 1*** under different embarked locations for the Titanic ship.
# 
# Nearly all females (except two), above the age of 18, survived in the passenger class 1 for embarked location S and C.
# 
# Using only this graph, it is not possible to assign missing values. Lets dig deeper using Fare constrain. 

# In[ ]:


train[(train['Pclass']==1) & (train['Sex']=='female') & (train['Survived']==1) & (train['Fare']<90) & (train['Fare']>70)]['Embarked'].value_counts()


# It is already known that both females bought ticket for the fare of 80 units. The result in previous cell indicates almost 
# equal number of females under both embarked locations S and C. 
# 
# Hence, still unable to deduce anything about the missing values. 

# In[ ]:


train[(train['Pclass']==1) & (train['Survived']==1)]['Embarked'].value_counts()


# In[ ]:


fig=plt.figure(figsize=(24,12))
cols=['S', 'C', 'Q']
j=1
for i in cols:
    fig.add_subplot(1,3,j)
    sns.histplot(data=train[(train['Pclass']==1) & (train['Embarked']==i)], x='Age', hue='Survived',multiple="dodge", bins=10, palette='Reds_r',shrink=.8)
    plt.title('Age distribution of the people Embarked at location {}'.format(i))
    j+=1


# From the results obtained in the last code cell, it is observed that more people did not survive at embarked location S for
# the age category around 60. Hence, let's assume this observation to marked as 'C'.
# 
# In addition, if the ratio of number of people survived to number of people did not survive for the same age category 
# (approximation used from the result graph obtained in the previous cell), it can deduced that more percentage of people survived who embarked at location C.
# 
# Hence, both the observations are marked as 'C' for the missing values in column 'Embarked'.

# In[ ]:


train['Embarked']=train['Embarked'].fillna('C')


# Since the column cabin has more than 60% missing values, it is decided to drop this column from the analysis.

# In[ ]:


train=train.drop(['Cabin'], axis=1)


# # Data Pre-Processing

# **Let's save the target variable as Y before preprocessing the data for indepedent features.**

# In[ ]:


Y=train['Survived'].values


# In[ ]:


train


# **PassengerId**: This column is used to note the row numbers. Hence this column is ***Dropped***
# 
# **Name**:This column only indicates name of the passenger. Hence this column is ***Dropped*** 
# 
# **Ticket**: Similar reasoning as above. Hence this column is ***Dropped***

# In[ ]:


train=train.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)


# **Re-arranging the columns**

# In[ ]:


train=train[['Age', 'Fare', 'Parch', 'SibSp', 'Sex', 'Pclass','Fam_Size', 'Embarked']]


# **Sex**: Categorical variable where 1 is male and 0 is female.
# 
# **Embarked**: This is also a categorical variable without ranking. Hence, it is encoded using pandas get_dummies().
# 
# **('drop_first' is set true to avoid collinearity amongst the introduced columns.)**

# In[ ]:


train['Sex']=train['Sex'].apply(lambda x: 1 if x=='male' else 0)


# In[ ]:


train=pd.get_dummies(train, drop_first=True)


# **Let's check skewness as a check for normality.**

# In[ ]:


train.skew().T


# **I applied box-cox transformation for the columns 'Fare', 'Parch' and 'SibSp'.**

# In[ ]:


skewed_columns=['Fare', 'Parch', 'SibSp']
from scipy.special import boxcox1p
lam=0.15
for i in skewed_columns:
    train[i]= boxcox1p(train[i],lam)


# In[ ]:


X=train.values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train[:,0:2]= sc.fit_transform(X_train[:, 0:2])
X_test[:,0:2]= sc.transform(X_test[:, 0:2])


# # Machine Learning Model

# In[ ]:


# Now we will implement model pipeline guidelines and replicate above models and learn about best model
# amongst these classifiers

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

model_pipeline=[]
model_pipeline.append(DecisionTreeClassifier(random_state=41))
model_pipeline.append(LogisticRegression(solver='saga', penalty='l1',random_state=42))
model_pipeline.append(KNeighborsClassifier())
model_pipeline.append(RandomForestClassifier(random_state=44))
model_pipeline.append(SVC(random_state=45))
model_pipeline.append(xgb.XGBClassifier())


# In[ ]:


model_list=['Decision Tree', 'Logistic Regression', 'K-Nearest Neighbors', 'Random_Forest_Classification', 'SVM', 'XG']
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn import metrics

acc=[]
cm=[]

for classifier in model_pipeline:
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    acc.append(round(accuracy_score(Y_test, Y_pred),2))
    cm.append(confusion_matrix(Y_test, Y_pred))


# In[ ]:


# Let us plot confusion matrix for all the model and compare.

fig=plt.figure(figsize=(20,10))

for i in range(0,len(cm)):
    cm_con=cm[i]
    model=model_list[i]
    sub_fig_title=fig.add_subplot(2,3,i+1).set_title(model)
    plot_map=sns.heatmap(cm_con,annot=True,cmap='Greens_r',fmt='g')
    plot_map.set_xlabel('Predicted_Values')
    plot_map.set_ylabel('Actual_Values')


# In[ ]:


result=pd.DataFrame({'Model': model_list, 'Accuracy': acc})
result


# # Grid-Search CV for different classification algorithms

# Since, all the accuracies are in the similar range, it becomes important to apply grid search cross-validation for every algorithm to hypertune all the parameters and understand standard deviation in accuracies using K-folds cross validation technique.

# ## Logistic Regression

# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'solver': ['sag','saga','lbfgs', 'newton-cg', 'liblinear'],
              'penalty': ['l1', 'l2', 'elasticnet' 'none'],
               
                }]
grid_search = GridSearchCV(estimator = LogisticRegression(),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LogisticRegression(random_state = 0, solver='sag', penalty='l2'), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ## Decision Tree Classifier

# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'criterion': ['gini', 'entropy'],
               'splitter': ['best', 'random'],
               'max_depth': [*range(1,10,1)],
               'min_samples_split': [2,5,6,7,8,9,10],
               'max_features': [1,2,3,4,5,7,9,11,13,15,'auto', 'sqrt', 'log2']
               
                }]
grid_search = GridSearchCV(estimator = DecisionTreeClassifier(random_state=1),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=3, min_samples_split=5, splitter='best'), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ## Random Forest Classification

# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [*range(100,1000,100)],
               'criterion': ['gini', 'entropy'],
               }]
grid_search = GridSearchCV(estimator = RandomForestClassifier(random_state=2),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = RandomForestClassifier(n_estimators=800, criterion='entropy'), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ## SVM Classifier

# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'gamma': [0.5,0.6,0.62,0.7,0.72, 0.8,0.85,0.9],
               'C': [1,2,3,4,5,6,7]
              }]
grid_search = GridSearchCV(estimator = SVC(random_state=4, kernel='rbf'),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = SVC(kernel='rbf', gamma=0.5, C=1, random_state=4), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ## XG-Boost

# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate': [0.01,0.02,0.03,0.04,0.05, 0.06,0.07],
               'gamma': [0.67,0.65,0.70, 0.6, 0.75]
              }]
grid_search = GridSearchCV(estimator = xgb.XGBClassifier(random_state=10, booster='gbtree'),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb.XGBClassifier(random_state=10, booster='gbtree', gamma=0.6, learning_rate=0.03), X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# ## Test Dataset

# In[ ]:


test.isna().sum()


# In[ ]:


Id=test['PassengerId']


# In[ ]:


test=test.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1)


# In[ ]:


test['Fam_Size']= test['Parch']+test['SibSp']
test['Fam_Size']=test['Fam_Size'].apply(lambda x: 1 if x>0 else 0)


# In[ ]:


test=test[['Age', 'Fare', 'Parch', 'SibSp', 'Sex', 'Pclass','Fam_Size', 'Embarked']]


# In[ ]:


test['Age'] = test['Age'].fillna(test.groupby(['Sex', 'Embarked', 'Pclass'])['Age'].transform('mean'))


# In[ ]:


test['Fare'] = test['Fare'].fillna(test['Fare'].median())


# In[ ]:


test['Sex']=test['Sex'].apply(lambda x: 1 if x=='male' else 0)


# In[ ]:


skewed_columns=['Fare', 'Parch', 'SibSp']
from scipy.special import boxcox1p
lam=0.15
for i in skewed_columns:
    test[i]= boxcox1p(test[i],lam)


# In[ ]:


test=pd.get_dummies(test, drop_first=True)


# In[ ]:


test=test.values


# In[ ]:


test[:,0:2]= sc.transform(test[:, 0:2])


# In[ ]:


classifier=xgb.XGBClassifier(random_state=10, booster='gbtree', gamma=0.67, learning_rate=0.03)
classifier.fit(X_train, Y_train)
predictions=pd.DataFrame(classifier.predict(test))


# In[ ]:


result=pd.concat([Id, predictions], axis=1)
result.columns=['PassengerId', 'Survived']


# In[ ]:


result


# In[ ]:


result.to_csv('prediction.csv', index=False)

