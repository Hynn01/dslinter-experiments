#!/usr/bin/env python
# coding: utf-8

# # 2020 annual CDC survey data of 400k adults related to their health status

# ## What topic does the dataset cover?

# According to the CDC, heart disease is one of the leading causes of death for people of most races in the US (African Americans, American Indians and Alaska Natives, and white people). About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicator include diabetic status, obesity (high BMI), not getting enough physical activity or drinking too much alcohol. Detecting and preventing the factors that have the greatest impact on heart disease is very important in healthcare. Computational developments, in turn, allow the application of machine learning methods to detect "patterns" from the data that can predict a patient's condition.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Importing the Libraries

# In[ ]:


# Libraries for Data Preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Libraries for Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Libraries for measuring accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing the Dataset

# In[ ]:


# read_csv function is used to read a csv file. It takes the filepath as argument
df = pd.read_csv('../input/personal-key-indicators-of-heart-disease/heart_2020_cleaned.csv')
# this prints the first 5 rows of the dataset by default
df.head()


# ## Changing Grid Style to Dark

# In[ ]:


# this changes the style of the plots in seaborn. Changing the grid to dark
sns.set_style("darkgrid", {"grid.color": ".6"})


# ## Counting Variation of all Categorical Variables

# In[ ]:


# This list of lists contains all the columns which have binary categorical values
colRange = [['Smoking','AlcoholDrinking','Stroke'],['DiffWalking','Sex','PhysicalActivity'],['Asthma','KidneyDisease','SkinCancer']]
# This function prints the countplots counting the number of people in each category
def printCount(cols):
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    row=0
    col=0
    p_count=1
    for row in range(3):
        for col in range(3):
            # reads column name from the list
            column = colRange[row][col]
            # plots the counts of the particular column
            sns.countplot(ax=axes[row,col],x=df[column],hue=df['HeartDisease'])
            # sets the title of the corresponding plot along with plot number
            axes[row,col].set_title("Counts of {} (Plot {})".format(column,p_count))
            p_count += 1
# Calling the function
printCount(colRange)


# #### Observation
# 1. Plot 1 - According to the plot people who smoke have a higher chance of Heart Diseases than people who don't
# 2. Plot 2 - According to the plot people who do not drink alcohol have a lower of Heart Disease 
# 3. Plot 3 - According to the plot people having Heart Disease have a lower chance of having a Stroke
# 4. Plot 4 - According to the plot people who do not have any difficulty in walking have a lower chance of Heart Disease 
# 5. Plot 5 - According to the plot Males have a higher chance of Heart Disease than Females
# 6. Plot 6 - According to the plot People who participate in Physical Activity cause Heart Diseases
# 7. Plot 7 - According to the plot People who have asthma have a lower chance of Heart Disease

# ## Checking Heart Disease among Different Races

# In[ ]:


# This statement enlarges the image
plt.figure(figsize=(12,6))
# countplot plots the counts of each type of value in a particular column
# hue property is used to color code the counts according to a second categorical variable.
# plotting the graph
sns.countplot(df['Race'],hue=df['HeartDisease'])
plt.title('Variation of Heart Disease amoung Races')


# #### Observation
# White races have a higher chance of heart disease

# ## Checking Spread of Heart Disease among Diabetic People

# In[ ]:


# This statement enlarges the image
plt.figure(figsize=(12,6))
# plotting the graph
sns.countplot(df['Diabetic'],hue=df['HeartDisease'])
plt.title('Variation of Heart Disease among Diabetic People')


# #### Observation
# People with no Diabetes have a higher chance of Heart Disease

# ## Plotting Variation of BMI with Heart Disease 

# In[ ]:


# This statement enlarges the image
plt.figure(figsize=(12,6))
# histplot plots the distribution of values in the particular column
# kde plots 'kernel distribution estimate' of that particular column
# plotting both graph on top of each other
sns.histplot(data=df[df['HeartDisease']=='Yes'],x='BMI',kde=True,color='red')
sns.histplot(data=df[df['HeartDisease']=='No'],x='BMI',kde=True,color='blue')
plt.title('Distribution of BMI Among People')


# #### Observation
# People with heart disease have a higher BMI than people who don't have heart disease

# ## Checking variation of Physical Health among people with and without Heart Disease

# In[ ]:


# This statement enlarges the image
plt.figure(figsize=(12,6))
# kdeplots show an estimated, smooth distribution of a single numerical variable
# plotting both graphs on top of each other
sns.kdeplot(df[df['HeartDisease']=='Yes']['PhysicalHealth'],shade=True,color='red')
sns.kdeplot(df[df['HeartDisease']=='No']['PhysicalHealth'],shade=True,color='blue')
plt.title('Physical Health Pattern')


# #### Observation
# People with Heart Disease have a lower Physical Health 

# ## Checking Variation of Mental health among people with and without Heart Diseases

# In[ ]:


# This statement enlarges the image
plt.figure(figsize=(12,6))
# plotting both graphs on top of each other
sns.kdeplot(df[df['HeartDisease']=='Yes']['MentalHealth'],shade=True,color='red')
sns.kdeplot(df[df['HeartDisease']=='No']['MentalHealth'],shade=True,color='blue')
plt.title('Variation of Mental Health')


# #### Observation
# People with Heart Disease have a lower Mental Health 

# ## Drawing the correlation graph

# In[ ]:


# importing matplotlib library
import matplotlib
# This statement reduces the size of image
plt.figure(figsize=(9,6))
# plotting the graph
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# ## Checking Columns of the Dataset

# In[ ]:


# the info() function gives information about all the columns in the dataset
df.info()


# ## Checking unique Values in Categorical Columns

# In[ ]:


# the unique() function prints the unique values in a particular column. It is used to check the values in a categorical column.
# printing all the unique values of each column 
print(df['Smoking'].unique())
print(df['AlcoholDrinking'].unique())
print(df['Stroke'].unique())
print(df['DiffWalking'].unique())
print(df['Sex'].unique())
print(df['AgeCategory'].unique())
print(df['Race'].unique())
print(df['Diabetic'].unique())
print(df['PhysicalActivity'].unique())
print(df['GenHealth'].unique())
print(df['Asthma'].unique())
print(df['KidneyDisease'].unique())
print(df['SkinCancer'].unique())


# ## Encoding all Binary Categorical Columns

# In[ ]:


# this is a user defined function that encodes the categorical columns
def change(col):
    if col=='Yes':
        return 1
    elif col=='No':
        return 0
# the apply() function puts value of each row one by one in a column to encode that whole column
# encoding all categorical columns having binary values
df['Smoking'] = df['Smoking'].apply(change)
df['AlcoholDrinking'] = df['AlcoholDrinking'].apply(change)
df['Stroke'] = df['Stroke'].apply(change)
df['DiffWalking'] = df['DiffWalking'].apply(change)
df['PhysicalActivity'] = df['PhysicalActivity'].apply(change)
df['Asthma'] = df['Asthma'].apply(change)
df['KidneyDisease'] = df['KidneyDisease'].apply(change)
df['SkinCancer'] = df['SkinCancer'].apply(change)
df['HeartDisease'] = df['HeartDisease'].apply(change)


# ## Checking State of dataset

# In[ ]:


df.head()


# ## Label Encoding all categorical columns with more than 2 values

# In[ ]:


# making an instance of the label encoder class
le = LabelEncoder()
# label encoding all the categorical columns that have more than 2 unique values
df['Sex']=le.fit_transform(df['Sex'])
df['AgeCategory']=le.fit_transform(df['AgeCategory'])
df['Race']=le.fit_transform(df['Race'])
df['Diabetic']=le.fit_transform(df['Diabetic'])
df['GenHealth']=le.fit_transform(df['GenHealth'])


# ## Checking state of Dataset

# In[ ]:


df.head()


# ## Dividing Dataset into Training and Test Set

# In[ ]:


# iloc[] function is used to select values from the dataset
# independent variables
X = df.iloc[:,1:].values
# dependent variable
y = df.iloc[:,0].values


# ## Splitting dataset into Training and Test Set

# In[ ]:


# train_test_split() is used to divide dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## Feature Scaling

# In[ ]:


# declaring an object of standardscaler class
sc = StandardScaler()
# fit_transform() method first trains the Scaler on dataset and then transforms it between 0 and 1
X_train = sc.fit_transform(X_train)
# transform() method only transforms the dataset based on what it has learnt on the dataset before
X_test = sc.transform(X_test)


# ## Training Models

# ### 1. Logistic Regression

# In[ ]:


# declaring a object of Logistic regression class
clf1 = LogisticRegression()
# fit() function trains the model
# fitting the object with the training data
clf1.fit(X_train, y_train)
# predict() function predicts results from validation data
# predicting result using the trained data
y_pred1 = clf1.predict(X_test)
# confusion_matrix() gives the true_positives, false positives, true negatives, false negatives
# making confusion matrix using predicted and given results in validation data
cm1=confusion_matrix(y_test,y_pred1)
# printing the confusion matrix
print(cm1)
# accuracy_score() is used to find the accuracy of the model
print(accuracy_score(y_test,y_pred1))


# ### 2. K Nearest Neighbors

# In[ ]:


# declaring a object of K Neighbors Classifier class
clf2 = KNeighborsClassifier()
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)
print(accuracy_score(y_test,y_pred2))


# ### 3. Naive Bayes

# In[ ]:


# declaring a object of GaussianNB class
clf3 = GaussianNB()
clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
cm3 = confusion_matrix(y_test,y_pred3)
print(cm3)
print(accuracy_score(y_test,y_pred3))


# ### 4. Support Vector Machine ( Not Recommended due to very long execution time {more than 4 hours} ) 

# In[ ]:


# clf4 = SVC()
# clf4.fit(X_train,y_train)
# y_pred4 = clf4.predict(X_test)
# cm4 = confusion_matrix(y_test,y_pred4)
# print(cm4)
# print(accuracy_score(y_test,y_pred4))


# In[ ]:


# rand_grid = {
#     'kernel':['linear','poly','rbf','sigmoid'],
#     'C':[int(x) for x in np.linspace(start = 2, stop = 10,num = 10)],
#     'gamma':[int(x) for x in np.linspace(start = 0.1, stop = 1, num = 5)]
# }


# In[ ]:


# rcv = RandomizedSearchCV(estimator=clf4,param_distributions=rand_grid,n_iter=100,cv=3,verbose=2,random_state=0,n_jobs=-1)
# rcv.fit(X_train, y_train)


# In[ ]:


# print(rcv.best_params_)
# print(rcv.best_estimator_)


# ### 5. Decision Tree

# In[ ]:


# declaring a object of Decision Tree Classifier class
clf5 = DecisionTreeClassifier(criterion='entropy')
clf5.fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)
cm5 = confusion_matrix(y_test,y_pred5)
print(cm5)
print(accuracy_score(y_test,y_pred5))


# ### 6. Random Forest

# In[ ]:


# declaring a object of Random Forest Classifier class
clf6 = RandomForestClassifier(criterion='entropy',n_estimators=50)
clf6.fit(X_train, y_train)
y_pred6 = clf6.predict(X_test)
cm6 = confusion_matrix(y_test,y_pred6)
print(cm6)
print(accuracy_score(y_test,y_pred6))


# ### Applying Hyperparameter Tuning on Random Forest

# ### RandomizedSearchCV

# ###     a) Making the Parameter Grid

# In[ ]:


# making the set of parameters to test the RandomizedSearchCV 
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=2,stop=100,num=10)],
    'max_features': ['auto','sqrt','log2'],
    'max_depth': [int(x) for x in np.linspace(10,1000,10)],
    'min_samples_split': [2,5,7,10,12,14],
    'min_samples_leaf': [1,2,4,6,8],
    'criterion': ['entropy','gini']
}
print(param_grid)


# ### b) Running RandomizedSearchCV to find best parameters

# In[ ]:


# RandomizedSearchCV randomly assigns a best class and checks if it is best by training the model on those parameters
# making an object of the RandomizedSearchCV class
rcv = RandomizedSearchCV(estimator=clf6,param_distributions=param_grid,n_iter=100,cv=5,verbose=2,n_jobs=-1)
# training the RandomizedSearchCV to find the best parameters
rcv.fit(X_train,y_train)


# ### c) Checking Best Parameters

# In[ ]:


# this best_params_ attribute prints the best attributes that were found by RandomizedSearchCV
rcv.best_params_


# ### d) Checking Best Estimator

# In[ ]:


# this best_estimator_ prints the best model that was found by RandomizedSearchCV
rcv.best_estimator_


# ### e) Training Random Forest with Best Parameters Found

# In[ ]:


# making another object of the Random Forest Classifier to test it with the best parameters
clf8 = RandomForestClassifier(criterion='gini',n_estimators=56,max_depth=10,max_features='log2',min_samples_split=5,min_samples_leaf=1)
clf8.fit(X_train, y_train)
y_pred8 = clf8.predict(X_test)
cm8 = confusion_matrix(y_test,y_pred8)
print(cm8)
print(accuracy_score(y_test,y_pred8))


# ### 7) XGBoost

# In[ ]:


# making an object of the XGBoost class 
clf7 = XGBClassifier()
clf7.fit(X_train, y_train)
y_pred7 = clf7.predict(X_test)
cm7 = confusion_matrix(y_test,y_pred7)
print(cm7)
print(accuracy_score(y_test,y_pred7))


# ## Here we are able to see that XGBoost Classifier beats even HyperTuned Random Forest Classifier

# ### XGBoost achieves an accuracy of 91.496%

# ## Congratulations you have reached the end of this notebook!!!

# # Please Upvote if you like it. It motivates me to do more.
