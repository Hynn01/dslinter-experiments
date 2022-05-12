#!/usr/bin/env python
# coding: utf-8

# # Introduction About titanic Ship

# ![](http://th.bing.com/th/id/OIP.AuRW9Bxj22afFaqfz_4MagHaEi?pid=ImgDet&w=576&h=353&rs=1)
# 

# RMS Titanic was a British passenger liner, operated by the White Star Line, which sank in the North Atlantic Ocean on 15 April 1912 after striking an iceberg during her maiden voyage from Southampton, UK, to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, which made the sinking possibly one of the deadliest for a single ship up to that time.[a] It remains to this day the deadliest peacetime sinking of a superliner or cruise ship.[4] The disaster drew much public attention, provided foundational material for the disaster film genre, and has inspired many artistic works.
# 
# RMS Titanic was the largest ship afloat at the time she entered service and the second of three Olympic-class ocean liners operated by the White Star Line. She was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, who was the chief naval architect of the shipyard at that time, died in the disaster.[5]
# 
# Titanic was under the command of Captain Edward Smith,[6] who went down with the ship. The ocean liner carried some of the wealthiest people in the world, as well as hundreds of emigrants from Great Britain and Ireland, Scandinavia and elsewhere throughout Europe, who were seeking a new life in the United States and Canada.

# # Importing all the modules

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# # Understanding the data

# Survival : 0 = No, 1 = Yes
# 
# Pclass : A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower
# 
# sibsp : The # of siblings / spouses aboard the Titanic Sibling = brother, sister, stepbrother, stepsister Spouse =
# 
# husband, wife (mistresses and fiancés were ignored)
# 
# parch : The # of parents / children aboard the Titanic Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.
# 
# Ticket : Ticket number
# 
# Fare : Passenger fare
# 
# Cabin : Cabin number embarked
# 
# Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton
# 
# Name, Sex , Age are self-explanatory

# # Load the data

# In[ ]:


train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()


# **Checking the data**

# In[ ]:


train_data.head()


# **Checking the shape of the data**

# In[ ]:


print("training data rows",train_data.shape[0])
print("training data column",train_data.shape[1])


# **Checking that how many null value present in the data or in which column**

# In[ ]:


train_data.isna().sum()


# # Visualizing and Analysing  the data

# **Visualising how many null value present**

# In[ ]:


plt.figure(figsize = (13,5))
plt.bar(train_data.columns, train_data.isna().sum())
plt.xlabel("Columns name")
plt.ylabel("Number of missing values in training data")
plt.show()


# # Visualising the data

# **Making the countplot using the seaborn for counting how many are survived**

# In[ ]:


sns.countplot('Survived', data = train_data)
plt.show()


# **Counting according to the embarked**

# In[ ]:


sns.countplot('Embarked', data = train_data)
plt.show()


# **Counting according to the gender**

# In[ ]:


sns.countplot('Survived', hue = 'Sex', data = train_data)
plt.plot()


# **Counting according to the Pclass**

# In[ ]:


sns.countplot('Survived',hue='Pclass',data=train_data)
plt.plot


# **Counting According to the Embarked and Survived**

# In[ ]:


sns.countplot('Survived',hue='Embarked',data=train_data)
plt.plot


# **Checking Using the boxplot**
# this shows that there were very few people who payed more than 100

# In[ ]:


sns.boxplot('Fare', data = train_data)
plt.show()


# **Checking According to the Age**
# this shows that there were very few people more than 65 years old in training data

# In[ ]:


sns.boxplot('Age', data = train_data)
plt.show()


# **Checking the numbers according to the age**
# this shows that around 400 people pay in between 20 and 40

# In[ ]:


interval = 10
value_for_bin = np.ceil((train_data.Age.max() - train_data.Age.min()) / interval).astype(int)

plt.hist(train_data.Age, bins = value_for_bin)
plt.xlabel("Age")
plt.ylabel("Number")
plt.show()


# **Checking the numbers according to the Fare**
# this shows that around 700 people pay in between 0 and 50

# In[ ]:


plt.figure(figsize = (10,4))
plt.hist(train_data.Fare, bins = 10, color = 'lime')
plt.xlabel("Fare")
plt.ylabel("Number")
plt.show()


# **This is plotting many histograms for comparing the data**
# and facerGrid is use for comparing all the graphs

# In[ ]:


grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()


# **This is plotting many barplot for comparing the data**
# and facetGrid is use for comparing all the graphs Accordint to the gender and fre it is give result about survived

# In[ ]:


grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()


# **In the last we visualising the corelation between the data using the heatmap**

# In[ ]:


sns.heatmap(train_data.corr())
plt.show()


# Survival rate for males and females

# **Inthis we are comparing that how many peoples are from diffrent gender and survived how much percent of them**

# In[ ]:


((train_data.groupby(['Sex','Survived']).Survived.count() * 100) / train_data.groupby('Sex').Survived.count())
# this shows that female have around 74% chance of survival while male have around 81% chance of death


# **Inthis we are comparing that how many peoples are from diffrent Pclass and survived how much percent of them**

# In[ ]:


((train_data.groupby(['Pclass','Survived']).Survived.count()*100/train_data.groupby('Pclass').Survived.count()))


# **Inthis we are comparing that how many peoples are from diffrent Embarked and survived how much percent of them**

# In[ ]:


((train_data.groupby(['Embarked','Survived']).Survived.count()*100/train_data.groupby('Embarked').Survived.count()))


# **In this we are finding that how much mean of the survived or non survives people according to the age**

# In[ ]:


train_data.groupby(by=['Survived']).mean()["Age"]
# this show that average age of people who survived was around 28 years old


# **We drop the Cabin feature bcoz we dosent required that feature and it has very few value have given**

# # Dealing with the missing Values

# In[ ]:


train_data.drop('Cabin', axis = 1, inplace = True)
test_data.drop('Cabin', axis = 1, inplace = True)


# **Finding the how much null have the dataset**

# In[ ]:


combined_data=[train_data,test_data]
for data in combined_data:
    print(data.isna().sum())
    print('-----------------')


# **Fill the value according to the mean in age and fare and we fill embarked with the S**

# In[ ]:


for data in combined_data:
    data.Age.fillna(data.Age.mean(),inplace=True)
    data.Fare.fillna(data.Fare.mean(),inplace=True)
    data.Embarked.fillna('S',inplace=True)


# In[ ]:


combined_data=[train_data,test_data]
for data in combined_data:
    print(data.isna().sum())
    print('-----------------')


# **Like the one hot encoding we also encoding the Sex Factor**

# In[ ]:


def change_gender(x):
    if x == 'male':
        return 0
    elif x == 'female':
        return 1
train_data.Sex = train_data.Sex.apply(change_gender)
test_data.Sex = test_data.Sex.apply(change_gender)


# **Like the Label encoding we also encoding the Embarked Factor**

# In[ ]:


change = {'S':1,'C':2,'Q':0}
train_data.Embarked = train_data.Embarked.map(change)
test_data.Embarked = test_data.Embarked.map(change)


# # Feature Extraction

# **We make One column from the SibSp and Parch bcoz they both are almost Same **

# In[ ]:


train_data['Alone'] = train_data.SibSp + train_data.Parch
test_data['Alone'] = test_data.SibSp + test_data.Parch

train_data.Alone = train_data.Alone.apply(lambda x: 1 if x == 0 else 0)
test_data.Alone = test_data.Alone.apply(lambda x: 1 if x == 0 else 0)


# In[ ]:


train_data.drop(['SibSp','Parch'], axis = 1, inplace = True)
test_data.drop(['SibSp','Parch'], axis = 1, inplace = True )


# # Creating new feature Title extracting from existing feature Name

# In[ ]:


train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False).unique().size


# Dropping name axis also bcosz it dosent required

# In[ ]:


for data in combined_data:
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand = False)
    data.drop('Name', axis = 1, inplace = True)
       


# **Find the unique value in the Title**

# In[ ]:


train_data.Title.value_counts()


# In[ ]:


test_data.Title.unique()


# **Converting the many column into the rare bcos it have not a sense create like that**

# In[ ]:


least_occuring = [ 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess','Dona',
       'Jonkheer']
for data in combined_data:
    data.Title = data.Title.replace(least_occuring, 'Rare')


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for data in combined_data:
    data['Title'] = data['Title'].map(title_mapping)


# **Converting age and the Fare**

# In[ ]:


for dataset in combined_data:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# In[ ]:


for data in combined_data:
    data.loc[data['Fare'] < 30, 'Fare'] = 1
    data.loc[(data['Fare'] >= 30) & (data['Fare'] < 50),'Fare'] = 2
    data.loc[(data['Fare'] >= 50) & (data['Fare'] < 100),'Fare'] = 3
    data.loc[(data['Fare'] >= 100),'Fare'] = 4


# In[ ]:


corr_train = train_data.corr()
sns.heatmap(corr_train)
plt.show()


# # Prepare the training data

# In[ ]:


columns_to_drop = ['PassengerId','Ticket']
train_data.drop(columns_to_drop, axis = 1, inplace = True)
test_data.drop(columns_to_drop[1], axis = 1, inplace = True)
X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]

print("shape of X_train",X_train.shape)
print("Shape of Y_train",Y_train.shape)


# # Creating Neural Network with tensorflow

# ![](https://th.bing.com/th/id/OIP.goFgCUHprcroxSLZvROjpgHaE7?pid=ImgDet&rs=1)

# ![](https://specials-images.forbesimg.com/imageserve/5fc09cf1950179a3f5f74874/960x0.jpg?fit=scale)

# In[ ]:


X_train.head()
Y_train.head()


# In[ ]:


import tensorflow as tf
import keras 
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential


# # Making Model

# In[ ]:


model = Sequential()
model.add(Dense(units = 32, input_shape = (7,), activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(units = 128, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
model.add(Dropout(0.1))
model.add(Dense(units = 64, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
model.add(Dropout(0.1))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(0.15))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(units = 8, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
model.add(Dense(units =1 , activation = 'sigmoid'))


# # Training data into Model

# In[ ]:


model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer = tf.keras.optimizers.Adam(), metrics = ['acc'])
model.fit(X_train, Y_train, batch_size = 32, verbose = 2, epochs = 50)


# # testing the Model

# In[ ]:


Y_test=test_data.drop('PassengerId',axis=1)
predict = model.predict(Y_test)
#since we have use sigmoid activation function in output layer
predict = (predict > 0.5).astype(int).ravel()
print(predict)


# In[ ]:


output = pd.DataFrame({"PassengerId":test_data.PassengerId, 'Survived':predict})
output.to_csv("submission.csv",index = False)


# In[ ]:


from sklearn import metrics
Y_pred_rand = (model.predict(X_train) > 0.5).astype(int)
print('Precision : ', np.round(metrics.precision_score(Y_train, Y_pred_rand)*100,2))
print('Accuracy : ', np.round(metrics.accuracy_score(Y_train, Y_pred_rand)*100,2))
print('Recall : ', np.round(metrics.recall_score(Y_train, Y_pred_rand)*100,2))
print('F1 score : ', np.round(metrics.f1_score(Y_train, Y_pred_rand)*100,2))
print('AUC : ', np.round(metrics.roc_auc_score(Y_train, Y_pred_rand)*100,2))


# In[ ]:


# plotting the confusion matrix in heatmap
matrix = metrics.confusion_matrix(Y_train, Y_pred_rand)
sns.heatmap(matrix, annot = True,fmt = 'g')
plt.show()


# **Any Suggestion for this model ius really appreciated**
# # Thank You

# In[ ]:




