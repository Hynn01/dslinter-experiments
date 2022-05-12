#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import all necessary library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split


# In[ ]:


#  Import dataset

train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.info()


# In[ ]:


# Data Preprocessing

train.drop('Cabin', axis=1, inplace=True)
train['Age'].fillna(method = 'bfill', inplace=True)
train['Embarked'].fillna(value='C', inplace=True)


# In[ ]:


test.drop('Cabin', axis=1, inplace=True)
test['Age'].fillna(method = 'ffill', inplace=True)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train


# In[ ]:


test


# **Perform EDA on train dataset**

# In[ ]:


train.head()


# In[ ]:


# bar graph on embarked distribution

train['Embarked'].unique()


# In[ ]:


ev = train['Embarked'].value_counts()
ev


# In[ ]:


fig = plt.figure(figsize = (8, 4))

# Creating a Bar plot
plt.bar(ev.index, ev, color='Green', width=0.3)

plt.title('Embarked Distribution')
plt.xlabel('Embarked')
plt.ylabel('Distribution')
plt.show()


# In[ ]:


# Plot a histogram on Age

plt.hist(train['Age'])


# In[ ]:


# Plot a piechart on survived people

survived = train[train['Survived'] == 1]
notsurvived = train[train['Survived'] == 0]

print("Survived: ", len(survived))
print("Not_Survived: ", len(notsurvived))


# In[ ]:


ppl_survived = pd.DataFrame([ len(survived), len(notsurvived)], index=['Survived', 'Not_Survived'])
ppl_survived.plot(kind = 'pie', subplots = True, figsize=(16,8), autopct = '%1.1f%%' )


# In[ ]:


# Find survived and not_survived in both gender

gender = train['Sex'].unique()
gender


# In[ ]:


len(train[(train['Survived'] == 1) & (train['Sex'] == 'male') ])


# In[ ]:


# return the survived people in both gender

def gendersurvived(gen):
    return len(train[(train['Survived'] == 1) & (train['Sex'] == gen) ])

# return the not_survived people in both gender

def gendernot_survived(gen):
    return len(train[(train['Survived'] == 0) & (train['Sex'] == gen) ])


# In[ ]:


# make a list of both gender, survived, not_survived and total people. Then merge it into one dataframe

gender_list = list()
genderSurvived_list = list()
genderNotsurvived_list = list()
genderTotal_list = list()

# append all details in respective list

for gen in gender:
    survived_gen = gendersurvived(gen)
    notsurvived_gen = gendernot_survived(gen)
    totalppl_gen = survived_gen + notsurvived_gen
    gender_list.append(gen)
    genderSurvived_list.append(survived_gen)
    genderNotsurvived_list.append(notsurvived_gen)
    genderTotal_list.append(totalppl_gen)


# In[ ]:


print('gender_list: ', gender_list)
print('genderSurvived_list: ', genderSurvived_list)
print('genderNotsurvived_list: ', genderNotsurvived_list)


# In[ ]:


train['Sex'].value_counts()


# In[ ]:


# Let's make a dataframe of it

gender_df = pd.DataFrame({
    "Gender" : gender_list,
    "Gender_Survived" : genderSurvived_list,
    "Gender_notSurvived" : genderNotsurvived_list,
    "Gender_Total" : genderTotal_list
}, columns=['Gender', 'Gender_Survived', 'Gender_notSurvived', 'Gender_Total'])

# Sorting into descending orders

gender_df.sort_values('Gender_Total', ascending=False, inplace=True)

gender_df.head()


# In[ ]:


def groupedGraph(start,end):
    # set width of bar
    barWidth = 0.20

    # set height of bar
    bars1 = gender_df['Gender_Survived'][start:end]
    bars2 = gender_df['Gender_notSurvived'][start:end]
    bars3 = gender_df['Gender_Total'][start:end]

    # Set position of bar on X axis
    r1 = np.arange(bars1.size)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#36688D', width=barWidth, edgecolor='white', label='Survived')
    plt.bar(r2, bars2, color='#F3CD05', width=barWidth, edgecolor='white', label='notSurvived')
    plt.bar(r3, bars3, color='#F49F05', width=barWidth, edgecolor='white', label='Total People')

    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth for r in range(len(bars1))], gender_df['Gender'][start:end])

    # Create legend & Show graphic
    plt.legend()


# In[ ]:


#Let's visualize this dataframe into the Grouped barplot

fig = plt.figure(figsize=(25,15))

plt.subplot(311)
groupedGraph(0,2)


# In[ ]:


# Label encoding on Embarked value

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
train['encode_embarked'] = label_encoder.fit_transform(train['Embarked'])
test['encode_embarked'] = label_encoder.fit_transform(test['Embarked'])


train.head()


# In[ ]:


test.head()


# In[ ]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
train['Gender'] = label_encoder.fit_transform(train['Sex'])
test['Gender'] = label_encoder.fit_transform(test['Sex'])

train.head()


# In[ ]:


test.head()


# In[ ]:


# feature selection

# # let's consider necessary columns

data1 = train[['Survived','Pclass','Gender','Age','Fare','encode_embarked']]
data1


# In[ ]:


data2 = test[['Pclass','Gender','Age','Fare','encode_embarked']]
data2


# In[ ]:


data1.info()


# In[ ]:


X = data1.drop('Survived', axis=1)
y = data1['Survived']


# In[ ]:


data1.head()


# In[ ]:


X.head()


# In[ ]:


# now, plot the dataset

plt.figure(figsize=(12,8))
ax = sns.heatmap(X.corr(), annot=True)
plt.show()


# In[ ]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[ ]:


corr_features = correlation(X, 0.5)
len(set(corr_features))


# In[ ]:


corr_features


# In[ ]:


X_corr = X.drop(corr_features,axis=1)
X_corr


# In[ ]:


test_data = data2.drop('Fare', axis=1)
test_data


# In[ ]:


x_train, x_test,y_train,y_test = train_test_split(X_corr,y,test_size =0.3)

# print the data
x_train


# In[ ]:


print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


# In[ ]:


# Model selection and implimentation (logistic regression`)

from sklearn.linear_model import LogisticRegression


# In[ ]:


clf = LogisticRegression()

clf.fit(x_train,y_train)


# In[ ]:


pred = clf.predict(x_test)
pred


# In[ ]:


# Model Evaluation

clf.score(x_test,y_test)


# In[ ]:


# Confusion Metrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
cm


# In[ ]:


# now, visualize confusion metrix

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');


## Display the visualization of the Confusion Matrix.
plt.show()


# In[ ]:


fnl = clf.predict(test_data)
fnl


# In[ ]:


fnl = pd.DataFrame(fnl,columns=['Survived'])
sub = pd.concat([test['PassengerId'],fnl],axis=1)

sub.set_index('PassengerId',inplace=True)

sub.to_csv("submission.csv")

