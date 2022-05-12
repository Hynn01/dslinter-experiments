#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[ ]:


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[ ]:





# ## Loading Datasets

# In[ ]:


space_df = pd.read_csv("../input/spaceship-titanic/train.csv")


# In[ ]:


space_df.head()


# In[ ]:


dimension = space_df.shape
print("Number of rows: ",dimension[0])
print("Number of columns: ",dimension[1])


# In[ ]:


pd.DataFrame(space_df.columns,columns=["Columns"])


# In[ ]:


# Number of missing values
space_df.isna().sum()


# In[ ]:


space_df.info()


# In[ ]:


space_df.head() 


# In[ ]:





# ## Descriptive Statistics

# In[ ]:


space_df.dtypes


# In[ ]:


space_df.describe()


# In[ ]:





# ## Data Cleaning

# In[ ]:


space_df.isna().sum()


# My plan is drop the missing boolean values, and replace the numeric variables with their mean

# In[ ]:


space_df.drop(space_df[space_df['CryoSleep'].isna()].index,inplace=True)


# In[ ]:


space_df.drop(space_df[space_df['Cabin'].isna()].index,inplace=True)


# In[ ]:


space_df.drop(space_df[space_df['Destination'].isna()].index,inplace=True)


# In[ ]:


space_df.drop(space_df[space_df['VIP'].isna()].index,inplace=True)


# In[ ]:


space_df.isna().sum()


# In[ ]:


space_df['RoomService'].fillna(np.mean(space_df['RoomService']),inplace=True)


# In[ ]:


space_df['FoodCourt'].fillna(np.mean(space_df['FoodCourt']),inplace=True)


# In[ ]:


space_df['ShoppingMall'].fillna(np.mean(space_df['ShoppingMall']),inplace=True)


# In[ ]:


space_df['Spa'].fillna(np.mean(space_df['Spa']),inplace=True)


# In[ ]:


space_df['VRDeck'].fillna(np.mean(space_df['VRDeck']),inplace=True)


# In[ ]:


space_df.isna().sum()


# All the null values but the name have been cleared. I did not bother with Name because we won't use it in our model creation

# ## Data Wrangling

# In[ ]:


space_df['HomePlanet'].unique()


# I would like to do further analysis with the null values of HomePlanet, so I will replace them with 'Planet X'

# In[ ]:


space_df['HomePlanet'] = space_df['HomePlanet'].fillna("Planet X")


# In[ ]:


plt.figure(figsize=(14,8))
plt.title("Number of travels by planet")
sns.countplot(x=space_df['HomePlanet'])


# In[ ]:


plt.figure(figsize=(14,8))
plt.title("Travels by planet vs Disappeared")
sns.countplot(x=space_df['HomePlanet'],hue=space_df['Transported'])


# From the graph, it looks like all passengers whose Home Planet wasn't specified where transported to another dimension. We also notice that of all the travels from earth, more than half of the travelers survived.

# In[ ]:


space_df['CryoSleep'].unique()


# In[ ]:


# We have about 200 null values for the CryoSleep
# Since the value is small when compared to the non-null values, we can drop them is the data transformation phase
plt.figure(figsize=(14,8))
sns.countplot(x=space_df['HomePlanet'],hue=space_df['CryoSleep'])
plt.title("Number of Cryo sleeps per planet")


# In[ ]:





# In[ ]:


space_df['Destination'].value_counts()


# In[ ]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
sns.countplot(x=space_df['Destination'],ax=ax[0])
sns.countplot(x=space_df['Destination'],hue=space_df['Transported'],ax=ax[1])


# In[ ]:


transported = space_df[space_df['Transported'] ==True]
not_transported = space_df[space_df['Transported'] == False]


# In[ ]:


fig,ax = plt.subplots(figsize=(14,7))
sns.countplot(x=transported['HomePlanet'],hue=transported['Destination'])


# From the barchart above, most of the people that were transported where those going to TRAPPIST-le

# In[ ]:





# Filling in the null values of the age column

# In[ ]:


space_df['Age'] = space_df['Age'].fillna(np.mean(space_df['Age']))


# In[ ]:


fig,ax = plt.subplots(figsize=(14,8))
sns.histplot(space_df['Age'])


# From the graph, most of the people that embarked on the journey were between the ages of 20 and 30

# In[ ]:


# Redefinging the variables, so as to recapture the updated 'Age' column
transported = space_df[space_df['Transported'] ==True]
not_transported = space_df[space_df['Transported'] == False]


# In[ ]:


fig,ax = plt.subplots(figsize=(14,8))
sns.histplot(transported['Age'])


# It also appears that most of the people that were transported were between the age range of 23 and 30

# In[ ]:


fig,ax = plt.subplots(figsize=(14,8))
sns.countplot(x=transported['VIP'])


# Very few of those who were VIPs were transported to another dimension

# In[ ]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
# sns.countplot(x=space_df['Destination'],ax=ax[0])
# sns.countplot(x=space_df['Destination'],hue=space_df['Transported'],ax=ax[1])
ax[0].set_title("Distribution of Room service expeses (Transported)")
ax[1].set_title("Distribution of Room service expeses (Not Transported)")
bins = np.linspace(0,4500,10)
sns.histplot(transported['RoomService'],bins=bins,ax=ax[0],color='red')
sns.histplot(not_transported['RoomService'],bins=bins,ax=ax[1])


# From the graph above, those that were not transported had higher expenses on room service

# In[ ]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
# sns.countplot(x=space_df['Destination'],ax=ax[0])
# sns.countplot(x=space_df['Destination'],hue=space_df['Transported'],ax=ax[1])
ax[0].set_title("Distribution of Food expenses (Transported)")
ax[1].set_title("Distribution of Food expenses (Not Transported)")
bins = np.linspace(0,4500,10)
sns.histplot(transported['FoodCourt'],bins=bins,ax=ax[0],color='red')
sns.histplot(not_transported['FoodCourt'],bins=bins,ax=ax[1])


# As expected, those who were not transported also spent more on food. Seems like the rich were favoured

# In[ ]:


'RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'


# In[ ]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
# sns.countplot(x=space_df['Destination'],ax=ax[0])
# sns.countplot(x=space_df['Destination'],hue=space_df['Transported'],ax=ax[1])
ax[0].set_title("Distribution of ShoppingMall expenses (Transported)")
ax[1].set_title("Distribution of ShoppingMall expenses (Not Transported)")
bins = np.linspace(0,4500,10)
sns.histplot(transported['ShoppingMall'],bins=bins,ax=ax[0],color='red')
sns.histplot(not_transported['ShoppingMall'],bins=bins,ax=ax[1])


# In[ ]:





# In[ ]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
# sns.countplot(x=space_df['Destination'],ax=ax[0])
# sns.countplot(x=space_df['Destination'],hue=space_df['Transported'],ax=ax[1])
ax[0].set_title("Distribution of Spa expenses (Transported)")
ax[1].set_title("Distribution of Spa expenses (Not Transported)")
bins = np.linspace(0,4500,10)
sns.histplot(transported['Spa'],bins=bins,ax=ax[0],color='red')
sns.histplot(not_transported['Spa'],bins=bins,ax=ax[1])


# In[ ]:





# In[ ]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
# sns.countplot(x=space_df['Destination'],ax=ax[0])
# sns.countplot(x=space_df['Destination'],hue=space_df['Transported'],ax=ax[1])
ax[0].set_title("Distribution of VRDeck expenses (Transported)")
ax[1].set_title("Distribution of VRDeck expenses (Not Transported)")
bins = np.linspace(0,4500,10)
sns.histplot(transported['VRDeck'],bins=bins,ax=ax[0],color='red')
sns.histplot(not_transported['VRDeck'],bins=bins,ax=ax[1])


# In[ ]:





# In[ ]:





# ## Data Transformation

# In[ ]:


# The Cabin is made up of 3 different values 
space_df['Cabin'][0]


# In[ ]:


space_df['Cabin'].str.extract(pat=r'([A-Z])').columns


# In[ ]:


deck = space_df['Cabin'].str.extract(pat=r'([A-Z])')
deck.columns = ['deck']


# In[ ]:


num = space_df['Cabin'].str.extract(pat=r'/([0-9])')
num.columns = ['num']


# In[ ]:


side = space_df['Cabin'].str.extract(pat=r'/([A-Z])')
side.columns = ['side']


# In[ ]:


space_df = pd.concat(objs=[space_df,deck,num,side],axis=1)


# In[ ]:


space_df.head()


# In[ ]:





# In[ ]:





# ## Model Creation

# The classifier I plan on using only takes numeric values. So I have to encode the labels

# In[ ]:


homeplanet = pd.get_dummies(space_df['HomePlanet'],drop_first=True)


# In[ ]:


cryosleep = pd.get_dummies(space_df['CryoSleep'],drop_first=True)


# In[ ]:


destination = pd.get_dummies(space_df['Destination'],drop_first=True)


# In[ ]:


vip = pd.get_dummies(space_df['VIP'],drop_first=True)


# In[ ]:


transported = pd.get_dummies(space_df['Transported'],drop_first=True)


# In[ ]:


deck = pd.get_dummies(space_df['deck'],drop_first=True)


# In[ ]:


side = pd.get_dummies(space_df['side'],drop_first=True)


# In[ ]:


num = pd.get_dummies(space_df['num'],drop_first=True)


# In[ ]:


space_df.columns


# In[ ]:


X = space_df.drop(columns=['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','VIP','Name','Transported','deck','num','side'],axis=1)


# In[ ]:


X.head()


# In[ ]:


X = pd.concat([X,homeplanet,cryosleep,destination,vip,transported,deck,side,num],axis=1)
y = transported


# In[ ]:





# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


logr = LogisticRegression()


# In[ ]:


logr.fit(X_train,y_train)


# In[ ]:





# ## Model Evaluation

# In[ ]:


# Predicting the training set
pred = logr.predict(X_train)


# In[ ]:


accuracy_score(y_train,pred)


# In[ ]:





# In[ ]:


# Predicting the test set
pred = logr.predict(X_test)


# In[ ]:


accuracy_score(y_test,pred)


# In[ ]:





# In[ ]:


from sklearn.model_selection import cross_val_score


# I'm intrested in finding out how correct on average my model is when different samples of the of predictor and response are taken

# In[ ]:


ans = cross_val_score(logr,X,y,cv=10)
ans


# In[ ]:


np.mean(ans)


# The average accuracy of the model is 0.9993

# In[ ]:





# In[ ]:


pd.read_csv('../input/spaceship-titanic/test.csv').isna().sum()

