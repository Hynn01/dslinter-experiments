#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Udemy courses analysis

# **Building a Regression model to predict the subscribers for the udemy courses**

# # Importing Libraries

# In[ ]:


#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# # Loading the Dataset

# In[ ]:


#Loading the dataset
data = pd.read_csv("../input/udemy-courses/udemy_courses.csv")
print("Number of datapoints:", len(data))
data[:3].style.set_properties(**{"background-color": "pink","color": "black", "border-color": "grey"})

#Hue setting
sns.set(rc={"axes.facecolor":"#FCE4DE","figure.facecolor":"#CABFC1"})
cmap = colors.ListedColormap(["#615154", "#F7B4A7", "#94DDDE", "#DCFFF5", "#F0ABC1"])
palette = ["#615154", "#F7B4A7", "#94DDDE", "#FCE4DE", "#DCFFF5", "#F0ABC1", "#CABFC1"]


# # Data cleaning

# In[ ]:


#Information on features 
data.info()


# * Dropping the unwanted features in the dataframe
# * Checking for null values
# * Outlier detection
# * Converting categorical data to numeric

# In[ ]:


#Dropping the unwanted features in the dataframe
df=data.drop(["course_id","course_title","url","published_timestamp"],axis=1)
df[:3].style.set_properties(**{"background-color": "pink","color": "black", "border-color": "grey"})


# In[ ]:


#Checking for null values
df.isnull().sum()


# In[ ]:


#outlier checking
df.describe().style.set_properties(**{"background-color": "grey","color": "black", "border-color": "grey"})


# In[ ]:


#Converting categorical data to numeric - is_paid, level, subject
#is_paid
df.is_paid.unique()


# In[ ]:


df.is_paid=df['is_paid'].apply(lambda x: 0 if x==False else 1)
df[:3].style.set_properties(**{"background-color": "pink","color": "black", "border-color": "grey"})


# In[ ]:


#One hot encoded
#level
df.level.unique()


# In[ ]:


dummies1=pd.get_dummies(df.level)
merged1=pd.concat([df,dummies1],axis=1)
merged1.drop(['Expert Level'], axis=1, inplace=True)
merged1[:3].style.set_properties(**{"background-color": "pink","color": "black", "border-color": "grey"})


# In[ ]:


#subject
dummies2=pd.get_dummies(merged1.subject)
merged2=pd.concat([merged1,dummies2],axis=1)
merged2.drop(['Musical Instruments'], axis=1, inplace=True)
df1=merged2
df1[:3].style.set_properties(**{"background-color": "pink","color": "black", "border-color": "grey"})


# In[ ]:


#outlier elimination
df1=df1[(df1['num_subscribers']<2600)]


#Creating Five stars based bins
bins = ['-1','0','120','920','2200','2600']
labels = ['0','120','920','2200','2500']
df1["Subscribers"] = pd.cut(data["num_subscribers"], bins=bins, labels=labels)
#Pairplot 
hue_C = ["#615154", "#F7B4A7", "#F0ABC1", "#94DDDE", "#B46B82"]
sns.pairplot(df1,hue= "Subscribers", palette= hue_C)


# In[ ]:


#outlier elimination
df1=df1[(df1['num_lectures']<400)]
df1=df1[(df1['content_duration']<40)]


To_plot = ["is_paid","price","num_reviews","num_lectures","content_duration"]
for i in To_plot:
    sns.jointplot(x=df1["num_subscribers"], y=df1[i], hue=df1["Subscribers"], palette= hue_C )
    plt.show()


# # Data Exploration
# 
# From above visuals
# * Most course durations lies between 1-5 hours
# * Usually around 10-50 lectures per course is more common.
# * Overall the review rate is very low irrespective of courses
# * The majority of courses are in the same range of subscribers. The instances farther up the scale were probably more successful or perhaps courses on a trending topic.
# * Assuming the currency is in USD, the most common price is around $25

# In[ ]:


#correlation matrix
corrmat= df1.corr()
plt.figure(figsize=(15,15))  
sns.heatmap(corrmat,annot=True, cmap=cmap)


# In[ ]:


plt.figure(figsize=(8,8))
sns.scatterplot(x=df1.num_subscribers, y=df1.level, color= palette[0])
plt.title("Disribution of course level Vs subscribers")
plt.xlabel("subscribers")
plt.ylabel("Course level")
plt.show()


# In[ ]:


plt.figure(figsize=(8,8))
sns.scatterplot(x=df1.num_subscribers, y=df1.subject, color= palette[0])
plt.title("Disribution of course level Vs subscribers")
plt.xlabel("subscribers")
plt.ylabel("Subject")
plt.xscale("linear")
plt.show()


# In[ ]:


# count number of instances
level_values = df1["level"].value_counts()

# count number of instances
subject_values = df1["subject"].value_counts()

# pie plot of course levels and subjects in data
fig, ax = plt.subplots(1,2, figsize=(20,5))
ax[0].pie(level_values, startangle=180, labels=level_values.index,autopct="%1.1f%%")
ax[0].set_title("Course Levels", size=20,color= "black")
ax[1].pie(subject_values, startangle=180, labels=subject_values.index, autopct="%1.1f%%")
ax[1].set_title("Course Subjects", size=20,color= "black")
plt.tight_layout()
plt.show()


# Other Observations:
# * Most of the course is All levels, over 50%
# * Similarly in course subject, most common one is Business finance
# 

# In[ ]:


df1.drop(["level","subject","Subscribers"],axis=1,inplace=True)


# In[ ]:


df1.content_duration=df['content_duration'].apply(lambda x: int(np.floor(x)))


# **Assign labels and targets**

# In[ ]:


# Assigning the features as X and trarget as y
X= df1.drop(["num_subscribers"],axis =1)
y= df1["num_subscribers"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)


# In[ ]:


test_set = pd.DataFrame({'org_target': y_test})
test_set.to_csv('./test_target.csv', index=False)


# # Model selection

# In[ ]:


#Building piplines for model selection

pipeline_lr=Pipeline([("scalar1",StandardScaler()),
                      ("pca1",PCA(n_components=6)),
                      ("LR",LinearRegression())])

pipeline_dt=Pipeline([("scalar2",StandardScaler()),
                      ("pca2",PCA(n_components=6)),
                      ("DT",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar3",StandardScaler()),
                      ("pca3",PCA(n_components=6)),
                      ("RF",RandomForestRegressor())])

pipeline_knn=Pipeline([("scalar4",StandardScaler()),
                       ("pca4",PCA(n_components=6)),("KN",KNeighborsRegressor())])


#List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_knn]

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors"}


# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

#Getting CV scores    
cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))


# **The best model is ... Random Forest 🏆**

# In[ ]:


# Model prediction on test data with best parameters
model = RandomForestRegressor(criterion='squared_error',max_depth= 6, n_estimators= 120, random_state= 2)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Model Evaluation
r2 = metrics.r2_score(y_test, pred)
Adjusted_r2 = 1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
mae = metrics.mean_absolute_error(y_test, pred)
mse = metrics.mean_squared_error(y_test, pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))

# initialise data of lists.
ResultData = [[r2],[Adjusted_r2],[mae],[mse],[rmse]]
# Creates pandas DataFrame.
Results = pd.DataFrame(ResultData,columns= ["Scores"] ,index = ["R-Squared","Adjusted R-Squared", "Mean Absolute Error","Mean Square Error","Root Mean Square Error"])
Results

