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


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Aquamarine;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# TPS MAY22📈
# </h1>
# </div>

# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 1. Data EDA
# </h1>
# </div>

# In[ ]:


df = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
df


# In[ ]:


df.info()


# insights:
# - no null values
# - one categorical value : f_27
# - Many Features => PCA? 
# - "id" Feature Drop

# In[ ]:


df = df.drop("id",axis = 1)


# In[ ]:


df.describe()


# In[ ]:


df.iloc[:, :-1].describe().T.sort_values(by='std' , ascending = False)                     .style.background_gradient(cmap='GnBu')                     .bar(subset=["max"], color='#BB0000')                     .bar(subset=["mean",], color='green')


# In[ ]:


df.hist(figsize = (20,15))


# In[ ]:


import matplotlib.pyplot as plt #data viz
# v Sets matplotlib figure size defaults to 25x20
plt.rcParams["figure.figsize"] = (25,20)

# Creates a figure with 30 subplots to plot each numeric feature
#on 'subplots'->https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.subplots.html
fig, ax = plt.subplots(#This functions lets us place many plots within a single figure
    5, #number of rows
    6  #number of columns
)

#adds title to figure            
fig.text(
    0.35, # text position along x axis
    1, # text position along y axis
    'EDA of Numeric Features', #title text
    {'size': 35} #Increase font size to 35
         )

#The below code will display all numeric feature distributions with a histogram

# subplots can be accessed with an index similar to python lists
i = 0 # subplot column index
j = 0 # subplot row index
for col in df.columns: #iterate thru all dataset columns
    if col not in ['f_27', 'target']: #dont plot f_27 or target feature-will error
        ax[j, i].hist(df[col], bins=100) #plots histogram on subplot [j, i]
        ax[j, i].set_title(col, #adds a title to the subplot
                           {'size': '14', 'weight': 'bold'}) 
        if i == 5: #if we reach the last column of the row, drop down a row and reset
            i = 0
            j += 1
        else: #if not at the end of the row, move over a column
            i += 1


plt.show() #displays figure


# In[ ]:


df['f_27'].value_counts()[:50].plot(kind='bar') #shows the top 50 common values
plt.title('f_27 Top 50 Most Common Values', {'size': '35'}) #Adds title


# ### insight
# - variable : 16
# - categorical : 12
# - object : 1
# - variable features are normal

# In[ ]:


import seaborn as sns

plt.figure(figsize=(30, 2))         #changes figure size

sns.heatmap(df.corr()[-1:],
            cmap="viridis",         
            annot=True          
           )


# ### insight
# - low corr => Do PCA?

# In[ ]:


int_features = list(df.select_dtypes(include='int').columns)
float_features = list(df.select_dtypes(include='float').columns)
object_features = list(df.select_dtypes(include='object').columns)


# In[ ]:


display(df[int_features].nunique())
display(df[float_features].nunique())
display(df[object_features].nunique())


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 2. Data Preprocessing
# </h1>
# </div>

# In[ ]:


train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
train.head()


# In[ ]:


train = train.drop("id",axis = 1)
target = train.pop("target")
display(train.head(), target.head())


# In[ ]:


test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")
test.head()


# In[ ]:


test = test.drop("id",axis = 1)
display(test.head())


# In[ ]:


# f_27 label encoding
for i in range(10):
    train["f_27_{}".format(i+1)] = train["f_27"].astype(str).apply(lambda x: x[i:i+1])
    test["f_27_{}".format(i+1)] = test["f_27"].astype(str).apply(lambda x: x[i:i+1])
display(train.head(),test.head())


# In[ ]:


# f_27 label encoding

from sklearn.preprocessing import LabelEncoder

for i in range(10):
    le = LabelEncoder()
    train["f_27_{}".format(i+1)] = le.fit_transform(train["f_27_{}".format(i+1)])
    
    for label in np.unique(test["f_27_{}".format(i+1)]):
        if label not in le.classes_: # unseen label 
            le.classes_ = np.append(le.classes_, label)
            
    test["f_27_{}".format(i+1)] = le.transform(test["f_27_{}".format(i+1)])
    
display(train.head(),test.head())


# In[ ]:


train = train.drop("f_27",axis = 1)
test = test.drop("f_27",axis = 1)


# In[ ]:


# scaling
float_features = list(df.select_dtypes(include='float').columns)

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
train[float_features] = mm.fit_transform(train[float_features])
test[float_features] = mm.transform(test[float_features])
display(train.head(),test.head())


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 3. Modelling (CatBoost) & Predict
# </h1>
# </div>

# In[ ]:


from catboost  import CatBoostClassifier

model = CatBoostClassifier(n_estimators = 1000, learning_rate = 0.05, random_state=43)
model.fit(train, target)

cat_pred = model.predict(test)


# In[ ]:


sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sub["target"] = cat_pred
sub


# In[ ]:


sub.to_csv('submission_Cat.csv', index = False)


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 4. Use AutoML
# </h1>
# </div>

# In[ ]:


## mljar 설치

get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@master')


# In[ ]:


X = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
y = X.target
X = X.drop(["id","target"],axis = 1)

test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
test = test.drop("id",axis = 1)

display(X.head(),test.head())


# In[ ]:


from supervised import AutoML

automl = AutoML(total_time_limit=60*10,
                model_time_limit = 60,
                mode = "Compete",
                eval_metric="accuracy",
                algorithms = ['Xgboost', 'LightGBM', 'CatBoost'],
                ml_task = 'binary_classification',
                train_ensemble=True,
                n_jobs = -1,
                random_state = 43)
automl.fit(train,target)

auto_pred = automl.predict(test)


# In[ ]:


sub["target"] = auto_pred
sub.head()


# In[ ]:


sub.to_csv('submission_auto.csv', index = False)


# Reference
# https://www.kaggle.com/code/calebreigada/getting-started-eda-preprocessing
