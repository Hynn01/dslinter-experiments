#!/usr/bin/env python
# coding: utf-8

# <h2 style="background-color:#101820FF;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 10px 10px;color:#FEE715FF">Welcome</h2>
# 
# <h4><center>In this notebook, we will do classification by tuning the parameters of the xgboost algorithm.</center></h4>
# <br>
# 
# <b>Dataset Abstract:</b> A great number of fruits are grown around the world, each of which has various types. The factors that determine the type of fruit are the external appearance features such as color, length, diameter, and shape. The external appearance of the fruits is a major determinant of the fruit type. Determining the variety of fruits by looking at their external appearance may necessitate expertise, which is time-consuming and requires great effort. The aim of this study is to classify the types of date fruit, that are, Barhee, Deglet Nour, Sukkary, Rotab Mozafati, Ruthana, Safawi, and Sagai by using three different machine learning methods. In accordance with this purpose, 898 images of seven different date fruit types were obtained via the computer vision system (CVS). Through image processing techniques, a total of 34 features, including morphological features, shape, and color, were extracted from these images. First, models were developed by using the logistic regression (LR) and artificial neural network (ANN) methods, which are among the machine learning methods. Performance results achieved with these methods are 91.0% and 92.2%, respectively. Then, with the stacking model created by combining these models, the performance result was increased to 92.8%. It has been concluded that machine learning methods can be applied successfully for the classification of date fruit types.

# <h2 style="background-color:#101820FF;font-family:newtimeroman;font-size:180%;border-radius: 5px 5px;color:#FEE715FF">Import</h2>

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

from xgboost import XGBClassifier


# In[ ]:


get_ipython().system('pip install openpyxl # to read excel file')

data_path = "../input/date-fruit-datasets/Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx"
data=pd.read_excel(data_path)

df = data.copy()


# <h2 style="background-color:#101820FF;font-family:newtimeroman;font-size:180%;border-radius: 5px 5px;color:#FEE715FF">First Look at The Dataset</h2>

# In[ ]:


def InfoData(dataframe,target_variable = None):
    
    print(f"""
== DATA INFO ==
* Shape: {dataframe.shape}
* Number of data = {dataframe.shape[0]}


== COLUMNS INFO ==
* Number of columns: {len(dataframe.columns)}
* Columns with dtype: 
{dataframe.dtypes}


== Missing / Nan Values ==
* Is there any missing value?: {dataframe.isnull().values.any()}
    """)
    
    if (target_variable != None) and (target_variable in dataframe.columns):    
        
        print(f"""
== TARGET VARIABLE ==

* Variable: {target_variable}
* Values of Variable: {" - ".join(df.Class.unique())}
* Count of Values: {len(df.Class.unique())}
""")
        
        
        
    elif (target_variable != None) and (target_variable not in dataframe.columns): 
        print("Please type correctly your target variable")
        


# In[ ]:


InfoData(df,"Class")


# In[ ]:


for i in df.columns[:-1]: # I dont want to / can not see last column("Class") because it is target variable and it is an object
    print(f"{i}: | Min: {df[i].min():.4f} | Max: {df[i].max():.4f} | Std: {df[i].std():.4f} | Mean: {(df[i].mean()):.4f}")


# <h2 style="background-color:#101820FF;font-family:newtimeroman;font-size:180%;border-radius: 5px 5px;color:#FEE715FF">Preprocessing</h2>

# In[ ]:


le = LabelEncoder()
target = df['Class']
target = le.fit_transform(target)


# In[ ]:


X = df.drop("Class",axis=1)


# In[ ]:


train_test_split_params = {"test_size":0.33,
                        "random_state":1}


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, target,
                                                    test_size=train_test_split_params["test_size"],
                                                    random_state=train_test_split_params["random_state"],
                                                    shuffle=True)


# In[ ]:


print(f"""
X_train shape: {X_train.shape}
X_test shape: {X_test.shape}
y_train shape: {y_train.shape}
y_test shape: {y_test.shape}
""")


# <h2 style="background-color:#101820FF;font-family:newtimeroman;font-size:180%;border-radius: 5px 5px;color:#FEE715FF">Model</h2>

# #### Notes:
# ##### * I will use "RandomizeGridSearch" to select parameters
# ##### * I will use "StratifiedKFold" because we are dealing with imbalanced class distributions. (You can see below)

# In[ ]:


sns.barplot(y=df["Class"].value_counts().index,x=df["Class"].value_counts().values);


# In[ ]:


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'eval_metric': ["mlogloss"],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [0, 3, 4]
        }


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)

randomized_search = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=5, cv=skf.split(X_train,y_train), verbose=3, random_state=1)

randomized_search.fit(X_train, y_train)


# In[ ]:


print('Best hyperparameters:', randomized_search.best_params_)


# In[ ]:


xgb = XGBClassifier(subsample = randomized_search.best_params_["subsample"],
                      min_child_weight = randomized_search.best_params_["min_child_weight"],
                      max_depth = randomized_search.best_params_["max_depth"],
                      learning_rate = randomized_search.best_params_["learning_rate"],
                      gamma = randomized_search.best_params_["gamma"],
                      eval_metric = randomized_search.best_params_["eval_metric"],
                      colsample_bytree = randomized_search.best_params_["colsample_bytree"])


# In[ ]:


xgb.fit(X_train, y_train)


# In[ ]:


train_pred = xgb.predict(X_train)
train_acc = accuracy_score(y_train,train_pred)
print('Train Accuracy: ', train_acc)
 
test_pred = xgb.predict(X_test)
test_acc = accuracy_score(y_test,test_pred)
print('Test Accuracy:', test_acc)


# <h2 style="background-color:#101820FF;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 10px 10px;color:#FEE715FF">What's Next?</h2>
# 
# * Model can be optimized with different hyperparameters.
# * A better result can be obtained with AutoML.
# * Apart from Accuracy, other metrics should also be checked.
# 
# You can do the above by using this notebook instead of leaving comments like "okay, good, can be improved" etc. by looking at this notebook.
# Let's get into action! Have Fun!
# If you have any questions or ideas, please do not forget to write a comment.
