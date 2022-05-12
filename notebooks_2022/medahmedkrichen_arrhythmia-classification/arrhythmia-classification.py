#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

import seaborn as sns
import scipy.stats as st

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#changable parameters
target = "diagnosis"
test_size = 0.2


# In[ ]:


df = pd.read_csv("/kaggle/input/cardiac-arrhythmia-database/data_arrhythmia.csv",sep=';')
df.dropna(axis=0, inplace=True)
j = []
for i in df.diagnosis:
    if i > 1 :
        j.append(1)
    else:
        j.append(0)
df.diagnosis = j
df.head()


# In[ ]:


#classification or regression
if (type(df[target][0]) == str) or (type(df[target][0]) == int) or (type(df[target][0]) == np.int64):
    models_type = 'classification'
else:
    models_type = 'regression'
    
    
print(models_type)


# In[ ]:


def label_encoding(old_column):
    le = LabelEncoder()
    le.fit(old_column)
    new_column = le.transform(old_column)
    return new_column


# In[ ]:


#encoding string parameters
for i in df.columns:
    if type(df[i][0]) == str:
        df[i] = label_encoding(df[i])


# In[ ]:


#extracting x and y
y = df[target].values
 
x = df.drop([target], axis=1).values


# In[ ]:


#spliting  data
X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=test_size)


# In[ ]:


#ensemble models for classification

if models_type == 'classification':

    model_1 = RandomForestClassifier()

    k_fold_cv = 5 
    params = {
     "n_estimators" : [10,50,100],
     "max_features" : ["auto", "log2", "sqrt"],
     "bootstrap" : [True, False]
     }
    clf1 = RandomizedSearchCV(model_1, param_distributions=params, cv=k_fold_cv,
     n_iter = 5, scoring="accuracy",verbose=2, random_state=42,
     n_jobs=-1, return_train_score=True)
    clf1.fit(X_train, y_train)



    #######################################


    model_2 = XGBClassifier(eval_metric='mlogloss')

    params = {  
                "n_estimators": st.randint(3, 40),
                "max_depth": st.randint(3, 40),
                "learning_rate": st.uniform(0.05, 0.4),
                "colsample_bytree": st.beta(10, 1),
                "subsample": st.beta(10, 1),
                "gamma": st.uniform(0, 10),
                'objective': ['binary:logistic'],
                'scale_pos_weight': st.randint(0, 2),
                "min_child_weight": st.expon(0, 50),

            }

    # Random Search Training with 5 folds Cross Validation
    clf2 = RandomizedSearchCV(model_2, params, cv=5,
                             n_jobs=1, n_iter=100) 

    clf2.fit(X_train, y_train)  



    #######################################


    model_3 = LogisticRegression(max_iter=160)
    model_3.fit(X_train,y_train)

    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }


    clf3 = RandomizedSearchCV(model_3, params, cv=5,
                             n_jobs=1, n_iter=100) 

    clf3.fit(X_train, y_train)  



    #######################################



    model_4 = KNeighborsClassifier()
    model_4.fit(X_train,y_train)

    param_grid=dict(n_neighbors=list(range(1,31)))
    print(param_grid)

    clf4 = RandomizedSearchCV(model_4, param_grid, cv=10, scoring='accuracy')
    clf4.fit(X_train, y_train)  



    #######################################



    train_results = np.array([clf1.predict(X_train), 
                        clf2.predict(X_train),
                        clf3.predict(X_train),
                        clf4.predict(X_train)]).T



    final_model = XGBClassifier()

    final_model = final_model.fit(train_results, y_train)

    test_results = np.array([clf1.predict(X_test), 
                        clf2.predict(X_test),
                        clf3.predict(X_test),
                        clf4.predict(X_test)]).T


    pred_final = final_model.predict(test_results)

    print("accuracy is: ",accuracy_score(y_test, pred_final))
    
    
    #######################################
    
    #save models

    filename = 'clf1.sav'
    pickle.dump(clf1, open(filename, 'wb'))
    
    filename = 'clf2.sav'
    pickle.dump(clf2, open(filename, 'wb'))
    
    filename = 'clf3.sav'
    pickle.dump(clf3, open(filename, 'wb'))
    
    filename = 'clf4.sav'
    pickle.dump(clf4, open(filename, 'wb'))
    
    filename = 'final_model.sav'
    pickle.dump(final_model, open(filename, 'wb'))
    


# In[ ]:


#ensemble models for regression

if models_type == 'regression':

    model_1 = RandomForestRegressor()

    k_fold_cv = 5 
    params = {
     "n_estimators" : [10,50,100],
     "max_features" : ["auto", "log2", "sqrt"],
     "bootstrap" : [True, False]
     }
    clf1 = RandomizedSearchCV(model_1, param_distributions=params, cv=k_fold_cv,
     n_iter = 5, scoring="neg_mean_squared_error",verbose=2, random_state=42,
     n_jobs=-1, return_train_score=True)
    clf1.fit(X_train, y_train)



    #######################################


    model_2 = XGBRegressor()

    params = {  
                "n_estimators": st.randint(3, 40),
                "max_depth": st.randint(3, 40),
                "learning_rate": st.uniform(0.05, 0.4),
                "colsample_bytree": st.beta(10, 1),
                "subsample": st.beta(10, 1),
                "gamma": st.uniform(0, 10),
                'scale_pos_weight': st.randint(0, 2),
                "min_child_weight": st.expon(0, 50),

            }

    # Random Search Training with 5 folds Cross Validation
    clf2 = RandomizedSearchCV(model_2, params, cv=5,
                             n_jobs=1, n_iter=100) 

    clf2.fit(X_train, y_train)  



    #######################################


    clf3 = LinearRegression()
    
    clf3.fit(X_train,y_train)





    #######################################



    model_4 = KNeighborsRegressor()
    model_4.fit(X_train,y_train)

    param_grid=dict(n_neighbors=list(range(1,31)))

    clf4 = RandomizedSearchCV(model_4, param_grid, cv=10, scoring='accuracy')
    clf4.fit(X_train, y_train)  



    #######################################



    train_results = np.array([clf1.predict(X_train), 
                        clf2.predict(X_train),
                        clf3.predict(X_train),
                        clf4.predict(X_train)]).T



    final_model = XGBRegressor()

    final_model = final_model.fit(train_results, y_train)

    test_results = np.array([clf1.predict(X_test), 
                        clf2.predict(X_test),
                        clf3.predict(X_test),
                        clf4.predict(X_test)]).T


    pred_final = final_model.predict(test_results)

    print("socre is: ",mean_squared_error(y_test, pred_final))
    
    
    #######################################
    
    #save models
    
    filename = 'clf1.sav'
    pickle.dump(clf1, open(filename, 'wb'))
    
    filename = 'clf2.sav'
    pickle.dump(clf2, open(filename, 'wb'))
    
    filename = 'clf3.sav'
    pickle.dump(clf3, open(filename, 'wb'))
    
    filename = 'clf4.sav'
    pickle.dump(clf4, open(filename, 'wb'))
    
    filename = 'final_model.sav'
    pickle.dump(final_model, open(filename, 'wb'))
    


# In[ ]:


#confusion matrix plot
if models_type == 'classification': 
    cm = confusion_matrix(y_test, pred_final)
    sns.set(rc={"figure.figsize":(4, 2)})
    sns.heatmap(cm, annot=True)

