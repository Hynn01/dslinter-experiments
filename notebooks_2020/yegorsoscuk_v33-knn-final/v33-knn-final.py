#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#3I0203 UI1


# In[ ]:


import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


#načítanie dát
train_df = pd.read_csv('../input/titanic/train.csv')
Xtest_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


#výpis stĺpcov
print(train_df.columns.values)


# In[ ]:


#zahodenie mena, čísla lístku, kabíny a čísla pasažiera, podľa mňa nemali vplyv na prežitie
train_df = train_df.drop('Name', axis=1,)
train_df = train_df.drop('Ticket', axis=1,)
train_df = train_df.drop('Cabin', axis=1,)


# In[ ]:


#výpis nových stĺpcov
print(train_df.columns.values)


# In[ ]:


#doplnienie prázdnych hodnôt
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Embarked"].mode()
train_df["Embarked"] = train_df["Embarked"].fillna("S")


# In[ ]:


#overenie doplnenia
feat_list = list(train_df.columns.values)

for feat in feat_list:
    print (feat,": ",sum(pd.isnull(train_df[feat])))


# In[ ]:


#informatívny výpis
train_df.head()


# In[ ]:


#informatívny výpis
train_df.info()


# In[ ]:


#deľba dátovej množiny
df_train, df_test = train_test_split(train_df, test_size=0.15,
                     stratify=train_df["Survived"], random_state=4)


# In[ ]:


#rozdelenie stĺpcov a predspracovanie, pipeline
categorical_inputs = ["Sex","Embarked"]  
numeric_inputs = ["Pclass","Age","SibSp","Parch","Fare","PassengerId"] 

output = "Survived"


# In[ ]:


input_preproc = make_column_transformer(
    (make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder()),
     categorical_inputs),
    
    (make_pipeline(
        SimpleImputer(),
        StandardScaler()),
     numeric_inputs)
)  
output_enc = OrdinalEncoder()


# In[ ]:


#transformátor
X_train = input_preproc.fit_transform(df_train[categorical_inputs+numeric_inputs])
Y_train = df_train[output].values.reshape(-1)

X_test = input_preproc.transform(df_test[categorical_inputs+numeric_inputs])
Y_test = df_test[output].values.reshape(-1)


# In[ ]:


#vytvorenie modelu (KNN)
model = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
model.fit(X_train, Y_train)


# In[ ]:


#vyskúšanie na skušovnych datach
y_test = model.predict(X_test)


# In[ ]:


cm = pd.crosstab(Y_test, y_test,
                 rownames=['actual'],
                 colnames=['predicted'])
print(cm)

acc = accuracy_score(Y_test, y_test)
print("Accuracy = {}".format(acc))
#vysledok na krizovej validacii


# In[ ]:


#spracovať data na zadanie
print(Xtest_df.columns.values)
Xtest_df = Xtest_df.drop('Name', axis=1,)
Xtest_df = Xtest_df.drop('Ticket', axis=1,)
Xtest_df = Xtest_df.drop('Cabin', axis=1,)


# In[ ]:


#doplnenie hodnôt
Xtest_df["Age"] = Xtest_df["Age"].fillna(Xtest_df["Age"].median())
Xtest_df["Embarked"].mode()
Xtest_df["Embarked"] = Xtest_df["Embarked"].fillna("S")


# In[ ]:


#informatívny výpis
Xtest_df.head()


# In[ ]:


#spracovanie
X_test1 = input_preproc.transform(Xtest_df[categorical_inputs+numeric_inputs])
Y_test1 = Xtest_df.values.reshape(-1)


# In[ ]:


#predikcia
y_test1 = model.predict(X_test1)


# In[ ]:


#odoslanie výsledkov
submission = pd.DataFrame({
        "PassengerId": Xtest_df["PassengerId"],
        "Survived": y_test1
    })
submission.to_csv('v3gender_submission.csv', index=False)

