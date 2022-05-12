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


# ### Process to solve code

# ### Load Libraries

# In[ ]:


# Data manipulation libraries
import pandas as pd
import numpy as np

##### Scikit Learn modules needed for Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler , StandardScaler
from sklearn.preprocessing import LabelEncoder,MinMaxScaler , StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Preprocessing of data

# In[ ]:


df = pd.read_csv("/kaggle/input/personal-key-indicators-of-heart-disease/heart_2020_cleaned.csv")
df.head()


# In[ ]:


print(f"Shape of dataframe: {df.shape}")


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=  "all")


# In[ ]:


def convert_y(x):
    if x == "No":
        x = 0
    else:
        x = 1
    return x


# In[ ]:


df.columns


# In[ ]:


df["HeartDisease"] = df["HeartDisease"].apply( lambda x: convert_y(x))


# In[ ]:


df.tail()


# ### Visualize Relationships between Input Features
# - Input Features = BMI, Smoking, AlcoholDrinking, PhysicalHealth, MentalHealth,Sex,Race, Physical Activity

# In[ ]:


# # Encoding of categorical class variable to integers which will be used for feeding the model
# le = LabelEncoder()
# le_encoded = le.fit_transform(df[["Smoking","AlcoholDrinking","Sex","Race", "PhysicalActivity"]])
# #le_encoded = le.transform(le)
# print((le.classes_))
# print(le_encoded)


# In[ ]:


def label_transformation(x):
    le = LabelEncoder()
    le_encoded = le.fit_transform(x)
    return le_encoded


# In[ ]:


df_transformed = df.copy()


# In[ ]:


categorical_columns = ["Smoking","AlcoholDrinking","Sex","Race", "PhysicalActivity"]
for x in categorical_columns:
    df_transformed[x] = label_transformation(df[x])


# In[ ]:


df_transformed.head()


# In[ ]:


scaler = StandardScaler()
df_transformed[["BMI","PhysicalHealth","MentalHealth"]] = scaler.fit_transform(df_transformed[["BMI","PhysicalHealth","MentalHealth"]])


# In[ ]:


df_transformed.head()


# In[ ]:


df[["BMI","PhysicalHealth","MentalHealth"]].corr()


# ##### *Based on Correation Matrix we do not observe correlation between input continuous variables*

# ### Build Model

# In[ ]:


# Train & Test split
x_train, x_test, y_train, y_test = train_test_split( df_transformed[["BMI","PhysicalHealth","MentalHealth","Smoking","AlcoholDrinking","Sex","Race", "PhysicalActivity"]],
                                                 df_transformed["HeartDisease"],test_size=0.20,
                                                    random_state=21)

print('Shape of Training Xs:{}'.format(x_train.shape))
print('Shape of Test Xs:{}'.format(x_test.shape))
print('Shape of Training y:{}'.format(y_train.shape))
print('Shape of Test y:{}'.format(y_test.shape))


# In[ ]:


model = LogisticRegression()
model.fit(x_train, y_train)
y_predicted_labels = model.predict(x_test)


# In[ ]:


model.score(x_test,y_test)


# ### Validate Performance - Confusion Matrix

# In[ ]:


#Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predicted_labels)
np.set_printoptions(precision=2)
cnf_matrix


# In[ ]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


#With Normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= [0,1],
                      title='Confusion matrix, without normalization')
# With normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= [0,1], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:


#With Normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= [0,1],
                      title='Confusion matrix, without normalization')
# With normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= [0,1], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ### Hyper Parameter Tunning

# 

# In[ ]:


df_transformed.columns


# In[ ]:


# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['BMI', 'PhysicalHealth','MentalHealth']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ['Smoking', 'AlcoholDrinking', 'Sex','Race','PhysicalActivity']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])


# In[ ]:


# Train & Test split
x_train, x_test, y_train, y_test = train_test_split( df[["BMI","PhysicalHealth","MentalHealth","Smoking","AlcoholDrinking","Sex","Race", "PhysicalActivity"]],
                                                 df["HeartDisease"],test_size=0.20,
                                                    random_state=21)

print('Shape of Training Xs:{}'.format(x_train.shape))
print('Shape of Test Xs:{}'.format(x_test.shape))
print('Shape of Training y:{}'.format(y_train.shape))
print('Shape of Test y:{}'.format(y_test.shape))


# In[ ]:


clf.fit(x_train, y_train)
print("model score: %.3f" % clf.score(x_test, y_test))


# #### Apply Hyper Paraneter Grid for tunning of the model

# In[ ]:


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__solver': ["newton-cg","lbfgs", "liblinear", "sag", "saga"],
    'classifier__max_iter' :[100,150,200]
}

grid_search = GridSearchCV(clf, param_grid, cv=2, verbose= 2 , n_jobs = -1)
grid_search.fit(x_train, y_train)

print(("best Logistic Regression from grid search: %.3f"
       % grid_search.score(x_test, y_test)))


# In[ ]:


grid_search.best_params_


# In[ ]:


import joblib


# In[ ]:


joblib.dump(grid_search, "logistic_model.md")


# In[ ]:


test_model = joblib.load('logistic_model.md')


# In[ ]:


y_prediction = test_model.predict(x_test)
y_prediction[0:100]


# In[ ]:


#Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_prediction)
np.set_printoptions(precision=2)
cnf_matrix


# In[ ]:


#With Normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= [0,1],
                      title='Confusion matrix, without normalization')
# With normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= [0,1], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ### Ways to Handle Imbalance in data
# - Reduce number of rows of high freq. lable by x%
# - Adding more samples with lables havings less freq
# - Hyper Parameter tunning - try different transformation methods, different model hyper parameters
# - Try other classification mdoels

# #### Accuracy = *f*(Time)

# In[ ]:




