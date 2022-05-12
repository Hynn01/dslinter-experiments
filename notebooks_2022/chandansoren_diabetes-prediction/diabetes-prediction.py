#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center">Diabetes Prediction</h1>

# <b>Importing the libraries</b>

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# To ignore the warnings
import warnings
warnings.filterwarnings('ignore')


# **Data collection and analysis**
# 
# *PIMA Diabetics*

# In[ ]:


# Loading the the dataset into a Pandas DataFrame
diabetes_dataset = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

# Printing the first 5 row of the dataset
diabetes_dataset.head()


# In[ ]:


# No. of rows and column in the dataset
diabetes_dataset.shape


# In[ ]:


# Statical measures in dataset
diabetes_dataset.describe()


# In[ ]:


diabetes_dataset['Outcome'].value_counts()


# 0---> Non- Diabetic
# 
# 0---> Diabetic

# In[ ]:


diabetes_dataset.groupby('Outcome').mean()


# In[ ]:


# Separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']


# In[ ]:


print(X)


# In[ ]:


print(Y)


# **Data Standardization**

# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(X)


# In[ ]:


standardized_data = scaler.transform(X)


# In[ ]:


print(standardized_data)


# In[ ]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[ ]:


print(X)
print(Y)


# **Train test split**

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2,stratify = Y, random_state=2)


# In[ ]:


print(X.shape, X_train.shape, X_test.shape)


# **Training the data**

# In[ ]:


classifier = svm.SVC(kernel='linear')


# In[ ]:


classifier.fit(X_train, Y_train)


# **Model Evaluation**
# 
# Accuracy score

# In[ ]:


# Accuracy Score in Training Data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[ ]:


print('Accuracy score of training data:', training_data_accuracy)


# In[ ]:


# Accuracy Score in Training Data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[ ]:


print('Accuracy score of test data:', test_data_accuracy)


# **Making prediction model**

# In[ ]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# **Saving the model**

# In[ ]:


import pickle


# In[ ]:


filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:


# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[ ]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

