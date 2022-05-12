#!/usr/bin/env python
# coding: utf-8

# # Build a Regression Model in Keras

# **A. Build a baseline model**
# 
# Use the Keras library to build a neural network with the following:
# 
# - One hidden layer of 10 nodes, and a ReLU activation function
# 
# - Use the adam optimizer and the mean squared error  as the loss function.
# 
# 1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the train_test_splithelper function from Scikit-learn.
# 
# 2. Train the model on the training data using 50 epochs.
# 
# 3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.
# 
# 4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.
# 
# 5. Report the mean and the standard deviation of the mean squared errors.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical


# In[ ]:


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[ ]:


from sklearn.metrics import mean_squared_error


# # Import DATA

# In[ ]:


data = pd.read_csv('../input/concrete-data/concrete_data.csv')


# In[ ]:


data.head()


# # X and y arrays

# In[ ]:


X = data.drop(['Strength'], axis=1)
y = data[['Strength']]


# In[ ]:


X


# In[ ]:


y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # A. Model

# In[ ]:


X_train.shape


# In[ ]:


def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=8, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[ ]:


model = regression_model()


# In[ ]:


model.fit(X_train, y_train, epochs=50)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


result = np.sqrt(mean_squared_error(y_test,y_pred))
result


# In[ ]:


model.summary()


# # A. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

# In[ ]:


MSE_List = []
for i in range(50):
    #1-Split Data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = regression_model()
    #2-Train:
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    #Prediction:
    y_pred = model.predict(X_test)
    
    #3-Evaluate_Model:
    result = np.sqrt(mean_squared_error(y_test,y_pred))
    print("{}: sqrt(mse) = {}".format(i+1,result))
    MSE_List.append(result)
    print("***_________________________________***\n\n\n")


# In[ ]:


MSE_List


# # A. Results
# **Report mean square MSE and std square MSE**

# In[ ]:


# Calculate the mean and the standard deviation of the metric on the 50 samplings
mean_sqmse_A = np.mean(MSE_List)
std_sqmse_A  = np.std(MSE_List)

# Generate a data frame to store the results of the differents parts of this project
df_results = pd.DataFrame.from_dict({"Part": ["A"],"mean_sq_mse": [mean_sqmse_A], "std_sq_mse": [std_sqmse_A]})
df_results


# # **Part B**

# **B. Normalize the data** 
# 
# Repeat Part A but use a normalized version of the data. Recall that one way to normalize the data is by subtracting the mean from the individual predictors and dividing by the standard deviation.

# # B. Normalize the data

# In[ ]:


from sklearn import preprocessing


# In[ ]:


X = preprocessing.normalize(X)
y = preprocessing.normalize(y, axis = 0)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # B. Model

# In[ ]:


X_train.shape


# In[ ]:


model = regression_model()


# In[ ]:


model.fit(X_train, y_train, epochs=50)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


result = np.sqrt(mean_squared_error(y_test,y_pred))
result


# In[ ]:


model.summary()


# # B. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

# In[ ]:


MSE_List = []
for i in range(50):
    #1-Split Data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = regression_model()
    #2-Train:
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    #Prediction:
    y_pred = model.predict(X_test)
    
    #3-Evaluate_Model:
    result = np.sqrt(mean_squared_error(y_test,y_pred))
    print("{}: sqrt(mse) = {}".format(i+1,result))
    MSE_List.append(result)
    print("***_________________________________***\n\n\n")


# In[ ]:


MSE_List


# # B. Results
# **Report mean square MSE and std square MSE**

# In[ ]:


# Calculate the mean and the standard deviation of the metric on the 50 samplings
mean_sqmse_B = np.mean(MSE_List)
std_sqmse_B  = np.std(MSE_List)

# Generate a data frame to store the results of the differents parts of this project
df_results = pd.DataFrame.from_dict({"Part": ["B"],"mean_sq_mse": [mean_sqmse_B], "std_sq_mse": [std_sqmse_B]})
df_results


# # B.Q: How does the mean of the mean squared errors compare to that from Step A?
# 
# ### Because of normalized data, mean_sq_mse and std_sq_mse have both become very small.

# # Part C
# 
# **C. Increate the number of epochs**
# 
# Repeat Part B but use 100 epochs this time for training.

# # C.Model

# In[ ]:


model = regression_model()


# In[ ]:


model.fit(X_train, y_train, epochs=100)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


result = np.sqrt(mean_squared_error(y_test,y_pred))
result


# In[ ]:


model.summary()


# # C. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

# In[ ]:


MSE_List = []
for i in range(50):
    #1-Split Data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = regression_model()
    #2-Train:
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    #Prediction:
    y_pred = model.predict(X_test)
    
    #3-Evaluate_Model:
    result = np.sqrt(mean_squared_error(y_test,y_pred))
    print("{}: sqrt(mse) = {}".format(i+1,result))
    MSE_List.append(result)
    print("***_________________________________***\n\n\n")


# In[ ]:


MSE_List


# # C. Results
# **Report mean square MSE and std square MSE**

# In[ ]:


# Calculate the mean and the standard deviation of the metric on the 50 samplings
mean_sqmse_C = np.mean(MSE_List)
std_sqmse_C  = np.std(MSE_List)

# Generate a data frame to store the results of the differents parts of this project
df_results = pd.DataFrame.from_dict({"Part": ["C"],"mean_sq_mse": [mean_sqmse_C], "std_sq_mse": [std_sqmse_C]})
df_results


# # C.Q: How does the mean of the mean squared errors compare to that from Step B?
# ### We had a decrease in mean_sq_mse, but not much, and std_sq_mse also increased slightly.

# # Part D
# **D. Increase the number of hidden layers**
# 
# Repeat part B but use a neural network with the following instead:
# 
# - Three hidden layers, each of 10 nodes and ReLU activation function.

# # D. Model

# In[ ]:


def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=8, activation='relu'))
    model.add(Dense(10, input_dim=8, activation='relu'))
    model.add(Dense(10, input_dim=8, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[ ]:


model = regression_model()


# In[ ]:


model.fit(X_train, y_train, epochs=50)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


result = np.sqrt(mean_squared_error(y_test,y_pred))
result


# In[ ]:


model.summary()


# # D. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

# In[ ]:


MSE_List = []
for i in range(50):
    #1-Split Data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = regression_model()
    #2-Train:
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    #Prediction:
    y_pred = model.predict(X_test)
    
    #3-Evaluate_Model:
    result = np.sqrt(mean_squared_error(y_test,y_pred))
    print("{}: sqrt(mse) = {}".format(i+1,result))
    MSE_List.append(result)
    print("***_________________________________***\n\n\n")


# In[ ]:


MSE_List


# # D. Result

# In[ ]:


# Calculate the mean and the standard deviation of the metric on the 50 samplings
mean_sqmse_D = np.mean(MSE_List)
std_sqmse_D  = np.std(MSE_List)

# Generate a data frame to store the results of the differents parts of this project
df_results = pd.DataFrame.from_dict({"Part": ["D"],"mean_sq_mse": [mean_sqmse_D], "std_sq_mse": [std_sqmse_D]})
df_results


# # **D.Q: How does the mean of the mean squared errors compare to that from Step B?**
# ### Both mean_sq_mse and std_sq_mse are reduced.

# # **All Results**

# In[ ]:


Results = pd.DataFrame.from_dict({"Part": ["A","B", "C", "D"],
                                  "mean_sq_mse": [mean_sqmse_A, mean_sqmse_B, mean_sqmse_C, mean_sqmse_D], 
                                  "std_sq_mse": [std_sqmse_A, std_sqmse_B, std_sqmse_C, std_sqmse_D]})
Results

