#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/kc-housesales-data/kc_house_data.csv')
df


# In[ ]:


house_price = df["price"].values
lot_sqft = df["sqft_lot15"].values
house_price
lot_sqft


# In[ ]:


plt.scatter(house_price, lot_sqft)


# In[ ]:


df.dropna


# In[ ]:


lot_sqft_vector = lot_sqft.reshape(-1,1)
lot_sqft_vector


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(lot_sqft_vector, house_price, train_size=.8, test_size=.2)
print(f"X_train shape {x_train.shape}")
print(f"y_train shape {y_train.shape}")
print(f"X_test shape {x_test.shape}")
print(f"y_test shape {y_test.shape}")


# In[ ]:


plt.scatter(x_train,y_train,color='red')
plt.xlabel('House Price')
plt.ylabel('Lot Size')
plt.title('Training data')
plt.show()


# In[ ]:


plt.scatter(x_test,y_test,color='blue')
plt.xlabel('Lot Size')
plt.ylabel('House Price')
plt.title('Testing data')
plt.show()


# In[ ]:


lm = LinearRegression()
lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)
print(f"Train accuracy {round(lm.score(x_train,y_train)*100,2)} %")
print(f"Test accuracy {round(lm.score(x_test,y_test)*100,2)} %")


# In[ ]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_test,y_predict)
plt.xlabel("Lot Size")
plt.ylabel("House Price")
plt.title("Trained model plot")
plt.plot


# In[ ]:


sqft_size = 5000
value_predict = lm.predict([[sqft_size]])[0]
print(f"Price of house with lot size {sqft_size} will be ${int(value_predict)}")


# # Conclusion
# 1. House prices of KC increases with lot size
# 2. House prices of KC lowers when lot size decreases
# 3. House prices seems to be clustered(not a lot of variance) up to 2000 sqft
