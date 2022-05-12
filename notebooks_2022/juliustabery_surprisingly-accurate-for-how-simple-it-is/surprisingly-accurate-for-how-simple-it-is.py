#!/usr/bin/env python
# coding: utf-8

# # Surprisingly Accurate for How Simple It Is
# Just taking a few averages and doing some linear algebra gets us to about 81% accuracy!

# #### Import modules and data, split into train and validation

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train = pd.read_csv("../input/digit-recognizer/train.csv").to_numpy()
X = train[:, 1:]
y = train[:, 0]
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)


# #### Find the "average" version of each digit

# In[ ]:


avg_digits = np.zeros((10, 784))
for digit in range(10):
    avg_digit = np.mean(X_train[y_train==digit], axis=0)
    avg_digits[digit] = avg_digit / np.linalg.norm(avg_digit)
    
# Plot the "average" digits
for digit in range(10):
    plt.subplot(2, 5, digit + 1)
    plt.imshow(avg_digits[digit].reshape((28, 28)), cmap='gray')


# #### Take the inner product with each digit, predict whichever has the highest score!

# In[ ]:


similarity_matrix = avg_digits @ X_val.T
predictions = similarity_matrix.argmax(axis=0)


# #### Results:

# In[ ]:


accuracy = np.mean(y_val == predictions)
print(f"Accuracy: {100 * accuracy : .1f}%")

