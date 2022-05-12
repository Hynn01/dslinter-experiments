#!/usr/bin/env python
# coding: utf-8

# # Dataset

# In[ ]:


import pandas as pd
df_insurance = pd.read_csv('../input/ushealthinsurancedataset/insurance.csv')
df_insurance


# In[ ]:


# Check missing value
df_insurance.isnull().sum()


# # Data Preprocessing

# In[ ]:


# Change string into numeric data value

df_insurance['sex'] = df_insurance['sex'].apply({'male':0,'female':1}.get) 
df_insurance['smoker'] = df_insurance['smoker'].apply({'yes':1, 'no':0}.get)
df_insurance['region'] = df_insurance['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[ ]:


df_insurance


# In[ ]:


# Data correlation

import matplotlib.pylab as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df_insurance.corr(),annot=True)
plt.show()


# Age to Charges = 0.3
# 
# BMI to Charges = 0.2
# 
# Smoker to Charges = 0.79
# 
# From those value we see Age, BMI and Smoker have a good correlation with Charges.

# # Spliting Train & Test Data

# In[ ]:


# Make new variable 
x = df_insurance[['age','bmi', 'smoker']]
y = df_insurance[['charges']]


# In[ ]:


# Split 50% with test_size=0.5
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
print(x.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# (1338, 3) is total original with 3 column (age, bmi, smoker)
# 
# (1338, 1) is total original with 1 column (charges)
# 
# (669, 3) is total train or test data with 3 column (age, bmi, smoker)
# 
# (669, 1) is total train or test data with 1 column (charges)

# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Accuracy: {r2_score(y_test, y_pred)}')


# # Predicting

# In[ ]:


import numpy as np
tes_x = np.array([24,20.1,0]).reshape(1,3)
print(model.predict(tes_x)[0,0])


# Age : 24, BMI : 20.1, Smoker : 0
# 
# We got Charges $ 740
# 
# do not worry with the warning

# In[ ]:


# Comparasion Between Real Insurance Price and Prediction Price
y_pred = model.predict(X_test)

plt.figure(figsize=(20,8))
plt.plot(np.arange(len(y_test)), y_test, label='Real')
plt.plot(np.arange(len(y_test)), y_pred, label='Prediction')
plt.title('Insurance Prediction Price vs Real Price')
plt.ylabel('Harga ($)')
plt.grid(True)
plt.legend()
plt.show()

