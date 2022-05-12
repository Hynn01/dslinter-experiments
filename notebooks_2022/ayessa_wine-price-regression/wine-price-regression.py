#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#FFB875'>|</span> Introduction</b>
# ![](https://i.pinimg.com/originals/3b/f6/08/3bf608b1f755d2ef1307ad913a9b58d5.jpg)
# 
# ### What to Expect?
# In this notebook, I will explore the [Spanish Wine Quality Dataset](https://www.kaggle.com/datasets/fedesoriano/spanish-wine-quality-dataset) and fit a regression model on the price column. I will use Scikit-Learn, Pandas, Numpy, Seaborn and Matplotlib.pyplot in this notebook
# 
# ### Dataset Description
# #### Context
# This dataset is related to red variants of spanish wines. The dataset describes several popularity and description metrics their effect on it's quality. The datasets can be used for classification or regression tasks. The classes are ordered and not balanced (i.e. the quality goes from almost 5 to 4 points). The task is to predict either the quality of wine or the prices using the given data.
# 
# #### Content
# The dataset contains 7500 different types of red wines from Spain with 11 features that describe their price, rating, and even some flavor description. 
# 
# #### Attribute Information
# * **winery**: Winery name
# * **wine**: Name of the wine
# * **year**: Year in which the grapes were harvested
# * **rating**: Average rating given to the wine by the users *[from 1-5]*
# * **num_reviews**: Number of users that reviewed the wine
# * **country**: Country of origin *[Spain]*
# * **region**: Region of the wine
# * **price**: Price in euros *[€]*
# * **type**: Wine variety
# * **body**: Body score, defined as the richness and weight of the wine in your mouth *[from 1-5]*
# * **acidity**: Acidity score, defined as wine's “pucker” or tartness; it's what makes a wine refreshing and your tongue salivate and want another sip *[from 1-5]*

# # <b>2 <span style='color:#FFB875'>|</span> Preparing the Data</b>

# In[ ]:


# Importing necessary libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv('../input/spanish-wine-quality-dataset/wines_SPA.csv')
df.head()


# #### First lets clean our data and do data validation

# In[ ]:


df.isnull().sum()


# There are some null values on our dataset so lets drop them

# In[ ]:


df = df.dropna()


# In[ ]:


df.shape


# Lets now see some simple statistical information of our numerical columns and lets also see the datasets info

# In[ ]:


df.describe()


# In[ ]:


df.info()


# I have also noticed that there is a string value that was stopping me to convert the year column into numerical datatype, and it was N.V. value. So I'm not sure what it meant so lets just drop them also since there not alot of 'N.V.' in the column anyways.

# In[ ]:


df['year'] = df['year'].replace('N.V.', np.NaN)
df = df.dropna()
df['year'] = df['year'].astype(np.int64)


# Additionally, the column country only has one value so it would not be helpful at all for our model so lets remove the country column.

# In[ ]:


df = df.drop(columns=['country'])
df.head()


# I think we are now ready for our next step

# # <b>3 <span style='color:#FFB875'>|</span> Exploratory Data Anaysis</b>

# In[ ]:


sns.heatmap(df.corr(), annot=True, cmap='Blues')


# Oooh seems like most of our numerical variables does not have much of a correlation on the price column except for the rating that has a weak to moderate positive correlation. The price and rating column has a positive correlation which means that when the rating is high, its more likely that the price is also high, which make sense (but not in all cases).

# #### Does the type of the wine affects the wines price?

# In[ ]:


fig, ax = plt.subplots(ncols=1, figsize=(18,7))
sns.boxplot(y='price', x='type', data=df, ax=ax)
plt.xticks(rotation=90)
plt.show()


# The boxplots are very close to each other so its quite hard to make a inference, but based on the boxplot above, the type of wine as a little to no relationship on the wines prices

# # <b>4 <span style='color:#FFB875'>|</span> Preprocessing the Data</b>
# #### Label Encoding Categorical Columns

# In[ ]:


print('Categorical columns: ')
for col in df.columns:
    if df[col].dtype == 'object':
        print(str(col))
        label = LabelEncoder()
        label = label.fit(df[col])
        df[col] = label.transform(df[col].astype(str))


# #### Standarization

# In[ ]:


df = (df-df.mean())/df.std()
df.head()


# #### Splitting the data using train test split

# In[ ]:


X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# # <b>5 <span style='color:#FFB875'>|</span> Training</b>
# #### Lets import models for regression

# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# #### Lets make a function that will train every regression model and choose which has the highest r2 score

# In[ ]:


models = {}
def train_validate_predict(regressor, x_train, y_train, x_test, y_test, index):
    model = regressor
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    models[index] = r2


# In[ ]:


model_list = [LinearRegression, Lasso, Ridge, BayesianRidge, DecisionTreeRegressor, LinearSVR, KNeighborsRegressor,
              RandomForestRegressor]
model_names = ['Linear Regression', 'Lasso', 'Ridge', 'Bayesian Ridge', 'Decision Tree Regressor', 'Linear SVR', 
               'KNeighbors Regressor', 'Random Forest Regressor']

index = 0
for regressor in model_list:
    train_validate_predict(regressor(), X_train, y_train, X_test, y_test, model_names[index])
    index+=1


# In[ ]:


models


# Here we can see that KNeighbors Regressor had the highest r2 score (0.6622171034392672), so lets use that model!

# # <b>6 <span style='color:#FFB875'>|</span> Evaluating</b>

# In[ ]:


model = KNeighborsRegressor()
model.fit(X_train, y_train)
    
y_pred = model.predict(X_test)
preds = pd.DataFrame({'y_pred': y_pred, 'y_test':y_test})
preds = preds.sort_values(by='y_test')
preds = preds.reset_index()


# #### Now lets visualize our models predictions

# In[ ]:


plt.figure(figsize=(15, 5))
plt.plot(preds['y_pred'], label='pred')
plt.plot(preds['y_test'], label='actual')
plt.legend()
plt.show()


# Oh look! The model does it job but not particularly good nor bad. But personally its kinda predictable that our model would do bad as how we saw that most columns has a very little to no relationship toward to the wines prices.
# 
# Surprisingly though, our model did alright at predicting low prices wines but did terrible at high prices wines, I think what caused this from happening according to our EDA earlier, that in our dataset, theres way more data on low prices wines but theres a little data from the high price wines.

# # <b>7 <span style='color:#FFB875'>|</span> Authors Message</b> 
# * If you find this helpful, I would really appreciate the upvote!
# * If you see something wrong please let me know.
# * And lastly Im happy to hear your thoughts about the notebook for me to also improve!
