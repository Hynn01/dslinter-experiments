#!/usr/bin/env python
# coding: utf-8

# ## Medical Insurance Cost Prediction
# In this Notebook, We are using a dataset called "Medical Personal Cost dataset" from kaggle, in which we have to predict the insurance cost using regression models.
# First, We have done Exploratory Data Analysis on the data to know more about the data and can infer some useful insights out of the data. Later, We have also used regression models like Linear Regression, Random Forest, XGBoost Regressor, Gradient Boosting Regressor etc on our data to understand which model work best for predicting the medical insurance cost. 

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("../input/insurance/insurance.csv")
df.head()


# In[ ]:


df.isnull().sum()


# So as there are no null values in the data, there in no need to worry about any missing values. Now we will use Label Encoder to encode all the categorical features.

# In[ ]:


df.info()


# In[ ]:


df.shape


# So, We have 1338 rows and 7 columns in our dataset.

# In[ ]:


df.describe()


# We observe that the range of values are different and there are categorical variable such as gender, smoker and region present.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
#sex
le = LabelEncoder()
le.fit(df.sex.drop_duplicates()) 
df.sex = le.transform(df.sex)
# smoker or not
le.fit(df.smoker.drop_duplicates()) 
df.smoker = le.transform(df.smoker)
#region
le.fit(df.region.drop_duplicates()) 
df.region = le.transform(df.region)


# In[ ]:


df.corr()['charges'].sort_values()


# Here, we calculated the correlation of all the features with our target feature which is "charges". We can understand it better with help of some kind of visualization. For that we will import matplotlib and seaborn, which are famous libraries used for data visualization.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


print(df.corr()) #We will visualize this data with help of seaborm heatmap.


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 7))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), 
            cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)


# A strong correlation is observed only with the fact of smoking the patient. Let's explore more about it.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
fig= plt.figure(figsize=(12,5))

ax=fig.add_subplot(121)
sns.distplot(df[(df.smoker == 1)]["charges"],color='c',ax=ax)
ax.set_title('Distribution of charges for smokers')

ax=fig.add_subplot(122)
sns.distplot(df[(df.smoker == 0)]['charges'],color='b',ax=ax)
ax.set_title('Distribution of charges for non-smokers')


# Smoking patients spend more on treatment. Let's check more on how many smokers and how many non-smokers are there in both genders.

# In[ ]:


sns.catplot(x="smoker", kind="count",hue = 'sex', palette="BuPu", data=df);


# Here, women are coded with the symbol " 1 "and men "0". Thus non-smoking people are more in number. Also we can notice that there are more male smokers than women smokers. 

# In[ ]:


sns.catplot(x="sex", y="charges", hue="smoker",
            kind="violin", data=df, palette = 'magma');


# Let's make a boxplot between smoker and charges for both the genders to understand the relationship better.

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Box plot for charges of women")
sns.boxplot(y="smoker", x="charges", data =  df[(df.sex == 1)] , orient="h", palette = 'magma');


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Box plot for charges of men")
sns.boxplot(y="smoker", x="charges", data =  df[(df.sex == 0)] , orient="h", palette = 'rainbow');


# Now let's pay attention to the age of the patients. First, let's look at how age affects the cost of treatment, and also look at patients of what age more in our data set.

# In[ ]:


import warnings
warnings.filterwarnings('ignore') 
#To ignore warnings so that our notebook is more presentable

plt.figure(figsize=(12,5))
plt.title("Distribution of age")
ax = sns.distplot(df["age"], color = 'violet')


# In[ ]:


sns.lmplot(x="age", y="charges", hue="smoker", data=df, palette = 'inferno_r', size = 6)
ax.set_title('Smokers and non-smokers');


# In non-smokers as well as smokers, we can see that the cost of treatment increases with age.

# ### Let's pay attention to BMI and its relationship with medical charges.

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of bmi")
ax = sns.distplot(df["bmi"], color = 'm')


# So we got a bell curve when we plotted the distribution of BMI in our data. We see that around 30 is our average BMI. Let's see medical charges based on BMI > 30 and BMI < 30.

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of charges for patients with BMI greater than 30")
ax = sns.distplot(df[(df.bmi >= 30)]['charges'], color = 'm')


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of charges for patients with BMI less than 30")
ax = sns.distplot(df[(df.bmi < 30)]['charges'], color = 'b')


# It is evident from the two graphs that patients with BMI above 30 spend more on treatment!

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.scatterplot(x='bmi',y='charges',data=df,palette='magma',hue='smoker')
ax.set_title('Scatter plot of charges and bmi');


# As we can see from the scatter plot, With the increase in BMI we see increase in medical charges. We also notice how being a smoker and being a non smoker affects the graph.

# Now, let's pay attention to the number of children, patients have, in our data and their relation with our target feature

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(data = df,x='children')
plt.title('Number of children')
plt.show()


# Most patients do not have children while on the other hand some even have 5 children. Let's see if the factor that patients have children affects the smoking.(Do people with 5 children smoke less?) 

# In[ ]:


sns.catplot(x="smoker", kind="count", palette="rainbow",hue = "sex",
            data=df[(df.children > 0)], size = 6)
ax.set_title('Smokers and non-smokers who have childrens');


# We see that patients who have children mostly don't smoke. This is seen in both of the genders.(This make absolute sense as, in general, people do become health consious if they have children)

# In[ ]:


type_value_count = df['region'].value_counts(normalize=True)*100
plt.figure(figsize=(10,6))
plt.pie(type_value_count,labels=['South East','North West','South West','North East'],autopct='%1.2f%%')
plt.title('Regions')
plt.show();


# ### Now we will predict Insurance with help of different Regression Models

# In[ ]:


#importing all the necessary libraries

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[ ]:


#We will read the dataset again
data1 = pd.read_csv("../input/insurance/insurance.csv")


# In[ ]:


data1.head()


# In[ ]:


data1.dtypes


# In[ ]:


data1.head()


# In[ ]:


data1['sex'] = pd.factorize(data1['sex'])[0] + 1
data1['region'] = pd.factorize(data1['region'])[0] + 1
data1['smoker'] = pd.factorize(data1['smoker'])[0] + 1


# In[ ]:


X = data1.drop('charges', axis = 1)
y = data1['charges']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


scaler= StandardScaler()
scaler.fit(X_train)
X_train_scaled= scaler.transform(X_train)
X_test_scaled= scaler.transform(X_test)


# ### Linear Regression

# In[ ]:


linear_reg_model= LinearRegression()
linear_reg_model.fit(X_train_scaled, y_train)


# In[ ]:


y_pred = linear_reg_model.predict(X_test_scaled)
y_pred = pd.DataFrame(y_pred)
MAE_li_reg= metrics.mean_absolute_error(y_test, y_pred)
MSE_li_reg = metrics.mean_squared_error(y_test, y_pred)
RMSE_li_reg =np.sqrt(MSE_li_reg)
pd.DataFrame([MAE_li_reg, MSE_li_reg, RMSE_li_reg], 
             index=['MAE_li_reg', 'MSE_li_reg', 'RMSE_li_reg'], 
             columns=['Metrics'])


# Here, we are comparing the original values of our target variable with the predicted value and then with help of these metrics we see how good or bad our model performed.

# In[ ]:


plt.scatter(x=y_pred,y=y_test);
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Predicted Values vs Real Values using Linear Regression");


# In[ ]:


scores = cross_val_score(linear_reg_model, X_train_scaled, y_train, cv=5)
print(np.sqrt(scores))


# In[ ]:


r2_score(y_test, linear_reg_model.predict(X_test_scaled))


# In[ ]:


linear_reg_model.coef_


# In[ ]:


pd.DataFrame(linear_reg_model.coef_, 
             X.columns, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)

# The coefficient for a term represents the change in 
# the mean response associated with a change in that term, 
# while the other terms in the model are held constant. 
# The sign of the coefficient indicates the direction of the 
# relationship between the term and the response.


# ### Gradient Boosting Regressor Model

# In[ ]:


Gradient_model = GradientBoostingRegressor()
Gradient_model.fit(X_train_scaled, y_train)


# In[ ]:


y_pred = Gradient_model.predict(X_test_scaled)
y_pred = pd.DataFrame(y_pred)
MAE_gradient= metrics.mean_absolute_error(y_test, y_pred)
MSE_gradient = metrics.mean_squared_error(y_test, y_pred)
RMSE_gradient =np.sqrt(MSE_gradient)
pd.DataFrame([MAE_gradient, MSE_gradient, RMSE_gradient], 
             index=['MAE_gradient', 'MSE_gradient', 'RMSE_gradient'], 
             columns=['Metrics'])


# In[ ]:


scores = cross_val_score(Gradient_model, X_train_scaled, y_train, cv=5)
print(np.sqrt(scores))


# In[ ]:


r2_score(y_test, Gradient_model.predict(X_test_scaled))


# In[ ]:


plt.scatter(x=y_pred,y=y_test);
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Predicted Values vs Real Values using Gradient Boosting Regressor Model");


# ### XGB Regressor Model

# In[ ]:


XGB_model =XGBRegressor()
XGB_model.fit(X_train_scaled, y_train);


# In[ ]:


y_pred = XGB_model.predict(X_test_scaled)
y_pred = pd.DataFrame(y_pred)
MAE_XGB= metrics.mean_absolute_error(y_test, y_pred)
MSE_XGB = metrics.mean_squared_error(y_test, y_pred)
RMSE_XGB =np.sqrt(MSE_XGB)
pd.DataFrame([MAE_XGB, MSE_XGB, RMSE_XGB], index=['MAE_XGB', 'MSE_XGB', 'RMSE_XGB'], 
             columns=['Metrics'])


# In[ ]:


scores = cross_val_score(XGB_model, X_train_scaled, y_train, cv=5)
print(np.sqrt(scores))


# In[ ]:


r2_score(y_test, XGB_model.predict(X_test_scaled))


# In[ ]:


plt.scatter(x=y_pred,y=y_test);
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Predicted Values vs Real Values using XGB Regressor Model");


# ### Random Forest Regressor Model

# In[ ]:


forest_reg_model =RandomForestRegressor()
forest_reg_model.fit(X_train_scaled, y_train);


# In[ ]:


y_pred = forest_reg_model.predict(X_test_scaled)
y_pred = pd.DataFrame(y_pred)
MAE_forest_reg= metrics.mean_absolute_error(y_test, y_pred)
MSE_forest_reg = metrics.mean_squared_error(y_test, y_pred)
RMSE_forest_reg =np.sqrt(MSE_forest_reg)
pd.DataFrame([MAE_forest_reg, MSE_forest_reg, RMSE_forest_reg], 
             index=['MAE_forest_reg', 'MSE_forest_reg', 'RMSE_forest_reg'], 
             columns=['Metrics'])


# In[ ]:


scores = cross_val_score(forest_reg_model, X_train_scaled, y_train, cv=5)
print(np.sqrt(scores))


# In[ ]:


r2_score(y_test, forest_reg_model.predict(X_test_scaled))


# In[ ]:


plt.scatter(x=y_pred,y=y_test);
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.title("Predicted Values vs Real Values using Random Forest Regressor Model");


# #### Gradient boosting model worked the best for our dataset.

# In[ ]:




