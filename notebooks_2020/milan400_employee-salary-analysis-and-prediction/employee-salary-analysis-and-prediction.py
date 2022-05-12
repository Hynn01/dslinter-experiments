#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bubbly')

# for some basic operations
import numpy as np 
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from bubbly.bubbly import bubbleplot

# for providing the path
import os
print(os.listdir("../input"))

import seaborn as sns
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
       # print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Reading Dataset

# In[ ]:


employee_salary = pd.read_csv('/kaggle/input/employee-salary-dataset/employee_data.csv')
employee_salary = employee_salary.drop(columns=['Unnamed: 0','id'])
employee_salary = employee_salary.astype({'groups':str, 'age':int, 'healthy_eating':int,'active_lifestyle':int, 'salary':int})
employee_salary.head()


# # Correlations between Data

# * **Positive correlation** is a relationship between two variables in which both variables move in tandem—that is, in the same direction. A positive correlation exists when one variable decreases as the other variable decreases, or one variable increases while the other increases.
# 
# * **Negative correlation** is a relationship between two variables in which one variable increases as the other decreases, and vice versa. In statistics, a perfect negative correlation is represented by the value -1, a 0 indicates no correlation, and a +1 indicates a perfect positive correlation.

# In[ ]:


# Employee salary vs age

plt.rcParams['figure.figsize'] = (15, 12)
sns.heatmap(employee_salary.corr(), cmap='gray', annot=True)
plt.show()


# * Salary is positively correlated with healthy eating heavily
# * Age and Active lifestyle are negatively correlated with Salary

# # Bubble Charts

# **Healthy Eating VS Salary VS Age**

# In[ ]:


fig = px.scatter(employee_salary, x='salary', y='healthy_eating',
                size='salary', color='age',
                hover_name = 'age', log_x=True, size_max=60)

fig.show()


# **Healthy Eating VS Salary VS Groups**

# In[ ]:


import plotly.express as px

fig = px.scatter(employee_salary, x= 'salary', y='healthy_eating',
                size='salary', color='groups',
                hover_name = 'groups', log_x=True, size_max=60)

fig.show()


# **Healthy Eating VS Salary VS Active lifestyle**

# In[ ]:


import plotly.express as px

fig = px.scatter(employee_salary, x= 'salary', y='healthy_eating',
                size='salary', color='active_lifestyle',
                hover_name = 'active_lifestyle', log_x=True, size_max=60)

fig.show()


# Bubble plot to depict the relation between Healthy Eating VS Salary and size of the bubbles is represented by the Salary and the color of the bubbles is represented by the Active lifestyle
# 
# * As the Healthy Eating score increses Salary increases
# * People with blood group 'B' are more in number
# * People with less Salary has more Active LifeStyle

# # Boxplot Charts

# In[ ]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Box(y = employee_salary['salary'], name='Salary'))
fig.show()


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Box(y = employee_salary['age'], name='Age'))
fig.show()


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Box(y = employee_salary['groups'], name='Groups'))
fig.add_trace(go.Box(y = employee_salary['healthy_eating'], name='Healthy Eating'))
fig.add_trace(go.Box(y = employee_salary['active_lifestyle'], name='Active lifestyle'))

fig.show()


# **Boxplots are a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”).**
# 
# * median (Q2/50th Percentile): the middle value of the dataset.
# 
# * first quartile (Q1/25th Percentile): the middle number between the smallest number (not the “minimum”) and the median of the dataset.
# 
# * third quartile (Q3/75th Percentile): the middle value between the median and the highest value (not the “maximum”) of the dataset.
# 
# * interquartile range (IQR): 25th to the 75th percentile.
# 
# * outliers (shown as green circles)
# 
# * “maximum”: Q3 + 1.5*IQR
# 
# * “minimum”: Q1 -1.5*IQR

# # Data Preprocessing

# In[ ]:


X = employee_salary.drop(columns=['salary'])
Y = employee_salary['salary']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#make the x for train and test (also called validation data) 
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.8,random_state=42)


# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()

xtrain['groups'] = label_encoder.fit_transform(xtrain['groups'])
xtest['groups'] = label_encoder.transform(xtest['groups'])


# # Model Creation and Training

# In[ ]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(xtrain, ytrain)


accuracy = reg.score(xtest, ytest)*100
print("Accuracy: {}".format(accuracy))
y_predict = reg.predict(xtest)


# # Distribution Plot of Test and Predict Value 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

hist1 = ytest
hist2 = y_predict

from scipy.stats import norm

sns.distplot(hist1,color="b",hist=False, label='True Value')
sns.distplot(hist2,hist=False,color='red', label='Predicted Value')


# In[ ]:


final_testwithpredict = pd.DataFrame()

final_testwithpredict = xtest
final_testwithpredict['Actual'] = ytest
final_testwithpredict['Predict'] = y_predict.astype(int)

final_testwithpredict['groups'] = label_encoder.inverse_transform(final_testwithpredict['groups'])


# # Scatter Plot: Actual Salary VS Predict Salary

# In[ ]:


fig = px.scatter(final_testwithpredict, x='Actual', y='Predict', color='groups')
fig.show()


# # Scatter 3D Plot: Actual Salary VS Predict Salary VS Groups

# In[ ]:


fig = px.scatter_3d(final_testwithpredict, x='Actual', y='Predict',z='groups', color='groups')
fig.show()

