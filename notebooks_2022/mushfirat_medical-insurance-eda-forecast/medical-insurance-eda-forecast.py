#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#ea4335'>|</span> Setup</b>

# ## Importing Libraries
# - **For ML Models**: sklearn  
# - **For Data Processing**: numpy, pandas, sklearn  
# - **For Data Visualization**: matplotlib, seaborn, plotly  

# In[ ]:


# For ML models
from sklearn.linear_model import LinearRegression, BayesianRidge, TweedieRegressor, LassoLars
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

# For Data Processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Miscellaneous
import os
import random
import math


# ## Setting up sklearnex to speed up training
# If you don't know about sklearnex, this is a tool you can use to speed up training sklearn models, without having to change any code.  
# A simple 2 line of code can speed up training by 2x.  
# You can follow [this kernel by Devlikamov Vlad](https://www.kaggle.com/code/lordozvlad/let-s-speed-up-your-kernels-using-sklearnex) to learn more about it

# In[ ]:


from sklearnex import patch_sklearn
patch_sklearn()


# # <b>2 <span style='color:#ea4335'>|</span> About the Dataset</b>

# In[ ]:


df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df


# ## Column Descriptions
# 
# - `age`: age of primary beneficiary
# - `sex`: insurance contractor gender, female, male
# - `bmi`: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# - `children`: Number of children covered by health insurance / Number of dependents
# - `smoker`: Smoking
# - `region`: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# - `charges`: Individual medical costs billed by health insurance
# 
# Categorical Features  
# `sex`, `smoker`, `region`  
# Continuous Features  
# `age`, `children`, `bmi`, `charges`

# ## Column Statistics (of numerical data)

# In[ ]:


df.describe()[1:][['age','children','bmi', 'charges']].T.style.background_gradient(cmap=sns.light_palette("#ea4335", as_cmap=True), axis=1)


# ## Column Statistics (of categorical data)

# In[ ]:


fig = make_subplots(
    rows=1, cols=3, subplot_titles=("sex", "smoker",
                                    "region"),
    specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}]],
)

colours = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']

fig.add_trace(go.Pie(labels=np.array(df['sex'].value_counts().index),
                     values=[x for x in df['sex'].value_counts()],
                     textinfo='label+percent', rotation=-45, hole=.35,
                     marker_colors=colours),
              row=1, col=1)

fig.add_trace(go.Pie(labels=np.array(df['smoker'].value_counts().index),
                     values=[x for x in df['smoker'].value_counts()],
                     textinfo='label+percent', hole=.35,
                     marker_colors=colours),
              row=1, col=2)

fig.add_trace(go.Pie(labels=np.array(df['region'].value_counts().index),
                     values=[x for x in df['region'].value_counts()],
                     textinfo='label+percent', rotation=-45, hole=.35,
                     marker_colors=colours),
              row=1, col=3)


fig.update_layout(height=450, font=dict(size=14), showlegend=False)

fig.show()


# # <b>3 <span style='color:#ea4335'>|</span> Exploratory Analysis</b>

# In[ ]:


fig = px.scatter(df, x="charges", y="age", color='smoker', color_continuous_scale='Blues', color_discrete_map={'yes':'#ea4335', 'no':'#4285f4'})
fig.update_layout(legend_title_text='Smoker')


# ### Insights
# - Individuals who smoke have a higher medical bill than individuals who do not smoke
# - We can see 3 categories of medical bill paid, 0-15k, 15k-32k, 32k-50k, but why?

# In[ ]:


fig = px.scatter(df, x="bmi", y="charges", color='age', color_continuous_scale='RdBu')
fig.show()


# ### Insights
# - Here, those 3 categories of charges appear again, with ranges: 0-15k, 15k-32k, 32k-50k  
# - It can be seen that those 3 categories of charges has a strong correlation with age
# - The correlation of charge and age is linear in each of the categories  
# - From this, we can determine that these 3 categories mean how bad the health condition of the patient is!

# In[ ]:


fig = go.Figure()
colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
for i,x in enumerate(df['region'].unique()):
    fig.add_trace(go.Box(
        x=df[df['region']==x]['bmi'],
        y=df[df['region']==x]['region'], name=x, marker_color=colors[i]
    ))

fig.update_layout(
    yaxis_title='region', xaxis_title='bmi'
)
fig.update_traces(orientation='h')
fig.update_layout(legend_title_text='region')
fig.show()


# ### Insights
# - Southeast-ern people have the highest bmi, a median bmi of 33.33
# - Northeast-ern people have the lowest bmi, a median bmi of 28.88

# In[ ]:


fig = px.histogram(df, x="bmi", color="sex", marginal='box', nbins=80, color_discrete_map = {'male':'#ea4335','female':'#4285f4'})
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()


# ### Insights
# - The bmi of a person is independent of their gender

# In[ ]:


fig = px.histogram(df, x="sex", color='smoker', color_discrete_map = {'yes':'#ea4335','no':'#4285f4'})
fig.show()


# ### Insights
# - Looks like people smoke regardless of their gender
# - More Males seem to be smokers than females, but the difference is minimal, also the number of females tested is slightly less than the number of males
# - It cannot be concluded that more males smoke because we do not have sufficient data

# In[ ]:


fig = px.histogram(df, x="children", color='smoker', barmode='group', color_discrete_map = {'yes':'#ea4335','no':'#4285f4'})
fig.show()


# ### Insights
# - Smokers usually have less children than non-smokers  
# 
# Nice to see that most parents do not smoke 😀

# # <b>4 <span style='color:#ea4335'>|</span> Data Cleaning & Preprocessing</b>

# <h2>4.1 <span style='color:#ea4335'>|</span> Encoding Categorical Features</h2>

# In[ ]:


print('\nCategorical Columns\n')
df.select_dtypes(include=['O']).nunique()


# `sex` and `smoker` have 2 unique values, and `region` have more than 2 unique values.  
# Here, I am converting the columns with 2 unique values to binary (either 1 or 0)  
# And one-hot encode the other categorical columns which has more than 2 unique values  

# In[ ]:


# Integer encode columns with 2 unique values
for col in ['sex', 'smoker']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
# One-hot encode columns with more than 2 unique values
df = pd.get_dummies(df, columns=['region'], prefix = ['region'])


# <h2>4.2 <span style='color:#ea4335'>|</span> Train-Val Split</h2>  

# In[ ]:


features = np.array(df[[col for col in df.columns if col!='charges']])
labels = np.array(df['charges'])

x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=0)


# # <b>5 <span style='color:#ea4335'>|</span> Models</b>

# **Metrics we will be using,**
# $$\text{Root Mean Squared Error, } \mathbf{RMSE} = \sqrt{\cfrac{\sum_{i=1}^N (y_i - \hat y_i)^2}{N}}$$
# 
# $$\text{Mean Absolute Percentage Error, } \mathbf{MAPE} = \cfrac{1}{N}\sum_{i=1}^N \Big|\cfrac{y_i - \hat y_i}{y_i}\Big|$$
# 
# $$\text{coefficient of determination, } \mathbf{R^2} = 1 - \cfrac{\sum_{i=1}^N (\hat y_i - y_i)^2}{\sum_{i=1}^N (y_i - \bar y)^2}$$
# 
# where,  
# $\hat y$ is the predicted variable, and $y$ is the target variable  
# $\bar y$ represents the mean of all values of $y$   
# $y_i$ = $i^\mathbf{th}$ sample of target variable $y$  
# $\hat y_i$ = $i^\mathbf{th}$ sample of predicted variable $\hat y$  
# $N$ = Number of training samples  

# In[ ]:


model_comparison = {} # We will use this to store the performance of different models on the validation dataset


# <h2>5.1 <span style='color:#ea4335'>|</span> RandomForestRegressor</h2>  
# 
# > A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the `max_samples` parameter if `bootstrap=True` (default), otherwise the whole dataset is used to build each tree.
# 
# Learn more about RandomForestRegressor at [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [wikipedia.org](https://en.wikipedia.org/wiki/Random_forest)

# In[ ]:


rf = RandomForestRegressor()

parameters = {'n_estimators': [160,180,200,220], 'max_depth':[16,18,20,22,24]}
clf = GridSearchCV(rf, parameters)
print("Searching for best hyperparameters ...")
clf.fit(x_train, y_train)
print(f'Best Hyperparameters: {clf.best_params_}')

y_pred = clf.predict(x_val)

rmse = math.sqrt(mean_squared_error(y_val,y_pred))
mape = mean_absolute_percentage_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print('\nRMSE:', rmse)
print('MAPE:', mape)
print('R2 Score:', r2)

model_comparison['RandomForestRegressor'] = [rmse, mape, r2]


# <h2>5.2 <span style='color:#ea4335'>|</span> LinearRegression</h2>  
# 
# > LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.  
# 
# Learn more about LinearRegression at [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [wikipedia.org](https://en.wikipedia.org/wiki/Linear_regression)

# In[ ]:


lr = LinearRegression().fit(x_train, y_train)

y_pred = lr.predict(x_val)

rmse = math.sqrt(mean_squared_error(y_val,y_pred))
mape = mean_absolute_percentage_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print('\nRMSE:', rmse)
print('MAPE:', mape)
print('R2 Score:', r2)

model_comparison['LinearRegression'] = [rmse, mape, r2]


# <h2>5.3 <span style='color:#ea4335'>|</span> DecisionTreeRegressor</h2>  
# 
# > Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.
# 
# Learn more about DecisionTreeRegressor at [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html), [wikipedia.org](https://en.wikipedia.org/wiki/Decision_tree_learning)

# In[ ]:


tree = DecisionTreeRegressor().fit(x_train, y_train)

y_pred = tree.predict(x_val)

rmse = math.sqrt(mean_squared_error(y_val,y_pred))
mape = mean_absolute_percentage_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print('\nRMSE:', rmse)
print('MAPE:', mape)
print('R2 Score:', r2)

model_comparison['DecisionTreeRegressor'] = [rmse, mape, r2]


# <h2>5.4 <span style='color:#ea4335'>|</span> BayesianRidge</h2>  
# 
# > Bayesian regression allows a natural mechanism to survive insufficient data or poorly distributed data by formulating linear regression using probability distributors rather than point estimates. The output or response 'y' is assumed to drawn from a probability distribution rather than estimated as a single value.
# 
# Learn more about BayesianRidge at [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html), [wikipedia.org](https://en.wikipedia.org/wiki/Bayesian_linear_regression)

# In[ ]:


br = BayesianRidge().fit(x_train, y_train)

y_pred = br.predict(x_val)

rmse = math.sqrt(mean_squared_error(y_val,y_pred))
mape = mean_absolute_percentage_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print('\nRMSE:', rmse)
print('MAPE:', mape)
print('R2 Score:', r2)

model_comparison['BayesianRidge'] = [rmse, mape, r2]


# <h2>5.5 <span style='color:#ea4335'>|</span> TweedieRegressor</h2>  
# 
# > Tweedie distribution is a special case of exponential dispersion models and is often used as a distribution for generalized linear models. It can have a cluster of data items at zero and this particular property makes it useful for modeling claims in the insurance industry.  
# 
# Learn more about TweedieRegressor at [wikipedia.org](https://en.wikipedia.org/wiki/Tweedie_distribution), [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html)

# In[ ]:


tr = TweedieRegressor().fit(x_train, y_train)

y_pred = tr.predict(x_val)

rmse = math.sqrt(mean_squared_error(y_val,y_pred))
mape = mean_absolute_percentage_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print('\nRMSE:', rmse)
print('MAPE:', mape)
print('R2 Score:', r2)

model_comparison['TweedieRegressor'] = [rmse, mape, r2]


# <h2>5.6 <span style='color:#ea4335'>|</span> LassoLars</h2>  
# 
# > LassoLars is a lasso model implemented using the LARS algorithm, and unlike the implementation based on coordinate descent, this yields the exact solution, which is piecewise linear as a function of the norm of its coefficients. Lasso model fit with Least Angle Regression a.k.a. Lars.
# 
# Learn more about LassoLars at [wikipedia.org](https://en.wikipedia.org/wiki/Least-angle_regression), [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html)

# In[ ]:


ll = LassoLars(alpha=.1, normalize=False).fit(x_train, y_train)

y_pred = ll.predict(x_val)

rmse = math.sqrt(mean_squared_error(y_val,y_pred))
mape = mean_absolute_percentage_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print('\nRMSE:', rmse)
print('MAPE:', mape)
print('R2 Score:', r2)

model_comparison['LassoLars'] = [rmse, mape, r2]


# <h2>5.7 <span style='color:#ea4335'>|</span> Model Comparison</h2>  

# In[ ]:


model_comparison_df = pd.DataFrame.from_dict(model_comparison).T
model_comparison_df.columns = ['MSE', 'MAPE', 'R2 Score']
model_comparison_df = model_comparison_df.sort_values('R2 Score', ascending=True)

model_comparison_df.style.background_gradient(cmap=sns.light_palette("#ea4335", as_cmap=True))


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='R2 Score', y=model_comparison_df.index, x=model_comparison_df['R2 Score'],
           orientation='h', marker_color=['#f5a19a', '#f28e86', '#f07b72', '#ee695d', '#ec5649', '#ea4335'])
])
fig.update_layout(barmode='group')
fig.show()


# ### Please Upvote this notebook as it encourages me in doing better.
# ![](http://68.media.tumblr.com/e1aed171ded2bd78cc8dc0e73b594eaf/tumblr_o17frv0cdu1u9u459o1_500.gif)
