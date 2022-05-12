#!/usr/bin/env python
# coding: utf-8

# ### Hello everyone! Welcome
# ### I have performed EDA and predicted the Life Expectancy according to the features given in the dataset. I wanted to both improve myself and make easier examples for those who are just at the beginning, like me.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


data.isna().sum()


# ### Using Interpolation method to deal with the null values

# In[ ]:


data1 = data.interpolate(method = 'linear', limit_direction = 'forward')


# In[ ]:


data1.isna().sum()


# In[ ]:


data1.columns = data1.columns.str.replace(' ','_')


# In[ ]:


data1.columns


# In[ ]:


data1 = data1.drop(['Country'], axis = 1)


# ### Changing the categories to binary values

# In[ ]:


data1.Status = data1.Status.map({'Developing':0, 'Developed': 1})


# In[ ]:


data1.head()


# In[ ]:


columns = {'Year':1,'Life_expectancy_':2,'Adult_Mortality':3,'infant_deaths':4,'Alcohol':5,'percentage_expenditure':6,
            'Hepatitis_B':7,'Measles_':8,'_BMI_':9,'under-five_deaths_':10,'Polio':11,'Total_expenditure':12,'Diphtheria_':13,'_HIV/AIDS':14,
            'GDP':15,'Population':16,'_thinness__1-19_years':17,'_thinness_5-9_years':18,'Income_composition_of_resources':19,'Schooling':20}


# In[ ]:


# let's see which columns have outliers
plt.figure(figsize = (20,30))
for var, i in columns.items():
    plt.subplot(5,4,i)
    plt.boxplot(data1[var], whis = 1.5)
    plt.title(var)
plt.show()


# ### Looks like we have a lot of outliers and need to deal with it!

# In[ ]:


data1.columns


# In[ ]:


# all columns with outliers - 
sns.boxplot(data1.Life_expectancy_)
plt.show()


# ### Let's use Cube root Transformation method to deal with outliers.

# In[ ]:


#Cube root transformation
plt.hist(data1['Life_expectancy_'])
plt.title('before transformation')
plt.show()
data1['Life_expectancy_'] = (data1['Life_expectancy_']**(1/3))
plt.hist(data1['Life_expectancy_'])
plt.title('after transformation')
plt.show()


# In[ ]:


sns.boxplot(data1.Life_expectancy_)
plt.show()


# ### This method does not eliminate the outliers completely but improves or reduces the outliers indeed.

# In[ ]:


# for Adult_Mortality
plt.hist(data1['Adult_Mortality'])
plt.title('before transf')
plt.show()
data1['Adult_Mortality'] = (data1.Adult_Mortality**(1/3))
plt.hist(data1['Adult_Mortality'])
plt.title('after transf')
plt.show()


# In[ ]:


sns.boxplot(data1['Adult_Mortality'])
plt.show()


# In[ ]:


plt.hist(data1['infant_deaths'])
plt.title('before transf')
plt.show()
data1['infant_deaths'] = (data1['infant_deaths']**(1/3))
plt.hist(data1['infant_deaths'])
plt.title('after transf')
plt.show()


# In[ ]:



plt.hist(data1['percentage_expenditure'])
plt.title('before transf')
plt.show()
data1['percentage_expenditure'] = (data1['percentage_expenditure']**(1/3))
plt.hist(data1['percentage_expenditure'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['Hepatitis_B'])
plt.title('before transf')
plt.show()
data1['Hepatitis_B'] = (data1['Hepatitis_B']**(1/3))
plt.hist(data1['Hepatitis_B'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['Measles_'])
plt.title('before transf')
plt.show()
data1['Measles_'] = (data1['Measles_']**(1/3))
plt.hist(data1['Measles_'])
plt.title('after transf')
plt.show()


# In[ ]:


sns.boxplot(data1['Measles_'])
plt.show()


# In[ ]:



plt.hist(data1['under-five_deaths_'])
plt.title('before transf')
plt.show()
data1['under-five_deaths_'] = (data1['under-five_deaths_']**(1/3))
plt.hist(data1['under-five_deaths_'])
plt.title('after transf')
plt.show()


# In[ ]:


sns.boxplot(data1['under-five_deaths_'])
plt.show()


# In[ ]:


plt.hist(data1['Polio'])
plt.title('before transf')
plt.show()
data1['Polio'] = (data1['Polio']**(1/3))
plt.hist(data1['Polio'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['Total_expenditure'])
plt.title('before transf')
plt.show()
data1['Total_expenditure'] = (data1['Total_expenditure']**(1/3))
plt.hist(data1['Total_expenditure'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['Diphtheria_'])
plt.title('before transf')
plt.show()
data1['Diphtheria_'] = (data1['Diphtheria_']**(1/3))
plt.hist(data1['Diphtheria_'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['_HIV/AIDS'])
plt.title('before transf')
plt.show()
data1['_HIV/AIDS'] = (data1['_HIV/AIDS']**(1/3))
plt.hist(data1['_HIV/AIDS'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['GDP'])
plt.title('before transf')
plt.show()
data1['GDP'] = (data1['GDP']**(1/3))
plt.hist(data1['GDP'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['Population'])
plt.title('before transf')
plt.show()
data1['Population'] = (data1['Population']**(1/3))
plt.hist(data1['Population'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['_thinness__1-19_years'])
plt.title('before transf')
plt.show()
data1['_thinness__1-19_years'] = (data1['_thinness__1-19_years']**(1/3))
plt.hist(data1['_thinness__1-19_years'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.hist(data1['_thinness_5-9_years'])
plt.title('before transf')
plt.show()
data1['_thinness_5-9_years'] = (data1['_thinness_5-9_years']**(1/3))
plt.hist(data1['_thinness_5-9_years'])
plt.title('after transf')
plt.show()


# In[ ]:


plt.figure(figsize = (20,30))
for var, i in columns.items():
    plt.subplot(5,4,i)
    plt.boxplot(data1[var])
    plt.title(var)
plt.show()


# ### There is definitely an improvement and the number of outliers have reduced significantly!

# In[ ]:


data1.head()


# In[ ]:


X = data1.drop(['Life_expectancy_'], axis= 1)
y = data1['Life_expectancy_']


# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3
                                                   , random_state = 0)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 200, random_state = 10, 
                            max_depth = 50)


# In[ ]:


reg.fit(x_train, y_train)


# In[ ]:


y_pred = reg.predict(x_test)


# In[ ]:


from sklearn import metrics
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))


# In[ ]:


print('R2 score: ', metrics.r2_score(y_test, y_pred))


# ### Ok so Random Forest gives a pretty good score but let's explore more with Cross validation.

# In[ ]:


y.shape


# In[ ]:


# Cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results = cross_val_score(reg, x_train, y_train, cv = KFold(10))
print(results)
print(np.mean(results))


# ### Great! We have 96 % score now!

# ### Let's check if Randomized Search CV gives us a better result.

# In[ ]:


# Randomized Search CV
from sklearn.model_selection import GridSearchCV
params = [{'n_estimators': [50, 100,200,300,400,500], 
          'criterion': ['squared_error', 'absolute_error'],
           'max_depth': [10, 15, 30, 50, 100, 200],
           'max_features':['auto','sqrt','log2'],
           'random_state':[0, 10, 20, 50, 70, 100],
           'n_jobs':[-1,  1]}]


# In[ ]:


# randomized Search CV
from sklearn.model_selection import RandomizedSearchCV
RScv = RandomizedSearchCV(reg, param_distributions = params, 
                          n_iter = 5, cv = 5)
RScv = RScv.fit(x_train, y_train)


# In[ ]:


RScv.best_score_


# In[ ]:


RScv.best_params_


# In[ ]:


reg_RS = RandomForestRegressor(random_state= 100,
 n_jobs= -1,
 n_estimators= 500,
 max_features= 'auto',
 max_depth= 100,
 criterion= 'mse')
reg_RS.fit(x_train, y_train)


# In[ ]:


RS_pred = reg_RS.predict(x_test)


# In[ ]:


print('MSE: ',metrics.mean_squared_error(y_test, RS_pred))
print('R2 score',metrics.r2_score(y_test, RS_pred))


# ### So there was no significant improvement after using RandomizedSearchCV

# ### Let us try other algorithms as well just in case

# In[ ]:


# trying other algorithms
get_ipython().system('pip install xgboost')


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model1 = XGBRegressor()


# In[ ]:


model1.fit(x_train, y_train)


# In[ ]:


xg_pred = model1.predict(x_test)


# In[ ]:


print('Mean Squared Error: ',metrics.mean_squared_error(y_test, xg_pred))
print('R2 score: ',metrics.r2_score(y_test, xg_pred))


# In[ ]:


# trying with params
model.fit(x_train,y_train)
xg_pred1 = model.predict(x_test)


# In[ ]:


print('Mean squared error: ',metrics.mean_squared_error(y_test,xg_pred1))
print('r2 score: ',metrics.r2_score(y_test, xg_pred1))


# ### So with XgBoost we are getting a better mse score

# ### Let us try Cross Validation here as well

# In[ ]:


# cross validation for xgboost
xg_result = cross_val_score(model1, x_train, y_train, cv = KFold(10))
print(xg_result)
print(np.mean(xg_result))


# ### Some more algorithms....

# ### Ransac algorithm is known to deal with outliers. Let's check if it works for us

# In[ ]:


# ransac regression
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(min_samples=10, max_trials=50, loss='absolute_loss', random_state=42, residual_threshold=50)
ransac.fit(x_train, y_train)
predsRR = ransac.predict(x_test)
mse = metrics.mean_squared_error(y_test, predsRR)
print("MSE : % f" %(mse))


# In[ ]:


print('R2 score', metrics.r2_score(y_test, predsRR))


# In[ ]:


get_ipython().system('pip install catboost')


# In[ ]:


from catboost import CatBoostRegressor
cat= CatBoostRegressor(loss_function='RMSE')
cat.fit(x_train, y_train, eval_set = (x_test, y_test), plot=True)


# ### Conclusion

# ### XgBoost - Mean squared error:  0.0014111426048558611  R2 score:  0.962317634886713
# ### Random Forest - Mean Squared Error:  0.0015578614449765099 R2 score - 0.960961324424398 
