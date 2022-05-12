#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from xgboost import XGBRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
df_shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
df_test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
df_sub = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')


def df_head(df):
    return df.head()

    
df_head(df_sales)


# In[ ]:


df_head(df_sub)


# In[ ]:


df_head(df_items)


# In[ ]:


df_head(df_shops)


# In[ ]:


df_head(df_test)


# # Descriptive and exploratory data analysis

# In[ ]:


df_sales.describe().T


# In[ ]:


df_test.describe().T


# In[ ]:


df_sales['item_price']


# In[ ]:


df_sales[['shop_id','item_id','item_price','item_cnt_day']].corr()


# In[ ]:


df_items.describe().T


# In[ ]:


plt.figure(figsize=(8,5))
plt.hist(df_sales['item_id'])
plt.show


# > Distribution of item id

# In[ ]:


plt.figure(figsize=(8,5))
plt.hist(df_items['item_category_id'])
plt.show


# > Distribution of item categories id

# In[ ]:


plt.figure(figsize=(8,5))
plt.hist(df_sales['item_cnt_day'])
plt.show


# In[ ]:


df_items.groupby('item_category_id').count()


# In[ ]:


df_items.groupby('item_category_id').mean()


# In[ ]:


df_items['diff_col_of_item_id'] = df_items.groupby('item_category_id')['item_id'].max() - df_items.groupby('item_category_id')['item_id'].min()

df_items.head()


# In[ ]:


#df_items.drop('diff_col', inplace=True, axis=1)
#df_items


# In[ ]:


df_items.head()


# > What we have found so far:
# 
# 1. item id and shop id will be only independent variable will predict target variable
# 2. we will drop item price column from train data set
# 3. shop ids are between 1 t0 60
# 4. item id and item price are correlate with each other
# 5. each item id fall into certain item category as item ids >> item category
# 6. we can assign new column item_category to each item id

# # > train and test data set preparation and pre-processing

# In[ ]:


df_sales.head()


# In[ ]:


df_sales.isnull().sum()


# In[ ]:


df_sales.drop_duplicates(keep='first', inplace=True, ignore_index=True)

df_sales.head()


# In[ ]:


df_sales[df_sales['item_price'] <0]


# In[ ]:


df_sales.drop(df_sales[df_sales['item_cnt_day'] <0].index , inplace=True)
df_sales.drop(df_sales[df_sales['item_price'] <0].index , inplace=True)

df_sales.shape


# # **>  outliers removal**

# In[ ]:


Q1 = np.percentile(df_sales['item_price'], 25.0)
Q3 = np.percentile(df_sales['item_price'], 75.0)

IQR = Q3 - Q1

df_sub1 = df_sales[df_sales['item_price'] > Q3 + 1.5*IQR]
df_sub2 = df_sales[df_sales['item_price'] < Q1 - 1.5*IQR]

df_sales.drop(df_sub1.index, inplace=True)

df_sales.shape


# In[ ]:


df_sales['date_block_num'].unique()


# In[ ]:


df_sales.groupby('date_block_num')['item_id'].mean()


# In[ ]:


price = round(np.array(df_sales.groupby('date_block_num')['item_price'].mean()).mean(),2)
print(price)


# In[ ]:


dict(round(df_sales.groupby('date_block_num')['item_price'].mean(),4))


# In[ ]:


df_sales.head()


# In[ ]:


df_test.head()


# # Feature engineering

# #FE workflow

# > create columns with mean price by date block in train and test dataset, remove item price from train
# 
# > remove data_ block column from train data set
# 
# > create new cloumns with mean price per shop id for both train and test dataset
# 
# > merge df_items table with train and test dataset on item id and create new column item category
# 

# In[ ]:


#df_sales.drop('mean_price_data_block', inplace=True, axis=1)

replace_dict = dict(round(df_sales.groupby('date_block_num')['item_price'].mean(),2))


# In[ ]:


df_sales['date_block_num'] = df_sales['date_block_num'].replace(replace_dict)

df_train = df_sales.copy()
df_train.drop(['date','item_price'], axis=1, inplace=True)
df_train.rename(columns = {'date_block_num':'mean_price_by_column'}, inplace=True)
df_train.head()


# In[ ]:


mean_price = np.array(df_sales.groupby('date_block_num')['item_price'].mean()).mean()
mean_price


# In[ ]:


df_test.shape


# In[ ]:


df_train.shape


# In[ ]:


#df_test.drop('ID', inplace=True, axis=1)
df_test.head()
com_df = pd.concat([df_train,df_test])

com_df['mean_price_by_column'] = com_df['mean_price_by_column'].fillna(value=price)
com_df['item_cnt_day'] = com_df['item_cnt_day'].fillna(value=0)

test_df = com_df[com_df['item_cnt_day'] == 0]
train_df = com_df[com_df['item_cnt_day'] != 0]


# In[ ]:


test_df.shape


# In[ ]:


testdf = test_df.copy()

testdf.drop('ID', inplace=True, axis=1)
testdf.drop('item_cnt_day', inplace=True, axis=1)
testdf


# In[ ]:


traindf = train_df.copy()

traindf.drop('ID', inplace=True, axis=1)


# In[ ]:


traindf.head()


# # >train data and test data for modelling and evalution

# In[ ]:


#test_df.drop('item_cnt_day', inplace=True, axis=1)
testdf['item_id'] = (testdf['item_id'] - testdf['item_id'].mean())/testdf['item_id'].std()
testdf.head()


# In[ ]:


traindf['item_id'] = (traindf['item_id'] - traindf['item_id'].mean())/traindf['item_id'].std()
traindf.head()


# # Model 1 

# In[ ]:


X = traindf.loc[:,['mean_price_by_column','shop_id','item_id']]
y = traindf.loc[:,'item_cnt_day']


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y,train_size=0.8, random_state= 42)

model1 = LinearRegression()

model1.fit(X_train,y_train)

print("regression coefficients are: " , model1.coef_)

y_pred = model1.predict(X_valid)


MSE = mean_squared_error(y_valid,y_pred)
MAE = mean_absolute_error(y_valid,y_pred)
R2  = r2_score(y_valid,y_pred)

print("MSE: ", MSE)
print("MAE: ", MAE)
print("R2: ", R2)


# In[ ]:


#model_sub_pred = model1.predict(testdf)

#sub_df = pd.DataFrame(data= {'ID':np.array(df_test['ID']),'item_cnt_month':model_sub_pred})

#sub_df.to_csv('submission.csv', index=False)


# # Model 2

# In[ ]:


# addding constant as statsmodels api does not include it!
X_new = X.copy()
X_new = sm.add_constant(X_new)
test_df_new = test_df.copy()
test_df_new = sm.add_constant(test_df_new)
X_train_new, X_valid_new, y_train_new, y_valid_new = train_test_split(X_new, y,train_size=0.8, random_state= 42)


# In[ ]:


#using statsmodel api

stats_model = sm.OLS(y_train_new, X_train_new)
stat_fit = stats_model.fit()
print("Model coeffieciants: ", stat_fit.params)

print("\nModel summary: ", stat_fit.summary2())


# > P value is less than 0.05 indicates that model is statistically significant
# 
# > F-stat is also very low
# 
# > R2 is almost 17%, that explains independent variables explain 17% of variation in dependent variable item count

# # > Residual analysis

# In[ ]:


model_residual = stat_fit.resid
probplot = sm.ProbPlot(model_residual)
plt.figure(figsize=(8,8))
probplot.ppplot(line = '45')
plt.title("Rsidual analysis of model 2")
plt.show()


# > theoritical probabilties and sample probabilities are not corellating with each other, it implies that resdiuls are not folloewing normal distribution.
# 
# > model is not a good fit to data as data is polynomial.

# # > Test of Homoscedasticity

# In[ ]:


def get_std_val(vals):
    return (vals - vals.mean())/vals.std()

plt.figure(figsize=(8,8))
plt.scatter(get_std_val(stat_fit.fittedvalues), get_std_val(model_residual))
plt.title("Residual plot")
plt.xlabel("predicted values")
plt.ylabel("Residuals")
plt.show()


# > There is a clear funnel shape is observed and we can see that there is couple of outliers in actual values of y.

# # > Outlier analysis

# In[ ]:


traindf['z_score_item'] = zscore(traindf['item_cnt_day'])


# In[ ]:


#outliears in y variable

traindf[(traindf['z_score_item'] > 3.0) | (traindf['z_score_item'] < -3.0)]


# > Total 9971 rows are having extream item sales which resulted in higher residuals

# # > Cook's distance

# In[ ]:


#item_influence = stat_fit.get_influence()
#(c, p) = item_influence.cooks_distance

#plt.stem(np.arange(len(X_train)),
   #      np.round(c, 3),
     #    markerfmt=',')
#plt.title("Cook's distance")
#plt.xlabel("row index")
#plt.ylabel('Cook\'s distance')
#plt.show


# > Using influence method we can plot cooks distance to find which observance has a most influence on output variable

# # > Leverage Values

# In[ ]:


#fig, ax = plt.subplots(figsize=(8,6))
#influence_plot(stat_fit, ax=ax)
#plt.title("Influence plot")
#plt.show()


# In[ ]:


pred_y = stat_fit.predict(X_valid_new)

r2 = r2_score(y_valid_new,pred_y)
mse = mean_squared_error(y_valid_new,pred_y)
mae = mean_absolute_error(y_valid_new,pred_y)

print(r2)
print(mse)
print(mae)


# In[ ]:


#model_sub = stat_fit.predict(testdf)

#sub_dff = pd.DataFrame(data= {'ID':np.array(df_test['ID']),'item_cnt_month':model_sub})

#sub_dff.to_csv('submission.csv', index=False) 


# # > Calculating prediction intervals

# In[ ]:


pred_y = stat_fit.predict(X_valid_new)

_, pred_y_low, pred_y_high = wls_prediction_std( stat_fit, 
                                                X_valid_new, 
                                                alpha = 0.1)

pred_int_df = pd.DataFrame({'item_id_z': X_valid['item_id'],
                            'pred_y': np.array(pred_y),
                            'Pred_y_low': pred_y_low,
                             'Pred_y_high': pred_y_high
                           })

pred_int_df.head(10)


# > Using statsmodels wls_prediction_std method we have calculated prediction interval for each predicted  value of y.

# # Model 3

# In[ ]:


model3 = XGBRegressor(n_estimators=50,
                      max_depth=3,
                      learning_rate = 0.01)

model3.fit(X_train, y_train)

prey = model3.predict(X_valid)

sq_error = mean_squared_error(y_valid, prey)

print(sq_error)


# In[ ]:


model3_sub = model3.predict(testdf)

sub_dff2 = pd.DataFrame(data= {'ID':np.array(df_test['ID']),'item_cnt_month':model3_sub})

sub_dff2.to_csv('submission.csv', index=False) 


# **> If you liked this notebook, Do upvote and share your feedback on the same**
