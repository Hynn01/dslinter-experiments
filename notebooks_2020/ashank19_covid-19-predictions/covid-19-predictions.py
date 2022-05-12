#!/usr/bin/env python
# coding: utf-8

# ## Will be updating the notebook with new Data Visualization Plot and new Prediction and Forecasting Models. Please upvote the kernel if you have liked the work. Also, any kind of suggestion and corrections would be highly appreciated!
#  # Stay Safe and follow basic hygiene Practices!

# # This notebook has been divided into three sections
# 
# ### Section 1- Simple time series analysis of covid cases using general forecasting models and ARIMA model with taking any exogenous features into account.
# 
# ### Section 2- Here the no of confirmed cases has been predicted taking into account the no of tests(extrapolating it five days down the line) and facebook's prophet library has also been used for forecasting.
# 
# ### Section 3- Here the average growth rate of covid cases,recovery rate and death rate has been analyzed and further time series analysis has been performed for prediction of deceased cases and recovered cases.

# # Introduction
# 
# > This notebook predicts outcome of confirmed cases in India in the coming 15 days.
# 
# > Different forecasting methods such as
#   
#   
#   -Simple Exponential Smoothing
#   
#   -Holt Winter's Method
#   
#   -SARIMA Model
#   
#   -Facebook Prophet library
#  
#  > have been used in predicting the outcomes.
#  
# > The data has been taken from www.covid19india.org
# 
# > This kernel is an updated version of my previous one
# 
# > I would like to thank those who expressed their gratitude by upvoting.

# In[ ]:


# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('/kaggle/input/covid18092020/case_time_series.csv')
testing=pd.read_csv('/kaggle/input/covid18092020/tested_numbers_icmr_data.csv')


# In[ ]:


df.info()


# In[ ]:


testing.info()


# In[ ]:


df['Date'] = df['Date'].str.replace(' ','-')
df['Date'] = df['Date'].str.replace('January','01')
df['Date'] = df['Date'].str.replace('February','02')
df['Date'] = df['Date'].str.replace('March','03')
df['Date'] = df['Date'].str.replace('April','04')
df['Date'] = df['Date'].str.replace('May','05')
df['Date'] = df['Date'].str.replace('June','06')
df['Date'] = df['Date'].str.replace('July','07')
df['Date'] = df['Date'].str.replace('August','08')
df['Date'] = df['Date'].str.replace('September','09')


# In[ ]:


df.head()


# In[ ]:


df.loc[:,'Date'] = df.loc[:,'Date']+'2020'


# In[ ]:


df.tail()


# In[ ]:


df['Date']=pd.to_datetime(df['Date'],format='%d-%m-%Y')


# In[ ]:


df.head()


# In[ ]:


# Making the date column as index
df.index=df['Date']
df.drop(['Date'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


# Plot of Daily Confirmed Cases in India
df['Daily Confirmed'].plot(figsize=(22,12),title='Daily Confirmed Cases');


# ## ETS Decomposition

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[ ]:


results = seasonal_decompose(df['Daily Confirmed'])


# In[ ]:


results.plot();


# In[ ]:


results.seasonal.plot(figsize=(15,10),title='Daily Confirmed Cases Seasonality');


# In[ ]:


results.trend.plot(figsize=(15,10),title='Daily Confirmed Cases Trend');


# In[ ]:


len(df)


# ### Splitting the data into training and testing set

# In[ ]:


train=df.iloc[:219]
test=df.iloc[219:]


# As there were no reported cases in India from 4th February 2020 to 1st March 2020. So the curve isn't strictly increasing as a result we cannot use multiplicative trend or assume seasonality to be multiplicative so we have only used additive trend.

# ### Simple Exponential Smoothing

# In[ ]:


# Simple Exponential Smoothing

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 7
alpha = 2/(span+1)

df['EWMA14'] = df['Daily Confirmed'].ewm(alpha=alpha,adjust=False).mean()
df['SES14']=SimpleExpSmoothing(df['Daily Confirmed']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
df.head()


# In[ ]:


df[['Daily Confirmed','EWMA14','SES14']].iloc[-14:].plot(figsize=(14,8),title='Comparison of last 14 days').autoscale(axis='x',tight=True);


# ### Double Exponential Smoothing

# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Double Exponential Smoothing

df['DESadd14'] = ExponentialSmoothing(df['Daily Confirmed'], trend='add').fit().fittedvalues.shift(-1)
df.head()


# In[ ]:


df[['Daily Confirmed','EWMA14','DESadd14']].iloc[-14:].plot(figsize=(14,8),title='Comparison of last 14 days').autoscale(axis='x',tight=True);


# Here we can see that Double Exponential Smoothing is a much better representation of the time series data than Simple Exponential Smoothing.<br>
# We cannot use multiplicative trend as the data is not strictly increasing.

# Building model using Double Exponential smoothing and calculating the rmse.

# In[ ]:


fitted_model = ExponentialSmoothing(train['Daily Confirmed'],trend='add').fit()


# In[ ]:


test_predictions = fitted_model.forecast(13).rename('Forecast')


# In[ ]:


test['Daily Confirmed'].plot(legend=True,label='TEST',figsize=(12,8),title='Comparison between testing set and actual values')
test_predictions.plot(legend=True,label='PREDICTION');


# In[ ]:


# Checking the RMSE for Double exponential exponential smoothing.

from statsmodels.tools.eval_measures import rmse,meanabs

print(rmse(test['Daily Confirmed'],test_predictions))


# ___
# ## Triple Exponential Smoothing
# Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and seasonality in the data. 
# 

# In[ ]:


df['TESadd14'] = ExponentialSmoothing(df['Daily Confirmed'],trend='add',seasonal='add',seasonal_periods=7).fit().fittedvalues
df.head()


# In[ ]:


df[['Daily Confirmed','TESadd14']].iloc[-14:].plot(figsize=(14,8),title='Comparison of last 14 days using Triple Exponential Smoothing').autoscale(axis='x',tight=True);


# Building model using Triple Exponential Smoothing.

# In[ ]:


fitted_model = ExponentialSmoothing(train['Daily Confirmed'],trend='add',seasonal='add').fit()


# In[ ]:


test_predictions = fitted_model.forecast(13).rename('Forecast')


# In[ ]:


test['Daily Confirmed'].plot(legend=True,label='TEST',figsize=(12,8),title='Comparison between testing set and actual values')
test_predictions.plot(legend=True,label='PREDICTION');


# In[ ]:


# Checking the RMSE for Triple exponential exponential smoothing.

print(rmse(test['Daily Confirmed'],test_predictions))


# ## Predicting the number cases till 30/09/2020 using Triple exponential smoothing method.

# In[ ]:


fitted_model = ExponentialSmoothing(df['Daily Confirmed'],trend='add',seasonal='add').fit()


# In[ ]:


test_predictions = fitted_model.forecast(13).rename('Forecast')


# In[ ]:


df['Daily Confirmed'].plot(legend=True,label='Actual',figsize=(12,8),title='Confirmed patients for covid-19 in India according to Triple Exponential Smoothening Model')
test_predictions.plot(legend=True,label='PREDICTION');


# In[ ]:


# Creating a new DataFrame for cumulative sum of confirmed cases in India.

date1 = '2020-01-31'
date2 = '2020-09-30'
mydates = pd.date_range(date1, date2).tolist()
len(mydates)


# In[ ]:


columns=['date','Patients','Total Confirmed']
final = pd.DataFrame(columns=columns)


# In[ ]:


final['date']=mydates

final.index=final['date']
final.drop(['date'],axis=1,inplace=True)


# In[ ]:


final['Patients']=df['Daily Confirmed']
final['Total Confirmed']=df['Total Confirmed']
final=final.reset_index()


# In[ ]:


for i in range(13):
    final.loc[231+i:,'Patients']=test_predictions[i]


# In[ ]:


final=final.round()


# In[ ]:


for i in range(13):
    final.loc[231+i,'Total Confirmed']=final.loc[231+i-1,'Total Confirmed']+final.loc[231+i,'Patients']

final.tail()


# ## Total predicted confirmed patients would be 6505139 by 30/09/2020 if similar trend of community spread continues according to Triple smoothening model.

# # Predicting cases using SARIMA Model

# In[ ]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[ ]:


adf_test(df['Daily Confirmed'])


# In[ ]:


from statsmodels.tsa.statespace.tools import diff
df['d1'] = diff(df['Daily Confirmed'],k_diff=1)

adf_test(df['d1'],'')


# In[ ]:


from statsmodels.tsa.statespace.tools import diff
df['d2'] = diff(df['Daily Confirmed'],k_diff=2)

adf_test(df['d2'],'')


# In[ ]:


pip install pyramid-arima


# In[ ]:


from pyramid.arima import auto_arima


# In[ ]:


train=df.iloc[:219]
test=df.iloc[219:]


# In[ ]:


# Auto Arima Model

stepwise_model = auto_arima(train['Daily Confirmed'], start_p=0, start_q=0, max_p=5, max_q=5,m=7,seasonality=True,
d=2,D=2,trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)

print(stepwise_model.aic())


# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:


model = SARIMAX(train['Daily Confirmed'],order=(1,2,1),seasonal_order=(0,2,2,7))
results = model.fit()
results.summary()


# In[ ]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False,typ='levels').rename('SARIMA(1,2,1)(0,2,2,7) Predictions')


# In[ ]:


title='Covid-19 India Daily Confirmed Cases'
ylabel='Persons'

ax = test['Daily Confirmed'].plot(legend=True,figsize=(14,8),title=title)
predictions.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


from statsmodels.tools.eval_measures import rmse,meanabs

error = rmse(test['Daily Confirmed'], predictions)
print(f'SARIMAX(1,2,1)(0,2,2,7) RMSE Error: {error:11.10}')


# As RMSE for SARIMA model is less than that of triple exponential model so proceeding ahead with SARIMA model.

# In[ ]:


model = SARIMAX(df['Daily Confirmed'],order=(1,2,1),seasonal_order=(0,2,2,7))
result = model.fit()
fcast = result.predict(len(df),len(df)+13).rename('SARIMAX(1,2,1)(0,2,2,7) Forecast')


# In[ ]:


fcast


# In[ ]:


title='Confirmed patients for covid-19 in India'
ylabel='Patients'
ax = df['Daily Confirmed'].plot(legend=True,figsize=(18,8),title=title)
fcast.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


# Creating a new DataFrame for cumulative sum of confirmed cases in India.

date1 = '2020-01-31'
date2 = '2020-09-30'
mydates = pd.date_range(date1, date2).tolist()
len(mydates)


# In[ ]:


columns=['date','Patients','Total Confirmed']
final = pd.DataFrame(columns=columns)


# In[ ]:


final['date']=mydates


# In[ ]:


final.index=final['date']
final.drop(['date'],axis=1,inplace=True)


# In[ ]:


final['Patients']=df['Daily Confirmed']
final['Total Confirmed']=df['Total Confirmed']
final=final.reset_index()


# In[ ]:


for i in range(13):
    final.loc[231+i:,'Patients']=fcast[i]


# In[ ]:


final=final.round()


# In[ ]:


for i in range(13):
    final.loc[231+i,'Total Confirmed']=final.loc[231+i-1,'Total Confirmed']+final.loc[231+i,'Patients']


# In[ ]:


final.tail()


# ## Total predicted confirmed patients would be 6503735 by 30/09/2020 if similar trend of community spread continues according to SARIMA model without taking into account the number of sample tests.

# # Section 2- Taking no of tests into account.

# In[ ]:


testing.head()


# In[ ]:


testing['Update Time Stamp']=testing['Update Time Stamp'].str.replace('/','-')
testing['Update Time Stamp']=testing['Update Time Stamp'].str.replace(' ','')
testing.head()


# In[ ]:


testing1=testing[['Update Time Stamp','Total Tested']]
testing1=testing1.rename(columns={'Update Time Stamp':'Date'})
testing1['Date']=pd.to_datetime(testing1['Date'],format='%d-%m-%Y')
testing1.info()


# In[ ]:


testing1=testing1[1:]


# In[ ]:


testing1=testing1.reset_index()


# In[ ]:


testing1.drop(['index'],axis=1,inplace=True)


# ## Extrapolating the no of tests thirteen days down the line.

# In[ ]:


testing1['dayofweek']=testing1['Date'].dt.dayofweek
testing1.head()


# In[ ]:


testing1.info()


# In[ ]:


train=testing1[:171]
test=testing1[171:]


# ### ETS decomposition of testing data.

# In[ ]:


testing1.index=testing1['Date']
testing1.drop(['Date'],axis=1,inplace=True)


# In[ ]:


testing1.info()


# In[ ]:


results = seasonal_decompose(testing1['Total Tested'])


# In[ ]:


results.seasonal.plot(figsize=(16,8),title='Seasonality of testing data');


# In[ ]:


results.trend.plot(figsize=(15,8),title='Trend of testing data');


# Simple Exponential Smoothing

# In[ ]:


# Simple Exponential Smoothing

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 7
alpha = 2/(span+1)

testing1['EWMA14'] = testing1['Total Tested'].ewm(alpha=alpha,adjust=False).mean()
testing1['SES14']=SimpleExpSmoothing(testing1['Total Tested']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
testing1.head()


# In[ ]:


testing1[['Total Tested','EWMA14','SES14']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days testing data').autoscale(axis='x',tight=True);


# Double Exponential Smoothing

# In[ ]:


# Double Exponential Smoothing

testing1['DESadd14'] = ExponentialSmoothing(testing1['Total Tested'], trend='add').fit().fittedvalues.shift(-1)
testing1.head()


# In[ ]:


testing1[['Total Tested','EWMA14','DESadd14']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days testing data using Double Exponential Smoothing').autoscale(axis='x',tight=True);


# Here we can see that Double Exponential Smoothing is a much better representation of the time series data than Simple Exponential Smoothing.
# Let's see if using a multiplicative trend adjustment helps.

# In[ ]:


testing1['DESmul14'] = ExponentialSmoothing(testing1['Total Tested'], trend='mul').fit().fittedvalues.shift(-1)
testing1.head()
testing1[['Total Tested','DESmul14','DESadd14']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days testing data using Triple Exponential Smoothing').autoscale(axis='x',tight=True);


# ## Triple Exponential Smoothing
# 
# Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and seasonality in the data.

# In[ ]:


testing1['TESadd14'] = ExponentialSmoothing(testing1['Total Tested'],trend='add',seasonal='add',seasonal_periods=7).fit().fittedvalues
testing1.head()
testing1[['Total Tested','TESadd14']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days testing data using additive seasonality').autoscale(axis='x',tight=True);


# In[ ]:


testing1['TESmul14'] = ExponentialSmoothing(testing1['Total Tested'],trend='mul',seasonal='mul',seasonal_periods=7).fit().fittedvalues
testing1.head()
testing1[['Total Tested','TESadd14','TESmul14']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days testing data using multiplicative seasonality').autoscale(axis='x',tight=True);


# In[ ]:


testing1.info()


# In[ ]:


train=testing1.iloc[:171]
test=testing1.iloc[171:]


# In[ ]:


fitted_model = ExponentialSmoothing(train['Total Tested'],trend='add').fit()
test_predictions = fitted_model.forecast(13).rename('Forecast')
test['Total Tested'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');


# In[ ]:


# Checking the RMSE for Double exponential smoothing.

from statsmodels.tools.eval_measures import rmse,meanabs

print(rmse(test['Total Tested'],test_predictions))


# In[ ]:


fitted_model = ExponentialSmoothing(train['Total Tested'],trend='mul').fit()
test_predictions = fitted_model.forecast(13).rename('Forecast')
test['Total Tested'].plot(legend=True,label='TEST',figsize=(12,8));
test_predictions.plot(legend=True,label='PREDICTION');


# In[ ]:


# Checking the RMSE for Double exponential smoothing with multiplicative trend.
print(rmse(test['Total Tested'],test_predictions))


# In[ ]:


fitted_model = ExponentialSmoothing(train['Total Tested'],trend='add',seasonal='add',seasonal_periods=7).fit()
test_predictions = fitted_model.forecast(13).rename('Forecast')
test['Total Tested'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');


# In[ ]:


# Checking the RMSE for Triple exponential smoothing.
print(rmse(test['Total Tested'],test_predictions))


# In[ ]:


fitted_model = ExponentialSmoothing(train['Total Tested'],trend='mul',seasonal='mul',seasonal_periods=7).fit()
test_predictions = fitted_model.forecast(13).rename('Forecast')
test['Total Tested'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');


# In[ ]:


# Checking the RMSE for Triple exponential smoothing multiplicative for both seasonality and trend.

print(rmse(test['Total Tested'],test_predictions))


# RMSE is too high for simple models so going for ARIMA Model.

# In[ ]:


# Checking for stationarity
adf_test(testing1['Total Tested'])


# In[ ]:


testing1['d1'] = diff(testing1['Total Tested'],k_diff=1)

adf_test(testing1['d1'],'')


# In[ ]:


testing1['d2'] = diff(testing1['Total Tested'],k_diff=2)

adf_test(testing1['d2'],'')


# In[ ]:


stepwise_model = auto_arima(train['Total Tested'], start_p=0, start_q=0, max_p=5, max_q=5, m=7,start_P=0, seasonal=True,
d=2, D=2, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True,exogenous=train[['dayofweek']])

print(stepwise_model.aic())


# In[ ]:


model = SARIMAX(train['Total Tested'],order=(0,2,2),seasonal_order=(0,2,2,7),exogenous=train[['dayofweek']],enforce_invertibility=True)
results = model.fit()
results.summary()


# In[ ]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False,typ='levels',exogenous=test[['dayofweek']]).rename('SARIMA(0,2,2)(0,2,2,7) Predictions')


# In[ ]:


title='Covid-19 India Daily Testing'
ylabel='Tests'

ax = test['Total Tested'].plot(legend=True,figsize=(16,8),title=title)
predictions.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


error = rmse(test['Total Tested'], predictions)
print(f'SARIMAX(0,2,2)(0,2,2,7) RMSE Error: {error:11.10}')


# In[ ]:


date1 = '2020-05-13'
date2 = '2020-09-30'
mydates = pd.date_range(date1, date2).tolist()
len(mydates)
a=pd.DataFrame()
a['Date']=mydates
a['dayofweek']=a['Date'].dt.dayofweek


# In[ ]:


model = SARIMAX(testing1['Total Tested'],order=(0,2,2),seasonal_order=(0,2,2,7),enforce_invertibility=True,exogenous=testing1[['dayofweek']])
results = model.fit()
fcast = results.predict(len(testing1),len(testing1)+13,exogenous=a[['dayofweek']]).rename('SARIMAX(0,2,2)(0,2,2,7) Forecast')


# In[ ]:


title='Number of testing for covid-19 in India'
ylabel='Tests'
ax = testing1['Total Tested'].plot(legend=True,figsize=(16,8),title=title)
fcast.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


date1 = '2020-03-18'
date2 = '2020-09-30'
mydates = pd.date_range(date1, date2).tolist()
len(mydates)


# In[ ]:


columns=['Date','Tests','dayofweek']
tests = pd.DataFrame(columns=columns)


# In[ ]:


tests['Date']=mydates


# In[ ]:


tests.info()


# In[ ]:


tests.index=tests['Date']
tests.drop(['Date'],axis=1,inplace=True)
tests['Tests']=testing1['Total Tested']


# In[ ]:


tests['dayofweek']=tests.index.dayofweek


# In[ ]:


tests.info()


# In[ ]:


len(testing1)


# In[ ]:


tests=tests.reset_index()


# In[ ]:


tests.info()


# In[ ]:


for i in range(13):
    tests.loc[184+i:,'Tests']=fcast[i]
tests.info()    


# In[ ]:


tests.index=tests['Date']
tests.drop(['Date'],axis=1,inplace=True)


# In[ ]:


tests=tests.round()


# In[ ]:


tests.tail()


# ### Now as we have our estimated tests by 30/09/2020 let's predict the total confirmed cases by 30/09/2020 taking the number of tests into account.

# Merging the tests dataframe and df dataframe for getting no of tests done.

# In[ ]:


df1=df['2020-03-18':]
df1=df1.reset_index()
df1.head()


# In[ ]:


covidtest=tests[:'2020-09-17']
exog_test=tests['2020-09-17':]
covidtest=covidtest.reset_index()
exog_test=exog_test.reset_index()


# In[ ]:


df1.tail()


# In[ ]:


df_clean=pd.merge(df1,covidtest,on='Date',how='inner')
df_clean.head()


# In[ ]:


df_clean.info()


# In[ ]:


train=df_clean[:171]
test=df_clean[171:]


# In[ ]:


adf_test(df_clean['Daily Confirmed'])


# In[ ]:


df_clean['d1'] = diff(df_clean['Daily Confirmed'],k_diff=1)


adf_test(df_clean['d1'],'')


# In[ ]:


df_clean['d2'] = diff(df_clean['Daily Confirmed'],k_diff=2)


adf_test(df_clean['d2'],'')


# In[ ]:


# Auto Arima Model

stepwise_model = auto_arima(train['Daily Confirmed'], start_p=0, start_q=0, max_p=5, max_q=5, m=6,start_P=0, seasonal=True,
d=2, D=2, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True,exogenous=train[['Tests']])

print(stepwise_model.aic())


# In[ ]:


model = SARIMAX(train['Daily Confirmed'],order=(2,2,3),seasonal_order=(2,2,2,6),enforce_invertibility=True,
                exogenous=train[['Tests']])
results = model.fit()
results.summary()


# In[ ]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False,typ='levels',
                              exogenous=test[['Tests']]).rename('SARIMA(2,2,3)(2,2,2,6) Predictions')


# In[ ]:


title='Covid-19 India Daily Confirmed Cases'
ylabel='Persons'

ax = test['Daily Confirmed'].plot(legend=True,figsize=(16,8),title=title)
predictions.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


from statsmodels.tools.eval_measures import rmse,meanabs

error = rmse(test['Daily Confirmed'], predictions)
print(f'SARIMAX(2,2,3)(2,2,2,6) RMSE Error: {error:11.10}')


# In[ ]:


model = SARIMAX(df_clean['Daily Confirmed'],order=(2,2,3),seasonal_order=(2,2,2,6),enforce_invertibility=True,
                exogenous=df_clean[['Tests']])
results = model.fit()
results.summary()


# In[ ]:


start=len(df_clean)
end=len(df_clean)+13
predictions = results.predict(start=start, end=end, dynamic=False,typ='levels',exogenous=exog_test[['Tests']]).rename('SARIMA(2,2,3)(2,2,2,6) Predictions')


# In[ ]:


title='Covid-19 India Daily Confirmed Cases'
ylabel='Persons'

ax = df_clean['Daily Confirmed'].plot(legend=True,figsize=(16,8),title=title)
predictions.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


predictions


# In[ ]:


# Creating a new DataFrame for cumulative sum of confirmed cases in India.

date1 = '2020-03-18'
date2 = '2020-09-30'
mydates = pd.date_range(date1, date2).tolist()
len(mydates)


# In[ ]:


columns=['date','Patients','Total Confirmed','Tests']
final = pd.DataFrame(columns=columns)


# In[ ]:


final['date']=mydates
final['Patients']=df_clean['Daily Confirmed']
final['Total Confirmed']=df_clean['Total Confirmed']


# In[ ]:


final['Tests']=df_clean['Tests']


# In[ ]:


final.info()


# In[ ]:


final.loc[184:,'Patients']=predictions


# In[ ]:


for i in range(13):
    final.loc[184+i,'Tests']=exog_test.loc[i,'Tests']


# In[ ]:


final.info()


# In[ ]:


final=final.round()


# In[ ]:


final.info()


# In[ ]:


for i in range(13):
    final.loc[184+i,'Total Confirmed']=final.loc[184+i-1,'Total Confirmed']+final.loc[184+i,'Patients']


# In[ ]:


final.index=final['date']
final.drop(['date'],axis=1,inplace=True)


# In[ ]:


final.tail()


# ## Taking no of samples tested into account the confirmed count of covid patients is 6205714 by 30-09-2020 approximately.

# In[ ]:


final.to_csv('output.csv')


# In[ ]:


tests=tests.reset_index()


# In[ ]:


tests.loc[0,'Total Tests']=tests.loc[0,'Tests']


# In[ ]:


for i in range(1,len(tests)):
    tests.loc[i,'Total Tests']=tests.loc[i,'Tests']+tests.loc[i-1,'Total Tests']


# In[ ]:


tests.info()


# In[ ]:


tests.tail()


# ### Total no of samples tested by 30-09-2020 would be 73546766 approximately.

# ## Predictions using Regression Models and Prophet library taking testing into account.

# In[ ]:


df_clean.info()


# In[ ]:


train=df_clean[:171]
test=df_clean[171:]


# In[ ]:


train['day'] = train['Date'].dt.day
train['month'] = train['Date'].dt.month
train['dayofyear'] = train['Date'].dt.dayofyear
train['quarter'] = train['Date'].dt.quarter
train['weekofyear'] = train['Date'].dt.weekofyear


# In[ ]:


train.columns


# In[ ]:


columns=['Daily Recovered','Daily Deceased', 'Tests','day', 'month', 'dayofweek',
         'dayofyear', 'quarter', 'weekofyear']
y=train['Daily Confirmed']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train=train[columns]
x_train,x_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=0)


# In[ ]:


models = []
mse = []
mae = []
rmse = []


# ## Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=1000,random_state=1)
reg.fit(x_train,y_train)


# In[ ]:


pred_RF=reg.predict(x_test)


# In[ ]:


# Importing the error metric
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[ ]:


models.append('Random Forest')
mse.append(round(mean_squared_error(pred_RF, y_test),2))
mae.append(round(mean_absolute_error(pred_RF, y_test),2))
rmse.append(round(np.sqrt(mean_squared_error(pred_RF, y_test)),2))


# ## XGB Regressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


# Training the algorithm
fit_GB = GradientBoostingRegressor(n_estimators=200)
fit_GB.fit(x_train, y_train)


# In[ ]:


pred_XGB=fit_GB.predict(x_test)


# In[ ]:


models.append('XGBoost')
mse.append(round(mean_squared_error(pred_XGB, y_test),2))
mae.append(round(mean_absolute_error(pred_XGB, y_test),2))
rmse.append(round(np.sqrt(mean_squared_error(pred_XGB, y_test)),2))


# ## LGBM Regressor

# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


lgbm = LGBMRegressor(n_estimators=1300)
lgbm.fit(x_train,y_train)
pred_LGBM = lgbm.predict(x_test)


# In[ ]:


models.append('LGBM')
mse.append(round(mean_squared_error(pred_LGBM, y_test),2))
mae.append(round(mean_absolute_error(pred_LGBM, y_test),2))
rmse.append(round(np.sqrt(mean_squared_error(pred_LGBM, y_test)),2))


# In[ ]:


import seaborn as sb


# In[ ]:


plt.figure(figsize= (15,10))
plt.xticks(rotation = 90 ,fontsize = 11)
plt.yticks(fontsize = 10)
plt.xlabel("Different Models",fontsize = 20)
plt.ylabel('RMSE',fontsize = 20)
plt.title("RMSE Values of different models" , fontsize = 20)
sb.barplot(x=models,y=rmse);


# ## Facebook prophet library

# In[ ]:


df=df.reset_index()
df.head()


# In[ ]:


from fbprophet import Prophet

data = pd.DataFrame(columns = ['ds','y','Total Confirmed'])
data['ds'] = df['Date']
data['y'] = df['Daily Confirmed']

prop=Prophet()
prop.fit(data)
future=prop.make_future_dataframe(periods=13)
prop_forecast=prop.predict(future)
forecast = prop_forecast[['ds','yhat']].tail(13)


# In[ ]:


# Prediction of covid-19 cases in  india by 30-09-2020 using Prophet library

prop.plot(prop_forecast,figsize=(15,10));


# In[ ]:


prop_forecast=prop_forecast.round()
prop_forecast['Total Confirmed']=df['Total Confirmed']


# In[ ]:


prop_forecast.info()


# In[ ]:


for i in range(13):
    prop_forecast.loc[232+i,'Total Confirmed']=prop_forecast.loc[232+i-1,'Total Confirmed']+prop_forecast.loc[232+i,'yhat']


# In[ ]:


prop_forecast.tail()


# ## Total confirmed cases by 29/08/2020 according to facebook prophet's library is 6549243.

# # Section 3- Analysis of Death and Recovery rate.
# 
# ## Let's have a look over the Average Growth rate of covid cases in India
#  
#  -Before first lockdown (upto 24 March 2020)
#  
#  -During first lockdown (from 25th March 2020 to 14th April 2020)
#  
#  -During second lockdown (from 15th April 2020 to 3rd May 2020)
#  
#  -In third lockdown (from 4th May 2020 to 17th May 2020)
#  
#  -In fourth lockdown (from 17th May 2020 to 31st May 2020)
#  
#  -In unlock-1(from 1st June 2020).
#  

# In[ ]:


df.index=df['Date']
df.drop(['Date'],axis=1,inplace=True)
df.info()


# In[ ]:


before_lockdown_growth = []
first_lockdown_growth = []
second_lockdown_growth = []
third_lockdown_growth = []
fourth_lockdown_growth = []
unlock_1_growth=[]

# As there the continuous reporting of cases have started from 2/3/2020 so truncating the dataframe accordingly.

Before_lockdown=df['2020-03-02':'2020-03-25']
Before_lockdown=Before_lockdown.reset_index()

# Calculating average growth rate before lockdown period

for i in range(1,len(Before_lockdown)):
    before_lockdown_growth.append(Before_lockdown.loc[i,'Daily Confirmed'] / Before_lockdown.loc[i-1,'Daily Confirmed'])


first_lockdown=df['2020-03-25':'2020-04-15']
first_lockdown=first_lockdown.reset_index()

# Calculating average growth rate in first lockdown

for i in range(1,len(first_lockdown)):
    first_lockdown_growth.append(first_lockdown.loc[i,'Daily Confirmed'] / first_lockdown.loc[i-1,'Daily Confirmed'])
    

second_lockdown=df['2020-04-15':'2020-05-04']
second_lockdown=second_lockdown.reset_index()

# Calculating average growth rate in second lockdown

for i in range(1,len(second_lockdown)):
    second_lockdown_growth.append(second_lockdown.loc[i,'Daily Confirmed'] / second_lockdown.loc[i-1,'Daily Confirmed'])


third_lockdown=df['2020-05-04':'2020-05-17']
third_lockdown=third_lockdown.reset_index()

# Calculating average growth rate in third lockdown

for i in range(1,len(third_lockdown)):
    third_lockdown_growth.append(third_lockdown.loc[i,'Daily Confirmed'] / third_lockdown.loc[i-1,'Daily Confirmed'])

    
fourth_lockdown=df['2020-05-17':'2020-05-31']
fourth_lockdown=fourth_lockdown.reset_index()    

# Calculating average growth rate in fourth lockdown

for i in range(1,len(fourth_lockdown)):
    fourth_lockdown_growth.append(fourth_lockdown.loc[i,'Daily Confirmed'] / fourth_lockdown.loc[i-1,'Daily Confirmed'])
    
unlock_1=df['2020-06-01':'2020-06-07']
unlock_1=unlock_1.reset_index()    

# Calculating average growth rate in unlock-1

for i in range(1,len(unlock_1)):
    unlock_1_growth.append(unlock_1.loc[i,'Daily Confirmed'] / unlock_1.loc[i-1,'Daily Confirmed'])
    

before_lockdown_growth_factor = sum(before_lockdown_growth)/len(before_lockdown_growth)
first_lockdown_growth_factor = sum(first_lockdown_growth)/len(first_lockdown_growth)
second_lockdown_growth_factor = sum(second_lockdown_growth)/len(second_lockdown_growth)
third_lockdown_growth_factor = sum(third_lockdown_growth)/len(third_lockdown_growth)
fourth_lockdown_growth_factor = sum(fourth_lockdown_growth)/len(fourth_lockdown_growth)
unlock_1_growth_factor = sum(unlock_1_growth)/len(unlock_1_growth)

print('Average growth factor before lockdown implemented ',before_lockdown_growth_factor)
print('Average growth factor in first lockdown ',first_lockdown_growth_factor)
print('Average growth factor in second lockdown ',second_lockdown_growth_factor)
print('Average growth factor in third lockdown ',third_lockdown_growth_factor)
print('Average growth factor in fourth lockdown ',fourth_lockdown_growth_factor)
print('Average growth factor in unlock_1 ',unlock_1_growth_factor)


# ## Let's have a look at the rate of average test per confirmed cases.

# In[ ]:


tests.tail()


# In[ ]:


df_clean.tail()


# In[ ]:


df_clean['Total Tests']=tests.loc[:171,'Total Tests']


# In[ ]:


df_clean['test_per_confirmed_cases']=df_clean['Total Tests']/df_clean['Total Confirmed']


# In[ ]:


df_clean['test_per_confirmed_cases'].plot(figsize=(20,8));


# ### The no of tests per confirmed cases of the entire country increased suddenly in the beginning during lockdown,after that it started decreasing as the lockdown regulations easened. Also the growth rate of covid cases in India is highest in the world presently. Also in the past 18 days the growth rate in India has not been as much as it was in the latter half of August.

# # ETS Decomposition for recovered cases

# In[ ]:


results = seasonal_decompose(df['Daily Recovered'])
results.plot();


# In[ ]:


results.seasonal.plot(figsize=(22,8),title='Seasonality of recovered data');


# In[ ]:


# Trend of Daily recovered cases
results.trend.plot(figsize=(16,8),title='Trend of recovered data');


# In[ ]:


# As the recovery of cases started from 23/03/2020
df1=df['2020-03-23':]
len(df1)


# In[ ]:


train=df1.iloc[:166]
test=df1.iloc[166:]


# In[ ]:


# Simple Exponential Smoothing

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 14
alpha = 2/(span+1)

df1['EWMA5'] = df1['Daily Recovered'].ewm(alpha=alpha,adjust=False).mean()
df1['SES5']=SimpleExpSmoothing(df1['Daily Recovered']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
df1.head()


# In[ ]:


df1[['Daily Recovered','EWMA5','SES5']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days recovery using simple exponential smoothing').autoscale(axis='x',tight=True);


# Double Exponential Smoothing

# In[ ]:


# Double Exponential Smoothing

df1['DESadd5_recovery'] = ExponentialSmoothing(df1['Daily Recovered'], trend='add').fit().fittedvalues.shift(-1)
df1.head()


# In[ ]:


df1[['Daily Recovered','EWMA5','DESadd5_recovery']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days recovery using Double Exponential Smoothing').autoscale(axis='x',tight=True);


# Here we can see that Double Exponential Smoothing is a much better representation of the time series data than Simple Exponential Smoothing.
# Let's see if using a multiplicative trend adjustment helps.

# In[ ]:


df1['DESmul5_recovery'] = ExponentialSmoothing(df1['Daily Recovered'], trend='mul').fit().fittedvalues.shift(-1)
df1[['Daily Recovered','DESadd5_recovery','DESmul5_recovery']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days recovery using Triple Exponential Smoothing').autoscale(axis='x',tight=True);


# ## Triple Exponential Smoothing
# 
# Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and seasonality in the data.

# In[ ]:


df1['TESadd5'] = ExponentialSmoothing(df1['Daily Recovered'],trend='add',seasonal='add',seasonal_periods=14).fit().fittedvalues
df1.head()


# In[ ]:


df1[['Daily Recovered','TESadd5']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days  recovery using Triple Exponential Smoothing').autoscale(axis='x',tight=True);


# In[ ]:


df1['TESmul5'] = ExponentialSmoothing(df1['Daily Recovered'],trend='mul',seasonal='mul',seasonal_periods=14).fit().fittedvalues
df1[['Daily Recovered','TESadd5','TESmul5']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days recovery using Triple Exponential Smoothing').autoscale(axis='x',tight=True);


# In[ ]:


fitted_model = ExponentialSmoothing(train['Daily Recovered'],trend='mul').fit()
test_predictions = fitted_model.forecast(13).rename('Forecast')
test['Daily Recovered'].plot(legend=True,label='TEST',figsize=(12,8));
test_predictions.plot(legend=True,label='PREDICTION');


# In[ ]:


np.sqrt(mean_squared_error(test['Daily Recovered'],test_predictions))


# Testing for stationarity of recovery data.

# In[ ]:


adf_test(df1['Daily Recovered'])


# In[ ]:


df1['d1_recovery'] = diff(df1['Daily Recovered'],k_diff=1)

adf_test(df1['d1_recovery'],'')


# In[ ]:


# Auto Arima Model

stepwise_model = auto_arima(train['Daily Recovered'], start_p=0, start_q=0, max_p=5, max_q=5,m=14,seasonality=True,
d=1,D=1,trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)

print(stepwise_model.aic())


# In[ ]:


model = SARIMAX(train['Daily Recovered'],order=(0,1,3),seasonal_order=(0,1,1,14),enforce_invertibility=True)
results = model.fit()
results.summary()


# In[ ]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False,typ='levels').rename('SARIMA(0,1,3)(0,1,1,14) Predictions')


# In[ ]:


title='Covid-19 India Daily Recovered Cases comparison of test and training set'
ylabel='Persons'

ax = test['Daily Recovered'].plot(legend=True,figsize=(16,8),title=title)
predictions.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


np.sqrt(mean_squared_error(test['Daily Recovered'], predictions))


# Predicting for 13 days ahead.

# In[ ]:


model = SARIMAX(df1['Daily Recovered'],order=(0,1,3),seasonal_order=(0,1,1,14),enforce_invertibility=True)
results = model.fit()
results.summary()


# In[ ]:


start=len(df1)
end=len(df1)+13
predictions = results.predict(start=start, end=end, dynamic=False,typ='levels').rename('SARIMA(0,1,3)(0,1,1,14) Predictions')


# In[ ]:


title='Covid-19 India Daily Recovered Cases'
ylabel='Persons'

ax = df1['Daily Recovered'].plot(legend=True,figsize=(16,8),title=title)
predictions.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


predictions


# In[ ]:


a=pd.DataFrame()
a['Date']=mydates
a.info()


# In[ ]:


a.index=a['Date']
a.drop(['Date'],axis=1,inplace=True)
a.info()


# In[ ]:


a['Recovered']=df['Daily Recovered']
a['Total Recovered']=df['Total Recovered']
a.info()


# In[ ]:


a=a.round()


# In[ ]:


a=a.reset_index()


# In[ ]:


for i in range(13):
    a.loc[184+i:,'Recovered']=predictions[i]


# In[ ]:


a.info()


# In[ ]:


for i in range(13):
    a.loc[184+i,'Total Recovered']=a.loc[184+i-1,'Total Recovered']+a.loc[184+i,'Recovered']
a.info()    


# In[ ]:


a=a.round()
a.tail()


# ## From ARIMA model total recovered cases would be 5282253 approximately by 30/09/2020.

# ## Let's have a look at the Death rate.

# In[ ]:


df.tail()


# # ETS Decomposition for death rate.

# In[ ]:


df.info()


# In[ ]:


results = seasonal_decompose(df['Daily Deceased'])
results.plot();


# In[ ]:


results.seasonal.plot(figsize=(16,8),title='Seasonality of Daily Deceased');


# In[ ]:


results.trend.plot(figsize=(16,8),title='Trend of Deceased Cases');


# In[ ]:


# As the deceased cases started from 22/03/2020
df1=df['2020-03-22':]
len(df1)


# In[ ]:


train=df1.iloc[:167]
test=df1.iloc[167:]


# Simple Exponential Smoothing

# In[ ]:


span = 14
alpha = 2/(span+1)

df1['EWMA7_deceased'] = df1['Daily Deceased'].ewm(alpha=alpha,adjust=False).mean()
df1['SES7_deceased']=SimpleExpSmoothing(df1['Daily Deceased']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
df1.head()


# In[ ]:


df1[['Daily Deceased','EWMA7_deceased','SES7_deceased']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days').autoscale(axis='x',tight=True);


# Double Exponential Smoothing

# In[ ]:


# Double Exponential Smoothing

df1['DESadd7_deceased'] = ExponentialSmoothing(df1['Daily Deceased'], trend='add').fit().fittedvalues.shift(-1)
df1.head()


# In[ ]:


df1[['Daily Deceased','EWMA7_deceased','DESadd7_deceased']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days').autoscale(axis='x',tight=True);


# Here we can see that Double Exponential Smoothing is a much better representation of the time series data than Simple Exponential Smoothing. Let's see if using a multiplicative trend adjustment helps.

# In[ ]:


df1['DESmul7_deceased'] = ExponentialSmoothing(df1['Daily Deceased'], trend='mul').fit().fittedvalues.shift(-1)
df1[['Daily Deceased','DESadd7_deceased','DESmul7_deceased']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days').autoscale(axis='x',tight=True);


# ## Triple Exponential Smoothing
# 
# Triple Exponential Smoothing, the method most closely associated with Holt-Winters, adds support for both trends and seasonality in the data.

# In[ ]:


df1['TESadd7_deceased'] = ExponentialSmoothing(df1['Daily Deceased'],trend='add',seasonal='add',seasonal_periods=14).fit().fittedvalues
df1.head()


# In[ ]:


df1[['Daily Deceased','TESadd7_deceased']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days').autoscale(axis='x',tight=True);


# In[ ]:


df1['TESmul7_deceased'] = ExponentialSmoothing(df1['Daily Deceased'],trend='mul',seasonal='mul',seasonal_periods=15).fit().fittedvalues
df1[['Daily Deceased','TESadd7_deceased','TESmul7_deceased']].iloc[-14:].plot(figsize=(16,8),title='Comparison of last 14 days').autoscale(axis='x',tight=True);


# Testing for stationarity of deceased data.

# In[ ]:


adf_test(df1['Daily Deceased'])


# In[ ]:


df1['d1_deceased'] = diff(df1['Daily Deceased'],k_diff=1)

adf_test(df1['d1_deceased'],'')


# In[ ]:


# Auto Arima Model

stepwise_model = auto_arima(train['Daily Deceased'], start_p=0, start_q=0, max_p=5, max_q=5,m=14,seasonality=True,
d=1,D=1,trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)

print(stepwise_model.aic())


# In[ ]:


model = SARIMAX(train['Daily Deceased'],order=(0,1,1),seasonal_order=(0,1,1,14),enforce_invertibility=True)
results = model.fit()
results.summary()


# In[ ]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False,typ='levels').rename('SARIMA(0,1,1)(0,1,1,14) Predictions')


# In[ ]:


title='Covid-19 India Daily Deceased Cases'
ylabel='Persons'

ax = test['Daily Deceased'].plot(legend=True,figsize=(16,8),title=title)
predictions.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


np.sqrt(mean_squared_error(test['Daily Deceased'], predictions))


# Predicting for 13 days ahead.

# In[ ]:


model = SARIMAX(df1['Daily Deceased'],order=(0,1,1),seasonal_order=(0,1,1,14),enforce_invertibility=True)
results = model.fit()
results.summary()


# In[ ]:


start=len(df1)
end=len(df1)+13
predictions = results.predict(start=start, end=end, dynamic=False,typ='levels').rename('SARIMA(0,1,1)(0,1,1,14) Predictions')


# In[ ]:


title='Covid-19 India Daily Deceased Cases'
ylabel='Persons'

ax = df1['Daily Deceased'].plot(legend=True,figsize=(16,8),title=title)
predictions.plot(legend=True);
ax.autoscale(axis='x',tight=True);
ax.set(ylabel=ylabel);


# In[ ]:


a=pd.DataFrame()
a['Date']=mydates
a.info()


# In[ ]:


a.index=a['Date']
a.drop(['Date'],axis=1,inplace=True)

a['Deceased']=df['Daily Deceased']
a['Total Deceased']=df['Total Deceased']
a.info()


# In[ ]:


a=a.round()

a.loc[184:,'Deceased']=predictions

a.info()


# In[ ]:


a=a.reset_index()
for i in range(13):
    a.loc[184+i,'Total Deceased']=a.loc[184+i-1,'Total Deceased']+a.loc[184+i,'Deceased']
a.info()


# In[ ]:


a=a.round()
a.tail()


# # From ARIMA model total deceased cases would be 99572 approximately by 30/09/2020.

# # Conclusion
# 
# ## In the last couple of months we have seen that though the infection spread has increased still the situation isn't alarming as it was in Italy,Spain.It is because of these reasons:-
# 
# - The recovery rate in India is pretty though we haven't reached our peak yet it is because more than 50% of our population is below the age of 25 and more than 65% below the age of 35.
# 
# - The death rate in India is also low due to the above statistic.
# 
# 
# ## Thus if we are careful it is possible that India may soon become free of corona.
# 
# ## If this doesn't happen then the daily active cases will keep increasing, though our recovery rate would be high but then corona would persist for a long time and have a damaging effect on our economy,healthcare etc. and the restoration to normalcy would be extended further.
# 
# ## But in the last 15 days the growth rate in India has decreased as compared to August which is an indication that people are becoming more aware of the virus and taking preventive measures accordingly.If similar trend continues then covid can be controlled in India much easily.
