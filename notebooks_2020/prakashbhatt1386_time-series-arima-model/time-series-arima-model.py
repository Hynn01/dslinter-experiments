#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[ ]:


ts=pd.read_csv("../input/MonthlyPassengers.csv",parse_dates=["Month"],index_col="Month")


# # parse_dates function used to change the datatype of month from object to datestamp and set month as index

# In[ ]:


ts.info()


# In[ ]:


ts.head()


# In[ ]:


# Rename column name to Passenger
ts=ts.rename(columns={"#Passengers":"Passenger"})


# In[ ]:


ts.head()


# #  Visualize the passanger frequency on monthly basis and it is growing every month as per the below graph.
# # By seeing the graph we can say the data is not stationary and to perform timeseries analysis we require stationary data.

# # Stationary data means mean , standard deviation and variance should constant over the time period

# In[ ]:


import matplotlib.pyplot as plt
ts.Passenger.resample("M").mean().plot()


# # Checking data stationarity by autocorelation method and imported below library
# 

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf


# In[ ]:


plot_acf(ts)


# # Converting series to stationary.
# # Stationary means mean, variance, covariance is constant over the periods
# 

# In[ ]:


ts_diff=ts.diff(periods=1)
ts_diff=ts_diff[1:]


# In[ ]:


plot_acf(ts_diff)

# now looking at below visualization, data seems to be converted to stationary data.


# In[ ]:


import matplotlib.pyplot as plt
ts_diff.Passenger.resample("M").mean().plot()


# # we will perform some other statistic technique to check stationarirty in data  like rolling stastistic and dickey fuller test

# In[ ]:


# Determine rolling statistics
rolmean=ts.rolling(window=12).mean()
rolstd=ts.rolling(window=12).std()
print(rolmean)
print(rolstd)



# In[ ]:


# plot rolling statistics
oringdnal=plt.plot(ts,color="blue",label="orignal")
mean=plt.plot(rolmean,color="black",label="rolmean")
std=plt.plot(rolstd,color="red",label="rolstd")
plt.legend(loc='best')


# In[ ]:


# Perform dickey fuller test

from statsmodels.tsa.stattools import adfuller

dftest = adfuller(ts["Passenger"],autolag='AIC')
print(dftest)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfoutput)


# In[ ]:


#critical values

dftest[4]


# # H0 = Data is not stationary
# # H1 = Data is stationary
# 
# # As per dickey fuller test p value >0.50, hence we can not reject the null hypothisis(H0) and conclude data is not stationary.
# # test statistic value is higher than 5% critical value, hence can not reject the null hypothisis

# # we need to estimate and eliminate the trend and seasonality from the data to make it stationary one

# # Transformation technique by taking log of data. we can also take square root or cube root. 
# # transformation technique penalize higher values more than smaller values.

# In[ ]:


ts_log=np.log(ts)
plt.plot(ts_log)
ts_log.head()


# In[ ]:


# now applying moving avg. technique to soothining the tend 

movavg=ts_log.rolling(window=12).mean()
movstd=ts_log.rolling(window=12).std()
oringdnal=plt.plot(ts_log,color="blue",label="orignal")
mean=plt.plot(movavg,color="black",label="rolmean")
std=plt.plot(movstd,color="red",label="rolstd")
plt.legend(loc='best')

# we can see that data is not stationary


# In[ ]:


logdiffmov=ts_log-movavg
plt.plot(logdiffmov)
logdiffmov.head()


# In[ ]:


logdiffmov.dropna(inplace=True)


# In[ ]:


logdiffmov.head()


# In[ ]:


movavg1=logdiffmov.rolling(window=12).mean()
movstd1=logdiffmov.rolling(window=12).std()
oringdnal=plt.plot(logdiffmov,color="blue",label="orignal")
mean=plt.plot(movavg1,color="black",label="movavg1")
std=plt.plot(movstd1,color="red",label="movstd1")
plt.legend(loc='best')


# In[ ]:


from statsmodels.tsa.stattools import adfuller

dftest1 = adfuller(logdiffmov["Passenger"],autolag='AIC')
print(dftest1)
dfoutput1 = pd.Series(dftest1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfoutput1)


# # Here we can see that p<0.5 and test statistic value < 5% of (crital value ),hence we reject the null hypothisis and accept alternate one
# # and consider data as stationary at a confidence level of 95%

# # Forecasting a Time Series by ARIMA technique
# # AR -Auto regressive and to be called as (p)
# # MA- Moving average and to be called as (q)
# # I (integrated) or number of differences (d)
# # To determine p,q value, we use auto correlation (ACF) and partial auto correlation(PACF) plotting technique
# 
# 

# In[ ]:



#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf


# In[ ]:


lag_acf = acf(logdiffmov, nlags=20)
lag_pacf = pacf(logdiffmov, nlags=20, method='ols')


# In[ ]:


# To determine value of Q when chart crosses the upper interval level for the fist time
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(logdiffmov)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(logdiffmov)),linestyle='--',color='gray')

# value of P = 2


# In[ ]:


# To determine value of P when chart crosses the upper interval level for the fist time

plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(logdiffmov)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(logdiffmov)),linestyle='--',color='green')

#Value of P =2


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit()  
plt.plot(logdiffmov)
plt.plot(results_ARIMA.fittedvalues, color='red')


# # Let convert the data in to series and convert the data in to its orignal form from where we have started.

# In[ ]:


series_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(series_ARIMA.head())


# In[ ]:


# the above series is not showing the first month so lets find cumulative sum

series_ARIMA_cumsum = series_ARIMA.cumsum()
series_ARIMA_cumsum.head()


# # we will add the above cumsum value of passenger with the orignal value where we put log on the passanger data for estimating trend.
# # By doing this we have revived 1st month ofpassenger's data, which was missing at the above cumsum stage.
# 

# In[ ]:


predictions_ARIMA_log = pd.Series(ts_log["Passenger"].iloc[0], index=ts_log.index)
series_predictions_ARIMA_log = predictions_ARIMA_log.add(series_ARIMA_cumsum,fill_value=0)
series_predictions_ARIMA_log.head()


# In[ ]:


# now lets make data as original and do compare by make a plot

predictions_orignal_data = np.exp(series_predictions_ARIMA_log)


# In[ ]:


predictions_orignal_data.head()


# In[ ]:


plt.plot(predictions_orignal_data)
plt.plot(ts)


# In[ ]:


ts_log.shape


# # Passenger forcasting for next 12 months only
# 

# In[ ]:


results_ARIMA.plot_predict(1,156) 


# In[ ]:


final_data=results_ARIMA.forecast(steps=24)[0]


# In[ ]:


final_data


# In[ ]:


passenger_projection= np.exp(final_data)


# In[ ]:


passenger_projection


# In[ ]:


passengers = pd.DataFrame({"Proj_passenger":[443.87099559, 470.03826902, 504.93502836, 540.45485906,
       567.73073473, 580.32812271, 577.00266573, 561.93886205,
       542.38464104, 525.56673272, 516.65911974, 518.14327442,
       529.9371715 , 549.71679011, 573.36070617, 595.78903063,
       612.33154129, 620.24253267, 619.59555149, 613.03677623,
       604.59668575, 598.2664506 , 596.94854054, 601.96248871]})


# In[ ]:


passengers


# In[ ]:


passengers.info()


# In[ ]:


#now lets create date range of projected 24 months and put passanger data infront of each month.


# In[ ]:


rng=pd.date_range(start="1/1/1961",periods=24,freq="M")


# In[ ]:


rng


# In[ ]:


Month = pd.DataFrame({"proj_month":['1961-01-31', '1961-02-28', '1961-03-31', '1961-04-30',
               '1961-05-31', '1961-06-30', '1961-07-31', '1961-08-31',
               '1961-09-30', '1961-10-31', '1961-11-30', '1961-12-31',
               '1962-01-31', '1962-02-28', '1962-03-31', '1962-04-30',
               '1962-05-31', '1962-06-30', '1962-07-31', '1962-08-31',
               '1962-09-30', '1962-10-31', '1962-11-30', '1962-12-31']})
print(Month)


# In[ ]:


Month.info()


# In[ ]:




