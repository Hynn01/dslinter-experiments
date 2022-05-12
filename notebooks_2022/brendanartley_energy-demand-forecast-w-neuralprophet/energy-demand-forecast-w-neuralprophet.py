#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from datetime import timedelta
from sklearn.metrics import mean_squared_error

# NeuralProphet
get_ipython().system('pip install neuralprophet[live] --quiet')

from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed

set_random_seed(0)


# ## Neural Prophet (Meta AI)
# 
# In this article, I will use NeuralProphet to forecast energy demand. Forecasting energy demand is extremely important as the demand for electricity increases. Knowing how much electricity is needed ahead of time has a significant impact on carbon emissions, energy cost, and policy decisions<sup>1</sup>.
# 
# On November 30, 2021, Meta AI (formerly Facebook) released NeuralProphet. NeuralProphet was built to bridge the gap between classical forecasting techniques and deep learning models. In this article, I will showcase the NeuralProphet framework and evaluate its performance against other forecasting techniques.
# 
# #### Useful Resources
# 
# * [NeuralProphet Documentation](https://neuralprophet.com/html/full_simple_model.html)
# * [NeuralProphet Release Blog Post](https://ai.facebook.com/blog/neuralprophet-the-neural-evolution-of-facebooks-prophet/)
# * [Is Facebook's "Prophet" the Time-Series Messiah, or Just a Very Naughty Boy?](https://www.microprediction.com/blog/prophet)
# 

# ## Loading Data
# 
# The dataset we are going to use contains electricity demand data from Victoria, Australia. Victoria is home to 6.7 million people. The dataset has daily data points from January 1st, 2015 to October 6th, 2020. This gives us enough samples to pick up on any seasonality and trends in the dataset.
# 
# ---
# 
# In the following cell, we load this dataset into pandas for preprocessing. Similar to Prophet, NeuralProphet requires a 'ds' column for the date/time stamp, and a 'y' column for the data value.

# In[ ]:


fpath = "../input/electricity-demand-in-victoria-australia/complete_dataset.csv"

df = pd.read_csv(fpath)
demand_df = df[['date','demand']].rename(columns={"date": "ds", "demand": "y"})
demand_df['ds'] = pd.to_datetime(demand_df['ds'])
demand_df.head()


# We can then use Matplotlib to visualize the data. In the first plot we can see all of the datapoints. It appears that there is some yearly seasonality in the data. The energy demand generally increases every year until June, where it then decreases for the rest of the year.
# 
# The second plot simply shows the first 100 days of data. We can see that energy demand is not consistent day to day. If there is any weekly seasonality in the data, it is difficult to identify from this plot.

# In[ ]:


plt.figure(figsize=(10, 4))
plt.title("All Data")
plt.plot(demand_df['ds'].dt.to_pydatetime(), demand_df['y'])
plt.show()

plt.figure(figsize=(10, 4))
plt.title("First 100 Days")
plt.plot(demand_df['ds'].dt.to_pydatetime()[:100], demand_df['y'][:100])
plt.show()


# In the next cell, we are simply creating a validation and testing set for the model. We could use the built-in .split_df() function, but I found that this duplicated rows in each set. For this reason, we will use simple indexing.
# 
# Note that the validation and testing set should always contain the most recent data points. This ensures that we do not train on data from the future and make predictions on the past.

# In[ ]:


train_size = 0.8

m = NeuralProphet()
df_train, df_valid = demand_df[:int(len(demand_df)*0.8)], demand_df[int(len(demand_df)*0.8):]
df_valid, df_test = df_valid[:len(df_valid)//2], df_valid[len(df_valid)//2:]

def train_valid_plot():
    """Visualizing the training + validation sets"""
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(df_train['ds'].dt.to_pydatetime(), df_train["y"], color='#1f76b4', label='Training Set')
    ax.plot(df_valid['ds'].dt.to_pydatetime(), df_valid["y"], color='#fc7d0b', label='Validation Set')
    ax.plot(df_test['ds'].dt.to_pydatetime(), df_valid["y"], color='#CDC7E5', label='Test Set')
    ax.legend()
    plt.show()
    
train_valid_plot()


# ## Defining a Model
# 
# NeuralProphet uses a very similar API to Prophet. If you have used Prophet before, then using NeuralProphet will be very intuitive.
# 
# In the following cell, we are simply defining a model and fitting this model to the data. We use 'D' to set the frequency of predictions as daily, and we use plot-all to visualize model performance live during training. The only other alteration we make is to specify Australian holidays.

# In[ ]:


m = NeuralProphet()

m.add_country_holidays(country_name='Australia')
metrics = m.fit(df=df_train, validation_df=df_valid, freq="D", progress="plot-all")
metrics[-1:]


# We can see from the graph above that the model is being overfit to the data. The model is fitting as low as it can on the training data, but we want the model to fit well on unseen data (ie. validation set). 
# 
# Looking at the metric plots above, we can see that the optimal parameters are reached around 25–30 epochs and then the model starts to overfit. We can combat this by specifying a number of epochs. A complete list of tuneable model parameters can be found [here](https://neuralprophet.com/html/forecaster.html).

# In[ ]:


m = NeuralProphet(epochs=30)
m.add_country_holidays(country_name='Australia')
metrics2 = m.fit(df=df_train, validation_df=df_valid, freq="D")
metrics2[-1:]


# By specifying the number of epochs, we can significantly reduce the validation RMSE. Even changing one parameter can improve our model significantly (as shown above). This suggests that using parameter tuning, and translating domain knowledge to the model can improve its performance.

# ## Evaluating a Model
# 
# Before we try and squeeze every ounce of performance out of our model, lets see how we can evaluate our model.
# 
# In the next cell, we are simply making a forecast that is the same length as the validation set. We can then visualize this using the .plot() function. This gives a decent visualization of the forecast, but does not provide a performance metric, nor can we see the predictions very clearly.

# In[ ]:


future = m.make_future_dataframe(df=df_train, periods=len(df_valid), n_historic_predictions=True)
forecast = m.predict(df=future)
fig_forecast = m.plot(forecast)


# To address the limitations of the built-in plot, I put together a customized plot using Matplotlib. The following cell plots the predictions with the true labels and shows the model metrics in the plot title. 

# In[ ]:


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.set_title("Train RMSE: {:.2f} --- Validation RMSE: {:.2f}".format(metrics2[-1:].RMSE.values[0], metrics2[-1:].RMSE_val.values[0]))
ax.plot(df_valid['ds'].dt.to_pydatetime(), df_valid["y"],'.k', label='True Value')
ax.plot(forecast[-len(df_valid):]['ds'].dt.to_pydatetime(), forecast[-len(df_valid):]["yhat1"], label='Predicted Value')
ax.legend()
plt.show()


# Next, we can look at the model parameters. This can give us a sense of seasonality patterns and the trend of the data. 
# 
# In the first and second plots, we can see that there was a spike in energy demand in 2018. Then, the demand dips and steadily increases throughout 2019 and 2020. This gives us a sense of how energy demand changes over time.
# 
# In the third plot, we are looking at the yearly seasonality. We can see that energy demand is at its lowest in April and October, and energy demand is at its highest in July. This makes sense, as July is the coldest month of the year in Australia. Interestingly, the warmest month is February, when we see a small spike in energy demand. This could indicate that people use electricity for Air Conditioning during the hottest month.
# 
# The fourth plot shows the weekly seasonality. This indicates that the energy consumption is at its lowest on Saturday and Sunday.
# 
# Finally, we have the plot of the additive events. This plot shows the effect of the Australian holidays that we added. We can see that on Holidays, the energy demand is typically lower than usual.

# In[ ]:


# fig_comp = m.plot_components(forecast) # Alternative: shows a slightly different figure
fig_param = m.plot_parameters()


# ## Adding AR-Net (AutoRegression)
# 
# One of the new additions in Prophet is AR-Net (Auto-Regressive Neural Network). This allows NeuralProphet to use observations from previous time steps when making a prediction. In our case, this means that the model can use the previous day's energy demands to make its predictions.
# 
# AR-Net can be enabled by setting an appropriate value to the n_lags parameter when creating the NeuralProphet Model. We are also increasing the checkpoints_range as we are making short-term predictions on the data.

# In[ ]:


m = NeuralProphet(n_forecasts=1, n_lags=3, epochs=30, changepoints_range=0.95)
m.add_country_holidays(country_name='Australia')
metrics3 = m.fit(df=df_train, validation_df=df_valid, freq="D")
metrics3[-1:]


# We can see from the metrics above that the validation RMSE decreased again. This is another significant gain in model performance we got by simply tuning two parameters.
# 
# If we use the same code that we did previously, only one prediction is made. It is unclear from the docs how to make "running" predictions when AR-Net is enabled, and therefore we can use the following code to make this possible. If anyone knows a built-in way to do this please let me know!

# In[ ]:


valid_preds = [] #list to store predictions
lags = 3

for d in df_valid['ds'].values:
    # getting necessary df rows
    date_index = demand_df.index[demand_df['ds'] == d][0]
    future = demand_df.iloc[date_index-lags:date_index]
    
    # adding new row
    entry = pd.DataFrame({
        'ds': [d],
        'y' : [np.nan]
    })
    future = pd.concat([future, entry], ignore_index = True, axis = 0)
    
    # making prediction
    forecast = m.predict(df=future)
    valid_preds.append(forecast.loc[lags]['yhat1'])


# We can then use the following code block to plot our predictions. We can see from the plot that the model is starting to pick up on outlying points.

# In[ ]:


# Creating DF for predictions
df_valid_copy = df_valid.copy()
df_valid_copy['yhat1'] = valid_preds
df_valid_copy.head()

# Plotting Predictions
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.set_title("Train RMSE: {:.2f} --- Validation RMSE: {:.2f}".format(metrics3[-1:].RMSE.values[0], metrics3[-1:].RMSE_val.values[0]))
ax.plot(df_valid_copy['ds'].dt.to_pydatetime(), df_valid_copy["y"],'.k', label='True Value')
ax.plot(df_valid_copy['ds'].dt.to_pydatetime(), df_valid_copy["yhat1"], label='Predicted Value')
ax.legend()
plt.show()


# If we then plot the model components, we can see that there is an additional plot shown. This plot shows how much each lagged term affects the prediction. In our case, we can see that the most recent days are the most important to the model. In most time series problems, this is often the case.

# In[ ]:


fig_param = m.plot_parameters()


# ## Hyperparameter tuning
# 
# Up to this point, we have been able to improve our validation RMSE manually. This is pretty good, but we only tuned a couple of parameters. What about other parameters? Consider the following list of tuneable parameters and their default values.
# 
# > *NeuralProphet(growth='linear', changepoints=None, n_changepoints=10, changepoints_range=0.9, trend_reg=0, trend_reg_threshold=False, yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto', seasonality_mode='additive', seasonality_reg=0, n_forecasts=1, n_lags=0, num_hidden_layers=0, d_hidden=None, ar_reg=None, learning_rate=None, epochs=None, batch_size=None, loss_func='Huber', optimizer='AdamW', newer_samples_weight=2, newer_samples_start=0.0, impute_missing=True, collect_metrics=True, normalize='auto', global_normalization=False, global_time_normalization=True, unknown_data_normalization=False)*
# 
# It would take a lot of time and effort to manually enter all the possible combinations of these parameters. We can combat this by implementing hyperparameter tuning. In this implementation, we are simply testing all possible combinations of parameters in the parameter grid. This means that the number of possible combinations grows exponentially as more parameters are added<sup>2</sup>.
# 
# This could potentially be improved using bayesian optimization to more efficiently search the parameter space, but adding this functionality is out of the scope of this notebook. In the following cell, we are creating a parameter grid and then training models using all the possible parameter combinations.

# In[ ]:


# Parameter Options
param_grid = {  
    'num_hidden_layers': [1,2],
    'changepoints_range': [0.95, 0.975, 0.99, 0.995, 0.999],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
results = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = NeuralProphet(**params, n_forecasts=1, newer_samples_weight=4, n_lags=3, learning_rate=0.02, epochs=50, batch_size=32)
    m.add_country_holidays(country_name='Australia')
    metrics4 = m.fit(df=df_train, validation_df=df_valid, freq="D")
    results.append(dict({"RMSE_val": metrics4['RMSE_val'].min(), "RMSE_train": metrics4['RMSE'][metrics4['RMSE_val'].idxmin()], "score_epoch_number": metrics4['RMSE_val'].idxmin()}, **params))


# Next, we are creating a Pandas dataframe to store the lowest RMSE value from each model training cycle. We can then sort by the validation RMSE value, to get a sense of which parameter combinations worked well. The training RMSE score and epoch where when the validation score was at its lowest are also stored.
# 
# This is done to ensure that the model is not overfitting to the validation set.

# In[ ]:


# Find the best parameters
results_df = pd.DataFrame.from_dict(results, orient='columns')
results_df = results_df.sort_values('RMSE_val')
results_df.head(10)


# Looking at the results above, we can see that the first and second rows appear to be overfitting the validation set. On the other hand, the third row shows a similar RMSE score on both the training and validation sets.
# 
# In the following cell we are re-entering high scoring model parameters that worked well. We can enable the progress plot to see more information on the model training, and we can make any further changes manually if needed.

# In[ ]:


m = NeuralProphet(newer_samples_weight=5, n_forecasts=1, n_lags=3, learning_rate=0.02, epochs=25, batch_size=32, num_hidden_layers=1, changepoints_range=0.995)
m.add_country_holidays(country_name='Australia')
metrics5 = m.fit(df=df_train, validation_df=df_valid, freq="D", progress="plot-all")
metrics5[-1:]


# We have reduced the RMSE even more! As we improve the model performance it becomes more and more difficult to make improvements. That being said we are looking for progress not perfection and will take improvements where we can.
# 
# The forecast can then be plotted in the same way as we did earlier in the notebook.

# In[ ]:


valid_preds = [] #list to store predictions
lags = 3

for d in df_valid['ds'].values:
    # getting necessary df rows
    date_index = demand_df.index[demand_df['ds'] == d][0]
    future = demand_df.iloc[date_index-lags:date_index]
    
    # adding new row
    entry = pd.DataFrame({
        'ds': [d],
        'y' : [np.nan]
    })
    future = pd.concat([future, entry], ignore_index = True, axis = 0)
    
    # making prediction
    forecast = m.predict(df=future)
    valid_preds.append(forecast.loc[lags]['yhat1'])


# In[ ]:


# Creating DF for predictions
df_valid_copy = df_valid.copy()
df_valid_copy['yhat1'] = valid_preds
df_valid_copy.head()

# Plotting Predictions
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.set_title("Train RMSE: {:.2f} --- Validation RMSE: {:.2f}".format(metrics5[-1:].RMSE.values[0], metrics5[-1:].RMSE_val.values[0]))
ax.plot(df_valid_copy['ds'].dt.to_pydatetime(), df_valid_copy["y"],'.k', label='True Value')
ax.plot(df_valid_copy['ds'].dt.to_pydatetime(), df_valid_copy["yhat1"], label='Predicted Value')
ax.legend()
plt.show()


# ## Model Performance Comparison
# 
# In the next cell, I am going to compare the NeuralProphet model with other common forecasting strategies.
# 
# - Predict Last Value
# - Exponential Smoothing
# - SARIMA
# - Neural Prophet
# 
# We can manually calculate the RMSE value for each model using sklearn. We can simply pass the parameter `squared=False` to get RMSE from the mean_squared_error function.
# 
# Firstly, we can calculate the RMSE if we just predicted the energy demand from the day before. 

# In[ ]:


# Last Value Method
last_val_df = demand_df.copy()

last_val_df['y_prev'] = last_val_df['y'].shift(1)
last_val_df = last_val_df[last_val_df['ds'].isin(df_test['ds'].values)]
last_val_preds = last_val_df['y'].values

last_val_rmse = mean_squared_error(last_val_preds, last_val_df['y_prev'], squared=False)
last_val_rmse


# Next, we can calculate a forecasting model using exponential smoothing. This model type uses the weighted averages of past observations, with weights decaying exponentially as observations get older.

# In[ ]:


# Exponential Smoothing Method
from statsmodels.tsa.api import SimpleExpSmoothing

exp_smooth_preds = []

for d in df_test['ds'].values:
    # Setting up Dataframe
    date_index = demand_df.index[demand_df['ds'] == d][0]
    future = demand_df.iloc[date_index-len(df_train):date_index]
    future = future.set_index('ds')
    future.index = pd.DatetimeIndex(future.index.values, freq='D')
    
    # Make Predictions
    fit = SimpleExpSmoothing(future, initialization_method="heuristic").fit(
            smoothing_level=0.2, optimized=False
        )
    
    exp_smooth_preds.append(fit.forecast(1).values[0])
    
exp_smooth_rmse = mean_squared_error(df_test['y'], exp_smooth_preds, squared=False)
exp_smooth_rmse


# Next, we can fit a SARIMA model to the data. This model acronym stands for "Seasonal Auto-Regressive Integrated Moving Average", and calculates its forecast exactly how it is named. For information on this model type, check out this great article [here](https://medium.com/@BrendanArtley/time-series-forecasting-with-arima-sarima-and-sarimax-ee61099e78f6).
# 
# This model is a little more complex, and we will break the training into code blocks. Firstly, the optimal model parameters are found using autoarima. This is essentially a hyper-parameter tuning package for ARIMA models.

# In[ ]:


get_ipython().system('pip install pmdarima --quiet')
import pmdarima as pm

# Seasonal - Find best parameters with Auto-ARIMA
SARIMA_model = pm.auto_arima(df_train["y"], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, 
                         m=7, #weekly frequency
                         start_P=0, 
                         seasonal=True, #set to seasonal
                         d=None, 
                         D=1, #order of the seasonal differencing
                         trace=False,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

SARIMA_model


# Next, we manually create an ARIMA model at each step with the optimal model parameters found above.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# SARIMA Method\nfrom statsmodels.tsa.arima.model import ARIMA\n\nsarima_preds = []\n\nfor d in df_test['ds'].values:\n    # Setting up dataframe\n    date_index = demand_df.index[demand_df['ds'] == d][0]\n    future = demand_df.iloc[date_index-len(df_train):date_index]\n    future = future.set_index('ds')\n    future.index = pd.DatetimeIndex(future.index.values, freq='D')\n    \n    # Fit model + make predictions\n    m_sarima = ARIMA(future['y'], order=(1,0,1), seasonal_order=(0, 1, 1, 7)).fit()\n    sarima_preds.append(m_sarima.forecast(1).values[0])\n    \nsarima_rmse = mean_squared_error(df_test['y'], sarima_preds, squared=False)\nsarima_rmse")


# Finally, we make predictions with the NeuralProphet Model.

# In[ ]:


m = NeuralProphet(newer_samples_weight=5, n_forecasts=1, n_lags=3, learning_rate=0.02, epochs=25, batch_size=32, num_hidden_layers=1, changepoints_range=0.995)
m.add_country_holidays(country_name='Australia')
metrics6 = m.fit(df=df_train, freq="D")

neuralprophet_preds = [] #list to store predictions
lags = 3

for d in df_test['ds'].values:
    # getting necessary df rows
    date_index = demand_df.index[demand_df['ds'] == d][0]
    future = demand_df.iloc[date_index-lags:date_index]
    
    # adding new row
    entry = pd.DataFrame({
        'ds': [d],
        'y' : [np.nan]
    })
    future = pd.concat([future, entry], ignore_index = True, axis = 0)
    
    # making prediction
    forecast = m.predict(df=future)
    neuralprophet_preds.append(forecast.loc[lags]['yhat1'])
    
neuralprophet_rmse = mean_squared_error(df_test['y'], neuralprophet_preds, squared=False)


# Now all the predictions are made, we can compare the RMSE scores on the test dataset.
# 
# The last value and exponential smoothing methods yield the highest error, SARIMA achieves the second-lowest error, and NeuralProphet performs the best. I was surprised how close the SARIMA forecast came to NeuralProphet. It would be interesting to take this a step further and see how these models perform on other time series tasks.

# In[ ]:


def comparison_plot():
    """Visualizing the training + validation sets"""
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title("Model Comparison")
    ax.plot(df_test["y"].values,'.k', label='True Value')
    ax.plot(last_val_preds, label="Last Value")
    ax.plot(exp_smooth_preds, label="Exponential Smoothing")
    ax.plot(sarima_preds, label="SARIMA")
    ax.plot(neuralprophet_preds, label="NeuralProphet")
    ax.legend()
    plt.show()
    
def score_plot():
    """Visualizing the training + validation sets"""
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    
    scores = {
        "Last Value": last_val_rmse,
        "Exponential Smoothing": exp_smooth_rmse,
        "SARIMA": sarima_rmse,
        "NeuralProphet": neuralprophet_rmse,
    }
    
    ax.set_title("Score Plot")
    ax.bar(scores.keys(), scores.values())
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.margins(y=0.1)
    plt.show()
    
comparison_plot()
score_plot()


# ### Closing Thoughts
# 
# NeuralProphet is a very intuitive framework that is still in the early stages of development. If you want to contribute to this project, you can do so on [Github](https://github.com/ourownstory/neural_prophet). You can also join the NeuralProphet community on [Slack](https://join.slack.com/t/neuralprophet/shared_invite/zt-18de4n6ef-XyJLYUmkL7ULcj77xcJrmQ)!
# 
# I did not include exogenous variables in this notebook but think that these would boost model performance. The predictions from the NeuralProphet model could be used as an input feature to an LGBM/XGBoost model. This would likely yield a very forecast.

# #### References
# 
# 1. [Energy for Sustainable Development - MD. Hasanuzzaman, Nasrudin Abd Rahim - 04/03/2017](https://www.sciencedirect.com/book/9780128146453/energy-for-sustainable-development)
# 2. [Handbook of Statistics - Jean-Marie Dufour, Julien Neves - 01/05/2019](https://www.sciencedirect.com/science/article/abs/pii/S0169716119300367?via%3Dihub)
