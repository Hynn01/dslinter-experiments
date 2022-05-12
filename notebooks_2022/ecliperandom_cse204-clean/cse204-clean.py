#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
import sklearn.tree
import sklearn.neighbors

import tensorflow as tf
import keras
import math

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


root_dir = '/kaggle/input'


# In[ ]:


def load_data(ticker):
    """
    Returns the data frame corresponding to the stock specified in ticker.
    """
    return pd.read_csv(f'{root_dir}/price-volume-data-for-all-us-stocks-etfs/Stocks/{ticker}.us.txt')

def date_to_weekday(df):
    """
    Returns a new data frame identical to df, except for the 'Date' column that
    will be substituted for binary columns 'Mon', 'Tue',... corresponding to the
    weekday.
    """
    
    df['Date'] = pd.Series(datetime.fromisoformat(date).weekday() for date in df['Date'])
    
    df['Mon'] = (df['Date'] == 0)*1.0
    df['Tue'] = (df['Date'] == 1)*1.0
    df['Wed'] = (df['Date'] == 2)*1.0
    df['Thu'] = (df['Date'] == 3)*1.0
    df['Fri'] = (df['Date'] == 4)*1.0
    
    df = df.drop('Date', axis=1)
    
    return df

def normalize_prices(df):
    """
    Returns a new stock dataframe where the opening prices have
    been normalized by the previous days prices and the 'High',
    'Low', 'Close' have been normalized by the opening price of 
    the same day.
    """
    
    df['High'] /= df['Open']
    df['Low'] /= df['Open']
    df['Close'] /= df['Open']
    df['Open'] = pd.Series(df['Open'].iloc[1:].to_numpy()/df['Open'].iloc[:-1].to_numpy())
    
    return df.iloc[1:]

def normalize_volume(df, scale_factor=1e8):
    """
    Divides the volume by scale_factor.
    """
    
    df['Volume'] = df['Volume']/scale_factor
    return df

def transform_dataframe(df):
    """
    Transforms the data frame by using normalize_volume and normalize_prices 
    and date_to_weekday
    """
    
    df = normalize_prices(df)
    df = normalize_volume(df)
    df = date_to_weekday(df)
    df.drop('OpenInt', axis=1)
    
    return df

def get_features_and_targets(df, binary_target=True, drop_features=None):
    """
    From df, construct the features and targets for the stock
    price prediction task.
    """
    
    if binary_target:
        y = df['Open'].iloc[1:].to_numpy() >= df['Close'].iloc[:-1].to_numpy()
    else:
        y = df['Open'].iloc[1:].to_numpy()
        
    x = df.iloc[:-1]
    
    if drop_features is None:
        drop_features = []
        
    x = x.drop(drop_features, axis=1)
    
    return x,y

def stock_train_test_split(x,y, train_test_proportion=0.8):
    """
    Splits x,y into training and test subsets. Notice that 
    all samples in x_test, y_test correspond to dates that occur
    after x_train, y_train (meaning that the test is 'the future')
    """
    train_test_split = int(train_test_proportion * len(x))
    
    try:
        x_train = x.iloc[:train_test_split]
    except AttributeError:
        x_train = x[:train_test_split]
        
    y_train =  1*y[:train_test_split]
    
    try:
        x_test = x.iloc[train_test_split:]
    except AttributeError:
        x_test = x[train_test_split:]
        
    y_test = 1*y[train_test_split:]
    
    return (x_train, y_train), (x_test, y_test)

def get_clean_data(ticker, binary_target=True, drop_features=None):
    """
    Returns (x_train, y_train), (x_test, y_test) after
    properly transforming the data.
    """
    
    df = load_data(ticker)
    df = transform_dataframe(df)
    x, y = get_features_and_targets(df, binary_target=binary_target, drop_features=drop_features)
    
    return stock_train_test_split(x, y)

def probability_to_binary(y):
    """
    For an array y with values in [0,1], rounds each
    entry to the closest number in {0,1}.
    """
    
    return (y > 0.5) * 1.0


# In[ ]:


def to_timeseries(x, y, window):
    """
    Transforms the data frame to a time series format.
    
    Now, each row of x will contain the data of window
    consecutive days.
    
    """
    try:
        x = x.to_numpy()
    except:
        pass
    
    x_out = []
    for time in range(x.shape[0]+1-window):
        x_out.append([])
        for i in range(window):
            x_out[-1] += list(x[time+i])
    
    x_out = np.array(x_out)
    
    y_out = y[window-1:]
        
    return x_out, y_out

def get_clean_timeseries_data(ticker, window, binary_target=True, drop_features=None):
    """
    Returns (x_train, y_train), (x_test, y_test) after
    properly transforming the data. But now the data is in a time-series format.
    """
        
    df = load_data(ticker)
    df = transform_dataframe(df)
    x, y = get_features_and_targets(df, binary_target=binary_target, drop_features=drop_features)
    
    # To timeseries
    x, y = to_timeseries(x,y,window)
    return stock_train_test_split(x,y)
    


# In[ ]:





# In[ ]:


def plot_stock_prices(ticker, log_scale=False, normalize=False):
    
    df = load_data(ticker)
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    if normalize:
        df = normalize_prices(df)
        
    n_samples = len(df)
    
    tab_time = np.arange(n_samples)
    plt.plot(tab_time, df['Open'], label='Open')
    
    if log_scale:
        plt.yscale('log')

    plt.xticks([])
    
    plt.title(f'{ticker} stock prices from {min_date} to {max_date}')
    plt.legend()

    plt.show()


# In[ ]:


plot_stock_prices('aapl')
plot_stock_prices('googl')
plot_stock_prices('fb')
plot_stock_prices('nflx')
plot_stock_prices('amzn')


# ## Predicting whether price will go up or down in the next day
# 
#     (1) Using Logistic Regression
#     (2) Using Random Forests
#     (3) Using KNN
#     (4) Using LSTM

# ### Logistic Regression

# In[ ]:


def train_and_test_model_binary(model, ticker, testing_stocks=None, verbosity=True, drop_features=None):
    
    if testing_stocks == None:
        testing_stocks = []
    
    (x_train, y_train), (x_test, y_test) = get_clean_data(ticker, binary_target=True, drop_features=drop_features)
    model.fit(x_train, y_train)
    
    if verbosity:
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        print(f"Training in stock {ticker}")
        print(f"Train accuracy: {sklearn.metrics.accuracy_score(y_train, y_train_pred)}")
        print(f"Test accuracy: {sklearn.metrics.accuracy_score(y_test, y_test_pred)}")
        
    for stock in testing_stocks:
        (x_train, y_train), (x_test, y_test) = get_clean_data(stock, binary_target=True, drop_features=drop_features)
        
        pred = model.predict(x_test)
        print(f"Accuracy in {stock}: {sklearn.metrics.accuracy_score(y_test, pred)}")


# In[ ]:


training_stock = 'aapl'
testing_stocks = ['fb', 'googl', 'amzn', 'nflx']


# In[ ]:


model_logistic_regression = sklearn.linear_model.LogisticRegression()
train_and_test_model_binary(model_logistic_regression, training_stock, testing_stocks)


# ### Random Forests

# In[ ]:


model_random_forest = sklearn.ensemble.RandomForestClassifier()
train_and_test_model_binary(model_random_forest, training_stock, testing_stocks)


# It looks like overfitting is occurring here. Let's tweak the hyperparameters.

# In[ ]:


model_random_forest_5 = sklearn.ensemble.RandomForestClassifier(max_depth=5)
model_random_forest_10 = sklearn.ensemble.RandomForestClassifier(max_depth=10)
model_random_forest_20 = sklearn.ensemble.RandomForestClassifier(max_depth=20)


# In[ ]:


train_and_test_model_binary(model_random_forest_5, training_stock, testing_stocks)


# In[ ]:


train_and_test_model_binary(model_random_forest_10, training_stock, testing_stocks)


# In[ ]:


train_and_test_model_binary(model_random_forest_20, training_stock, testing_stocks)


# In[ ]:


model_random_forest_3 = sklearn.ensemble.RandomForestClassifier(max_depth=3)
model_random_forest_4 = sklearn.ensemble.RandomForestClassifier(max_depth=4)
model_random_forest_6 = sklearn.ensemble.RandomForestClassifier(max_depth=6)


# In[ ]:


train_and_test_model_binary(model_random_forest_3, training_stock, testing_stocks)


# In[ ]:


train_and_test_model_binary(model_random_forest_4, training_stock, testing_stocks)


# In[ ]:


train_and_test_model_binary(model_random_forest_6, training_stock, testing_stocks)


# Now let's try some more sophisticated ensemble methods.

# In[ ]:


model_adaboost = sklearn.ensemble.AdaBoostClassifier()
train_and_test_model_binary(model_adaboost, training_stock, testing_stocks)


# In[ ]:


model_bagging = sklearn.ensemble.BaggingClassifier()
train_and_test_model_binary(model_bagging, training_stock, testing_stocks)


# In[ ]:


model_bagging_6 = sklearn.ensemble.BaggingClassifier(sklearn.tree.DecisionTreeClassifier(max_depth = 6))
train_and_test_model_binary(model_bagging_6, training_stock, testing_stocks)


# ### k-NN

# In[ ]:


model_knn = sklearn.neighbors.KNeighborsClassifier()
train_and_test_model_binary(model_knn, training_stock, testing_stocks)


# K-NN has a very poor performance. 
# 
# Possibly this is happening because of the weekdays. When two data points contain two different weekdays, this difference will be responsible for most of the distance.
# 
# After further consideration, I realized that volume was also a big responsible for the distance function.
# 
# Below I implement knn without volume or weekdays.

# In[ ]:


def train_and_test_model_binary_knn(model, ticker, testing_stocks=None, verbosity=True):
    
    if testing_stocks == None:
        testing_stocks = []
    
    (x_train, y_train), (x_test, y_test) = get_clean_data(ticker, binary_target=True)
    
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    drop_features = ['Volume'] + weekdays
    
    x_train = x_train.drop(drop_features, axis=1)
    x_test = x_test.drop(drop_features, axis=1)
    
    model.fit(x_train, y_train)
    
    if verbosity:
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        print(f"Training in stock {ticker}")
        print(f"Train accuracy: {sklearn.metrics.accuracy_score(y_train, y_train_pred)}")
        print(f"Test accuracy: {sklearn.metrics.accuracy_score(y_test, y_test_pred)}")
        
    for stock in testing_stocks:
        (x_train, y_train), (x_test, y_test) = get_clean_data(stock, binary_target=True)
        
        x_train = x_train.drop(drop_features, axis=1)
        x_test = x_test.drop(drop_features, axis=1)
        
        pred = model.predict(x_test)
        print(f"Accuracy in {stock}: {sklearn.metrics.accuracy_score(y_test, pred)}")
        
model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=12)
train_and_test_model_binary_knn(model_knn, training_stock, testing_stocks)


# ### Using Time Series to Predict Stock Movement

# Now we are going to put the data from several consecutive days as features to predict whether the stock price will go up or down. This improvement will be fundamental in the implementation of the LSTM.

# In[ ]:


def train_and_test_model_timeseries_binary(model, ticker, window=14, testing_stocks=None, verbosity=True, drop_features=None, reshape=False):
    
    if testing_stocks is None:
        testing_stocks = []
    
    (x_train, y_train), (x_test, y_test) = get_clean_timeseries_data(ticker, window, binary_target=True, drop_features=drop_features)
    
    if reshape:
        x_train = x_train.reshape((*x_train.shape, 1))
        x_test = x_test.reshape((*x_test.shape, 1))
        
    model.fit(x_train, y_train)
    
    if verbosity:
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        y_train_pred = probability_to_binary(y_train_pred)
        y_test_pred = probability_to_binary(y_test_pred)
                
        print(f"Training in stock {ticker}")
        print(f"Train accuracy: {sklearn.metrics.accuracy_score(y_train, y_train_pred)}")
        print(f"Test accuracy: {sklearn.metrics.accuracy_score(y_test, y_test_pred)}")
        
    for stock in testing_stocks:
        (x_train, y_train), (x_test, y_test) = get_clean_timeseries_data(ticker, window, binary_target=True, drop_features=drop_features)
        if reshape:
            x_train = x_train.reshape((*x_train.shape, 1))
            x_test = x_test.reshape((*x_test.shape, 1))
        
        pred = model.predict(x_test)
        pred = probability_to_binary(pred)
        
        print(f"Accuracy in {stock}: {sklearn.metrics.accuracy_score(y_test, pred)}")


# In[ ]:


model_rf = sklearn.ensemble.RandomForestClassifier(max_depth=10)
train_and_test_model_timeseries_binary(model_rf, training_stock, window=10, testing_stocks=testing_stocks)


# Let's first try to see what happens if we only use the opening price to make predictions.

# In[ ]:


features_to_drop = ['High', 'Low', 'Close', 'Volume', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']


# In[ ]:


model_rf = sklearn.ensemble.RandomForestClassifier(max_depth=10)
train_and_test_model_timeseries_binary(model_rf, training_stock, window=10, testing_stocks=testing_stocks, drop_features=features_to_drop)


# ### Using LSTMs to predict stock price movements

# In[ ]:


def train_and_test_lstm(model, ticker, window=14, testing_stocks=None, verbosity=True, drop_features=None, reshape=True, epochs=100, batch_size=32):
    
    if testing_stocks is None:
        testing_stocks = []
    
    (x_train, y_train), (x_test, y_test) = get_clean_timeseries_data(ticker, window, binary_target=True, drop_features=drop_features)
    
    if reshape:
        x_train = x_train.reshape((*x_train.shape, 1))
        x_test = x_test.reshape((*x_test.shape, 1))
        
    model.fit(x_train, y_train, epochs=epochs, batch_size=32)
    
    if verbosity:
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        y_train_pred = probability_to_binary(y_train_pred)
        y_test_pred = probability_to_binary(y_test_pred)
                
        print(f"Training in stock {ticker}")
        print(f"Train accuracy: {sklearn.metrics.accuracy_score(y_train, y_train_pred)}")
        print(f"Test accuracy: {sklearn.metrics.accuracy_score(y_test, y_test_pred)}")
        
    for stock in testing_stocks:
        (x_train, y_train), (x_test, y_test) = get_clean_timeseries_data(ticker, window, binary_target=True, drop_features=drop_features)
        if reshape:
            x_train = x_train.reshape((*x_train.shape, 1))
            x_test = x_test.reshape((*x_test.shape, 1))
        
        pred = model.predict(x_test)
        pred = probability_to_binary(pred)
        
        print(f"Accuracy in {stock}: {sklearn.metrics.accuracy_score(y_test, pred)}")


# In[ ]:


if 0:
    window = 10

    model_lstm = keras.models.Sequential()

    model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
    model_lstm.add(keras.layers.Dropout(0.2))

    model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
    model_lstm.add(keras.layers.Dropout(0.2))

    model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
    model_lstm.add(keras.layers.Dropout(0.2))

    model_lstm.add(keras.layers.LSTM(units=50, return_sequences=False, input_shape=(window, 1)))
    model_lstm.add(keras.layers.Dropout(0.2))

    model_lstm.add(keras.layers.Dense(units=1, activation='sigmoid'))

    model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
    train_and_test_lstm(model_lstm, training_stock, window=window, testing_stocks=testing_stocks,
                                           drop_features=features_to_drop, reshape=True)


# In[ ]:


if 0:
    window = 20

    model_lstm = keras.models.Sequential()

    model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
    model_lstm.add(keras.layers.Dropout(0.2))

    model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
    model_lstm.add(keras.layers.Dropout(0.2))

    model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
    model_lstm.add(keras.layers.Dropout(0.2))

    model_lstm.add(keras.layers.LSTM(units=50, return_sequences=False, input_shape=(window, 1)))
    model_lstm.add(keras.layers.Dropout(0.2))

    model_lstm.add(keras.layers.Dense(units=1, activation='sigmoid'))

    model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
    train_and_test_lstm(model_lstm, training_stock, window=window, testing_stocks=testing_stocks,
                                           drop_features=features_to_drop, reshape=True)


# In[ ]:


window = 10

model_lstm = keras.models.Sequential()

model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
model_lstm.add(keras.layers.Dropout(0.2))

model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
model_lstm.add(keras.layers.Dropout(0.2))

model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
model_lstm.add(keras.layers.Dropout(0.2))

model_lstm.add(keras.layers.LSTM(units=50, return_sequences=False, input_shape=(window, 1)))
model_lstm.add(keras.layers.Dropout(0.2))

model_lstm.add(keras.layers.Dense(units=1, activation='sigmoid'))

model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
train_and_test_lstm(model_lstm, training_stock, window=window, testing_stocks=testing_stocks,
                                       drop_features=None, reshape=True, epochs=400)


# In[ ]:


window = 20

model_lstm = keras.models.Sequential()

model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
model_lstm.add(keras.layers.Dropout(0.2))

model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
model_lstm.add(keras.layers.Dropout(0.2))

model_lstm.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(window, 1)))
model_lstm.add(keras.layers.Dropout(0.2))

model_lstm.add(keras.layers.LSTM(units=50, return_sequences=False, input_shape=(window, 1)))
model_lstm.add(keras.layers.Dropout(0.2))

model_lstm.add(keras.layers.Dense(units=1, activation='sigmoid'))

model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
train_and_test_lstm(model_lstm, training_stock, window=window, testing_stocks=testing_stocks,
                                       drop_features=features_to_drop, reshape=True, epochs=400)


# #### Summary:
# 
# 
