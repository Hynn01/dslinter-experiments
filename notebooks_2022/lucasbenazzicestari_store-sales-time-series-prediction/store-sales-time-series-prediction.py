#!/usr/bin/env python
# coding: utf-8

# **Disclaimer** - This notebook will focus only on making predictions for the Store Sales - Time Series Forecasting competition; it won't do any in-depth analysis as there are already plenty of well-made resources available for that.
# 
# # **Introduction**
# The predictions will be made using the already available datasets from the competition. The sales information is divided into product families, so each day there's information on how much a product family is sold as well as how many were on promotion. There is also oil price information available, which will be useful as Ecuador is an oil-dependent country. And finally, there's information on all holidays and relevant events for the country.
# 
# # **Objective**
# The goal of this notebook will be to create multiple models for each store, average their predictions, and make a competition submission. In order to do this, we will preprocess the data and create a Store class to make individual predictions.
# 
# # **Relevant datasets**
# **Oil** - This dataset consists only of the oil prices for each given day. This information needs some preprocessing as there are some missing values. In addition to this, two columns will be added, one with the moving average and another with the moving standard deviation.
# 
# **Holidays Events** - This dataset has a description of all events and holidays and which regions of the country were affected. Since we have the region for each store, we can see which holidays and events affected which stores. The dataset still needs to have transformations done in order to get a list of affected stores. Alongside of that, there are many different types of holidays and events that happened during this period, so there will be many simplifications. The goal will be to determine, for each store, if a given day had an event or holiday, and classify that date as "Non working day".
# 
# **Stores** - For each store, there's information on product families being sold and being on promotion. Since the goal is to predict sales, product family sales will be used as the label and promotion information will be used as input for training the model.
# 
# # **Prediction**
# Two models will be trained: XGBoost and Random Forest regressor. The final prediction will be the average of the predictions of each model.

# # Imports

# In[ ]:



import numpy as np
import pandas as pd

from calendar import monthrange

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


# # Loading Data

# In[ ]:


df_holidays_events = pd.read_csv("../input/store-sales-time-series-forecasting/holidays_events.csv",
                                 parse_dates = ['date'])

df_oil = pd.read_csv("../input/store-sales-time-series-forecasting/oil.csv",
                     parse_dates = ['date'])

df_stores = pd.read_csv("../input/store-sales-time-series-forecasting/stores.csv")

df_test = pd.read_csv("../input/store-sales-time-series-forecasting/test.csv",
                      parse_dates = ['date'])

df_train = pd.read_csv("../input/store-sales-time-series-forecasting/train.csv",
                      parse_dates = ['date'])

df_sample_submission = pd.read_csv("../input/store-sales-time-series-forecasting/sample_submission.csv")


# # Preparing Oil data

# In[ ]:


# importing oil data and adds two columns
oil = df_oil.copy()
oil = oil.set_index('date')
oil = oil['dcoilwtico'].resample('D').sum().reset_index()
oil = oil.replace({0:np.nan})
oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(limit_direction = 'both')
oil['dcoilwtico mean'] = oil['dcoilwtico'].rolling(7).mean().interpolate(limit_direction = 'both')
oil['dcoilwtico std'] = oil['dcoilwtico'].rolling(7).std().interpolate(limit_direction = 'both')


# # Preparing Holiday and Events data

# In[ ]:


# defining which stores were affected by which event or holiday
holidays_events = df_holidays_events.copy()

# the next couple of changes were made analysing the data. There was an official transfer for new years eve holiday, but that didn't translate on a difference in sales.
holidays_events.loc[297, 'transferred'] = False
holidays_events = holidays_events.loc[~(holidays_events.index == 298)]

holidays_events = holidays_events.loc[holidays_events['transferred'] == False].drop('transferred', axis = 1)
holidays_events = holidays_events.loc[holidays_events['type'] != "Work Day"]

stores = df_stores.copy()

def affected_stores(holiday_locale, holiday_locale_name):
    if holiday_locale == 'National':
        return stores['store_nbr'].unique()
    
    elif holiday_locale == 'Local':
        return stores['store_nbr'].loc[stores['city'] == holiday_locale_name].to_numpy()
    elif holiday_locale == 'Regional':
        return stores['store_nbr'].loc[stores['state'] == holiday_locale_name].to_numpy()
    else:
        return []

holidays_events['cities'] = holidays_events.apply(lambda x : affected_stores(x['locale'],x['locale_name']), axis = 1)

holidays_events = holidays_events.drop(columns = ['type','locale','locale_name','description'], axis = 0)

cities_dummies = cities_dummies = pd.get_dummies(holidays_events['cities'].explode()).sum(level=0)
holidays_with_dummies = pd.concat([holidays_events, cities_dummies], axis = 1).drop(['cities'], axis = 1)

holidays_events = pd.melt(holidays_with_dummies, id_vars = 'date', var_name = 'store_nbr').drop(['value'], axis = 1)
holidays_by_store = holidays_events.groupby(['date','store_nbr']).sum().reset_index() 


# # Functions class

# In[ ]:


class AuxiliaryFunctions():
    """Class with auxiliary functions to be used in the
    Store class.
    
    """
    def get_first_day_sold(self, store_nbr):
        """Gets the day the given store had its first
        sale.
        
        Args:
            store_nbr (int): Unique id of the store.
        Returns:
            Datetime of the first sale.
        
        """
        
        df = df_train.copy()
        df = df.loc[df['store_nbr'] == store_nbr, ['date','sales']]
        df = df.groupby('date').sum()
        return df.ne(0).idxmax().values[0] + np.timedelta64(1,'D')

    def prepare_test_inputs(self, store_nbr, start_date, end_date):
        """Gets the input and label values from the train
        dataframe with their relevant values unstacked. It
        takes two dates as input to use as a date range.
        
        X input has the family column unstacked with the
        promotion values.
        y input has the family column unstacked with the
        sales values.
        
        Args:
            start_date (Datetime): Day from which to start.
            end_date (Datetime): Day from which to end.
            
        Returns:
            The inputs (X) and labels (y) for training.
        """
        
        df = df_train.copy()

        df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.set_index(['family','date']).sort_index().drop('id', axis = 1)
        
        X = df.loc[df['store_nbr'] == store_nbr].drop(['store_nbr','sales'], axis = 1).unstack('family')
        X.columns = [name for _, name in X.columns]
        
        y = df.loc[df['store_nbr'] == self.store_nbr].drop(['store_nbr','onpromotion'], axis = 1).unstack('family')
        y.columns = [name for _, name in y.columns]
        
        X = X.reset_index()
        return X, y      
    
    def add_information(self, dataframe):
        """Adds relevant data not directly present in the
        train dataframe.

        It adds data related to oil prices, time information,
        and which dates to consider as work days.

        Args:
            dataframe (DataFrame): Base dataframe which will
                have the information added.

        Returns:
            A dataframe with its base values and extra
            information combined.

        """

        X = dataframe.copy()

        X = X.merge(oil, on = ['date'], how = 'left')

        timestamp_s = X['date'].map(pd.Timestamp.timestamp)

        day = 24 * 60 * 60
        week = day * 7
        year = 365.2425 * day
        quarter = year / 4
        half_decade = year * 5

        X['week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
        X['week cos'] = np.cos(timestamp_s * (2 * np.pi / week))

        X['quarter sin'] = np.sin(timestamp_s * (2 * np.pi / quarter))
        X['quarter cos'] = np.cos(timestamp_s * (2 * np.pi / quarter))

        X['year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        X['year cos'] = np.cos(timestamp_s * (2 * np.pi / year))    

        X['half decade sin'] = np.sin(timestamp_s * (2 * np.pi / half_decade))
        X['half decade cos'] = np.cos(timestamp_s * (2 * np.pi / half_decade))    

        X['day of week'] = X['date'].dt.dayofweek

        X['is new year'] = 0
        X.loc[X['date'].dt.dayofyear == 1, 'is new year'] = 1

        X['is work day'] = 1
        X['quarter'] = X['date'].dt.month // 4

        store_holidays = holidays_by_store.loc[holidays_by_store['store_nbr'] == self.store_nbr, 'date']
        X.loc[
            (X['date'].isin(store_holidays)) | (X['day of week'] > 4) |
            (X['date'].dt.day == 15)
            ,
            'is work day'] = 0

        X['is payday'] = X['date'].map(
            lambda date: 1 if (date.day == monthrange(date.year, date.month)[1] or date.day == 15) else 0
        )
        X = pd.get_dummies(X, columns = ['day of week'])

        return X.set_index('date')

    def train_test_split(self, store_nbr, number_of_days = 15):
        """Creates a train test split based on a given number
        of days, which specifies the number of days used for
        testing. The split isn't randomized and the test data
        is always after the training data.
        
        Args:
            store_nbr (int): Unique id of the store.
            number_of_days (int): Number of days to be use for the
                test data.
        
        """
        
        train_start_date = self.get_first_day_sold(store_nbr)
        train_end_date = df_train['date'].max() - np.timedelta64(number_of_days, 'D')
        
        X_train, y_train = self.prepare_test_inputs(store_nbr, train_start_date, train_end_date)
        X_train = self.add_information(X_train)
        
        val_start_date = train_end_date + np.timedelta64(1, 'D')
        val_end_date = df_train['date'].max()
    
        X_test, y_test = self.prepare_test_inputs(store_nbr, val_start_date, val_end_date)
        X_test = self.add_information(X_test)
    
        return X_train, y_train, X_test, y_test
    
    def get_input_for_prediction(self):
        """Gets the inputs from the test dataframe that will
        be used to make the final prediction. This function
        also calls the self.add_information method to prepare
        the dataframe.
        
        Returns:
            Dataframe will all necessary information for
            prediction.
        
        """
        
        df = df_test.copy()
        df = df.set_index(['family','date']).sort_index().drop('id', axis = 1)
        
        X = df.loc[df['store_nbr'] == self.store_nbr].drop(['store_nbr'], axis = 1).unstack('family')
        X.columns = X.columns.map(' - '.join).str.strip(' - ')
        X = X.reset_index()
        X = self.add_information(X)
        return X
    
    def merge_to_prediction(self, y_pred, y_columns, X_pred):
        """This function returns the prediction in the same
        format as the sample_submission file.
        
        Args:
            y_pred (np.array): Values predicted.
            y_columns (list): Column names.
            X_pred (DataFrame): Dataframe used for prediction.
        
        Returns:
            Section of submission for the given store.
        
        """
        
        prediction = pd.DataFrame(y_pred, columns = y_columns, index = X_pred.index).unstack()
        prediction = prediction.reset_index().set_index('date').rename(columns = {0: "sales", "level_0":"family"})
        prediction['store_nbr'] = self.store_nbr
        submission = df_test.copy()
        current_submission = submission.merge(prediction.reset_index(), how = 'inner')
        
        return current_submission
    


# # Individual Store models class

# In[ ]:


class Store(AuxiliaryFunctions):
    """Class for handling the training and prediction of a
    XGBRegressor and RandomForestRegressor models. Both
    model predictions are averaged out before submitting
    the results.
    
    """
    
    def __init__(self, store_nbr):
        """Setting up all necessary attributes for training
        and prediction.
        
        Args:
            store_nbr (int): Unique id of the store.
            
        Attributes:
            store_nbr (int): Unique id of the store.
            random_state (int): Set seed for the random
                number generator.
            X_train, y_train (DataFrame): Dataframes for
                training the model.
            X_test, y_test (DataFrame): Dataframes for
                testing the model.
            X, y (DataFrame): Dataframes that are a
                combination of training and testing data.
            xgb_model: XGBoost regressor model.
            random_forest_model: Random Forest regressor
                model.
        
        """
        
        self.store_nbr = store_nbr
        
        self.random_state = 42
        
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_test_split(self.store_nbr)
        
        self.X, self.y, _, _ = self.train_test_split(self.store_nbr, 0)
        
        self.xgb_model = MultiOutputRegressor(XGBRegressor(booster = "gbtree", random_state = self.random_state))
        self.random_forest_model = RandomForestRegressor(n_estimators = 150, random_state = self.random_state)
        
    def train_models(self):
        """Trains both models with the training data.
        
        """
        
        self.xgb_model.fit(self.X_train, self.y_train)
        self.random_forest_model.fit(self.X_train, self.y_train)
    
    def evaluate_models(self):
        """Uses the testing data to evaluate each model, as
        well as the combination of the two.
        
        Returns:
            evaluation_xgb (float): mean squared log error
                of the XGBoost regressor model.
            evaluation_forest (float): mean squared log error
                of the Random Forest regressor model.
            evaluation_average (float): mean squared log error
                of the average prediction of the two models.
        
        """
        
        y_pred_xgb = self.xgb_model.predict(self.X_test).clip(0.0)
        y_pred_forest = self.random_forest_model.predict(self.X_test).clip(0.0)
        
        y_pred_average = (y_pred_xgb + y_pred_forest) / 2
        
        evaluation_xgb = mean_squared_log_error(y_pred_xgb, self.y_test.to_numpy())
        evaluation_forest = mean_squared_log_error(y_pred_forest, self.y_test.to_numpy())
        
        evaluation_average = mean_squared_log_error(y_pred_average, self.y_test.to_numpy())
        
        return evaluation_xgb, evaluation_forest, evaluation_average
    
    def train_and_predict(self):
        """Trains both models using all available testing data,
        and returns a formatted prediction.
        
        Returns:
            A dataframe with the current predictions with the
            same formating as the sample_submission.csv file.
        
        """
        
        self.xgb_model.fit(self.X, self.y)
        
        self.random_forest_model.fit(self.X, self.y)
        
        X_pred = self.get_input_for_prediction()
        
        y_pred_xgb_final = self.xgb_model.predict(X_pred).clip(0.0)
        y_pred_random_forrest_final = self.random_forest_model.predict(X_pred).clip(0.0)
        
        y_pred = (y_pred_xgb_final + y_pred_random_forrest_final) / 2
        
        return self.merge_to_prediction(y_pred, self.y.columns, X_pred)


# # Training all models and submitting prediction

# In[ ]:


def make_prediction():
    """Trains a model for each store and saves the prediction
    results into a csv file.
    
    """
    
    prediction_list = []
    for store_nbr in stores['store_nbr'].unique():
        store = Store(store_nbr = store_nbr)
        prediction_list.append(store.train_and_predict())
        print(f'finished predicting store {store_nbr}')

    prediction = prediction_list[0]
    for i in range(len(prediction_list) - 1):
        prediction = prediction.append(prediction_list[i + 1])

    final_submission = prediction[['id','sales']].sort_values('id').reset_index().drop('index', axis = 1)
    final_submission.to_csv("submission.csv", index=False)
    return final_submission


# In[ ]:


submission = make_prediction()


# # Some possible improvements
# * Use grid search to choose better hyperparameters, which also implies setting more hyperparameters for training the XGBoost and Random Forest models. However, this is expected to take a really long time.
# * Explore other types of models for prediction such as Support Vector Machines and Linear Regression, or a combination of them.
# * Create different models for each product family as opposed to each store.
# * Explore more preprocessing options to deal with the available data, such as changing how the model deals with events and holidays.
