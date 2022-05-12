#!/usr/bin/env python
# coding: utf-8

# # Machine Learning model to Retail Industry

# # Section 1: Business understanding

# This project presents a Machine Learning model (Random Forest) to predict whether or not a customer will use an offer sent by a business of the Retail Industry.

# The project presents answers to the following questions:
# 
# Question 1: Will a customer respond to an offer sent by the commerce?

# # Section 2: Data Understanding

# ### Gather process

# In[ ]:


import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# read in the json files
portfolio = pd.read_json('../input/starbucks-app-customer-reward-program-data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('../input/starbucks-app-customer-reward-program-data/profile.json', orient='records', lines=True)
transcript = pd.read_json('../input/starbucks-app-customer-reward-program-data/transcript.json', orient='records', lines=True)


# ### Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record

# # Section 3: Prepare Data

# ### Explore: Transcript (transactions)

# In[ ]:


transcript.head()


# In[ ]:


# we have dictionaries in the value column
list(transcript.value.iloc[12658].keys())


# In[ ]:


# there are 4 types of events
transcript.event.unique()


# ### Prepare data: clean process (transcript df)

# In this dataframe we can apply the following modifications:
# 
# - we can separate the column value into three more columns: "offer_id", "amount" e "reward"
# - we can separate the column value into three more columns (one hot encoding): "offer_received", "offer_viewed", "transaction" e "offer_completed"

# In[ ]:


# we have three different keys in the value column (dictionary): offer_id, amount and reward
# let's create functions to extract data from these dictionaries' keys

def expand_transcript_offer_id():
    '''
    this function extract the data: "offer_id" from the value column
    '''
    offer_id_list = []

    for dic in transcript.value.iloc[0:]:
        if 'offer id' in list(dic.keys()):
            offer_id_list.append(dic['offer id'])
        elif 'offer_id' in list(dic.keys()):
            offer_id_list.append(dic['offer_id'])
        else:
            offer_id_list.append(np.nan)
    
    return offer_id_list
    
def expand_transcript_amount(entry):
    '''
    this function extract the data: "amount" from the value column
    '''
    try:
        if type(entry['amount']) == float or int:
            return entry['amount']
    except:
        return np.nan
    
def expand_transcript_reward(entry):
    '''
    this function extract the data: "reward" from the value column
    '''
    try:
        if type(entry['reward']) == float or int:
            return entry['reward']
    except:
        return np.nan


# In[ ]:


def expand_transcript_all(transcript=transcript):
    '''
    this function use the following functions:
    - expand_transcript_offer_id()
    - expand_transcript_amount(entry)
    - expand_transcript_reward(entry)
    to create new columns in the transcript dataframe from the data in the value column
    '''
    
    # then we create three new columns     
    offer_id_list = expand_transcript_offer_id()
    transcript['offer_id'] = offer_id_list
    transcript['amount'] = transcript['value'].apply(lambda x: expand_transcript_amount(x))
    transcript['reward'] = transcript['value'].apply(lambda x: expand_transcript_reward(x))
    
    # and then drop the "value" column
    transcript.drop(columns=["value"], inplace=True)
    return transcript

transcript = expand_transcript_all(transcript=transcript)


# In[ ]:


transcript.head()


# In[ ]:


transcript.hist(bins=100, figsize=(10, 5));


# In[ ]:


# we can see that one person can have more than one transaction (event) and there are 17000 clients
len(transcript), len(transcript.person.unique())


# In this dataset we have information of the user's transactions and the offers the users receive. So, we can inquire which offers the users received, viewed and completed and which they just received and didn't used. It can be interesting if we apply one hot encoding to each event: received, viewed and completed. So, we will separate the dataset by event and then merge to apply one hot encoding.
# 
# We also have the event: transaction. Using this event we can see the number of transactions of each user and the total amount the users have spent. This information can be valuable to complement the profile dataset.

# ### events: received, viewed, completed, transaction

# In[ ]:


def separate_transcript(transcript=transcript):
    '''
    this function separates the transcript df into 4 df according to
    the 4 types of events
    '''
    # the received df just tells us that a user received an offer
    # so, it doesn't have amount and reward information, let's drop
    received = transcript[transcript['event'] == "offer received"]
    received.drop(columns=['amount', 'reward'], inplace=True)
    
    # the received df just tells us that a user viewed an offer
    # so, it doesn't have amount and reward information, let's drop
    viewed = transcript[transcript['event'] == "offer viewed"]
    viewed.drop(columns=['amount', 'reward'], inplace=True)
    
    # the completed df tells us that a user completed (responded to) an offer
    # so, it just have reward information, let's drop amount column
    completed = transcript[transcript['event'] == "offer completed"]
    completed.drop(columns=['amount'], inplace=True)
    
    # the transaction df tells us the amount of each transaction
    # so, itjust have amount information, let's drop offer_id and reward columns 
    transaction = transcript[transcript['event'] == "transaction"]
    transaction.drop(columns=['offer_id', 'reward'], inplace=True)
    
    #change the columns' names
    received.columns = ['customer_id', 'event', 'time', 'offer_id']
    viewed.columns = ['customer_id', 'event', 'time', 'offer_id']
    completed.columns = ['customer_id', 'event', 'time', 'offer_id', 'reward']
    transaction.columns = ['customer_id', 'event', 'time', 'amount']
    
    return received, viewed, completed, transaction

received, viewed, completed, transaction = separate_transcript(transcript=transcript)


# Now we are going to apply one hot encoding to the events: received, viewed and completed by merging the dataframes with a new column with a 1 value depending on the event

# In[ ]:


def one_hot_events(received=received, viewed=viewed, completed=completed):
    '''
    this function creates a new column with one values according to the event
    and then merge the dataframes
    '''
    # column with 1 value
    received['offer_received'] = 1
    viewed['offer_viewed'] = 1
    completed['offer_completed'] = 1
    
    # merge the three dfs
    event_df = received.merge(
        viewed, how="left", on=['customer_id', "offer_id"]).merge(
        completed, how="left", on=['customer_id', "offer_id"])
    event_df.drop(columns=["event_x", "event_y", "event"], inplace=True)
    event_df.columns = ['customer_id', 'time_received', 'offer_id', 'offer_received',
                        'time_viewed', 'offer_viewed', 'time_completed', 'reward', 'offer_completed']
    
    return event_df


# In[ ]:


event_df = one_hot_events(received=received, viewed=viewed, completed=completed)


# Since we are interested in whether or not someone will respond to an offer, we are going to drop offers that weren't seen.

# Also, we can see that there are some offers that were completed before viewed, so the customer completed the offer but he didn't know it exists. Since we are just interested in the offers that the customers responded to, we are going to drop offers completed before they were seen.

# In[ ]:


def clean_event_df(event_df=event_df):
    '''
    this function will create a dataframe that indicate whether or not the
    customer responded to an offer with binary data. Columns:
    *customer_responded:
    1 - the customer viewed and completed the offer
    0 - the customer viewed but didn't used the offer
    *customer_id
    *offer_id
    '''
    event_df = event_df[event_df['offer_viewed'] == 1]
    
    # we replace the nan with 1000 to then apply the customer_responded
    # function
    event_df.time_completed = event_df.time_completed.fillna(1000)
    
    def customer_responded(event_df=event_df):
        '''
        This function returns the values 0, 1 or nan depending on the values
        of the 'time_completed' and 'time_viewed' columns
        '''
        if event_df['time_completed'] == 1000:
            return 0
        elif event_df['time_completed'] > event_df['time_viewed']:
            return 1
        else:
            return np.nan
    
    # we create a new column applying the customer_responded function
    event_df['customer_responded'] = event_df.apply(customer_responded, axis=1)
    
    # drop customers who viewed the offer after completed it
    event_df.dropna(subset=['customer_responded'], inplace=True)
    
    # replace 1000 with nan again
    event_df.time_completed = event_df.time_completed.replace(1000, np.nan)
    
    # drop the columns that we won't use to build the ML model
    event_df.drop(columns=['time_received','offer_received','time_viewed','offer_viewed',
                          'time_completed','reward','offer_completed'], inplace=True)
    
    return event_df


# In[ ]:


event_df = clean_event_df(event_df=event_df)


# In[ ]:


event_df.head()


# ## Explore: Portfolio (offers' data) 

# In[ ]:


portfolio.head()


# In[ ]:


portfolio.shape


# In[ ]:


portfolio[['difficulty', 'duration', 'reward']].hist(bins=50, figsize=(10, 5));


# ## Prepare data: clean process (Portfolio df)

# In this dataframe we can make the following modifications:
# 
# - binary data of each channel (one hot encoding)
# - binary data of each offer_type (one hot encoding)

# In[ ]:


def one_hot_channels_offer_type(portfolio=portfolio):
    '''
    this function applies one hot encoding in the channels column of the portfolio df
    '''
    channels_list = ['web', 'email', 'mobile', 'social']
    portfolio['web_channel'] = portfolio['channels'].apply(lambda x: 1 if channels_list[0] in x else 0)
    portfolio['email_channel'] = portfolio['channels'].apply(lambda x: 1 if channels_list[1] in x else 0)
    portfolio['mobile_channel'] = portfolio['channels'].apply(lambda x: 1 if channels_list[2] in x else 0)
    portfolio['social_channel'] = portfolio['channels'].apply(lambda x: 1 if channels_list[3] in x else 0)
    
    # drop the channels columns
    portfolio.drop(columns=["channels"], inplace=True)
    
    # one hot offer_type
    try:
        portfolio = pd.get_dummies(portfolio, columns=['offer_type'], prefix="offer_type_")
    except:
        pass
    
    portfolio.columns = ['reward', 'difficulty', 'duration', 'offer_id', 'web_channel', 'email_channel', 
                     'mobile_channel', 'social_channel', 'offer_type__bogo','offer_type__discount',
                     'offer_type__informational']
    
    return portfolio

portfolio = one_hot_channels_offer_type(portfolio=portfolio)


# In[ ]:


portfolio[['difficulty', 'duration', 'reward']].hist(bins=50, figsize=(10, 5));


# In[ ]:


portfolio.head()


# ## Explore: Profile (customer's data)

# In[ ]:


profile.head()


# In this dataframe we can do the following modifications:
# 
# - total time using the app
# - one hot encoding of gender

# In[ ]:


profile.age.describe()


# We can see that probably customers who didn't inform their age, are registered as 118 years old, which don't make sense.

# In[ ]:


(profile['age'] == 118).sum()/len(profile), profile['income'].isna().sum()/len(profile)


# ## Prepare data: clean process (profile df)

# We can see that the percentage of people who didn't inform age is the same as people who didn't inform their income and their gender as well. And, since this percentage is relatively small (13%) it makes sense if we drop the these rows, because we're not going to lose too much information.

# In[ ]:


def modify_profile(profile=profile):
    try:
        profile = pd.get_dummies(profile, columns=['gender'], dummy_na=False, prefix="gender_")
    except:
        pass
    
    profile.became_member_on = pd.to_datetime(profile.became_member_on.apply(str), format='%Y%m%d')
    
    profile['now'] = pd.to_datetime('now')
    profile['days_as_member'] = (profile['now'] - profile['became_member_on']).dt.days
    profile.drop(columns=['became_member_on', 'now'], inplace=True)
    
    # drop nan
    profile.dropna(subset=['income'], inplace=True)
    
    return profile

profile = modify_profile()
profile.head()


# In[ ]:


profile[['age', 'income', 'days_as_member']].hist(bins=30, figsize=(10, 5));


# In[ ]:


# we can use the transactions df that we extracted from the transcript df to add valuable 
# information to the profile df, which contains customer's data
transaction.head()


# ## Prepare data: merge data (transcript df)

# We're going to calculate the number of transactions per user and the total amount per user. This information will be valuable to add to the profile df:
# 
# - Number of transactions
# - Total amount

# In[ ]:


def merge_profile_df(profile=profile, transaction=transaction):
    '''
    this function merge the profile df with data from transaction df. Columns:
    - age: customer's age
    - customer_id
    - income: customer's income
    - gender: M, F, O with one-hot-encoding
    - days_as_member: time in days using the app
    - number_of_transactions: total transactions using the app
    - total_amount: total amount of money spent on the app
    '''
    # number of transactions per customer
    number_of_transactions = pd.DataFrame(
        transaction.groupby(['customer_id']).size(), columns=['number_of_transactions']).reset_index(level=0)
    number_of_transactions.columns = ['customer_id', 'number_of_transactions']
    
    # merge the number of transactions on customer_id
    profile.columns = ['age', 'customer_id', 'income', 'gender__F', 'gender__M', 'gender__O', 'days_as_member']
    profile = profile.merge(number_of_transactions, how='left', on='customer_id')
    
    # total amount spent per customer
    total_amount = pd.DataFrame(transaction.groupby(['customer_id']).sum()['amount'])
    total_amount = total_amount.reset_index(level=0)
    total_amount.columns = ['customer_id', 'total_amount']
    
    # merge the total_amount on customer_id
    profile = profile.merge(total_amount, how='left', on='customer_id')
    
    return profile


# Since we have just 2.2% of missing values after the merge operation, it does make sense to drop the rows with missing values.
# 
# The profile df after the merge provides interesting information of the consumption of the customers based on demographic data. So, we can make some visuals with this df.

# In[ ]:


profile = merge_profile_df(profile=profile, transaction=transaction)
profile.head()


# In[ ]:


profile[['age', 'income', 'days_as_member', 'number_of_transactions', 'total_amount']].hist(bins=50, figsize=(10, 8));


# ## Merge all dataframes

# Let's add the customer's information to the event_df dataframe

# In[ ]:


df = event_df.merge(profile, how='left', on='customer_id').dropna()


# And then, let's add the offer's information

# In[ ]:


df = df.merge(portfolio, how='left', on='offer_id')


# In[ ]:


df.head()


# In[ ]:


# Let's see if the data is balanced
df.customer_responded.sum() / len(df)


# # Section 4: Data Modeling 

# Then, let's separate the df into X and Y (target) variables

# In[ ]:


int_cols = ['age', 'income', 'gender__F', 'gender__M', 'gender__O', 'days_as_member',
     'number_of_transactions', 'reward', 'difficulty', 'duration', 'web_channel',
     'email_channel', 'mobile_channel', 'social_channel', 'offer_type__bogo', 
     'offer_type__discount', 'offer_type__informational']


# In[ ]:


df.columns[3:]


# In[ ]:


X = df[df.columns[3:]]
X[int_cols] = X[int_cols].astype(int)
y = df['customer_responded'].astype('category')


# In[ ]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, shuffle=True)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

clf = RandomForestClassifier()
clf.fit(train_X, train_y)


# In[ ]:


pred = clf.predict(val_X)
pred


# # Section 5: Evaluate the Model

# In[ ]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(val_y, pred)
print('Accuracy: ', accuracy)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(val_y,pred))


# # Section 6: Hyperparameter tuning

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 4)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 4)]
max_depth.append(None)
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'bootstrap': bootstrap}

print(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=10, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_X, train_y)


# ## Evaluate the tuned model

# In[ ]:


pred2 = np.around(rf_random.predict(val_X))
pred2


# In[ ]:


accuracy2 = accuracy_score(val_y, pred2)
print('Accuracy of tuned model: ', accuracy2)


# In[ ]:


print(classification_report(val_y,pred2))


# In[ ]:


print('Accuracy improvement of: {0:.3g} % with hyperparameter tuning (GridSearchCV)'.format((accuracy2-accuracy)*100))


# # Section 6: Conclusions

# - The data preprocessing methods allowed us to build a highly accurate model
# - The Random Forest Regressor classifier algorithm showed to be effective for the binary classification in this business case since the base model showed an accuracy of 92.5% and precision, recall and f1-score metrics almost all over 90%
# - Performing hyperparameter tuning improved the model accuracy in 0.344%
