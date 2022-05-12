#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning with Vizier #
# 
# In the second half of this notebook, we'll demonstrate Vertex Vizier, Vertex AI's hyperparameter optimization service. Among its capabilities are a Bayesian Optimization algorithm to search efficiently within a hyperparameter space, transfer learning to make use of information from previous hyperparameter studies, and automated early stopping when tuning models that train incrementally, like neural nets with stochastic gradient descent or gradient boosted trees. Google Research has a great whitepaper describing the capabilities of Vizier in detail: Google Vizier: A Service for Black-Box Optimization. Also see the Vizier guide for a nice overview.

# In[ ]:


X_test = df_test.loc[:, ['latitude', 'longitude']]

idx_pred = nhbrs.kneighbors(X_test, return_distance=False).squeeze()
idx_pred = idx_pred[:, 1:]  # don't include self


# In[ ]:


df_train = pd.read_csv(data_dir / 'train.csv', index_col='id')
df_pairs = pd.read_csv(data_dir / 'pairs.csv')

df_test = pd.read_csv(data_dir / 'test.csv', index_col='id')
submission = pd.read_csv(data_dir / 'sample_submission.csv', index_col='id')


# In[ ]:


X_pairs.join(y_pairs).groupby('match').mean()


# In[ ]:


# Select matching pairs for submission
submission = (
    matches
    .groupby('id_1')['id_2']
    .apply(lambda x: ' '.join(x))
    .rename_axis('id')
    .rename('matches')
)
display(submission)


# # Create Study #
# 
# Now we'll create a study. A study conducts trials in order to optimize one or more metrics. A trial is a selection of hyperparameter values together with the outcome they produce. In our case, the hyperparameters define a neural net architechture and training regimen, and will produce a validation loss, the metric we hope to minimize.

# In[ ]:


display(df_train)


# In[ ]:


# Create submission
submission.to_csv('submission.csv')


# In[ ]:


# Key: Entry id, Value: Neighbor ids
neighbors = {entry: df_test.index[nhbr].to_list() for entry, nhbr in zip(df_test.index, idx_pred)}
neighbors = pd.Series(neighbors, name='id_2').rename_axis('id_1').explode().reset_index()
neighbors


# In[ ]:


X_pairs = make_similarity_df(df_pairs)
y_pairs = df_pairs.loc[:, 'match'].astype(int)

X_pairs.join(y_pairs)


# In[ ]:


def category_similarity(df):
    X = df['categories_1'].fillna('').str.split(',').combine(
        df['categories_2'].fillna('').str.split(','),
        lambda c1, c2: len(set(c1) & set(c2)),
    ).rename('category_similarity')
    return X.mask(
        df[['categories_1', 'categories_2']].isna().any(axis=1),
        0,
    )


def string_similarity(df, attr, fn):
    X = df[f'{attr}_1'].fillna('').combine(
        df[f'{attr}_2'].fillna(''),
        lambda n1, n2: fn(n1, n2)
    ).rename(f'{attr}_similarity')
    return X.mask(
        df[[f'{attr}_1', f'{attr}_2']].isna().any(axis=1),
        0,
    )
        


# # Notebook Setup #
# 
# 1. Download this Notebook
# Start by creating your own copy of this notebook. Click the Copy and Edit button to the upper right. Now, in the menubar above, click File -> Download Notebook and save a copy of the notebook to your computer. We will reupload this in an AI Notebooks instance to take advantage of the Explainable AI service.
# 
# 2. Download Kaggle API Key
# We'll use the Kaggle API to download the competition data to the notebook instance. You'll need a copy of your Kaggle credentials to authenticate your account.
# 
# From the site header, click on your user profile picture, then on “My Account” from the dropdown menu. This will take you to your account settings at https://www.kaggle.com/account. Scroll down to the section of the page labelled API.
# 
# To create a new token, click on the “Create New API Token” button. This will download a fresh authentication token onto your machine.
# 
# 3. Sign up for Google Cloud Platform
# If you don't have a GCP account already, go to https://cloud.google.com/ and click on “Get Started For Free". This is a two step sign up process where you will need to provide your name, address and a credit card. The starter account is free and it comes with $300 credit that you can use. For this step you will need to provide a Google Account (i.e. your Gmail account) to sign in.
# 
# 4. Create a Project and Enable the Notebook API
# Follow the directions at https://cloud.google.com/notebooks/docs/before-you-begin to setup a notebook project.
# 
# 5. Create a Notebook Instance
# Next, go to https://notebook.new. Enter an Instance name of your choice and then click the blue CREATE button at the end of the page. Be sure to keep the default TensorFlow Enterprise environment. You'll be redirected to a page with a list of your notebook instances. It may take a few minutes for the instance you just created to start up.
# 
# Once the notebook instance is running, click OPEN JUPYTERLAB just to the right of the instance name. You should be redirected to a JupyterLab environment.
# 
# 6. Upload MLB Notebook and API Key
# From inside JupyterLab, click the "Upload Files" (up arrow) button in the file browser on the left and upload the files kaggle.json and vertex-ai-with-mlb-player-digital-engagement.ipynb.
# 
# 7. Authenticate Kaggle API and Download MLB Date
# Run the next cell to download the competition data.

# In[ ]:


def make_similarity_df(df_pairs):
    from fuzzywuzzy import fuzz
    X_pairs = pd.DataFrame(index=df_pairs.index)
    # Categories
    X_pairs = X_pairs.join(category_similarity(df_pairs))
    # Names
    X_pairs = X_pairs.join(string_similarity(df_pairs, attr='name', fn=fuzz.partial_ratio))
    return X_pairs


# # MLB Getting Started #
# 
# The first part of this notebook reproduces the data and model setup of the Getting Started notebook.
# 
# The results of Explainable AI will be easier to understand if we restrict our analysis to a single player. The next cell has a helper function to load data for only a single player, by default Aaron Judge of the NY Yankees, who had the highest overall engagement during the training period.
# 
# We've picked out a few features from the playerBoxScores dataframe, but there are lots more you could try (see the data documentation for a complete description). Increase the number of Fourier components to model seasonality with in more detail. You could also look at explanations for other players -- the players dataframe can tell you the playerId for each player.

# In[ ]:


from sklearn.neighbors import NearestNeighbors

X_test = df_test.loc[:, ['latitude', 'longitude']]

nhbrs = NearestNeighbors(
    n_neighbors=5,
    n_jobs=-1,
).fit(X_test)


# In[ ]:


import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path('../input/foursquare-location-matching')


# In[ ]:


def metric_fn(params):
    # Parse hyperparameters
    units = int(params['units'])
    dropout = params['dropout']
    optimizer = params['optimizer']
    batch_size = int(params['batch_size'])
    # Create and train model
    INPUTS = X_train.shape[-1]
    OUTPUTS = y_train.shape[-1]
    early_stopping = keras.callbacks.EarlyStopping(patience=10,
                                                   restore_best_weights=True)
    model = keras.Sequential([
        layers.InputLayer(name='numpy_inputs', input_shape=(INPUTS, )),
        layers.Dense(units, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(units, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(OUTPUTS),
    ])
    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae'],
    )
    model.fit(X_train,
              y_train,
              validation_data=(X_valid, y_valid),
              batch_size=batch_size,
              epochs=50,
              callbacks=[early_stopping],
              verbose=0)
    # Optimize the metric monitored by `early_stopping` (`val_loss` by default)
    # The metric needs to be reported in this format
    return {'metric_id': early_stopping.monitor, 'value': early_stopping.best}


# # Explain #
# Explainable AI on Vertex AI Notebooks lets you compute feature attributions for neural networks. Feature attributions describe the contribution each features makes to the final prediction relative to a baseline. Feature attributions can help you tune your model by indicating which features are important and which are not. Features with little importance you could consider dropping from your feature set.
# 
# Read more about Vertex Explainable AI here: Introduction to Vertex Explainable AI for Vertex AI. In JupyterLab on Vertex AI Notebooks, you can also review a tutorial on XAI in the tutorials > explainable_ai > sdk_tutorial.ipynb file
# 
# Now we can look at explanations using the explainable_ai_sdk library. Run the following cell on AI Notebooks with a Cloud TF image to see model explanations.

# In[ ]:


matches = neighbors.loc[model.predict(X_pairs_test).astype('bool'), :].append(
    pd.DataFrame({'id_1': df_test.index, 'id_2': df_test.index})  # Everything matches itself
)
matches


# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
).fit(X_pairs, y_pairs)


# # Getting Started on Vertex AI Notebooks #
# 
# This notebook demonstrates how to do the following on Vertex AI, Google's powerful new machine learning platform:
# 
# run the getting started notebook on Vertex AI Notebooks, to load the data, create a model & generate predictions
# explore explainable AI on Vertex AI to refine your features
# tune hyperparameters with Vizier
# It is a complement to the Getting Started with MLB Digital Engagement tutorial which was designed to be run on Kaggle Notebooks.
# 
# This tutorial uses Cloud Notebooks, a billable component of Google Cloud. Learn more about Notebooks pricing.
