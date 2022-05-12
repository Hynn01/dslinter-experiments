#!/usr/bin/env python
# coding: utf-8

# ### Update:
# Added ppscore library : a Python implementation of the Predictive Power Score (PPS)

# In[ ]:


# Import basic libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


# List files available
print(os.listdir("../input/"))


# In this kernel,I am trying to compile some useful Python libraries for data science tasks other than the commonly used ones like **pandas**, **scikit-learn**, **matplotlib**, etc. My idea is to regularly update the kernel to include some awesome Python libraries which can real come in handy for the Data Analysis and Machine learning tasks. The following libraries have been included :
# 
# 
# <div class="alert alert-block alert-warning"><h3> 1. dabl(Data Analysis Baseline Library)</h3> <h3> 2. missingno </h3>
#                                           <h3> 3. Emot </h3>
#                                           <h3> 4. FlashText </h3>
#                                           <h3> 5. PyFlux </h3>
#                                           <h3> 6. Bamboolib </h3>
#                                           <h3> 7. AutoViz </h3>
#                                           <h3> 8. Numerizer</h3>
#                                           <h3> 9. ppscore</div>
# 

# <div class="alert alert-block alert-info">
# <h1> 1. dabl(Data Analysis Baseline Library)</h1></div>
# 
# 
# **dabl** has been created by [Andreas Mueller](https://amueller.github.io/) and it tries to help make supervised machine learning more accessible for beginners, and reduce boiler plate for common tasks. Dabl takes inspirations from scikit-learn and auto-sklearn. Refer to the official [website](https://amueller.github.io/dabl/dev/index.html) for more info.
# 
# 

# In[ ]:


# Installing and loading the library
get_ipython().system('pip install dabl')

import dabl


# ### 1.1 Automated Preprocessing with dabl
# 
# As part of the preprocessing,  dabl will attempt to identify missing values, feature types and erroneous data. if the detection of semantic types (continuous, categorical, ordinal, text, etc) fails, the user can provide `type_hints`. Let's demo the library with the help of the titanic dataset.

# In[ ]:


titanic_df = pd.read_csv('../input/titanic/train.csv')

# A first look at data
titanic_df.shape


# In[ ]:


titanic_df.head()


# In[ ]:


titanic_df_clean = dabl.clean(titanic_df, verbose=1)


# In[ ]:


types = dabl.detect_types(titanic_df_clean)
print(types) 


# ### 1.2 Exploratory Data analysis with dabl
# 
# dabl provides a high-level interface that summarizes several common high-level plots. For low dimensional datasets, all features are shown; for high dimensional datasets, only the most informative features for the given task are shown

# In[ ]:


dabl.plot(titanic_df, target_col="Survived")


# * **Initial Model Building with dabl**
# We can find an initial model for our data. The SimpleClassifier implements the familiar scikit-learn API of fit and predict.

# In[ ]:


ec = dabl.SimpleClassifier(random_state=0).fit(titanic_df, target_col="Survived") 


# The SimpleClassifier first tries several baseline and instantaneous models, potentially on subsampled data, to get an idea of what a low baseline should be.

# <div class="alert alert-block alert-info">
# <h1>2. missingno </h1> </div>
# 
# 
# Data in real life is messy and [missingno](https://github.com/ResidentMario/missingno) helps us to deal with missing values in a dataset with the help of visualisations. With over 2k stars on github, this library is already very popular.

# In[ ]:


# installation and importing the library
get_ipython().system('pip install missingno')
import missingno as msno


# In[ ]:


# Let's check out the missing values first with the train.info() method
titanic_df.info()


# clearly, the `Age`, `Cabin` and the `Embarked` column have missing values. Now lets use missingno to visualise this information.

# ### 2.1 Matrix
# 
# The [ matrix](https://github.com/ResidentMario/missingno#matrix) is a data-dense display which lets you quickly visually pick out patterns in data completion.

# In[ ]:


msno.matrix(titanic_df)


# We can clearly see that Cabin indeed has a lot of missing values. There is also a bar chart on the right.It summarizes the general shape of the data completeness and points out the rows with the maximum and minimum nullity in the dataset.
# 
# We can also sample the data to only show few data points.

# In[ ]:


msno.matrix(titanic_df.sample(50))


# This shows for the first 50 data points, the `Embarked` column has no missing value.

# ### 2.2 Bar Chart
# 
# Alternatively, you can also plot a barchart to show the missing values
# 

# In[ ]:


msno.bar(titanic_df)


# ### 2.3 Heatmap
# 
# [`missingno.heatmap`](https://github.com/ResidentMario/missingno#heatmap) measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another:

# In[ ]:


msno.heatmap(titanic_df)


# <div class="alert alert-block alert-info">
# <h1>3. Emot</h1> </div>
# 
# 
# [Emot](https://github.com/NeelShah18/emot) is an Emoji and Emoticons detection package for Python. It can come in real handy when we have to preprocess our text data to get rid of the emoticons.

# In[ ]:


# installation and importing the library
get_ipython().system('pip install emot')
import emot


# In[ ]:


text = "The weather is ☁️, we might need to carry our ☂️ :("
emot.emoji(text)


# In[ ]:


emot.emoticons(text)


# <div class="alert alert-block alert-info">
# <h1>4. Flashtext</h1> </div>
# 
# 
# [Flastext](https://github.com/vi3k6i5/flashtext) lets you extract Keywords from sentence or Replace keywords in sentences.It is based on the [FlashText algorithm](https://arxiv.org/abs/1711.00046) and is considerably faster than Regular Expressions for NLP tasks.

# In[ ]:


# installation and importing the library
get_ipython().system('pip install flashtext')
from flashtext import KeywordProcessor


# The dataset is from the competition : [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) where our job is to create a ML model to predict whether the test set tweets belong to a disaster or not, in the form of 1 or 0.This is a classic case of a Binary Classification problem.

# In[ ]:


twitter_df =  pd.read_csv('../input/nlp-getting-started/train.csv')
twitter_df.head()


# Lets create a corpus of all the tweets in the training set

# In[ ]:


corpus = ', '.join(twitter_df.text)
corpus[:1000]


# ### 4.1 Extract Keywords or searching for words in a corpus

# In[ ]:


# How many times does the word 'flood' appear in the corpus?
processor = KeywordProcessor()
processor.add_keyword('flood')
found = processor.extract_keywords(corpus)
print(len(found))
  


# ### 4.2 Replacing words in a text document

# In[ ]:


# Replacing all occurences of word 'forest fire'(case insensitive) with fire

processor = KeywordProcessor(case_sensitive = False)
processor.add_keyword('forest fire','fire')
found = processor.replace_keywords(corpus)
print(found[:100])


# The word `Forest Fire` gets replaced with only fire. In the same way we could also replace special characters, hyperlinks etc from a document.

# <div class="alert alert-block alert-info">
# <h1>5. PyFlux</h1> </div>
# 
# 
# Time series analysis is one of the most frequently encountered problems in the Machine learning domain. **[PyFlux](https://pyflux.readthedocs.io/en/latest/getting_started.html)** is an open source library in Python explicitly built for working with time series problems. The library has an excellent array of modern time series models including but not limited to ARIMA, GARCH, and VAR models. In short, PyFlux offers a probabilistic approach to time series modeling. Worth trying out.
# 
# To see how this library works, I'll be using the [NIFTY-50 Stock Market Data (2000-2019)](https://www.kaggle.com/rohanrao/nifty50-stock-market-data) dataset. The data is the price history and trading volumes of the fifty stocks in the index NIFTY 50 from NSE (National Stock Exchange) India. Let's use just stocks of just one company.

# In[ ]:


# installing and importing the library

get_ipython().system('pip install pyflux')

import pyflux as pf


# In[ ]:


maruti = pd.read_csv("../input/nifty50-stock-market-data/MARUTI.csv")
# Convert string to datetime64
maruti ['Date'] = maruti ['Date'].apply(pd.to_datetime)


# ### 5.1 Visualise the data

# In[ ]:


maruti_df = maruti[['Date','VWAP']]

#Set Date column as the index column.
maruti_df.set_index('Date', inplace=True)
maruti_df.head()


# In[ ]:


plt.figure(figsize=(15, 5))
plt.ylabel("Volume Weighted Average Price'")
plt.plot(maruti_df)


# ### 5.2 Modelling
# Let's run an **ARIMA** Model. We can build an ARIMA model as follows, specifying the order of model we want, as well as a pandas DataFrame or numpy array carrying the data.

# In[ ]:


my_model = pf.ARIMA(data=maruti_df, ar=4, ma=4, family=pf.Normal())
print(my_model.latent_variables)

result = my_model.fit("MLE")
result.summary()

my_model.plot_z(figsize=(15,5))
my_model.plot_fit(figsize=(15,10))
my_model.plot_predict_is(h=50, figsize=(15,5))
my_model.plot_predict(h=20,past_values=20,figsize=(15,5))


# <div class="alert alert-block alert-info">
# <h1>6. Bamboolib</h1> </div>
# 
# **bamboolib** is a GUI for data exploration & transformation in Python Pandas. It is compatible with Jupyter Notebook and JupyterLab. Bamboolib is an otherwise closed source library but can be used for free for open data via Binder or Kaggle.  For further information on Bamboolib, please visit their  [official website](https://bamboolib.8080labs.com/).

# In[ ]:


# installing and importing the library

#!pip install --upgrade bamboolib>=1.2.1


# After the installation, which takes some time,reload this browser page (don't press the Restart Session button). 
# 
# Try installing this library in a new kernel where it works fine(https://www.kaggle.com/tkrabel/bamboolib-titanic-df-wrangling/). It shows some error in this current kernel.

# In[ ]:


# Importing the necessary libraries 
#import bamboolib as bam
#bam.enable()


#Importing the training dataset
#df = pd.read_csv('../input/titanic/train.csv')
#df


# Import the Titanic dataset( or any other depending on your choice)
# 
# On running the above cell, magic bamboolib button appears that looks like the image below:
# ![](https://cdn-images-1.medium.com/max/800/1*cviqH-lxhV1jSZbfbTFojg.png)
# 
# Click on the green `Show Bamboolib UI` button to see the various available options in the bamboolib library.
# 
# 
# Some of the available features are:
# 
# ### 6.1 Interactive UI
# 
# 
# Interact with the pandas dataframe easily by scrolling and selecting, with a number of options.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*CwhOsrH-6P_5tyJ5ixq2RA.gif)
# 
# ### 6.2 Split strings
# 
# Let's split the Name of the passengers into last name and first name by simply highlighting the separator with the mouse.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*r_9OUdxHUat7MUFwFME8dQ.gif)
# 
# ### 6.3 Dropping Missing values
# Easily drop the missing values
# 
# ![](https://cdn-images-1.medium.com/max/800/1*UHAqeQkg4QXUbVhQta93MQ.gif)
# 
# ### 6.4 Live code export
# 
# ![](https://cdn-images-1.medium.com/max/800/1*PgBwE32hqJg196WxaDNFIw.gif)
# 
# ### 6.5 Data Visualization
# Various types of visualizations can be obtained by clicking the Visualise Dataframe, button. 
# 
# * ### Glimpse
# 
# gives an overview of all the columns, datatypes, unique and missing values
# ![](https://cdn-images-1.medium.com/max/800/1*9REEl9AtWDZOs54Sm0z5fA.png)
# 
# * ### Columns 
# Give information about individual columns giving univariate and bivariate summaries. For instance, for the 'Survived' column, we can see the distribution which is 60 to 40. We can also look at the predictor and the best predictor for the Survived column is 'Sex' of the passenger. The visualizations are interactive and can be clicked to understand the relationships between different columns.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*a_s02VoSDzJLi-VBOrdYxQ.gif)
# 
# * ### Predictor patterns:
# 
# This value is obtained when each cell shows the normalized predictors score of the feature on the x-axis for the target on the y-axis.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*uNKRmul_KATuNLdf1u5GHQ.gif)
# 
# * ### Correlation Matrix :
# 
# ![](https://cdn-images-1.medium.com/max/800/1*RLz-XZVIL06fyw3ZZQ1mRA.gif)
# 
# Some people may argue that it is better to code than use the point and click options, but this might come handy for the people who want to have a quick look at the data and get an overview instantly.
# 

# <div class="alert alert-block alert-info">
# <h1>7. AutoViz</h1> </div>
# 
# 
# [AutoViz](https://github.com/AutoViML/AutoViz) automatically visualizes any dataset, any size with a single line of code.It performs automatic visualization of any dataset with one line. Give any input file (CSV, txt or json) and AutoViz will visualize it.
# 
# To see how this library works, I'll be using the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset. With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# In[ ]:


# Installing the library
get_ipython().system('pip install autoviz')


# In[ ]:


# Instantiate the library
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()


# In[ ]:


# Reading the dataset
house_price = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
house_price.head(3)


# In[ ]:


sep = '\,'
target = 'SalePrice'
datapath = '../input/house-prices-advanced-regression-techniques/'
filename = 'train.csv'
df = pd.read_csv(datapath+filename,sep=sep,index_col=None)
df = df.sample(frac=1.0,random_state=42)
print(df.shape)
df.head()


# In[ ]:


dft = AV.AutoViz(datapath+filename, sep=sep, depVar=target, dfte=None, header=0, verbose=0,
                            lowess=False,chart_format='svg',max_rows_analyzed=1500,max_cols_analyzed=30)


# 

# <div class="alert alert-block alert-info">
# <h1>8. Numerizer</h1> </div>
# 
# [Numerizer](https://github.com/jaidevd/numerizer) is a Python module for converting natural language numbers into ints and floats. It is a port of the Ruby gem [numerizer](https://github.com/jduff/numerizer). This could be really useful when preprocessing text data.

# In[ ]:


get_ipython().system('pip install numerizer')


# In[ ]:


from numerizer import numerize
numerize('forty two')


# In[ ]:


numerize('forty-two')


# In[ ]:


numerize('four hundred and sixty two')


# In[ ]:


numerize('twenty one thousand four hundred and seventy three')


# In[ ]:


numerize('one billion and one')


# In[ ]:


numerize('nine and three quarters')


# In[ ]:


numerize('platform nine and three quarters')


# <div class="alert alert-block alert-info">
# <h1>9. ppscore</h1> </div>
# 
# [ppscore](https://github.com/8080labs/ppscore)brought to you by the makers of Bamboolib, is a Python implementation of the Predictive Power Score (PPS). The PPS is an asymmetric, data-type-agnostic score that can detect linear or non-linear relationships between two columns. The score ranges from 0 (no predictive power) to 1 (perfect predictive power). It can be used as an alternative to the correlation (matrix).
# 
# Let us see it demo through the Titanic Dataset

# In[ ]:


get_ipython().system('pip install ppscore')


# In[ ]:


import ppscore as pps

def heatmap(df):
    ax = sns.heatmap(df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    ax.set_title('PPS matrix')
    ax.set_xlabel('feature')
    ax.set_ylabel('target')
    return ax


def corr_heatmap(df):
    ax = sns.heatmap(df, vmin=-1, vmax=1, cmap="BrBG", linewidths=0.5, annot=True)
    ax.set_title('Correlation matrix')
    return ax


# ### Single Predictive Power Score
# 
# How well can Sex predict the Survival probability?

# In[ ]:


titanic_df_subset = titanic_df[["Survived", "Pclass", "Sex", "Age", "Ticket", "Fare", "Embarked"]]
pps.score(titanic_df_subset, "Sex", "Survived")


# ### PPS matrix
# 
# which predictive patterns exist between the columns?

# In[ ]:



matrix = pps.matrix(titanic_df_subset)
heatmap(matrix)


# Let's compare the PPS matrix to the basic correlation matrix

# In[ ]:


# Correlation Matrix
f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)
corr_heatmap(titanic_df_subset.corr())

f.add_subplot(1,2, 2)
matrix = pps.matrix(titanic_df_subset)
heatmap(matrix)


# These were some of the interesting and  useful python libraries for data science, that I have come across recently. In case you know about others which can be added to the list, do mention them in the comments below. 
