#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg">
#     
# ## [mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course 
# 
# Author: [Yury Kashnitsky](https://yorko.github.io). Translated and edited by [Christina Butsko](https://www.linkedin.com/in/christinabutsko/), [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/), [Anastasia Manokhina](https://www.linkedin.com/in/anastasiamanokhina), Sergey Isaev, and [Artem Trunov](https://www.linkedin.com/in/datamove/). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# # <center> Topic 1. Exploratory data analysis with Pandas
# 
# <img align="center" src="https://habrastorage.org/files/10c/15f/f3d/10c15ff3dcb14abdbabdac53fed6d825.jpg"  width=50% />
# 
# ## Article outline
# 1. [Demonstration of main Pandas methods](#1.-Demonstration-of-main-Pandas-methods)
# 2. [First attempt at predicting telecom churn](#2.-First-attempt-at-predicting-telecom-churn)
# 3. [Assignments](#3.-Assignments)
# 4. [Useful resources](#4.-Useful-resources)

# ## 1. Demonstration of main Pandas methods
# Well... There are dozens of cool tutorials on Pandas and visual data analysis. If you are already familiar with these topics, you can wait for the 3rd article in the series, where we get into machine learning.  
# 
# **[Pandas](http://pandas.pydata.org)** is a Python library that provides extensive means for data analysis. Data scientists often work with data stored in table formats like `.csv`, `.tsv`, or `.xlsx`. Pandas makes it very convenient to load, process, and analyze such tabular data using SQL-like queries. In conjunction with `Matplotlib` and `Seaborn`, `Pandas` provides a wide range of opportunities for visual analysis of tabular data.
# 
# The main data structures in `Pandas` are implemented with **Series** and **DataFrame** classes. The former is a one-dimensional indexed array of some fixed data type. The latter is a two-dimensional data structure - a table - where each column contains data of the same type. You can see it as a dictionary of `Series` instances. `DataFrames` are great for representing real data: rows correspond to instances (examples, observations, etc.), and columns correspond to features of these instances.

# In[ ]:


import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)


# We'll demonstrate the main methods in action by analyzing a [dataset](https://bigml.com/user/francisco/gallery/dataset/5163ad540c0b5e5b22000383) on the churn rate of telecom operator clients. Let's read the data (using `read_csv`), and take a look at the first 5 lines using the `head` method:

# In[ ]:


df = pd.read_csv("../input/telecom_churn.csv")
df.head()


# <details>
# <summary>About printing DataFrames in Jupyter notebooks</summary>
# <p>
# In Jupyter notebooks, Pandas DataFrames are printed as these pretty tables seen above while `print(df.head())` is less nicely formatted.
# By default, Pandas displays 20 columns and 60 rows, so, if your DataFrame is bigger, use the `set_option` function as shown in the example below:
# 
# ```python
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_rows', 100)
# ```
# </p>
# </details>
# 
# Recall that each row corresponds to one client, an **instance**, and columns are **features** of this instance.

# Let’s have a look at data dimensionality, feature names, and feature types.

# In[ ]:


print(df.shape)


# From the output, we can see that the table contains 3333 rows and 20 columns.
# 
# Now let's try printing out column names using `columns`:

# In[ ]:


print(df.columns)


# We can use the `info()` method to output some general information about the dataframe: 

# In[ ]:


print(df.info())


# `bool`, `int64`, `float64` and `object` are the data types of our features. We see that one feature is logical (`bool`), 3 features are of type `object`, and 16 features are numeric. With this same method, we can easily see if there are any missing values. Here, there are none because each column contains 3333 observations, the same number of rows we saw before with `shape`.
# 
# We can **change the column type** with the `astype` method. Let's apply this method to the `Churn` feature to convert it into `int64`:

# In[ ]:


df["Churn"] = df["Churn"].astype("int64")


# The `describe` method shows basic statistical characteristics of each numerical feature (`int64` and `float64` types): number of non-missing values, mean, standard deviation, range, median, 0.25 and 0.75 quartiles.

# In[ ]:


df.describe()


# In order to see statistics on non-numerical features, one has to explicitly indicate data types of interest in the `include` parameter.

# In[ ]:


df.describe(include=["object", "bool"])


# For categorical (type `object`) and boolean (type `bool`) features we can use the `value_counts` method. Let's have a look at the distribution of `Churn`:

# In[ ]:


df["Churn"].value_counts()


# 2850 users out of 3333 are *loyal*; their `Churn` value is `0`. To calculate fractions, pass `normalize=True` to the `value_counts` function.

# In[ ]:


df["Churn"].value_counts(normalize=True)


# 
# ### Sorting
# 
# A DataFrame can be sorted by the value of one of the variables (i.e columns). For example, we can sort by *Total day charge* (use `ascending=False` to sort in descending order):
# 

# In[ ]:


df.sort_values(by="Total day charge", ascending=False).head()


# We can also sort by multiple columns:

# In[ ]:


df.sort_values(by=["Churn", "Total day charge"], ascending=[True, False]).head()


# ### Indexing and retrieving data
# 
# A DataFrame can be indexed in a few different ways. 
# 
# To get a single column, you can use a `DataFrame['Name']` construction. Let's use this to answer a question about that column alone: **what is the proportion of churned users in our dataframe?**

# In[ ]:


df["Churn"].mean()


# 14.5% is actually quite bad for a company; such a churn rate can make the company go bankrupt.
# 
# **Boolean indexing** with one column is also very convenient. The syntax is `df[P(df['Name'])]`, where `P` is some logical condition that is checked for each element of the `Name` column. The result of such indexing is the DataFrame consisting only of rows that satisfy the `P` condition on the `Name` column. 
# 
# Let's use it to answer the question:
# 
# **What are average values of numerical features for churned users?**

# In[ ]:


df[df["Churn"] == 1].mean()


# **How much time (on average) do churned users spend on the phone during daytime?**

# In[ ]:


df[df["Churn"] == 1]["Total day minutes"].mean()


# 
# **What is the maximum length of international calls among loyal users (`Churn == 0`) who do not have an international plan?**
# 
# 

# In[ ]:


df[(df["Churn"] == 0) & (df["International plan"] == "No")]["Total intl minutes"].max()


# DataFrames can be indexed by column name (label) or row name (index) or by the serial number of a row. The `loc` method is used for **indexing by name**, while `iloc()` is used for **indexing by number**.
# 
# In the first case below, we say *"give us the values of the rows with index from 0 to 5 (inclusive) and columns labeled from State to Area code (inclusive)"*. In the second case, we say *"give us the values of the first five rows in the first three columns"* (as in a typical Python slice: the maximal value is not included).

# In[ ]:


df.loc[0:5, "State":"Area code"]


# In[ ]:


df.iloc[0:5, 0:3]


# If we need the first or the last line of the data frame, we can use the `df[:1]` or `df[-1:]` construct:

# In[ ]:


df[-1:]


# 
# ### Applying Functions to Cells, Columns and Rows
# 
# **To apply functions to each column, use `apply()`:**
# 

# In[ ]:


df.apply(np.max)


# The `apply` method can also be used to apply a function to each row. To do this, specify `axis=1`. Lambda functions are very convenient in such scenarios. For example, if we need to select all states starting with W, we can do it like this:

# In[ ]:


df[df["State"].apply(lambda state: state[0] == "W")].head()


# The `map` method can be used to **replace values in a column** by passing a dictionary of the form `{old_value: new_value}` as its argument:

# In[ ]:


d = {"No": False, "Yes": True}
df["International plan"] = df["International plan"].map(d)
df.head()


# The same thing can be done with the `replace` method:

# In[ ]:


df = df.replace({"Voice mail plan": d})
df.head()


# 
# ### Grouping
# 
# In general, grouping data in Pandas works as follows:
# 

# 
# ```python
# df.groupby(by=grouping_columns)[columns_to_show].function()
# ```

# 
# 1. First, the `groupby` method divides the `grouping_columns` by their values. They become a new index in the resulting dataframe.
# 2. Then, columns of interest are selected (`columns_to_show`). If `columns_to_show` is not included, all non groupby clauses will be included.
# 3. Finally, one or several functions are applied to the obtained groups per selected columns.
# 
# Here is an example where we group the data according to the values of the `Churn` variable and display statistics of three columns in each group:

# In[ ]:


columns_to_show = ["Total day minutes", "Total eve minutes", "Total night minutes"]

df.groupby(["Churn"])[columns_to_show].describe(percentiles=[])


# Let’s do the same thing, but slightly differently by passing a list of functions to `agg()`:

# In[ ]:


columns_to_show = ["Total day minutes", "Total eve minutes", "Total night minutes"]

df.groupby(["Churn"])[columns_to_show].agg([np.mean, np.std, np.min, np.max])


# 
# ### Summary tables
# 
# Suppose we want to see how the observations in our sample are distributed in the context of two variables - `Churn` and `International plan`. To do so, we can build a **contingency table** using the `crosstab` method:
# 
# 

# In[ ]:


pd.crosstab(df["Churn"], df["International plan"])


# In[ ]:


pd.crosstab(df["Churn"], df["Voice mail plan"], normalize=True)


# We can see that most of the users are loyal and do not use additional services (International Plan/Voice mail).
# 
# This will resemble **pivot tables** to those familiar with Excel. And, of course, pivot tables are implemented in Pandas: the `pivot_table` method takes the following parameters:
# 
# * `values` – a list of variables to calculate statistics for,
# * `index` – a list of variables to group data by,
# * `aggfunc` – what statistics we need to calculate for groups, ex. sum, mean, maximum, minimum or something else.
# 
# Let's take a look at the average number of day, evening, and night calls by area code:

# In[ ]:


df.pivot_table(
    ["Total day calls", "Total eve calls", "Total night calls"],
    ["Area code"],
    aggfunc="mean",
)


# 
# ### DataFrame transformations
# 
# Like many other things in Pandas, adding columns to a DataFrame is doable in many ways.
# 
# For example, if we want to calculate the total number of calls for all users, let's create the `total_calls` Series and paste it into the DataFrame:
# 
# 

# In[ ]:


total_calls = (
    df["Total day calls"]
    + df["Total eve calls"]
    + df["Total night calls"]
    + df["Total intl calls"]
)
df.insert(loc=len(df.columns), column="Total calls", value=total_calls)
# loc parameter is the number of columns after which to insert the Series object
# we set it to len(df.columns) to paste it at the very end of the dataframe
df.head()


# It is possible to add a column more easily without creating an intermediate Series instance:

# In[ ]:


df["Total charge"] = (
    df["Total day charge"]
    + df["Total eve charge"]
    + df["Total night charge"]
    + df["Total intl charge"]
)
df.head()


# To delete columns or rows, use the `drop` method, passing the required indexes and the `axis` parameter (`1` if you delete columns, and nothing or `0` if you delete rows). The `inplace` argument tells whether to change the original DataFrame. With `inplace=False`, the `drop` method doesn't change the existing DataFrame and returns a new one with dropped rows or columns. With `inplace=True`, it alters the DataFrame.

# In[ ]:


# get rid of just created columns
df.drop(["Total charge", "Total calls"], axis=1, inplace=True)
# and here’s how you can delete rows
df.drop([1, 2]).head()


# ## 2. First attempt at predicting telecom churn
# 
# 
# Let's see how churn rate is related to the *International plan* feature. We'll do this using a `crosstab` contingency table and also through visual analysis with `Seaborn` (however, visual analysis will be covered more thoroughly in the next article).
# 

# In[ ]:


pd.crosstab(df["Churn"], df["International plan"], margins=True)


# In[ ]:


# some imports to set up plotting
import matplotlib.pyplot as plt
# pip install seaborn
import seaborn as sns

# Graphics in retina format are more sharp and legible
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


sns.countplot(x="International plan", hue="Churn", data=df);


# 
# We see that, with *International Plan*, the churn rate is much higher, which is an interesting observation! Perhaps large and poorly controlled expenses with international calls are very conflict-prone and lead to dissatisfaction among the telecom operator's customers.
# 
# Next, let's look at another important feature – *Customer service calls*. Let's also make a summary table and a picture.

# In[ ]:


pd.crosstab(df["Churn"], df["Customer service calls"], margins=True)


# In[ ]:


sns.countplot(x="Customer service calls", hue="Churn", data=df);


# Although it's not so obvious from the summary table, it's easy to see from the above plot that the churn rate increases sharply from 4 customer service calls and above.
# 
# Now let's add a binary feature to our DataFrame – `Customer service calls > 3`. And once again, let's see how it relates to churn. 

# In[ ]:


df["Many_service_calls"] = (df["Customer service calls"] > 3).astype("int")

pd.crosstab(df["Many_service_calls"], df["Churn"], margins=True)


# In[ ]:


sns.countplot(x="Many_service_calls", hue="Churn", data=df);


# 
# Let's construct another contingency table that relates *Churn* with both *International plan* and freshly created *Many_service_calls*.
# 
# 

# In[ ]:


pd.crosstab(df["Many_service_calls"] & df["International plan"], df["Churn"])


# Therefore, predicting that a customer is not loyal (*Churn*=1) in the case when the number of calls to the service center is greater than 3 and the *International Plan* is added (and predicting *Churn*=0 otherwise), we might expect an accuracy of 85.8% (we are mistaken only 464 + 9 times). This number, 85.8%, that we got through this very simple reasoning serves as a good starting point (*baseline*) for the further machine learning models that we will build. 
# 
# As we move on in this course, recall that, before the advent of machine learning, the data analysis process looked something like this. Let's recap what we've covered:
#     
# - The share of loyal clients in the sample is 85.5%. The most naive model that always predicts a "loyal customer" on such data will guess right in about 85.5% of all cases. That is, the proportion of correct answers (*accuracy*) of subsequent models should be no less than this number, and will hopefully be significantly higher;
# - With the help of a simple forecast that can be expressed by the following formula: "International plan = True & Customer Service calls > 3 => Churn = 1, else Churn = 0", we can expect a guessing rate of 85.8%, which is just above 85.5%. Subsequently, we'll talk about decision trees and figure out how to find such rules **automatically** based only on the input data;
# - We got these two baselines without applying machine learning, and they'll serve as the starting point for our subsequent models. If it turns out that with enormous effort, we increase the share of correct answers by 0.5% per se, then possibly we are doing something wrong, and it suffices to confine ourselves to a simple model with two conditions;
# - Before training complex models, it is recommended to manipulate the data a bit, make some plots, and check simple assumptions. Moreover, in business applications of machine learning, they usually start with simple solutions and then experiment with more complex ones.
# 
# ## 3. Assignments
# ### Demo-version
# To practice with Pandas and EDA, you can complete [this demo assignment](https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset) where you'll be analyzing socio-demographic data. The assignment is just for you to practice, and goes with [solution](https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset-solution).
# 
# ### Bonus version
# You can also choose a ["Bonus Assignments" tier](https://www.patreon.com/ods_mlcourse) (details are outlined on the [mlcourse.ai](https://mlcourse.ai/) main page) and get a non-demo version of the assignment where the history of Olympic Games is analyzed with Pandas. 
# 
# ## 4. Useful resources
# 
# * The same notebook as an interactive web-based [Kaggle Notebook](https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas)
# * ["Merging DataFrames with pandas"](https://nbviewer.jupyter.org/github/Yorko/mlcourse.ai/blob/master/jupyter_english/tutorials/merging_dataframes_tutorial_max_palko.ipynb) – a tutorial by Max Plako within mlcourse.ai (full list of tutorials is [here](https://mlcourse.ai/tutorials))
# * ["Handle different dataset with dask and trying a little dask ML"](https://nbviewer.jupyter.org/github/Yorko/mlcourse.ai/blob/master/jupyter_english/tutorials/dask_objects_and_little_dask_ml_tutorial_iknyazeva.ipynb) – a tutorial by Irina Knyazeva within mlcourse.ai
# * Main course [site](https://mlcourse.ai), [course repo](https://github.com/Yorko/mlcourse.ai), and YouTube [channel](https://www.youtube.com/watch?v=QKTuw4PNOsU&list=PLVlY_7IJCMJeRfZ68eVfEcu-UcN9BbwiX)
# * Official Pandas [documentation](http://pandas.pydata.org/pandas-docs/stable/index.html)
# * Course materials as a [Kaggle Dataset](https://www.kaggle.com/kashnitsky/mlcourse)
# * Medium ["story"](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-1-exploratory-data-analysis-with-pandas-de57880f1a68) based on this notebook
# * If you read Russian: an [article](https://habrahabr.ru/company/ods/blog/322626/) on Habr.com with ~ the same material. And a [lecture](https://youtu.be/dEFxoyJhm3Y) on YouTube
# * [10 minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
# * [Pandas cheatsheet PDF](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
# * GitHub repos: [Pandas exercises](https://github.com/guipsamora/pandas_exercises/) and ["Effective Pandas"](https://github.com/TomAugspurger/effective-pandas)
# * [scipy-lectures.org](http://www.scipy-lectures.org/index.html) — tutorials on pandas, numpy, matplotlib and scikit-learn
# 
# ## Support course creators
# <br>
# <center>
# You can make a monthly (Patreon) or one-time (Ko-Fi) donation ↓
# 
# <br>
# <br>
# 
# <a href="https://www.patreon.com/ods_mlcourse">
# <img src="https://habrastorage.org/webt/zc/11/0y/zc110yh0u3kgnlmay1gwbekk0ys.png" width=20% />
# 
# <br>
# 
# <a href="https://ko-fi.com/mlcourse_ai">
# <img src="https://habrastorage.org/webt/8r/ml/xf/8rmlxfpdzukegpxa62cxlfvgkqe.png" width=20% />
#     
# </center>
