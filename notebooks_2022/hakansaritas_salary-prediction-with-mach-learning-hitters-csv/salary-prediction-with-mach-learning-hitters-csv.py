#!/usr/bin/env python
# coding: utf-8

# # SALARY PREDICTION WITH MACHINE LEARNING - HITTERS DATASET

# ![](https://www.newyorkalmanack.com/wp-content/uploads/2021/11/1885-86-Cuban-Giants.jpg)

# **AIM** : Develop a machine learning model to estimate the salary of baseball players whose salary information and career statistics for 1986 are shared.

# **SOURCE**: 
# * This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.  
# * This is part of the data that was used in the 1988 ASA Graphics Section Poster Session.
# * The salary data were originally from Sports Illustrated, April 20, 1987. 
# * The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.

# **ALL VARIABLES**:
# 
# *A data frame with 322 observations of major league players on the following 20 variables.*
# 
# * **AtBat**: Number of times at bat in 1986
# * **Hits**: Number of hits in 1986
# * **HmRun**: Number of home runs in 1986
# * **Runs**: Number of runs in 1986
# * **RBI**: Number of runs batted in in 1986
# * **Walks**: KNumber of walks in 1986
# * **Years**: Number of years in the major leagues
# * **CAtBat**: Number of times at bat during his career
# * **CHits**: Number of hits during his career
# * **CHmRun**: Number of home runs during his career
# * **CRuns**: Number of runs during his career
# * **CRBI**: Number of runs batted in during his career
# * **CWalks**: Number of walks during his career
# * **League**: A factor with levels A and N indicating player's league at the end of 1986
# * **Division**: A factor with levels E and W indicating player's division at the end of 1986
# * **PutOuts**: Number of put outs in 1986
# * **Assits**: Number of assists in 1986
# * **Errors**: Number of errors in 1986
# * **Salary**:  1987 annual salary on opening day in thousands of dollars
# * **NewLeague**: A factor with levels A and N indicating player's league at the beginning of 1987
#     
# **TARGET VARIABLE**:
# 
# * **Salary**

# **INTRODUCTION**
# 
# * In this salary prediction machine learning project, linear regression modeling method was used. RMSE error values were compared using different testing methods.
# 
# * The project was created under the main headings "Modelling using feature engineering" and "Modelling without feature engineering".
# 
# * Under each main heading, error calculation methods "without Train and Test datasets", "with Train and test sets" and "k-fold cross validation" were used. R2 and RMSE results from these were compared.

# ------------

# **RESULTS:**
# * RMSE error values decreased significantly in the model created by feature engineering.
# * R2 values increased in models created after feature engineering. This shows that the explainability of the data increases.

# -----------------------

# # 1-Discovery Data Analysis

# ## 1.1 - Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score,mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

from warnings import filterwarnings
filterwarnings('ignore')


# ----------------------

# ## 1.2 - Dataset Display Settings

# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# -------------

# ## 1.3 - Reading the Dataset

# In[ ]:


df_ = pd.read_csv(r"../input/hitters-baseball-data/Hitters.csv")
df = df_.copy()


# -----------------

# ## 1.4 - Dataset Review

# In[ ]:


# The dataset has 322 observations and 20 variables.

df.shape


# In[ ]:


# check the dataset with the index values given

df.take([0,50,100,150,200,250,300,320])


# In[ ]:


# type information of the variables in the dataset. There are 3 categorical and 17 numerical variables.

df.info()


# In[ ]:


# number of unique values in each variable

df.nunique()


# In[ ]:


# categorical and numeric variables

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
num_cols = [col for col in df.columns if df[col].dtypes != "O"]

print("cat_cols:",cat_cols)
print("num_cols:",num_cols)


# ### 1.4.1 Analysis of Categoric Variables 

# In[ ]:


# unique classes and their frequencies in categorical variables 

for i in cat_cols:
    
    print(df[i].value_counts())
    
    fig, ax = plt.subplots(figsize=(5,3))
    
    sns.countplot(x=df[i], data=df, ax=ax)
    
    plt.show()
    
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")


# ### 1.4.2 Analysis of Numeric Variables

# In[ ]:


quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
display(df.describe(quantiles).T)

for i in num_cols:
    
    df[i].hist(bins=20)
    plt.xlabel(i)
    plt.title(i)
    plt.show()


# ### 1.4.3 Queries

# In[ ]:


# Is there anyone who plays in "league" A now and will play in "league" N next year or vice versa? yes there is
# if yes how many people? 19 people

display(df.loc[df["League"] != df["NewLeague"]])

print("#####################################")

print(len(df.loc[df["League"] != df["NewLeague"]]))


# In[ ]:


# "Salary" information according to the level of players playing in the E and W divisions of the 86 league

df.groupby(["League","Division"])[["Salary"]].agg(["min","max","mean"])


# In[ ]:


# "Salary" information according to the level of play who will play in the E and W divisions of the 87 league

df.groupby(["NewLeague","Division"])[["Salary"]].agg(["min","max","mean"])


# In[ ]:


# how many rookies, how many experienced players are there?. there is 24 years old player! 4-year-experinced players are the most numbers, rookies (1.2) quite a lot

df["Years"].value_counts()


# In[ ]:


# The min, mean and max value for the "Salary" variable
# The salary of the 24-year-experience player is above the average, but the old school is not valued :(

display(df[["Salary"]].agg(["min","mean","max"]).T)

print("###############")

display(df[["Years","Salary"]].loc[df["Years"] == 24])


# In[ ]:


# Statistics of experience, number of errors and salary of the top 10 players who made the most errors
# The minimum salary in the dataset is 67.5, but the top 10 observations of those who make the most mistakes do not have this minimum value
# there are other factors that affect the low salary

df[["Years","Errors","Salary"]].sort_values("Errors",ascending =False).head(10)


# In[ ]:


# those who have neither helped their friends, nor assisted or made errors during the season.

df.loc[(df["PutOuts"] == 0) & (df["Assists"] == 0) & (df["Errors"] == 0)]


# -----------------

# ## 1.5 - Correlation of Variables

# In[ ]:


df.corr()


# ### 1.5.1 Heatmap

# In[ ]:


# The first three variables that show the best positive correlation with the "Salary" variable are "CRBI", "CRuns", "CHits"
# The first three variables that show the weakest positive correlation with the "Salary" variable are "Assists", "PutOuts", "HmRuns"
# Variable showing negative correlation with "Salary" is "Errors" variable

fig, ax = plt.subplots(figsize=(25,10)) 
sns.heatmap(df.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()


# -------------------

# ## 1.6 - Outliers

# ### 1.6.1 Graphs outliers by BoxPlot 

# In[ ]:


## Function that displays numeric variables as boxplot graphs, respectively

def graph_outliers(dataframe):
    
    for col in num_cols:
        
        sns.boxplot(x= df[col]);
        
        plt.show()


# In[ ]:


# There are outliers in numerical variables after "HmRuns" variable

graph_outliers(df)


# ### 1.6.2 Up and Down Values for Thresholds

# In[ ]:


# Finding Thresholds values (limit percentages for Threshold 0.25-0.75)

def outlier_thresholds(dataframe, col_name, q1=0.20, q3=0.75):
    
    quartile1 = dataframe[col_name].quantile(q1)
    
    quartile3 = dataframe[col_name].quantile(q3)
    
    interquantile_range = quartile3 - quartile1
    
    up_limit = quartile3 + 1.5 * interquantile_range
    
    low_limit = quartile1 - 1.5 * interquantile_range
    
    return low_limit, up_limit


# In[ ]:


# threshold values for each variable according to the given percentile 

for col in df.columns:
    
    if (df[col].dtype != "O") and (df[col].nunique() > 10):
        
        print(f"{col}------> Low: {round(outlier_thresholds(df,col)[0])}         ---------Up:{round(outlier_thresholds(df,col)[1])}")


# ### 1.6.3 Variables with Outliners and Replacing

# In[ ]:


# the function for find variable with outliners

def find_fill_outliers(dataframe, col_name, fill=False):
    
    """
    INFO
        this function is for check outliners of a given dataframe and 
        fill the outliners in dataframe with up and down thresholds values if you wish
    
    PARAMETERS
        dataframe: pandas dataframe 
        col_name: numeric variables, in list format 
        fill:  if True, replace outliners with up and down thresholds values, default=False

    """
    
    outliers_col = []
    
    for col in col_name:
        
        low_limit, up_limit = outlier_thresholds(dataframe, col)
    
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
        
            outliers_col.append(col)
            
        if fill: 
        
            dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
    
            dataframe.loc[(dataframe[col] > up_limit), col] = up_limit
        
    
    return outliers_col 


# In[ ]:


# variables with outliers (down :0.25, up: 0.75 thresholds): 13 variables

find_fill_outliers(df, num_cols)


# In[ ]:


# replace outliners with up and down thresholds values

outlier_col = find_fill_outliers(df, num_cols, fill=True)


# In[ ]:


# variables with outliers after run the function

find_fill_outliers(df, num_cols)


# --------------------

# ## 1.7- Missing Values

# In[ ]:


# "Salary" variable has 59 missing observations

df.isnull().sum()


# ### 1.7.1 Filling Missing Values

# **Queries**

# In[ ]:


# Average of "Salary" variable in "League" and "Division" groupby

df.groupby(["League","Division"])["Salary"].mean()


# In[ ]:


# Among the players who are in League A and playing in the E division, those who are missing the Salary variable

df.loc[(df["Salary"].isnull()) & (df["League"] == "A") & (df["Division"] == "E")].head()


# In[ ]:


# Average salary of League A players playing in Division E

df.groupby(["League","Division"])["Salary"].mean()["A","E"]


# In[ ]:


# Among the players who are in League N and playing in the W division, those who are missing the Salary variable

df.loc[(df["Salary"].isnull()) & (df["League"] == "N") & (df["Division"] == "W")].head()


# In[ ]:


# Average salary of League N players playing in Division W

df.groupby(["League","Division"])["Salary"].mean()["N","W"]


# **Fill the Missing Salary Variable**

# In[ ]:


# Eksik değişkenleri League ve Division kırılımana göre doldurma (A,E)(A,W)(N,E)(N,W)

df.loc[(df["Salary"].isnull()) & (df["League"] == "A") & (df["Division"] == "E"), "Salary"] = df.groupby(["League","Division"])["Salary"].mean()["A","E"]
df.loc[(df["Salary"].isnull()) & (df["League"] == "A") & (df["Division"] == "W"), "Salary"] = df.groupby(["League","Division"])["Salary"].mean()["A","W"]
df.loc[(df["Salary"].isnull()) & (df["League"] == "N") & (df["Division"] == "E"), "Salary"] = df.groupby(["League","Division"])["Salary"].mean()["N","E"]
df.loc[(df["Salary"].isnull()) & (df["League"] == "N") & (df["Division"] == "W"), "Salary"] = df.groupby(["League","Division"])["Salary"].mean()["N","W"]


# In[ ]:


# Salary değişkeni kontrol

df["Salary"].isnull().any()


# -----------------

# # 2 - Salary Estimation Results Without Using Feature Engineering
# 
# _problem is a salary estimation, it becomes a regression problem_

# ## 2.1 - Graphical Output by joinplot

# In[ ]:


# Graphical representation of the relationship between dependent and independent variables with joinplot
# Blue dots actual intersection values
# Blue line trend
# A histogram of the distribution of data on the upper and right axis
# According to graphics "Assists" and "Errors" are not linear relationship with "Salary"
# https://seaborn.pydata.org/generated/seaborn.jointplot.html

for i in num_cols:
    
    sns.jointplot(x=i, y="Salary", kind ="reg", data=df);


# -------------------

# ## 2.2 - Multiple Linear Regression

# In[ ]:


# Assign dependent and independent variables X and y.
# Subtract the categorical and dependent variable from the independent variables

X = df.drop(["League","Division","NewLeague","Salary"], axis =1)
y = df[["Salary"]]


# In[ ]:


X.head(2)


# In[ ]:


y.head(2)


# ### 2.2.1 Setting up the Model without Separating as Test and Train

# In[ ]:


lm  = LinearRegression()
model = lm.fit(X,y)


# In[ ]:


# Intercept Value

np.round(model.intercept_[0],2)


# In[ ]:


# coefficients of the variables. We have 16 numeric variables excluding the Salary variable

np.round(model.coef_[0],2)


# ### Formül
# 
# **Salary = 267 + (-1.08 * AtBat) + (3.07 * Hits) + (2.26 * HmRun) + (0.61 * Runs) + (0.32 * RBI) + (1.68 * Walks) + (-13.45 * Years) + (-0.35 * CAtBat) + (1.06 * CHits) + (-0.5 * CHmRun) + (0.57 * CRuns) + (0.59 * CRBI) + (0.2 * CWalks) + (0.33 * PutOuts) + (0.33 * Assists) + (-3.66 * Errors)**

# In[ ]:


# R2 value: what percentage of the change in the dependent variable can be explained by the independent variables.

np.round(model.score(X,y),2)


# -------------

# ### 2.2.2 Success of the Prediction Model

# * With RMSE, we calculate the average error value per unit of our errors, that is, the differences between the actual values and the estimated values.

# In[ ]:


# root_mean_squared_error 

RMSE = np.sqrt(mean_squared_error(y,model.predict(X)))
np.round(RMSE,2)


# -----------------

# ### Example Estimate

# In[ ]:


# creation of a new player's information in a dataset

player_data1 =[[310],[105],[15],[80],[35],[20],[4],[1500],[800],[100],[330],[350],[260],[0],[0],[10]]
new_data = pd.DataFrame(player_data1).T

player_data2 =[[0],[0],[15],[80],[350],[20],[4],[1500],[1000],[100],[330],[350],[260],[200],[100],[1]]
new_data1 = pd.DataFrame(player_data2).T

display(new_data)
display(new_data1)


# In[ ]:


# Salary estimate of this player: 1007.6 units

print("player 1:", round(model.predict(new_data)[0][0],2))
print("player 2:",round(model.predict(new_data1)[0][0],2))


# ----------------

# ### 2.2.3 - Setting up the Model with Separating as Test and Train

# In[ ]:


# random_state is for getting the same train test combination in each time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# Set up model

reg_model = LinearRegression().fit(X_train, y_train)


# In[ ]:


# size of train and test datasets

print("X_test data shape:",X_test.shape)
print("X_train data shape:", X_train.shape)
print("y_test data shape:",y_test.shape)
print("y_train data shape:", y_train.shape)


# In[ ]:


# intercept and coeficiant of independet variables

print(np.round(reg_model.intercept_[0],2))
print(np.round(reg_model.coef_[0],2))


# ### Example Estimate

# In[ ]:


# Salary estimate of the same players whose values are described above

print("player 1:", round(reg_model.predict(new_data)[0][0],2))
print("player 2:",round(reg_model.predict(new_data1)[0][0],2))


# -----------------------

# ### 2.2.4 - Evaluating Model Forecasting Success

# ### *2.2.4.1 With Test Set (holdout)*

# ### A - train error

# In[ ]:


# predicted y value
y_pred = reg_model.predict(X_train)

# The difference between the actual and the predicted y-value. The goal is minimize the difference
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred))

# RMSE train value
np.round(RMSE_train,2)


# In[ ]:


# R2 Tarin
np.round(reg_model.score(X_train, y_train),3)


# ### B - test error

# In[ ]:


# predicted y value
y_pred = reg_model.predict(X_test)

# The difference between the actual and the predicted y-value. The goal is minimize the difference
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))

# rmse value
np.round(RMSE_test,2)


# In[ ]:


# R2 Test

np.round(reg_model.score(X_test, y_test),3)


# ### *2.2.4.2  with K-Fold Cross Validation*

# In[ ]:


# cv :Fold numbers

cv=cross_val_score(model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error")
cv


# In[ ]:


# Root mean square error (RMSE) for train set - validation error

RMSE_cross = np.sqrt(np.mean(-cv))
np.round(RMSE_cross,2)


# -------------------

# # 3 - Modeling Data Using Feature Engineering

# ## 3.1 - Creating New Variables

# ### 1. Calculation the annual average statistical values of each player. For example, the number of hits he made this year / the number of hits in his collecting career

# In[ ]:


df["AVG_AtBat"] = df["CAtBat"] / df["Years"] 
df["AVG_CHits"] = df["CHits"] / df["Years"] 
df["AVG_CHmRun"] = df["CHmRun"] / df["Years"] 
df["AVG_CRuns"] = df["CRuns"] / df["Years"] 
df["AVG_CRBI"] = df["CRBI"] / df["Years"] 
df["AVG_CWalks"] = df["CWalks"] / df["Years"]


# In[ ]:


df.head(3)


# ### 2. How many percent increase or decrease each player's performance in this year compared to the average performance of all years

# In[ ]:


df["PERF_AtBat"] = (( df["AtBat"] - df["AVG_AtBat"]) / df["AVG_AtBat"]) * 100
df["PERF_Hits"] =  (( df["Hits"] - df["AVG_CHits"]) / df["AVG_CHits"]) * 100
df["PERF_HmRun"] = (( df["HmRun"] - df["AVG_CHmRun"]) / df["AVG_CHmRun"]) * 100  
df["PERF_Runs"] = (( df["Runs"] - df["AVG_CRuns"]) / df["AVG_CRuns"]) * 100 
df["PERF_RBI"] = (( df["RBI"] - df["AVG_CRBI"]) / df["AVG_CRBI"]) * 100  
df["PERF_Walks"] = (( df["Walks"] - df["AVG_CWalks"]) / df["AVG_CWalks"]) * 100 


# In[ ]:


df.head()


# ### 3. Separation of each player as "Rookie","Mid","Senior","Expert" according to the year spent in baseball

# In[ ]:


df["PLAYER_LEVEL"] = pd.qcut(df["Years"], 4, labels=["Rookie","Mid","Senior","Expert"])
df.head()


# ### 4. Creating a new variable based on the player's next season League change

# In[ ]:


df.loc[df["League"] != df["NewLeague"],"LEAGUE_CHANGE"] = "changed"
df.loc[df["League"] == df["NewLeague"],"LEAGUE_CHANGE"] = "not_changed"


# In[ ]:


df.head()


# ### 5. Evaluation of the players' performance this year based on their Total performance (by İzzet Topçuoğlu)

# In[ ]:


df["DIV_CAtBat"] = df["AtBat"]/df["CAtBat"] 
df["DIV_CHits"] = df["Hits"]/df["CHits"] 
df["DIV_CHmRun"] = df["HmRun"]/df["CHmRun"] 
df["DIV_Cruns"] = df["Runs"]/df["CRuns"] 
df["DIV_CRBI"] = df["RBI"]/df["CRBI"] 
df["DIV_CWalks"] = df["Walks"]/df["CWalks"]


# In[ ]:


df.head()


# ### 6. Create a variable that combines the players' level with the region they are playing (by Mehmet Akturk)

# In[ ]:


df.loc[(df["PLAYER_LEVEL"] == "Rookie") & (df["Division"] == "E"), 'LEVEL_DIV'] = "Rookie-East"
df.loc[(df["PLAYER_LEVEL"] == "Rookie") & (df["Division"] == "W"), 'LEVEL_DIV'] = "Rookie-West"
df.loc[(df["PLAYER_LEVEL"] == "Mid")    & (df["Division"] == "E"), 'LEVEL_DIV'] = "Mid-East"
df.loc[(df["PLAYER_LEVEL"] == "Mid")    & (df["Division"] == "W"), 'LEVEL_DIV'] = "Mid-West"
df.loc[(df["PLAYER_LEVEL"] == "Senior") & (df["Division"] == "E"), 'LEVEL_DIV'] = "Senior-East"
df.loc[(df["PLAYER_LEVEL"] == "Senior") & (df["Division"] == "W"), 'LEVEL_DIV'] = "Senior-West"
df.loc[(df["PLAYER_LEVEL"] == "Expert") & (df["Division"] == "E"), 'LEVEL_DIV'] = "Expert-East"
df.loc[(df["PLAYER_LEVEL"] == "Expert") & (df["Division"] == "W"), 'LEVEL_DIV'] = "Expert-West"


# In[ ]:


df.head()


# ----------------------

# ## 3.2 - Encoding

# ### 3.2.1 Lable Encoding

# In[ ]:


# label encoding 

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# In[ ]:


# Label Encoding of categoric variable having two class as binary format (0,1)

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


# In[ ]:


df.head()


# ### 3.2.2 One Hot Encoder

# In[ ]:


# A function for one hot encoding 

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    
    return dataframe


# In[ ]:


# take the columns with number of unique value bigger than 2, lower than 10. 
# these variables are for one hot encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


# In[ ]:


df = one_hot_encoder(df, ohe_cols, drop_first=True)


# In[ ]:


df.head()


# -----------------

# ## 3.3 - Missing Values after New Variables

# In[ ]:


# There are missing values in newly created variables cause of zero divisions.

for i in df.columns:
    if df[i].isnull().any():
        print(i, df[i].isnull().sum())


# In[ ]:


# Fill with Zero

df = df.fillna(0)


# In[ ]:


df.isnull().any().any()


# ---------------------------------

# ## 3.4 Scaling the Independent Variables with Roboust

# In[ ]:


# Remove the Dependent variable from numeric columns 
# RobustScaler

num_cols = [col for col in df.columns if df[col].dtypes != "O"]

num_cols.remove("Salary")


# In[ ]:


for col in num_cols:
    
    transformer = RobustScaler().fit(df[[col]])
    
    df[col] = transformer.transform(df[[col]])


# ---------------------

# ## 3.5 - Multiple Linear Regression

# In[ ]:


X = df.drop(["Salary"], axis =1)
y = df[["Salary"]]


# In[ ]:


X.head(2)


# In[ ]:


y.head(2)


# ### 3.5.1 Setting up the Model without Separating as Test and Train

# In[ ]:


lm  = LinearRegression()
model = lm.fit(X,y)


# In[ ]:


np.round(model.intercept_[0])


# In[ ]:


np.round(model.coef_[0],3)


# In[ ]:


# R2

np.round(model.score(X,y),2)


# -------------

# ### 3.5.2 Success of the Prediction Model

# In[ ]:


# root_mean_squared_error 

RMSE = np.sqrt(mean_squared_error(y,model.predict(X)))
np.round(RMSE,2)


# -----------------

# ### 3.5.3 Setting up the Model with Separating as Test and Train

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

reg_model = LinearRegression().fit(X_train, y_train)


# In[ ]:


print(reg_model.intercept_)
print(reg_model.coef_)


# -----------------------

# ### 3.5.4  Evaluating Model Forecasting Success

# ### *3.5.4.1 With Test Set (holdout)*

# ### A - train errors

# In[ ]:


y_pred = reg_model.predict(X_train)

RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred))

np.round(RMSE_train,2)


# In[ ]:


# R2 
reg_model.score(X_train, y_train)


# ### B - test errors

# In[ ]:


y_pred = reg_model.predict(X_test)

RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))

np.round(RMSE_test,2)


# In[ ]:


# R2

reg_model.score(X_test, y_test)


# ### *3.5.4.2 with K-Fold Cross Validation*

# In[ ]:


cv=cross_val_score(model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error")
cv


# In[ ]:


RMSE_cross = np.sqrt(np.mean(-cv))
np.round(RMSE_cross,2)


# -------------

# # 4 - RESULTS
# 
# ---------------------------------------------------------------- -----------
# * Model errors without feature engineering
# 
#      * Without separating as Test Train
#     
#          * RMSE: 272
#          * R2: 0.46
# *-----------------------------------------*
# 
#      * By separating as Test Train
#     
#          * RMSE_train: 273
#          * R2_train: 0.5
#          *----------------* 
#          * RMSE_test: 283
#          * R2_test: -0.09
#          * RMSE_cross: 299
# ---------------------------------------------------------------- ------------
# * Model errors achieved using feature engineering
# 
#      * Without separating as Test Train
#     
#          * RMSE: 221
#          * R2: 0.64
# *-----------------------------------------*
# 
#      * By separating as Test Train
#     
#          * RMSE_train: 211
#          * R2_train: 0.700
#          *----------------*    
#          * RMSE_test: 283
#          * R2_test: -0.08
#          * RMSE_cross: 275
# --------------------------------------------------------------
# * RMSE error values decreased significantly in the model created by feature engineering.
# * R2 values increased in models created after feature engineering. This shows that the explainability of the data increases.

# ## 4.1  Results in dataframe

# In[ ]:


results = {'Feature_Eng': {0: 'No', 1: 'No ', 2: 'Yes', 3: 'Yes'},
 'X_y': {0: 'No', 1: 'Yes', 2: 'No', 3: 'Yes'},
 'R2': {0: 0.46, 1: "nan", 2: 0.64, 3: "nan"},
 'R2_train': {0: "nan", 1: 0.5, 2: "nan", 3: 0.7},
 'R2_test': {0: "nan", 1: -0.09, 2: "nan", 3: -0.08},
 'RMSE': {0: 271.96, 1: "nan", 2: 221.3, 3: "nan"},
 'RMSE_train': {0: "nan", 1: 273.41, 2: "nan", 3: 211.19},
 'RMSE_test': {0: "nan", 1: 283.54, 2: "nan", 3: 283.33},
 'RMSE_cross': {0: "nan", 1: 299.32, 2: "nan", 3: 275.41}}


# In[ ]:


df_results = pd.DataFrame(results)


# In[ ]:


df_results

