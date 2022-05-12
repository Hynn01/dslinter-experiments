#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


titanic_test=pd.read_csv("../input/titanic/test.csv")
titanic_train=pd.read_csv("../input/titanic/train.csv")


# In[ ]:


titanic_train.head()


# Defining the problem statement
# Create a predective model which can tell if a person can survive the Titanic crash or not ?
# 
# * Target Variable: Survival
# * Predictive: age,Class, Gender ect
#     
# * Survival =0 The passenger died
# * Survival =1 The passenger survived
# 

# Determining the type of Machine Learning 
# 
# Based on the problem statement  you can understand that we need to create a suppervised ML classification model, as the target variable is categorical.
# 

# Looking at the distribution of Target variable
# 
# * if the predictive variable is too skewed then predective modeling will not be possiable 
# * Bell curve is desirable but slightly positive or negative skew is also fine
# * When performing classification, make sure there is a balance in the distribution of each class otherwise it impacts the machine learning algorithms ability to learn all the classes
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Creating Bar chart as the Target variable is Categorical
GroupedData=titanic_train.groupby("Survived").size()
GroupedData.plot(kind="bar", figsize=(4,3))


# Basic Data Exploration
# 
# * Head(): This help to see a few sample row of data
# * info(): This provide the summarized infomation of the  data 
# * Describe() : This Provide the descriptive stastical details of the data
# * nunique() : This help us to identify is a columns is categorical or continuous    
# 

# In[ ]:


# Looking at sample rows in the data
titanic_train.head()


# In[ ]:


titanic_train.info()


# In[ ]:


titanic_train.describe(include="all")


# In[ ]:


titanic_train.nunique()


# #### Basic Data Exploration Results
# 
# The selected columns in this step are not final, further study will be done and then a final list will be created
# 
# 
# * PassengerId:Qualitative.Rejected.    
# * Survived   :Categorical.Selected. This is the target variable
# * Pclass     :Categorical.Selected
# * Name       :Qualitative.Rejected. 
# * Sex        :Categorical.Selected
# * Age        :Continuous.selected
# * SibSp      :Categorical. Selected
# * Parch      :Categorical. Selected
# * Ticket     :Qualitative.Rejected.
# * Fare       :Continuous.Selected
# * Cabin      :Qualitative. Rejected. Also, this has too many missing values
# * Embarked   :Categorical.Selected
# 
# 

# Removing useless columns from the data

# In[ ]:


# Deleting those columns which are not useful in predictive analysis because these variables are qualitative
UselessColumns= ["PassengerId","Name","Ticket","Cabin"]
titanic_train = titanic_train.drop(UselessColumns,axis=1)
titanic_train.head()


# #### Visual Exploratory Data analysis
# 
# * Categorical variable: Bar plot
# * Continuous Variable: Histogram
# 
# 
# Visualize distribution of all the categorical predictor variable in the data using bar plots
# 
# We can spot a categorical variable in the data by looking at the unique values in them. Typically categorical variable contains less then 20 unique values And There is repatition of values, which mean the data can be grouped by those unique values.
# 
# 
# based on the basic Data Exploration above, we have spotted five categorical predictors in the data
# 
# Categorcal Predictors: Pclass,Sex, Sibsp,Parch,Embarked
# 
# We use bar chart to see how data is distributed for these categorical columns.

# In[ ]:


def PlotBarCharts(InpData,colsToPlot):
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    import matplotlib.pyplot as plt
    
    # Generating multiple subplots
    fig,subplot=plt.subplots(nrows=1,ncols=len(colsToPlot),figsize=(20,5))
    fig.suptitle("Bar Chart of:"+ str(colsToPlot))
    
    for colName, PlotNumber in zip(colsToPlot,range(len(colsToPlot))):
        InpData.groupby(colName).size().plot(kind="bar", ax=subplot[PlotNumber])

    


# In[ ]:


# Calling the function
PlotBarCharts(InpData=titanic_train, colsToPlot=["Pclass","Sex", "SibSp","Parch","Embarked"])


# Selected Categorical Variables: All the Categorical variables are selected for further analysis.
# 
# "Pclass","Sex", "SibSp","Parch","Embarked"
# 

# Visualize distribution of all the continuous Predictor Variables in the Data using Histograms
# 
# Based on the Basic Data Exploration, there are two continuous predictor variables "Age" and "Fare"

# In[ ]:


titanic_train.hist(["Age","Fare"], figsize=(20,7))


# #### Selected Continuous Variables:
# 
# * Age: Selected. the Distribution is good
# * Fare: Selected. Outliers seen beyond 300, need to treat them.
# 
# 
# There are below two options to treat outliers in the data.
# 
# * Option-1: Deleting the outlier Records. Only if there are just few rows lost.
# * Option-2: impute the outlier values with a logical business values
# 
# 
# below we are finding out the most logical value to be replaced in place of outlier by looking at the histogram. 

# Replacing outliers for 'Fare'

# In[ ]:


# Finding nearest values to 300 mark
titanic_train["Fare"][titanic_train["Fare"]<300].sort_values(ascending=False)


# Above result shows the nearest logical value is 263.0, hence replacing any values above 300 with it.

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None


# In[ ]:


# Replacing outliers with nearest possibe value
titanic_train["Fare"][titanic_train["Fare"]>300]=263.0


# In[ ]:


titanic_train.hist(["Age","Fare"],figsize=(18,5))


# 
# 

# #### Missing Values Treatment
# 
# Missing values are treated for each columns separately.
# 
# if a columns as more than 30% of data missing, then missing value treatment cannot be done. that columns must be rejected because to much information is missing 
# 
# There are below option to treating missing values in data
# 
# * Delete the missing values rows if there are only few records
# * imput the missing values with MEDIAN value for continuous variables
# * imput the missing values with MODE value for categorical variables
# * interpolate the values based on nearby values
# * interpolate the values based on business logic

# In[ ]:


# Finding how many missing values are there for each column

titanic_train.isnull().sum()


# #### I am using Median and Mode value for missing value replacement 
# 

# In[ ]:


# Replacing missing values of Age with median value
titanic_train["Age"].fillna(titanic_train["Age"].median(), inplace = True)

# Replacing missing values of Embarked with Mode value
titanic_train["Embarked"].fillna(titanic_train["Embarked"].mode()[0], inplace=True)


# In[ ]:


titanic_train.isnull().sum()


# ### Feature Selection
# 
# Now it is time to finally choose the best columns(Feature) which are correlated to target variable. this can be done directly by measuring the correaltion values or Anova/Chi-Square test. However, it is always helpful to visualize the relation between the target variable and each of the predictors to get a better sense of data.
# 
# I have listed below the techniques used for visualizing relationship between two variables as well as measuring the strength statistically.
# 
# 
# ### Visual exploration of relationship between varianbles
# 
# * continuous vs Continuous -- Correlation Matrix
# * Categorical VS Continuous --Anova Test
# * Categorical VS Categorical --Chi-SquareTest
# 
# 
# In this case study the target variable is categorica, hence below two scenarios will be present
# 
# * Categorical target variable vs Continuous Predictor
# * Categorical target variable vs categorical predictor
# 

# ### Relationship exploration: Categorical VS Continuous -- Box Plots
# 
# When the target variable is Categorical and predictor variable is continuous we analyze the relation using bar plot/boxplots and measure the strength of relation using Anova test

# In[ ]:


#Boxplot for categorical target variable "Survived" and continuous predictors
continuousColsList=["Age","Fare"]

import matplotlib.pyplot as plt
fig,PlotCanvas=plt.subplots(nrows=1,ncols=len(continuousColsList), figsize=(18,5))

#creating box plots for each continuous predictor against the target variable "Survived"
for PredictorCol, i in zip(continuousColsList,range(len(continuousColsList))):
    titanic_train.boxplot(column=PredictorCol, by="Survived",figsize=(5,5),
                          vert=True, ax=PlotCanvas[i])
    
    


# #### Box-Plots interpretation
# 
# These plots gives an idea about the data distribution of continuous predictor in the Y-axis for each of the category in the X-Axis.
# 
# if the distrubution looks similar for each category, that means the continuous variable has No effect on the target variable. hence the variables are not correlated to each other.
# 
# 
# The other chart exhibit opposite characteristics. Means the data distribution is different for each category of Survival.It hints that these variables meight be correlated with Survival.
# 
# We confirm this by looking at the result of Anova test below

# ## Statistical Feature Selection (Categorical Vs Continuous) using ANOVA test
# 
# Analysis of variance is performed to check if there is any relationship between the given continuous and categorical variable
# 
# * Assumption(H0):There is No relation between the given variables(i.e The average(mean) values of the numeric predictor variable is same for all the group in the categorical target variable
# 
# * Anova Test Results: Probability og H0 being true

# In[ ]:


# Defining a function to find the statistical relationship with all the categorical variables
def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    from scipy.stats import f_oneway
           
    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    
    print("#### ANOVA Results ####\n")
    for Predictor in ContinuousPredictorList:
        CategoricalGroupLists=inpData.groupby(TargetVariable)[Predictor].apply(list)
        AnovaResults=f_oneway(*CategoricalGroupLists)
        
        
        # If the ANOVA P-Value is <0.05, that means we reject H0
        if(AnovaResults[1]<0.05):
            print(Predictor,"is correlated with",TargetVariable,"| P-Value:", AnovaResults[1])
            SelectedPredictors.append(Predictor)
        else:
            print(Predictor,"is not correlated with",TargetVariable,"| P-Value:", AnovaResults[1])
    return(SelectedPredictors)        
            


# In[ ]:


# Calling the function to check which categorical variables are correlated with target
ContinuousVariables=["Age","Fare"]
FunctionAnova(inpData=titanic_train,TargetVariable="Survived",
              ContinuousPredictorList=ContinuousVariables)


# The result of Anova confirm our visual analysis using box plots above.
# 
# Loke at the P value of age it got rejected by a little margine. in such scenarios you may decide to include the variable which is at the boundry line and see if it helps to increase the accuracy.
# 
# ### Final selection continuous columns
# "Fare"

# ### Relationship exploration: Categorical VS Categorical --Grouped Bar charts
# 
# When the target variable and predictor variable are categorical then we explore the correlation between them visually using Barplots and Statistically using Chi-square test

# In[ ]:


# Cross tablulation between two categorical variables
CrossTabResult=pd.crosstab(index=titanic_train["Sex"],columns=titanic_train["Survived"])
CrossTabResult


# In[ ]:


# Visual Inference using Grouped Bar charts
CategoricalColsList=["Pclass","Sex", "SibSp","Parch","Embarked"]

import matplotlib.pyplot as plt
fig, PlotCanvas=plt.subplots(nrows=len(CategoricalColsList), ncols=1, figsize=(10,30))


# Creating Grouped bar plots for each categorical predictor against the Target Variable "Survived"
for CategoricalCol, i in zip(CategoricalColsList, range(len(CategoricalColsList))):
    CrossTabResult=pd.crosstab(index=titanic_train[CategoricalCol],columns=titanic_train["Survived"])
    CrossTabResult.plot.bar(color=["red","green"],ax=PlotCanvas[i], title= CategoricalCol+ "VS " + "Survived")


# ### Grouped Bar Charts Interpretation
# 
# These grouped bar charts show the frequesncy in the Y-Axis and the category in the X-Axis. if the ratio of bars is similer across all categories, then the two columns are not correlated.
# 
# on the other hand, look at the sex vs Survived plot. The bars are diffrent for each category, Hence two columns are correlated with each other

# #### Statistical Feature Selection (Categorical Vs Categorical) using Chi-Square Test
# 
# Chi-Square test is conducted to check the correlation between two categorical variables
# 
# Assumption(H0): The two columns are NOT related to each other
# Result of Chi-Sq Test: The Probability of H0 being Tru

# In[ ]:


# Writing a function to find the correlation of all categorical variables with the Target variable
def FunctionChisq(inpData,TargetVariable, CategoricalVariablesList):
    from scipy.stats import chi2_contingency
    
    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    
    for Predictor in CategoricalVariablesList:
        CrossTabResult=pd.crosstab(index=inpData[TargetVariable],columns=inpData[Predictor])
        ChiSqResults = chi2_contingency(CrossTabResult)
        
        # If the ChiSq P-Value is <0.05, that means we reject H0
        if (ChiSqResults[1]<0.05):
            print(Predictor,"is correlated with", TargetVariable, "| P-Value:",ChiSqResults[1])
            SelectedPredictors.append(Predictor)
        else:
            print(Predictor,"is not correlated with", TargetVariable, "| P-Value:",ChiSqResults[1])

    return(SelectedPredictors)    


# In[ ]:


CategoricalVariables=["Pclass","Sex", "SibSp","Parch","Embarked"]

# Calling the function

FunctionChisq(inpData=titanic_train,
              TargetVariable="Survived",
              CategoricalVariablesList=CategoricalVariables)


# #### Finally selected Categorical variables
# 
# 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'
# 
# #### Selecting final predicotrs for machine learning
# 
# Based on the above test, selecting the final columns for machine learning
# 
# Instead of original "education" columns, i am selecting the "education_num". Which represents the ordinal property of the data.
# 

# In[ ]:


SelectedColumns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

# Selecting final columns
DataForML=titanic_train[SelectedColumns]
DataForML.head()


# # Saving this final data for reference during deployment
# DataForML.to_pickle("DataForML.pkl")

# ### Data Pre-Processing for Machine Learning 
# 
# List of steps performed on predictor variable before data can be used for Machine learning 
# 
# 1.Convert each Ordinal categorical columns to numeric
# 2.convert Binary nominal categorical columns to numeric using 1/0 mapping
# 3.convert all other nominal categorical columns to numeric using pd.get_dummies()
# 4.Data Transformation(optional): Standardization/Normalization/log/sqrt. Important if you are using distance based algorithms like KNN,or Neural Networks
# 
