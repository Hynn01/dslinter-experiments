#!/usr/bin/env python
# coding: utf-8

# # Analyzing Kaggle Survey Data Through the Years
# Each year, Kaggle conducts an Machine Learning and Data Science Survey to its users. Data available on Kaggle spans from 2017 to 2021. This notebook intends to:
# - Read data from each year's survey
# - Combine data from each year into a dataset
# - Analyzing the trends of survey results

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import os
import warnings

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load data
df2021 = pd.read_csv('/kaggle/input/kaggle-survey-2021/kaggle_survey_2021_responses.csv')
df2020 = pd.read_csv('/kaggle/input/kaggle-survey-2020/kaggle_survey_2020_responses.csv')
df2019 = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
df2018 = pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')
df2017 = pd.read_csv('/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding='ISO-8859-1')


# In[ ]:


# Print dimension of each year
print('2021: ' + str(df2021.shape))
print('2020: ' + str(df2020.shape))
print('2019: ' + str(df2019.shape))
print('2018: ' + str(df2018.shape))
print('2017: ' + str(df2017.shape))


# # Basic info of respondents
# 
# In this section, we combine basic info of respondents of each year into one dataset. Basic info includes:
# - Age
# - Country of origin
# - Gender
# - Occupation
# - Education qualification
# - Coding Experience
# 
# A note on 'Coding Experience': The question wordings asked each year is different. In the following analysis, we extract answers to the following question as 'Coding Experience' value for each year:
# - 2017: How long have you been learning data science?
# - 2018: How long have you been writing code to analyze data?
# - 2019: How long have you been writing code to analyze data (at work or at school)?
# - 2020 and 2021: For how many years have you been writing code and/or programming?

# In[ ]:


basic = pd.DataFrame(columns = ['year','age','country','gender','occupation', 'education', 'experience'])


# In[ ]:


# Get 2021 data
basic2021 = df2021.loc[1:,['Q1','Q3','Q2','Q5','Q4','Q6']]
basic2021.columns = ['age','country','gender','occupation', 'education', 'experience']
basic2021['year']=2021
basic2021.head()


# In[ ]:


# Get 2020 data
basic2020 = df2020.loc[1:,['Q1','Q3','Q2','Q5','Q4','Q6']]
basic2020.columns = ['age','country','gender','occupation', 'education', 'experience']
basic2020['year']=2020
basic2020.head()


# In[ ]:


# Get 2019 data
basic2019 = df2019.loc[1:,['Q1','Q3','Q2','Q5','Q4','Q15']]
basic2019.columns = ['age','country','gender','occupation', 'education', 'experience']
basic2019['year']=2019
basic2019.head()


# In[ ]:


# Get 2018 data
basic2018 = df2018.loc[1:,['Q2','Q3','Q1','Q6','Q4','Q24']]
basic2018.columns = ['age','country','gender','occupation', 'education', 'experience']
basic2018['year']=2018
basic2018.head()


# In[ ]:


# Get 2017 data
basic2017 = df2017.loc[:,['Age','Country','GenderSelect','CurrentJobTitleSelect','FormalEducation','LearningDataScienceTime']]
basic2017.columns = ['age','country','gender','occupation', 'education', 'experience']
basic2017['year']=2017
basic2017.head()


# In[ ]:


basic = pd.concat([basic, basic2021, basic2020, basic2019, basic2018, basic2017])
basic.shape


# Let's quickly investigate distribution for each year:

# In[ ]:


import seaborn as sns

pd.pivot_table(basic, index='age',columns='year', values='country', aggfunc='count')


# In[ ]:


pd.pivot_table(basic, index='gender',columns='year', values='country', aggfunc='count')


# In[ ]:


pd.pivot_table(basic, index='country',columns='year', values='gender', aggfunc='count')


# In[ ]:


pd.pivot_table(basic, index='occupation',columns='year', values='country', aggfunc='count')


# In[ ]:


pd.pivot_table(basic, index='education',columns='year', values='country', aggfunc='count')


# In[ ]:


pd.pivot_table(basic, index='experience',columns='year', values='country', aggfunc='count')


# # Cleaning data and basic visualization
# 
# As seen above, there is much need to clean up the data for more sensible analysis.

# In[ ]:


# Clean up gender
basic['gender'] = basic['gender'].replace({'Man':'Male','Woman':'Female','Prefer to self-describe':'Undisclosed'})
basic['gender'][~basic['gender'].isin(['Male','Female','Undisclosed'])] = 'Others'

gender_table = pd.pivot_table(basic, index='gender',columns='year', values='country', aggfunc='count').fillna(0)
gender_table = gender_table / gender_table.sum(axis=0)
gender_table.style.format("{:.2%}")


# Gender proportion is quite stable across the years, with male made roughly 80% of respondents.
# 
# Continue next time!
