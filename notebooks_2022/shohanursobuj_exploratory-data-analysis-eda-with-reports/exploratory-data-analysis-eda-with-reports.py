#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data loading and overview
# 
# <font color='cyan'>Let's start by reading in the Diabetes Health Indicators Dataset csv file into a pandas dataframe.</font>

# In[ ]:


df = pd.read_csv("../input/diabetes-health-indicators-dataset/diabetes_012_health_indicators_BRFSS2015.csv")


# In[ ]:


##shape of the dataset
print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.') 


# In[ ]:


#print first five rows
df.head()


# In[ ]:


#Check the info of the data set which describes null values, data type, memory usage
df.info()


# ``float64`` is the data types of our features. We can easily see if there are any missing values. Here, there are none because each column contains 253680 observations, the same number of rows we saw before with shape

# In[ ]:


df.describe()


# > <font color='cyan'>  Check the description of the data set which describes the minimum value, maximum value, mean value, total count, standard deviation etc. and visualise the Diabetes of how many persons have diabetes or how many persons have no diabetes and how many have pre-diabetes. </font>

# Exploratory Data Analysis# Exploratory Data Analysis
# 
# <font color='cyan'>In this section, we will be doing some basic Exploratory Data Analysis to get the "feel" of the data, we will be checking the distributions, the correlations etc of the different columns</font>

# ### Missing Data
# 
# <font color='cyan'>We can use seaborn to create a simple heatmap to see where we are missing data!</font>

# In[ ]:


def msv_1(df, thresh = 20, color = 'black', edgecolor = 'black', height = 3, width = 15):
    
    plt.figure(figsize = (width, height))
    percentage = (df.isnull().mean()) * 100
    percentage.sort_values(ascending = False).plot.bar(color = color, edgecolor = edgecolor)
    plt.axhline(y = thresh, color = 'r', linestyle = '-')
    
    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )
    
    plt.text(len(df.isnull().sum()/len(df))/1.7, thresh+2.5, f'Columns with more than {thresh}% missing values', fontsize=12, color='crimson',
         ha='left' ,va='top')
    plt.text(len(df.isnull().sum()/len(df))/1.7, thresh - 0.5, f'Columns with less than {thresh}% missing values', fontsize=12, color='green',
         ha='left' ,va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight ='bold')
    
    return plt.show()
msv_1(df, 20, color=sns.color_palette('Reds',15))


# In[ ]:


print(f'There are {df.isnull().any().sum()} columns in diabetes dataset with missing values.')


# > Here from the above code we first checked that is there any null values from the IsNull() function then we are going to take the sum of all those missing values from the sum() function and the inference we now get is that there are no missing values.
# 
# 

# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# > <font color='cyan'><font>Roughly no data is missing.
# 
# Let's continue on by visualizing some more of the data! Check out the video for full explanations over these plots, this code is just to serve as reference.

# ## Data Exploration

# ### Counts How many have diabetes or not

# In[ ]:


#Renaming Diabetes type from int to string for better consistency
df['Diabetes_012_str'] = df['Diabetes_012'].replace({0.0:'Healthy', 1.0:'Pre-diabetic', 2.0:'Diabetic'})


# > 213703 persons out of 253680 are ``Healthy``; 35346 have ``diabetic`` and rest of 4631 have ``Pre-diabetic`` phase. 

# In[ ]:


countHealthy = len(df[df.Diabetes_012 == 0])
countHavePreDiabetic = len(df[df.Diabetes_012 == 1])
countDiabteic = len(df[df.Diabetes_012 == 2])
print("Percentage of Patients Are Healthy: {:.2f}%".format((countHealthy / (len(df.Diabetes_012))*100)))
print("Percentage of Patients Have Pre-Diabetic: {:.2f}%".format((countHavePreDiabetic / (len(df.Diabetes_012))*100)))
print("Percentage of Patients Have Diabetic: {:.2f}%".format((countDiabteic / (len(df.Diabetes_012))*100)))


# > * We have 213703 persons out of 253680 are Healthy; 35346 have diabetic and rest of 4631 have Pre-diabetic phase, so our problem is imbalanced.

# In[ ]:


# countplot----Plot the frequency of the Diabetes_012

fig1, ax1 = plt.subplots(1,2,figsize=(8,8))

#It shows the count of observations in each categorical bin using bars

sns.countplot(df['Diabetes_012'],ax=ax1[0])

#Find the % of diabetic and Healthy person

labels = 'Healthy','Diabetic', 'Pre-Diabetic'

df.Diabetes_012.value_counts().plot.pie(labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)


# ### Plots to analyze the Dataset 

# In[ ]:


# Histogram 

df.hist(figsize=(30,20))


# > From the above HIstogram we can easily visualize that these ```'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age','Education', 'Income'```  variables are continuous and the rest of the variable are dicrete data. 

# ### **Correlation with Target**

# In[ ]:


df_corr = df.corr().transpose()
df_corr


# In[ ]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(22, 10))
ax = sns.heatmap(corr_matrix,annot=True,linewidths=0.5,fmt=".2f",cmap="YlGn");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5);


# The correlation plot shows the relation between the parameters.
# 

# In[ ]:


df.drop('Diabetes_012', axis=1).corrwith(df.Diabetes_012).plot(kind='bar', grid=True, figsize=(20, 8), title="Correlation with target",color="lightgreen");


# ---
# ***Observations from correlation:***
# - *``AnyHealthCare``, ``NoDocbcCost`` and ``Sex`` are the least correlated with the target variable.*
# - *All other variables have a significant correlation with the target variable.*
# ---
# 

# ### Relationship Between Age vs Diabetes

# In[ ]:


def mean_target(var):
    """
    A function that will return the mean values for 'var' column depending on whether the person
    is diabetic or not
    """
    return pd.DataFrame(df.groupby('Diabetes_012_str').count()[var])


# In[ ]:


sns.boxplot(x = 'Diabetes_012_str', y = 'Age', data = df)
plt.title('Age vs Diabetes_012_str')
plt.show()


# > We know that as the age increases, the chances of diabetes also commonly increases. From above we can say, the median of the age of diabetic people is greater than that of non-diabetic people.

# In[ ]:


pd.crosstab(df.Age,df.Diabetes_012_str).plot(kind="bar",figsize=(20,6))
plt.title('Diabetes Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


pd.crosstab(df["Diabetes_012_str"], df["Age"])


# In[ ]:


mean_target("Age")


# > As we can see, age has a significant impact on diabetic disease, with the greatest impact occurring between the ages of 9 and 10.

# ### Relationship Between Smoker vs Diabetes

# In[ ]:


pd.crosstab(df.Smoker, df.Diabetes_012_str).plot(kind = 'bar')
plt.title("Diabetes Frequency for Smoker")


# In[ ]:


mean_target("Smoker")


# In[ ]:


pd.crosstab(df["Diabetes_012_str"], df["Smoker"])


# > According to the table, persons with diabetes are the most likely to smoke.

# ### Relationship Between Sex and Diabetes

# In[ ]:


pd.crosstab(df.Sex,df.Diabetes_012_str).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111', '#FFA500' ])
plt.title('Diabetes Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Have Diabetic", "Healthy", "Have Pre-Diabetic"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


mean_target("Sex")


# In[ ]:


pd.crosstab(df["Diabetes_012_str"], df["Sex"])


# > We can see that women are not just more diabetic than men, but they are also healthier.

# ### Relationship Between HighBP and Diabetes

# In[ ]:


pd.crosstab(df.HighBP,df.Diabetes_012_str).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111', '#FFA500' ])
plt.title('Diabetes Disease Frequency for HighBP')
plt.xlabel('HighBP')
plt.xticks(rotation=0)
plt.legend(["Have Diabetic", "Healthy", "Have Pre-Diabetic"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


mean_target("HighBP")


# In[ ]:


pd.crosstab(df["Diabetes_012_str"], df["HighBP"])


# > People with high blood pressure are more likely to have diabetes.

# ### Relationship Between HighChol and Diabetes

# In[ ]:


pd.crosstab(df.HighChol,df.Diabetes_012_str).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111', '#FFA500' ])
plt.title('Diabetes Disease Frequency for HighChol')
plt.xlabel('HighChol')
plt.xticks(rotation=0)
plt.legend(["Have Diabetic", "Healthy", "Have Pre-Diabetic"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


mean_target("HighChol")


# In[ ]:


pd.crosstab(df["Diabetes_012_str"], df["HighChol"])


# > People with high high cholesterol are more likely to have diabetes.

# ### Relationship Between BMI and Diabetes
# 
# 

# In[ ]:


pd.crosstab(df.BMI,df.Diabetes_012_str).plot(kind="bar",figsize=(30,12),color=['#1CA53B','#AA1111', '#FFA500' ])
plt.title('Diabetes Disease Frequency for BMI')
plt.xlabel('BMI')
plt.xticks(rotation=0)
plt.legend(["Have Diabetic", "Healthy", "Have Pre-Diabetic"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


pd.crosstab(df["Diabetes_012_str"], df["BMI"])


# In[ ]:


mean_target("BMI")


# > As we can see people range between 24-33 BMI have more likely to have Diabetic.

# ### Relationship Between MentlHlth and Diabetes
# 
# 

# In[ ]:


pd.crosstab(df.MentHlth,df.Diabetes_012_str).plot(kind="bar",figsize=(30,12),color=['#1CA53B','#AA1111', '#FFA500' ])
plt.title('Diabetes Disease Frequency for MentHlth')
plt.xlabel('MentHlth')
plt.xticks(rotation=0)
plt.legend(["Have Diabetic", "Healthy", "Have Pre-Diabetic"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


pd.crosstab(df["Diabetes_012_str"], df["BMI"])


# In[ ]:


mean_target("MentHlth")


# > From figure we can say that Menthlth Group 0-5 have impact on Diabetic

# ### Relationship Between Income and Diabetes
# 
# 

# In[ ]:


pd.crosstab(df.Income,df.Diabetes_012_str).plot(kind="bar",figsize=(30,12),color=['#1CA53B','#AA1111', '#FFA500' ])
plt.title('Diabetes Disease Frequency for Income')
plt.xlabel('Income')
plt.xticks(rotation=0)
plt.legend(["Have Diabetic", "Healthy", "Have Pre-Diabetic"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


mean_target("Income")


# In[ ]:


pd.crosstab(df["Diabetes_012_str"], df["Income"])


# >  We can deduce from the table that persons with greater income have more Diabetic than those with lower income.

# In[ ]:


df.groupby('Diabetes_012_str').mean()


# ### In this section we will see the continuous variable have *Outliers* or not.

# In[ ]:


plt.figure(figsize = (15,15))
for i,col in enumerate(['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age','Education', 'Income']):
    plt.subplot(4,2,i+1)
    sns.boxplot(x = col, data = df)
plt.show()


# > As we can see from the Boxplot ```BMI, GentHlth, MenHlth, PhysHlth``` have some outliers.

# > As we can see, the smoker have high effect on Diabetes disease.

# In[ ]:


mean_target('MentHlth')


# > The value demonstrates that there is a similar distribution of mental health severity across the diabetes spectrum, implying that mental health does not offer value in terms of clinical scenario.
# 

# ## Conclusion from EDA
# 
# 
# 
# 1.   The dataset contains 7 continuos feature variable and 15 Discrete type features variable.
# 2.   The target feature is ``` Diabetes_012 ```.
# 3.   ```float64``` is the data types of our features.
# 4.   The parameters do not contain any null values (missing values).
# 5.   The ```Diabetes_012``` feature shows that there are 213703 persons out of 253680 are Healthy; 35346 have diabetic and rest of 4631 have Pre-diabetic phase. It means 84.24% people are Healthy
# Percentage of Patients Have Pre-Diabetic: 1.83% people are Pre-Diabetic Phase, and 13.93% people Have Diabetic.
# 6.   There are very less number of outliers in all features.
# 7.   There is no apparent linear correlation between feature variable according to the heatmap.
# 8. With age increases, the chances of diabetes also increases.
# 9. persons with diabetes are the most likely to smoke.
# 10. women are not just more diabetic than men, but they are also healthier.
# 11. People with high blood pressure are more likely to have diabetes.
# 12. People with high high cholesterol are more likely to have diabetes.
# 13. people range between 24-33 BMI have more likely to have Diabetic.
# 14. Menthlth Group 0-5 have impact on Diabetic
# 15.  persons with greater income have more Diabetic than those with lower income.

# <font color='red'> # Thank You </font>
