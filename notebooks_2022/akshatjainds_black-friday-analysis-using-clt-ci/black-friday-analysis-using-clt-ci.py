#!/usr/bin/env python
# coding: utf-8

# In[ ]:





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


# ### Objective 
# - The purpose of this notebook is to cover analysis based on the Central Limit Theorem and CI

# ### Importing the required libraries or packages for EDA 

# In[ ]:


#Importing packages
import numpy as np
import pandas as pd

# Importing matplotlib and seaborn for graphs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import kstest
import statsmodels.api as sm

# Importing Date & Time util modules
from dateutil.parser import parse

import statistics
from scipy.stats import norm


# ## Utility Functions - Used during Analysis

# ### Missing Value - Calculator

# In[ ]:


def missingValue(df):
    #Identifying Missing data. Already verified above. To be sure again checking.
    total_null = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
    print("Total records = ", df.shape[0])

    md = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
    return md


# ### Categorical Variable Analysis 
#   - Bar plot - Frequency of feature in percentage
#   - Pie Chart

# In[ ]:


# Frequency of each feature in percentage.
def cat_analysis(df, colnames, nrows=2,mcols=2,width=20,height=30, sortbyindex=False):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height))  
    fig.set_facecolor(color = 'lightgrey')
    string = "Frequency of "
    rows = 0                          
    for colname in colnames:
        count = (df[colname].value_counts(normalize=True)*100)
        string += colname + ' in (%)'
        if sortbyindex:
                count = count.sort_index()
        count.plot.bar(color=sns.color_palette("Paired"),ax=ax[rows][0])
        ax[rows][0].set_ylabel(string, fontsize=14,family = "Comic Sans MS")
        ax[rows][0].set_xlabel(colname, fontsize=14,family = "Comic Sans MS")      
        count.plot.pie(colors = sns.color_palette("Paired"),autopct='%0.0f%%',
                       textprops={'fontsize': 14,'family':"Comic Sans MS"},ax=ax[rows][1])        
        string = "Frequency of "
        rows += 1


# ### Frequency Graph in percentage

# In[ ]:


# Frequency of each feature in percentage.
def bar_plot_percentage(df, colnames, sortbyindex=False):
    fig = plt.figure(figsize=(32, 36))
    fig.set_facecolor("lightgrey")
    string = "Frequency of "
    for colname in colnames:
        plt.subplot(5,2,colnames.index(colname)+1)
        count = (df[colname].value_counts(normalize=True)*100)
        string += colname + ' in (%)'
        if sortbyindex:
                count = count.sort_index()
        count.plot.bar(color=sns.color_palette('Paired'))
        plt.xticks(rotation = 70,fontsize=14,family="Comic Sans MS")
        plt.yticks(fontsize=14,family="Comic Sans MS")
        plt.ylabel(string, fontsize=14,family = "Comic Sans MS")
        plt.xlabel(colname, fontsize=14,family = "Comic Sans MS")
        string = "Frequency of "


# ### Bi-Varainte Analysis for Numerical and Categorical variables
#  - Used Box plot

# In[ ]:


def num_cat_bi(df,col_cat,col_num,nrows=1,mcols=2,width=15,height=6):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height),squeeze=False)
    sns.set(style='white')
    fig.set_facecolor("lightgrey")
    rows = 0
    i = 0
    while rows < nrows:
        sns.boxplot(x = col_cat[i],y = col_num, data = df,ax=ax[rows][0],palette="Paired")
        ax[rows][0].set_xlabel(col_cat[i], fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][0].set_ylabel(col_num,fontweight="bold", fontsize=14,family = "Comic Sans MS")
        i += 1
        sns.boxplot(x = col_cat[i],y = col_num, data = df,ax=ax[rows][1],palette="Paired")
        ax[rows][1].set_xlabel(col_cat[i], fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][1].set_ylabel(col_num,fontweight="bold", fontsize=14,family = "Comic Sans MS") 
        i += 1
        rows += 1
    plt.show()


# ### Distribution plot based on the Male and Female

# In[ ]:


def bar_M_vs_F(colname):
    fig = plt.figure(figsize=(16,6))

    male = retail_data_v1[retail_data_v1["Gender"]=='M'][colname].value_counts().reset_index()
    male["percentage"]  = (male[colname]*100/male[colname].sum())
    male["legends"]        = "Male"


    female = retail_data_v1[retail_data_v1["Gender"]=='F'][colname].value_counts().reset_index()
    female["percentage"] = (female[colname]*100/female[colname].sum())
    female["legends"]    = "Female"

    m_f_status = pd.concat([female,male],axis=0)

    ax = sns.barplot("index","percentage",data=m_f_status,hue="legends",palette="Blues_d")
    plt.xlabel(colname)
    fig.set_facecolor("white")
    plt.title(colname + "percentage in data with respect to churn status")
    plt.show()


# ### Multi-Varainte Analysis for Numerical and Categorical variables
#  - Used Box plot

# In[ ]:


def num_cat_bi_grpby(df,colname,category,groupby,nrows=1,mcols=2,width=18,height=6):
    fig , ax = plt.subplots(nrows,mcols,figsize=(width,height),squeeze=False)
    sns.set(style='white')
    fig.set_facecolor("lightgrey")
    rows = 0
    for var in colname:
        sns.boxplot(x = category,y = var,hue=groupby, data = df,ax=ax[rows][0],palette="Set3")
        sns.lineplot(x=df[category],y=df[var],ax=ax[rows][1],hue=df[groupby],palette="bright") 
        ax[rows][0].set_ylabel(var, fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][0].set_xlabel(category,fontweight="bold", fontsize=14,family = "Comic Sans MS")
        ax[rows][0].legend(loc='lower right')
        ax[rows][1].set_ylabel(var, fontweight="bold",fontsize=14,family = "Comic Sans MS")
        ax[rows][1].set_xlabel(category,fontweight="bold", fontsize=14,family = "Comic Sans MS") 
        rows += 1
    plt.show()


# ### Function for Booststrapping technique to calculate the CI

# In[ ]:


def bootstrapping(sample1,sample2,smp_siz=500,itr_size=5000,confidence_level=0.95,no_of_tails=2):
    
    smp1_means_m = np.empty(itr_size)
    smp2_means_m = np.empty(itr_size)
    for i in range(itr_size):
        smp1_n = np.empty(smp_siz)
        smp2_n = np.empty(smp_siz)
        smp1_n = np.random.choice(sample1, size = smp_siz,replace=True)
        smp2_n = np.random.choice(sample2, size = smp_siz,replace=True)
        smp1_means_m[i] = np.mean(smp1_n)
        smp2_means_m[i] = np.mean(smp2_n)
        
    #Calcualte the Z-Critical value
    alpha = (1 - confidence_level)/no_of_tails
    z_critical = stats.norm.ppf(1 - alpha)
        
    # Calculate the mean, standard deviation & standard Error of sampling distribution of a sample mean
    mean1  = np.mean(smp1_means_m)
    sigma1 = statistics.stdev(smp1_means_m)
    sem1   = stats.sem(smp1_means_m)
    
    lower_limit1 = mean1 - (z_critical * sigma1)
    upper_limit1 = mean1 + (z_critical * sigma1)
    
    # Calculate the mean, standard deviation & standard Error of sampling distribution of a sample mean
    mean2  = np.mean(smp2_means_m)
    sigma2 = statistics.stdev(smp2_means_m)
    sem2   = stats.sem(smp2_means_m)
    
    lower_limit2 = mean2 - (z_critical * sigma2)
    upper_limit2 = mean2 + (z_critical * sigma2)
        
    fig, ax = plt.subplots(figsize=(14,6))
    sns.set_style("darkgrid")
    
    sns.kdeplot(data=smp1_means_m,color="#467821",fill=True,linewidth=2)
    sns.kdeplot(data=smp2_means_m,color='#e5ae38',fill=True,linewidth=2)
    
    label_mean1=("μ (Males) :  {:.2f}".format(mean1))
    label_ult1=("Lower Limit(M):  {:.2f}\nUpper Limit(M):   {:.2f}".format(lower_limit1,upper_limit1))
    label_mean2=("μ (Females):  {:.2f}".format(mean2))
    label_ult2=("Lower Limit(F):  {:.2f}\nUpper Limit(F):   {:.2f}".format(lower_limit2,upper_limit2))
    
    plt.title(f"Sample Size: {smp_siz}, Male Avg: {np.round(mean1, 2)}, Male SME: {np.round(sem1,2)},Female Avg:{np.round(mean2, 2)}, Female SME: {np.round(sem2,2)}",
              fontsize=14,family = "Comic Sans MS")
    plt.xlabel('Purchase')
    plt.axvline(mean1, color = 'y', linestyle = 'solid', linewidth = 2,label=label_mean1)
    plt.axvline(upper_limit1, color = 'r', linestyle = 'solid', linewidth = 2,label=label_ult1)
    plt.axvline(lower_limit1, color = 'r', linestyle = 'solid', linewidth = 2)
    plt.axvline(mean2, color = 'b', linestyle = 'dashdot', linewidth = 2,label=label_mean2)
    plt.axvline(upper_limit2, color = '#56B4E9', linestyle = 'dashdot', linewidth = 2,label=label_ult2)
    plt.axvline(lower_limit2, color = '#56B4E9', linestyle = 'dashdot', linewidth = 2)
    plt.legend(loc='upper right')

    plt.show()
    
    return smp1_means_m,smp2_means_m ,np.round(lower_limit1,2),np.round(upper_limit1,2),np.round(lower_limit2,2),np.round(upper_limit2,2)


# In[ ]:


def bootstrapping_m_vs_um(sample1,sample2,smp_siz=500,itr_size=5000,confidence_level=0.95,no_of_tails=2):
    
    smp1_means_m = np.empty(itr_size)
    smp2_means_m = np.empty(itr_size)
    for i in range(itr_size):
        smp1_n = np.empty(smp_siz)
        smp2_n = np.empty(smp_siz)
        smp1_n = np.random.choice(sample1, size = smp_siz,replace=True)
        smp2_n = np.random.choice(sample2, size = smp_siz,replace=True)
        smp1_means_m[i] = np.mean(smp1_n)
        smp2_means_m[i] = np.mean(smp2_n)
        
    #Calcualte the Z-Critical value
    alpha = (1 - confidence_level)/no_of_tails
    z_critical = stats.norm.ppf(1 - alpha)
        
    # Calculate the mean, standard deviation & standard Error of sampling distribution of a sample mean
    mean1  = np.mean(smp1_means_m)
    sigma1 = statistics.stdev(smp1_means_m)
    sem1   = stats.sem(smp1_means_m)
    
    lower_limit1 = mean1 - (z_critical * sigma1)
    upper_limit1 = mean1 + (z_critical * sigma1)
    
    # Calculate the mean, standard deviation & standard Error of sampling distribution of a sample mean
    mean2  = np.mean(smp2_means_m)
    sigma2 = statistics.stdev(smp2_means_m)
    sem2   = stats.sem(smp2_means_m)
    
    lower_limit2 = mean2 - (z_critical * sigma2)
    upper_limit2 = mean2 + (z_critical * sigma2)
        
    fig, ax = plt.subplots(figsize=(14,6))
    sns.set_style("darkgrid")
    
    sns.kdeplot(data=smp1_means_m,color="#467821",fill=True,linewidth=2)
    sns.kdeplot(data=smp2_means_m,color='#e5ae38',fill=True,linewidth=2)
    
    label_mean1=("μ (Married) :  {:.2f}".format(mean1))
    label_ult1=("Lower Limit(M):  {:.2f}\nUpper Limit(M):   {:.2f}".format(lower_limit1,upper_limit1))
    label_mean2=("μ (Unmarried):  {:.2f}".format(mean2))
    label_ult2=("Lower Limit(F):  {:.2f}\nUpper Limit(F):   {:.2f}".format(lower_limit2,upper_limit2))
    
    plt.title(f"Sample Size: {smp_siz}, Married Avg: {np.round(mean1, 2)}, Married SME: {np.round(sem1,2)},Unmarried Avg:{np.round(mean2, 2)}, Unmarried SME: {np.round(sem2,2)}",
              fontsize=14,family = "Comic Sans MS")
    plt.xlabel('Purchase')
    plt.axvline(mean1, color = 'y', linestyle = 'solid', linewidth = 2,label=label_mean1)
    plt.axvline(upper_limit1, color = 'r', linestyle = 'solid', linewidth = 2,label=label_ult1)
    plt.axvline(lower_limit1, color = 'r', linestyle = 'solid', linewidth = 2)
    plt.axvline(mean2, color = 'b', linestyle = 'dashdot', linewidth = 2,label=label_mean2)
    plt.axvline(upper_limit2, color = '#56B4E9', linestyle = 'dashdot', linewidth = 2,label=label_ult2)
    plt.axvline(lower_limit2, color = '#56B4E9', linestyle = 'dashdot', linewidth = 2)
    plt.legend(loc='upper right')

    plt.show()
    
    return smp1_means_m,smp2_means_m ,np.round(lower_limit1,2),np.round(upper_limit1,2),np.round(lower_limit2,2),np.round(upper_limit2,2)


# In[ ]:


def bootstrapping_age(sample,smp_siz=500,itr_size=5000,confidence_level=0.95,no_of_tails=2):
    
    smp_means_m = np.empty(itr_size)
    for i in range(itr_size):
        smp_n = np.empty(smp_siz)
        smp_n = np.random.choice(sample, size = smp_siz,replace=True)
        smp_means_m[i] = np.mean(smp_n)
        
    #Calcualte the Z-Critical value
    alpha = (1 - confidence_level)/no_of_tails
    z_critical = stats.norm.ppf(1 - alpha)
        
    # Calculate the mean, standard deviation & standard Error of sampling distribution of a sample mean
    mean  = np.mean(smp_means_m)
    sigma = statistics.stdev(smp_means_m)
    sem   = stats.sem(smp_means_m)
    
    lower_limit = mean - (z_critical * sigma)
    upper_limit = mean + (z_critical * sigma)
       
    fig, ax = plt.subplots(figsize=(14,6))
    sns.set_style("darkgrid")
    
    sns.kdeplot(data=smp_means_m,color="#7A68A6",fill=True,linewidth=2)
    
    label_mean=("μ :  {:.2f}".format(mean))
    label_ult=("Lower Limit:  {:.2f}\nUpper Limit:   {:.2f}".format(lower_limit,upper_limit))
    
    plt.title(f"Sample Size: {smp_siz},Mean:{np.round(mean,2)}, SME:{np.round(sem,2)}",fontsize=14,family="Comic Sans MS")
    plt.xlabel('Purchase')
    plt.axvline(mean, color = 'y', linestyle = 'solid', linewidth = 2,label=label_mean)
    plt.axvline(upper_limit, color = 'r', linestyle = 'solid', linewidth = 2,label=label_ult)
    plt.axvline(lower_limit, color = 'r', linestyle = 'solid', linewidth = 2)
    plt.legend(loc='upper right')

    plt.show()
    
    return smp_means_m ,np.round(lower_limit,2),np.round(upper_limit,2)


# ## Exploratory Data Analysis

# ### Loading and inspecting the Dataset

# In[ ]:


retail_data = pd.read_csv('../input/black-friday/train.csv')
retail_data.head()


# #### Checking Shape and Column names

# In[ ]:


retail_data.shape


# In[ ]:


retail_data.columns


# #### Validating Duplicate Records

# In[ ]:


retail_data.duplicated().sum()


# ### Inference
#   - No dupicates records found.

# #### Missing Data Analysis

# In[ ]:


missingValue(retail_data).head(5)


# ### Inference
#   - No missing value found.

# #### Unique values (counts) for each Feature

# In[ ]:


retail_data.nunique()


# ### Inference
#  - The total number of records exceeds five million but the UserIDs are only 5891, meaning customers have visited multiple times in order to buy products.

# ### A deep dive into User ID
#  - Based on 5891 user IDs, how many are married, male or female, or the age of the users.

# In[ ]:


retail_data.groupby(['Gender'])['User_ID'].nunique()


# In[ ]:


print("Females are ", 1666/5891)
print("Females are ", 4225/5891)


# ### Inference
#  - The percentage of women customers is only 28%
#  - Around 72% of customers are male

# In[ ]:


retail_data.groupby(['Age'])['User_ID'].nunique()


# In[ ]:


retail_data.groupby(['City_Category'])['User_ID'].nunique()


# In[ ]:


retail_data.groupby(['Stay_In_Current_City_Years'])['User_ID'].nunique()


# In[ ]:


retail_data.groupby(['Marital_Status'])['User_ID'].nunique()


# ### Unique values (names) are checked for each Features

# In[ ]:


colname = ['Gender','Age','City_Category','Stay_In_Current_City_Years','Marital_Status','Occupation']
for col in colname:
    print("\nUnique values of ",col," are : ",list(retail_data[col].unique()))


# ### Inferences 
#  - All the values looks good.

# ### DataType Validation

# In[ ]:


retail_data.info()


# ### Inferencce
#  - **'User_ID','Product_ID','Gender', 'Age','City_Category','Marital_Status'** are categorical variables. As a result, we need to change the datatype to category.

# In[ ]:


cols = ['User_ID','Product_ID','Gender', 'Age','City_Category','Marital_Status']
for col_name in cols:
    retail_data[col_name] = retail_data[col_name].astype("category")


# In[ ]:


retail_data.info()


# ### Basic Statistics Analysis - count, min, max, and mean

# In[ ]:


retail_data.describe().T


# In[ ]:


retail_data.describe(include=['object','category']).T


# In[ ]:


retail_data.groupby(['Gender'])['Purchase'].describe()


# In[ ]:


retail_data.groupby(['Marital_Status'])['Purchase'].describe()


# In[ ]:


retail_data.groupby(['Age'])['Purchase'].describe()


# In[ ]:


retail_data.groupby(['City_Category'])['Purchase'].describe()


# In[ ]:


retail_data.groupby(['City_Category'])['User_ID'].nunique()


# ### Inferences
#  - There are more single people than married people.
#  - Most mall customers are between the ages of **26 and 35**.
#  - The majority of our customers come from **city category B ** but customers come from **City category C spent more as mean is 9719**.
#  - Male customers tend to spend more than female customers, as the mean is higher for male customers.
#  - The majority of users come from **City Category C, but more people from City Category B tend to purchase**, which suggests the same users visit the mall multiple times in City Category B.

# ### Correlation Analysis

# In[ ]:


plt.figure(figsize = (10, 7))
ax = sns.heatmap(retail_data.corr(),
            annot=True,cmap='Greens',square=True)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=40,fontsize=16,family = "Comic Sans MS",
    horizontalalignment='right')

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,fontsize=16,family = "Comic Sans MS",
    horizontalalignment='right')
    
plt.show()


# In[ ]:


sns.pairplot(retail_data)


# ### Inference
#   - Mostly features are categorical and not much correlation can be observed from above graphs.

# ## Univariante Analysis
#   - Numerical Variables
#   - Categorial variables 

# ### Numerical Variables - Outlier detection

# In[ ]:


retail_data.columns


# In[ ]:


# Visualizing our dependent variable for Outliers and Skewness
fig = plt.figure(figsize=(15,5))
fig.set_facecolor("lightgrey")

plt.subplot(1,2,1)
sns.boxplot(retail_data["Purchase"],color='m')
plt.title("Boxplot for outliers detection", fontweight="bold",fontsize=14)
plt.xlabel('Purchase', fontsize=12,family = "Comic Sans MS")

plt.subplot(1,2,2)
sns.distplot(retail_data["Purchase"],color='y')

plt.title("Distribution plot for skewness", fontweight="bold",fontsize=14)
plt.ylabel('Density', fontsize=12,family = "Comic Sans MS")
plt.xlabel('Purchase', fontsize=12,family = "Comic Sans MS")
plt.axvline(retail_data["Purchase"].mean(),color="g")
plt.axvline(retail_data["Purchase"].median(),color="b")
plt.axvline(retail_data["Purchase"].mode()[0],color="r")

plt.show()


# ### Inferences
#  - Above graphs ;ooks like "right-skewed distribution" which means the mass of the distribution is concentrated on the left of the figure.
#  - **Majority of Customers** purchase within the **5,000 - 20,000** range.

# ### Handling outliers

# In[ ]:


retail_data_v1 = retail_data.copy()


# In[ ]:


#Outlier Treatment: Remove top 5% & bottom 1% of the Column Outlier values
Q3 = retail_data_v1['Purchase'].quantile(0.75)
Q1 = retail_data_v1['Purchase'].quantile(0.25)
IQR = Q3-Q1
retail_data_v1 = retail_data_v1[(retail_data_v1['Purchase'] > Q1 - 1.5*IQR) & (retail_data_v1['Purchase'] < Q3 + 1.5*IQR)]


# In[ ]:


# Visualizing our dependent variable for Outliers and Skewness
fig = plt.figure(figsize=(15,5))
fig.set_facecolor("lightgrey")

plt.subplot(1,2,1)
sns.boxplot(retail_data_v1["Purchase"],color='m')
plt.title("Boxplot for outliers detection", fontweight="bold",fontsize=14)
plt.xlabel('Purchase', fontsize=12,family = "Comic Sans MS")

plt.subplot(1,2,2)
sns.distplot(retail_data_v1["Purchase"],color='y')

plt.title("Distribution plot for skewness", fontweight="bold",fontsize=14)
plt.ylabel('Density', fontsize=12,family = "Comic Sans MS")
plt.xlabel('Purchase', fontsize=12,family = "Comic Sans MS")
plt.axvline(retail_data_v1["Purchase"].mean(),color="g")
plt.axvline(retail_data_v1["Purchase"].median(),color="b")
plt.axvline(retail_data_v1["Purchase"].mode()[0],color="r")

plt.show()


# ### Categorical variable Uni-variante Analysis

# In[ ]:


cat_colnames = ['Gender','Age','City_Category','Stay_In_Current_City_Years','Marital_Status']
cat_analysis(retail_data_v1,cat_colnames,5,2,12,32)


# ### Inferences
#   - Males clearly purchase more than females. **75%** of men and only **25%** of women purchase products.
#   - **60%** of purchases are made by people between the ages of 26 and 45
#   - City Category **B accounts for 42%**, City Category **C 31%**, and City Category **A represents 27%** of all customer purchases.

# In[ ]:


retail_data.columns


# In[ ]:


col_cat = ['Gender', 'Age','City_Category','Marital_Status']
num_cat_bi(retail_data_v1,col_cat,'Purchase',2,2,15,12)


# In[ ]:


col_num = [ 'Purchase']
num_cat_bi_grpby(retail_data_v1,col_num,"City_Category",'Gender')


# In[ ]:


col_num = [ 'Purchase']
num_cat_bi_grpby(retail_data_v1,col_num,"Marital_Status",'Gender')


# In[ ]:


col_num = [ 'Purchase']
num_cat_bi_grpby(retail_data_v1,col_num,"Age",'Gender')


# In[ ]:


col_num = [ 'Purchase']
num_cat_bi_grpby(retail_data_v1,col_num,"Marital_Status",'Gender')


# In[ ]:


col_num = [ 'Purchase']
num_cat_bi_grpby(retail_data_v1,col_num,"Age",'City_Category')


# In[ ]:


col_num = [ 'Purchase']
num_cat_bi_grpby(retail_data_v1,col_num,"Age",'Marital_Status')


# ### Inferences
#  - Purchases are high in city category C
#  - Purchase is the same for all age groups
#  - Most of the customers are 55+ and live in city category B
#  - City category C has more customers between the ages of 18 and 45.

# In[ ]:


bar_M_vs_F('City_Category')


# In[ ]:


bar_M_vs_F('Age')


# In[ ]:


bar_M_vs_F('Stay_In_Current_City_Years')


# ### Inferences 
#  - In City Category C, there are slightly more female customers.

# In[ ]:


print(retail_data_v1.groupby(['Gender','City_Category'])['User_ID'].count())


# In[ ]:


fig = plt.figure(figsize=(25,10))
fig.set_facecolor("lightgrey")
sns.set(style='dark')
sns.displot(x= 'Purchase',data=retail_data_v1,hue='Gender',bins=25)
plt.show()


# ### Inference
#  - The amount of money spent by women is less than that spent by men

# In[ ]:


retail_data_v1.sample(500,replace=True).groupby(['Gender'])['Purchase'].describe()


# ### Inference
#  - Even the sample mean shows that males spend more than females.

# In[ ]:


retail_data_v1.groupby(['Gender'])['Purchase'].describe()


# ### Inference
#  - Given the sample size of 5.4 Million data for customer purhase history with 1.3M Females and 4.1 Males

# In[ ]:


retail_data_smp_male = retail_data_v1[retail_data_v1['Gender'] == 'M']['Purchase']
retail_data_smp_female = retail_data_v1[retail_data_v1['Gender'] == 'F']['Purchase']


# In[ ]:


print("Male Customers : ",retail_data_smp_male.shape[0])
print("Female Customers : ",retail_data_smp_female.shape[0])


# ## Calculate Confidence Interval (CI) - to estimate the mean weight of the expenses by female and male customers.

# ### Central limit Theorem
# The central limit theorem states that **the sampling distribution of a sample mean** is approximately **normal** if the sample size is large enough, even if the **population distribution is not normal.**

# ### Assumptions
# - **Randomization:** The data must be sampled randomly such that every member in a population has an equal probability of being selected to be in the sample.
# - **Independence:** The sample values must be independent of each other.
# - **The 10% Condition:** When the sample is drawn without replacement, the sample size should be no larger than 10% of the population.
# - **Large Sample Condition:** The sample size needs to be sufficiently large.

# ### Calculate CI using Bootstrapping
#  - We will be using Bootstrapping method to estimate the confidence interval of the population mean of the expenses by female and Male customers.

# ### Bootstrapping
# Bootstrapping is a method that can be used to estimate the standard error of any statistic and produce a confidence interval for the statistic.
# 
# The basic process for bootstrapping is as follows:
# 
# - Take k repeated samples with replacement from a given dataset.
# - For each sample, calculate the statistic you’re interested in.
# - This results in k different estimates for a given statistic, which you can then use to calculate the standard error of the statistic and create a confidence interval for the statistic.

# ### CLT Analysis for mean purchase with confidence 90% - Based on Gender
#  - Analysis of the true mean of purchase values by gender with a 90% confidence

# In[ ]:


itr_size = 1000
size_list = [1, 10, 30, 300, 1000, 100000]
ci = 0.90

array = np.empty((0,7))

for smp_siz in size_list:
    m_avg, f_avg, ll_m, ul_m, ll_f, ul_f = bootstrapping(retail_data_smp_male,retail_data_smp_female,smp_siz,itr_size,ci)

    array = np.append(array, np.array([['M', ll_m, ul_m, smp_siz, ([ll_m,ul_m]) ,(ul_m-ll_m),90]]), axis=0)
    array = np.append(array, np.array([['F', ll_f, ul_f, smp_siz, ([ll_f,ul_f]) ,(ul_f-ll_f),90]]), axis=0)

overlap = pd.DataFrame(array, columns = ['Gender','Lower_limit','Upper_limit','Sample_Size','CI','Range','Confidence_pct'])
print()


# In[ ]:


overlap.loc[(overlap['Gender'] == 'M') & (overlap['Sample_Size'] >= 300)]


# In[ ]:


overlap.loc[(overlap['Gender'] == 'F') & (overlap['Sample_Size'] >= 300)]


# ### Inferences
#  - As the sample size increases, the two groups start to become distinct 
#  - With increasing sample size, Standard error of the mean in the samples decreases.
#  - For sample size **100000 is 0.49**
#  - For Female (sample size 100000) range for mean purchase with confidence interval 90% is **[8645.68, 8696.14]**
#  - For Male range for mean purchase with confidence interval 90% is **[9341.03, 9393.94]**

# ### CLT Analysis for mean purchase with confidence 95% - Based on Gender
#  - Analysis of the true mean of purchase values by gender with a 95% confidence

# In[ ]:


itr_size = 1000
size_list = [1, 10, 30, 300, 1000, 100000]
ci = 0.95

array = np.empty((0,7))

for smp_siz in size_list:
    m_avg, f_avg, ll_m, ul_m, ll_f, ul_f = bootstrapping(retail_data_smp_male,retail_data_smp_female,smp_siz,itr_size,ci)

    array = np.append(array, np.array([['M', ll_m, ul_m, smp_siz, ([ll_m,ul_m]) ,(ul_m-ll_m),95]]), axis=0)
    array = np.append(array, np.array([['F', ll_f, ul_f, smp_siz, ([ll_f,ul_f]) ,(ul_f-ll_f),95]]), axis=0)

overlap_95 = pd.DataFrame(array, columns = ['Gender','Lower_limit','Upper_limit','Sample_Size','CI','Range','Confidence_pct'])
overlap = pd.concat([overlap, overlap_95], axis=0)


# In[ ]:


overlap_95.loc[(overlap_95['Gender'] == 'M') & (overlap_95['Sample_Size'] >= 300)]


# In[ ]:


overlap_95.loc[(overlap_95['Gender'] == 'F') & (overlap_95['Sample_Size'] >= 300)]


# ### Inferences
#  - Using confidence interval 95%, the mean purchase value by gender shows a similar pattern to that found with confidence interval 90%- 
#  - As the sample size increases, the Male and female groups start to become distinct 
#  - With increasing sample size, Standard error of the mean in the samples decreases. For sample size **100000 is 0.47**
#  - For Female (sample size 100000) range for mean purchase with confidence interval 90% is **[8642.58, 8701.58]**
#  - For Male range for mean purchase with confidence interval 95% is **[9336.23, 9397.53]**
#  - Overlappings are increasing with a confidence interval of 95%. Due to the increasing CI, we consider higher ranges within which the actual population might fall, so that both mean purchase are more likely to fall within the same range.

# ### CLT Analysis for mean purchase with confidence 99% - Based on Gender
#  - Analysis of the true mean of purchase values by gender with a 99% confidence.

# In[ ]:


itr_size = 1000
size_list = [1, 10, 30, 300, 1000, 100000]
ci = 0.99

array = np.empty((0,7))

for smp_siz in size_list:
    m_avg, f_avg, ll_m, ul_m, ll_f, ul_f = bootstrapping(retail_data_smp_male,retail_data_smp_female,smp_siz,itr_size,ci)

    array = np.append(array, np.array([['M', ll_m, ul_m, smp_siz, ([ll_m,ul_m]) ,(ul_m-ll_m),99]]), axis=0)
    array = np.append(array, np.array([['F', ll_f, ul_f, smp_siz, ([ll_f,ul_f]) ,(ul_f-ll_f),99]]), axis=0)

overlap_99 = pd.DataFrame(array, columns = ['Gender','Lower_limit','Upper_limit','Sample_Size','CI','Range','Confidence_pct'])
overlap = pd.concat([overlap, overlap_99], axis=0)


# In[ ]:


overlap_99.loc[(overlap_99['Gender'] == 'M') & (overlap_99['Sample_Size'] >= 300)]


# In[ ]:


overlap_99.loc[(overlap_99['Gender'] == 'F') & (overlap_99['Sample_Size'] >= 300)]


# In[ ]:


overlap.loc[(overlap['Gender'] == 'M') & (overlap['Sample_Size'] >= 10000)]


# In[ ]:


overlap.loc[(overlap['Gender'] == 'F') & (overlap['Sample_Size'] >= 10000)]


# ### Inferences
#  - Using confidence interval 99%, the mean purchase value by gender shows a similar pattern to that found with confidence interval 90% & 95%- 
#  - As the sample size increases, the Male and female groups start to become distinct 
#  - With increasing sample size, Standard error of the mean in the samples decreases. For sample size **100000 is 0.45**
#  - For Female (sample size 100000) range for mean purchase with confidence interval 99% is **[8634.54, 8707.85]**
#  - For Male range for mean purchase with confidence interval 90% is **[9328.03, 9409.07]**
#  - When the confidence percentage increases, the spread, that is the difference between the upper and lower limits, also increases. For Female Confidence percent as **[90,95,99]** have difference between the upper & lower limits as **[50.46,59,73.31]**

# ### Recommendations 
#  - In light of the fact that females spend less than males on average, management needs to focus on their specific needs differently. Adding some additional offers for women can increase their spending on Black Friday.

# ### Calculate Confidence Interval (CI) - to estimate the mean weight of the expenses by married and unmarried customers.¶

# ### CLT Analysis for mean purchase with confidence 99% - Based on Marital Status
#  - Analysis of the true mean of purchase values by marital Status with a 99% confidence.

# In[ ]:


retail_data_v1['Marital_Status'].replace(to_replace = 0, value = 'Unmarried', inplace = True)
retail_data_v1['Marital_Status'].replace(to_replace = 1, value = 'Married', inplace = True)


# In[ ]:


retail_data_v1.sample(500,replace=True).groupby(['Marital_Status'])['Purchase'].describe()


# In[ ]:


sns.displot(data = retail_data_v1, x = 'Purchase', hue = 'Marital_Status',bins = 25)
plt.show()


# In[ ]:


retail_data_smp_married = retail_data_v1[retail_data_v1['Marital_Status'] == 'Married']['Purchase']
retail_data_smp_unmarried = retail_data_v1[retail_data_v1['Marital_Status'] == 'Unmarried']['Purchase']


# In[ ]:


itr_size = 1000
size_list = [1, 10, 30, 300, 1000, 100000]
ci = 0.99

array = np.empty((0,7))

for smp_siz in size_list:
    m_avg, f_avg, ll_m, ul_m, ll_u, ul_u = bootstrapping_m_vs_um(retail_data_smp_married,retail_data_smp_unmarried,smp_siz,itr_size,ci)

    array = np.append(array, np.array([['Married', ll_m, ul_m, smp_siz, ([ll_m,ul_m]) ,(ul_m-ll_m),99]]), axis=0)
    array = np.append(array, np.array([['Unmarried', ll_u, ul_u, smp_siz, ([ll_u,ul_u]) ,(ul_u-ll_u),99]]), axis=0)

overlap = pd.DataFrame(array, columns = ['Marital_Status','Lower_limit','Upper_limit','Sample_Size','CI','Range','Confidence_pct'])


# In[ ]:


overlap.head()


# In[ ]:


overlap.loc[(overlap['Marital_Status'] == 'Married') & (overlap['Sample_Size'] >= 300)]


# In[ ]:


overlap.loc[(overlap['Marital_Status'] == 'Unmarried') & (overlap['Sample_Size'] >= 300)]


# ### Inference
# - Overlapping is evident for married vs single customer spend even when more samples are analyzed, which indicates that customers spend the same regardless of whether they are single or married.
# - For Unmarried customer (sample size 100000) range for mean purchase with confidence interval 99% is **[9162.0, 9241.98]**
# - For married customer range for mean purchase with confidence interval 90% is **[9148.09, 9227.05]**

# ### CLT Analysis for mean purchase with confidence 99% - Based on Age Group
#  - Analysis of the true mean of purchase values by Age Group with a 99% confidence.

# In[ ]:


itr_size = 1000
smp_size = 1000
ci = 0.99
age_list =['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'] 

array = np.empty((0,8))

for age in age_list:
    mean, ll_m, ul_m = bootstrapping_age(retail_data_v1[retail_data_v1['Age'] == age]['Purchase'],smp_siz,itr_size,ci)

    array = np.append(array, np.array([[age,np.round(mean,2), ll_m, ul_m, smp_siz, ([ll_m,ul_m]) ,(ul_m-ll_m),99]]), axis=0)

age_data = pd.DataFrame(array, columns = ['Age_Group','Mean','Lower_limit','Upper_limit','Sample_Size','CI','Range','Confidence_pct'])


# In[ ]:


age_data.head(7)


# ### Checking the Sampling distribution of a sample mean for each Age Group

# In[ ]:


age_dict = {}
age_list = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'] 
for i in range(len(age_data)):
    age_dict[age_list[i]] = age_data.loc[i, "Mean"]


# In[ ]:


fig, ax = plt.subplots(figsize=(14,6))
sns.set_style("darkgrid")
for label_val in age_dict.keys():
    sns.kdeplot(age_dict[label_val], shade = True, label = label_val)

plt.title("Sampling distribution of a sample mean for each Age",fontsize=14,family="Comic Sans MS")
plt.xlabel('Purchase')
plt.legend(loc='upper right')


# ### Inferences 
# - Spending by Age_group 0-17 is low compared to other age groups.
# - Customers in Age_group 51-55 spend the most between **9381.9 and 9463.7**

# ### Recommendations 
#  - Management should come-up with some games in the mall to attract more younger generation will can help them to increase the sale.
#  - The management should have some offers on kids (0-17 years) in order to increase sales. 
#  - In order to attract more young shoppers, they can offer some games for the younger generation.

# ## Inferences & Recommendations 

# ### Inferences 
# 
# #### Based on EDA
# 
#  - The majority of our customers come from **city category B ** but customers come from **City category C spent more as mean is 9719**.
#  - The majority of users come from **City Category C, but more people from City Category B tend to purchase**, which suggests the same users visit the mall multiple times in City Category B.
#  - **Majority of Customers** purchase within the **5,000 - 20,000** range.
#  - Males clearly purchase more than females. **75%** of men and only **25%** of women purchase products.
#  - Most mall customers are between the ages of **26 and 35**.**60%** of purchases are made by people between the ages of 26 and 45
#  - City Category **B accounts for 42%**, City Category **C 31%**, and City Category **A represents 27%** of all customer purchases.Purchases are high in city category C
#  - Most mall customers are between the ages of **26 and 35**.City category C has more customers between the ages of **18 and 45.**
#  - In City Category C, there are slightly more female customers. 
# 
# #### Based on Statistical Analysis (using CLT & CI
#  - As the sample size increases, the two groups start to become distinct. With increasing sample size, Standard error of the mean in the samples decreases. For sample size **100000 is 0.49** with confidence is 90%.
#  - Overlappings are increasing with a confidence interval of 95%. Due to the increasing CI, we consider higher ranges within which the actual population might fall, so that both mean purchase are more likely to fall within the same range.
#  - Using confidence interval 99%, the mean purchase value by gender shows a similar pattern to that found with confidence interval 90% & 95%
#  - For Female (sample size 100000) range for mean purchase with confidence interval 99% is **[8634.54, 8707.85]**
#  - For Male range for mean purchase with confidence interval 99% is **[9328.03, 9409.07]**
#  - When the confidence percentage increases, the spread, that is the difference between the upper and lower limits, also increases. For Female Confidence percent as **[90,95,99]** have difference between the upper & lower limits as **[50.46,59,73.31]**
#  - Overlapping is evident for married vs single customer spend even when more samples are analyzed, which indicates that customers spend the same regardless of whether they are single or married.
#  - Spending by Age_group 0-17 is low compared to other age groups.
# - Customers in Age_group 51-55 spend the most between **9381.9 and 9463.7**

# ### Recommendations
# 
#  - In light of the fact that females spend less than males on average, management needs to focus on their specific needs differently. Adding some additional offers for women can increase their spending on Black Friday.
#  - Management should come-up with some games in the mall to attract more younger generation will can help them to increase the sale.
#  - The management should have some offers on kids (0-17 years) in order to increase sales. 
#  - In order to attract more young shoppers, they can offer some games for the younger generation..

# ### Please leave an Upvote if you like this notebook!

# In[ ]:




